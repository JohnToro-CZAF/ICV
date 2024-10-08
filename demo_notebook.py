import os
import re
import tqdm
import gguf
import torch
import numpy as np
from typing import Union

from tasks import load_task
from utils.pca import PCA

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.control_vectors.request import ControlVectorRequest
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from typing import Dict, List, Union
import concurrent

from sllm.value_functions.step_wise import StepWiseValueFunction
from sllm.envs.math.utils import is_equiv
from sllm.envs.base_env import BaseEnv

ANS_RE = re.compile("\\\\boxed{(.*)}")

class MATHEnv(BaseEnv):
    def __init__(self, config, logger):
        super().__init__()
        limit = config.limit
        prm_dataset = load_dataset('lighteval/MATH')
        test = prm_dataset['test']
        # suffle the dataset
        test = test.shuffle(seed=42)
        filter_test = []
        data_statistic = {
            "1": {}, 
            "2": {},
            "3": {},
            "4": {},
            "5": {}
        }
        for i in range(len(test)):
            full_ans = test[i]["solution"]
            match = ANS_RE.search(full_ans)
            if match:
                level = test[i]["level"].split(" ")[1]
                prob_type = test[i]["type"]
                if prob_type in data_statistic[level]:
                    data_statistic[level][prob_type] += 1
                else:
                    data_statistic[level][prob_type] = 1
                data = {
                    "problem": test[i]["problem"],
                    "solution": match.group(1).strip(),
                    "full_solution": full_ans,
                    "level": level
                }
                filter_test.append(data)
        self.prm_dataset = [problem for problem in filter_test if problem["level"] == str(config.level)][:limit]
        self.problem_idx = 0
        self.problem = None
        self.results = []
        self.logger = logger
        self.experiment_name = "default"

    @property
    def problems(self):
        return self.prm_dataset

    def set_problem(self, idx):
        self.problem_idx = idx
        self.problem = self.prm_dataset[idx]

    def get_problem(self, idx):
        return self.prm_dataset[idx]
    
    def get_answer(self, idx):
        return self.prm_dataset[idx]

    def get_problem_statement(self):
        return self.problem["problem"]

    def get_problem_idx(self):
        return self.problem_idx
    
    def check_extracted(self, extracted_solution, problem):
        if extracted_solution == "":
            return False
        return is_equiv(extracted_solution, problem["solution"])

    def extract_answer(self, proposed_solutions: Union[List[str], str]):
        answers = []
        if isinstance(proposed_solutions, str):
            solution = proposed_solutions
            answer_prefix = "The answer is: "
            if answer_prefix in solution:
                start = solution.find(answer_prefix) + len(answer_prefix)
                end_id = None
                for id in range(start, len(solution)):
                    if solution[id] == "к":
                        end_id = id
                        break
                final_answer = solution[start:end_id]
                final_answer = final_answer.replace("ки", "")
                final_answer = final_answer.strip()
            else:
                final_answer = ""
            return final_answer
        
        for solution in proposed_solutions:
            answer_prefix = "The answer is: "
            if answer_prefix in solution:
                start = solution.find(answer_prefix) + len(answer_prefix)
                end_id = None
                for id in range(start, len(solution)):
                    if solution[id] == "к":
                        end_id = id
                        break
                final_answer = solution[start:end_id]
                final_answer = final_answer.replace("ки", "")
                final_answer = final_answer.strip()
            else:
                final_answer = ""
            answers.append(final_answer)
        return answers
        
    def accumulate_result(self, result: Dict):
        self.results.append(result)
    
    def finalize(self):
        self.result = {"correct": sum([result["is_correct"] for result in self.results]), "total": len(self.results)}
        if "n_generated_token" in self.results[0]:
            self.result["n_generated_token"] = sum([result["n_generated_token"] for result in self.results])
            self.result["avg_generated_token"] = self.result["n_generated_token"] / self.result["total"]
    
class Config:
    def __init__(self):
        self.limit = 5000
        self.level = 5

def extract_answer(text: str) -> str:
    ANS_RE = re.compile("\\\\boxed{(.*)}")
    match = ANS_RE.search(text)
    answer = match.group(1) if match else ''
    return answer
    
cfg = Config()
env = MATHEnv(cfg, logger=None)

class RewardConfig:
    def __init__(self):
        self.abc = False
        self.model = "peiyi9979/math-shepherd-mistral-7b-prm"
        self.reward_device = "cuda:1"
        self.good_token = "+"
        self.bad_token = "-"
        self.value_token = "ки"

reward_model = StepWiseValueFunction(RewardConfig())

def compute_reward(problem, steps, current_step, value_token = "ки"):
    prompt = problem
    for idx, step in enumerate(steps):
        if idx == 0:
            prompt += (" " + step.strip() + f". {value_token} ")
        elif idx < len(steps) - 1:
            prompt += (step.strip() + f". {value_token} ")
        else:
            prompt += step.strip() + f" {value_token}"
    rewards = reward_model.compute([prompt])[0]
    return rewards[current_step:].mean()

def tokenize_each_demonstration(demonstration_list, tokenizer, dataset_name=None, prefix = None):
    def strip_special_characters(input_string):
        return input_string.strip()

    tokenized_demonstration_list = []
    for exp_id in range(len(demonstration_list)):
        if prefix is not None:
            demonstration_list[exp_id] = (prefix[0] + strip_special_characters(demonstration_list[exp_id][0]), prefix[1] + strip_special_characters(demonstration_list[exp_id][1]))
        else:
            demonstration_list[exp_id] = (strip_special_characters(demonstration_list[exp_id][0]), strip_special_characters(demonstration_list[exp_id][1]))
        e_original = tokenizer(demonstration_list[exp_id][0]) 
        e_rewrite = tokenizer(demonstration_list[exp_id][1])
        tokenized_demonstration_list.append((e_original, e_rewrite)) 
    return tokenized_demonstration_list


MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
cv_path = "/datadrive5/huypn16/ICV/controlvector.gguf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
SEED = 42

def sampling_icv(engine, steps):
    text = "".join(steps)
    text += f"### Step {len(steps)}: "
    prompts = [(text,
                SamplingParams(temperature=0.65,
                               max_tokens=600,
                               stop=["###"]),
                ControlVectorRequest("chaotic", 1, cv_path, scale=0.12))]

    request_id = 0
    results = set()
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, cv_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               control_vector_request=cv_request)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished or request_output.outputs[0].prompt_hidden_states != None:
                results.add((request_output.request_id, request_output.outputs[0].text, request_output.outputs[0].hidden_states.shape, request_output.outputs[0].prompt_hidden_states.shape if request_output.outputs[0].prompt_hidden_states != None else None))
    
    # print(results)
    return list(results)

def export_gguf(path: os.PathLike[str] | str, directions: list[torch.Tensor], model):
    arch = "controlvector"
    directions = [directions[i] for i in range(1, len(directions))]
    writer = gguf.GGUFWriter(path, arch)
    writer.add_string(f"{arch}.model_hint", model.config.model_type)
    writer.add_uint32(f"{arch}.layer_count", len(directions))
    for layer, _ in enumerate(directions):
        writer.add_tensor(f"direction.{layer}", directions[layer].numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

def sampling(engine, steps:List[str]) -> Union[str, torch.Tensor]:
    # first prompt with a control vector and second without.
    text  = "".join(steps)
    text += f"### Step {len(steps)}: "
    prompts = [(text,SamplingParams(temperature=0.65,max_tokens=700, stop=["###"]), None)]
    request_id = 0
    results = set()
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, cv_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               control_vector_request=cv_request)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished or request_output.outputs[0].prompt_hidden_states != None:
                results.add((request_output.request_id, request_output.outputs[0].text, request_output.outputs[0].hidden_states, request_output.outputs[0].prompt_hidden_states if request_output.outputs[0].prompt_hidden_states != None else None))
    prompt_hidden = None
    sampled_text  = ""
    sampled_hidden = None
    for result in results:
        if result[-1] is not None:
            prompt_hidden = result[-1]
        else:
            sampled_text = result[1]                                                    
            sampled_hidden = result[2]
    # assert prompt_hidden is not None and sampled_text is not None
    assert sampled_text is not None, f"Sampled text is None, results: {results}"
    # return f"### Step {len(steps)}: " + sampled_text, prompt_hidden, sampled_hidden
    return sampled_text, prompt_hidden, sampled_hidden

def multisampling(engine, steps:List[str], nsteps: int) -> Union[str, torch.Tensor]:
    # first prompt with a control vector and second without.
    text  = "".join(steps)
    text += f"### Step {len(steps)}: "
    prompts = [(text,SamplingParams(temperature=0.65,max_tokens=700, stop=["###"]), None) for _ in range(nsteps)]
    request_id = 0
    results = set()
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, cv_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               control_vector_request=cv_request)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished or request_output.outputs[0].prompt_hidden_states != None:
                results.add((request_output.request_id, request_output.outputs[0].text, request_output.outputs[0].hidden_states, request_output.outputs[0].prompt_hidden_states if request_output.outputs[0].prompt_hidden_states != None else None))
    
    responses = []
    for req_id in range(nsteps):
        req_results = []
        for res in results:
            if res[0] == str(req_id):
                req_results.append(res)
        # print("Req_id: ", req_id, "Req_results: ", req_results)
        prompt_hidden = None
        sampled_text  = ""
        sampled_hidden = None
        for result in req_results:
            if result[-1] is not None:
                prompt_hidden = result[-1]
            else:
                sampled_text = result[1]                                                    
                sampled_hidden = result[2]
        # assert prompt_hidden is not None and sampled_text is not None
        assert sampled_text is not None, f"Sampled text is None, results: {results}"
        # return f"### Step {len(steps)}: " + sampled_text, prompt_hidden, sampled_hidden
        responses.append((sampled_text, prompt_hidden, sampled_hidden))
    return responses

engine_args = EngineArgs(model=MODEL_PATH, enable_control_vector=True, gpu_memory_utilization=0.95, return_hidden_states=True, disable_log_stats=True)
engine = LLMEngine.from_engine_args(engine_args)
n_samples = 4
num_problems = 100
SEED = 42
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    token="",
    torch_dtype="float16",
)
model.eval()
TaskHandler = load_task("reward")
task_agent = TaskHandler("1.0.0")

example = """To solve for the perimeter of pentagon \\(ABCDP\\), we need to determine the lengths of all its sides. 
            We are given that \\(P\\) is the midpoint of \\(\\overline{BD}\\), \\(AP = BP = 4\\), \\(\\overline{AP} \\perp \\overline{BD}\\), \\(\\overline{BD} \\perp \\overline{DC}\\), and \\(\\overline{AB} \\perp \\overline{BC}\\).\n\n
            ### Step 1: Determine the coordinate
            ### Step 2: Determine the length of \\(\\overline{BD}\\)\nSince \\(P\\) is the midpoint, the coordinates of \\(P\\) being \\((4, 0)\\) imply:\n\\[\nP = \\left( \\frac{4 + 4}{2}, \\frac{y_B + (-y_B)}{2} \\right) = (4, 0)\n\\]\nThis confirms the midpoint calculation.\n\n
            ### Step 3: Determine the length of \\(\\overline{BD}\\)\nUsing the distance formula for \\(\\overline{BD}\\):\n\\[\nBD = \\sqrt{(4 - 4)^2 + (y_B - (-y_B))^2} = \\sqrt{0 + (2y_B)^2} = 2y_B\n\\]\n\nGivenes of points\nGiven that \\(AP = BP = 4\\) and \\(\\overline{AP} \\perp \\overline{BD}\\), we place point \\(A\\) at the origin \\((0, 0)\\) and point \\(P\\) at \\((4, 0)\\) since \\(AP\\) is horizontal and equal to 4.\n\nSince \\(P\\) is the midpoint of \\(\\overline{BD}\\), we denote the coordinates of \\(B\\) and \\(D\\) as follows:\n- Let \\(B = (4, y_B)\\)\n- Let \\(D = (4, -y_B)\\)\n\n \\(BP = 4\\):\n\\[\nBP = \\sqrt{(4 - 4)^2 + (y_B - 0)^2} = y_B = 4\n\\]\n\nThus, \\(BD = 2y_B = 2 \\times 4 = 8\\).\n\n
            ### Step 4: Determine the length of \\(\\overline{DC}\\)\nSince \\(\\overline{BD} \\perp \\overline{DC}\\), \\(D = (4, -4)\\) and \\(C\\) is directly below \\(D\\) with the same x-coordinate, we place \\(C\\) at \\((4, -8)\\).\n\n
            ### Step 5: Determine the length of \\(\\overline{BC}\\)\nUsing the distance formula for \\(\\overline{BC}\\):\n\\[\nBC = \\sqrt{(4 - 4)^2 + (-8 - 4)^2} = \\sqrt{0 + (-12)^2} = 12\n\\]\n\n
            ### Step 6: Determine the length of \\(\\overline{AB}\\)\nSince \\(\\overline{AB} \\perp \\overline{BC}\\) and \\(A = (0, 0)\\), \\(B = (4, 4)\\):\n\\[\nAB = \\sqrt{(4 - 0)^2 + (4 - 0)^2} = \\sqrt{16 + 16} = \\sqrt{32} = 4\\sqrt{2}\n\\]\n\n
            ### Step 7: Calculate the perimeter of pentagon \\(ABCDP\\)\nThe perimeter is the sum of all sides:\n\\[\nAB + BP + PD + DC + CA\n\\]\n\nWe already know:\n\\[\nAB = 4\\sqrt{2}, \\quad BP = 4, \\quad PD = 4, \\quad DC = 12, \\quad CA = 4\n\\]\n\nThus, the perimeter is:\n\\[\n4\\sqrt{2} + 4 + 4 + 12 + 4 = 4\\sqrt{2} + 24\n\\]\n\n
            ### Final Answer\n\\[\n\\boxed{4\\sqrt{2} + 24}\n\\]
            """
            
prompt = "You are a great mathematics solver that consider reflection, analogy and multiple approaches to solve the problem. Here is an example of a formatted solution to a math problem: \n\n"
solved_prompt = ".Solve the following math problem step-by-step. \n\nProblem: {problem} \n\nSolution:"
task_agent.set_seed(SEED)

def process_problem_icv_prefix(problem_id):
    problem = env.get_problem(problem_id)
    print("Problem: ", problem["problem"], "Solution: ", problem["solution"])
    steps = [prompt + example + solved_prompt.format(problem=problem["problem"]) + "\n\n"]
    step_prefix = "### Step "
    solved = False
    while not solved:
        hidden_states = []
        possible_steps = []
        # print("Steps: ", steps)
        
        # def sampling_func():
        #     text, xs, ys = sampling(engine, steps)
        #     return text, xs, ys
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(sampling_func) for _ in range(n_samples)]
        #     # Collect the results as they complete
        #     for future in concurrent.futures.as_completed(futures):
        #         result = future.result()
        #         if result:
        #             text, xs, ys = result
        #             possible_steps.append(text)
        #             x,y = xs[-1], ys[-1]
        #             hidden_states.append((x, y))
        import time
        t0 = time.time()
        u = multisampling(engine, steps, n_samples)
        for s in u:
            text, xs, ys = s
            # text, x, y = s
            possible_steps.append(text)
            x,y = xs[-1], ys[-1]
            # hidden_states.append((x,y))
            hidden_states.append(y)
        t1 = time.time()
        print("Time: ", t1 - t0)
        # for _ in range(n_samples):
        #     text, xs, ys = sampling(engine, steps)
        #     # xs: prompt_hidden_states, ys: sampled_hidden_states
        #     possible_steps.append(text)
        #     print("====================================")
        #     print(text)
        #     print("====================================")
        #     print(xs.shape)
        #     print(ys.shape)
        #     x, y = xs[-1], ys[-1] # taking the last token
        #     hidden_states.append((x, y))
        rewards = []
        for text in possible_steps:
            rewards.append(compute_reward(problem["problem"], steps + [text], -1).item())
        icvs = task_agent.obtain_icv_rewarded_vllm(hidden_states, rewards)
        # icvs = task_agent.obtain_icv_vllm(hidden_states)
        export_gguf(cv_path, icvs[0], model) # export the control vector for the first PCA axis only
        # resampling with ICV
        result = sampling_icv(engine, steps)
        # print("Result: ", result)
        idx = 0
        for id, res in enumerate(result):
            if res[-1] == None:
                idx = id
        if "\\box" in result[idx][1] or "box" in result[idx][1]:
            solved = True
            print("Solved")
        # print("Next step ICV:", result[idx][1])
        if result[idx][1] is not None:
            reward_steered = compute_reward(problem["problem"], steps + [result[idx][1]], -1).item()
            highest_reward_base = max(rewards)
            highest_reward_base_idx = rewards.index(highest_reward_base)
            
            steps.append(step_prefix + str(len(steps)) + ": " + result[idx][1])
            
            if reward_steered > highest_reward_base:
                print("Steered reward: ", reward_steered, "Highest base reward: ", highest_reward_base)
                print("We are steering the model to the right direction")
                # steps.append(step_prefix + str(len(steps)) + ": " + result[idx][1])
            else:
                print("Steered reward: ", reward_steered, "Highest base reward: ", highest_reward_base)
                print("We are steering to base step")
                # steps.append(step_prefix + str(len(steps)) + ": " + possible_steps[highest_reward_base_idx])
        else:
            steps.append(step_prefix + str(len(steps)) + ": " + possible_steps[0])
    
    answer = extract_answer("".join(steps[1:]))
    print("Answer: ", answer, "Solution: ", problem["solution"])
    correct = env.check_extracted(answer, problem)
    return int(correct)

if __name__ == "__main__":
    acc = 0
    for problem_id in tqdm.tqdm(range(num_problems)):
        correct = process_problem_icv_prefix(problem_id)
        acc += correct
        print(f"Accuracy: {acc / (problem_id+1)}")

    print(f"Accuracy: {acc / num_problems}")