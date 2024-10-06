import os
import gguf
import torch
import numpy as np
from typing import Union

from tasks import load_task
from utils.pca import PCA

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.control_vectors.request import ControlVectorRequest
from transformers import AutoTokenizer, AutoModelForCausalLM

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


MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
cv_path = "/datadrive5/huypn16/ICV/controlvector.gguf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
SEED = 42

def sampling_icv(engine, text):
    prompts = [(text,
                SamplingParams(temperature=0.65,
                               max_tokens=100),
                ControlVectorRequest("chaotic", 1, cv_path, scale=1.0))]

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
    
    print(results)
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

def sampling(engine, text) -> Union[str, torch.Tensor]:
    # first prompt with a control vector and second without.
    prompts = [(text,SamplingParams(temperature=0.65,max_tokens=100), None)]
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
    sampled_text  = None
    sampled_hidden = None
    for result in results:
        if result[-1] is not None:
            prompt_hidden = result[-1]
        else:
            sampled_text = result[1]
            sampled_hidden = result[2]
    # assert prompt_hidden is not None and sampled_text is not None
    return sampled_text, prompt_hidden, sampled_hidden

def test_cv_adapter():
    engine_args = EngineArgs(model=MODEL_PATH, enable_control_vector=True, gpu_memory_utilization=0.95, return_hidden_states=True)
    engine = LLMEngine.from_engine_args(engine_args)
    n_samples = 4
    n_steps = 4
    directions = []
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        token="",
        torch_dtype="float16",
    )
    model.eval()
    TaskHandler = load_task("reward")
    task_agent = TaskHandler("1.0.0")
    task_agent.set_seed(SEED)
    steps = ["Solving the following mathematical problem. Problem: Calculate the following expression: (10222 + 23123123 * 4 - 1 ) * 5 - 6. Step 1:"]
    for _ in range(n_steps):
        hidden_states = []
        possible_steps = []
        for _ in range(n_samples):
            text, xs, ys = sampling(engine, "".join(steps))
            # xs: prompt_hidden_states, ys: sampled_hidden_states
            x, y = xs[-1], ys[-1] # taking the last token
            hidden_states.append((x, y))
            possible_steps.append(text)
            print("====================================")
            print(text)
            print("====================================")
        
        icvs = task_agent.obtain_icv_vllm(hidden_states)
        export_gguf(cv_path, icvs[0], model) # export the control vector for the first PCA axis only
        # resampling with ICV
        result = sampling_icv(engine, "".join(steps))
        print("Next step ICV:", result[1][1])
        steps.append(result[1][1])
    
    print(steps)
    
if __name__ == "__main__":
    test_cv_adapter()