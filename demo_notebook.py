import gc
import json
import os
import textwrap
from typing import Union
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import setup_env, mk_parser
from models import build_model_signature, build_tokenizer, build_model
from tasks import load_task
from utils.logger import tabular_pretty_print
from utils.tools import ensure_folder
from utils.pca import PCA
from utils.llm_layers import add_icv_layers, remove_icv_layers

import numpy as np
torch.cuda.is_available()

def tokenize_each_demonstration(demonstration_list, tokenizer, dataset_name=None, prefix = None):
    def strip_special_characters(input_string):
        # for char in special_characters:
        #     input_string = input_string.replace(char.strip(), '')
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

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.control_vectors.request import ControlVectorRequest
from transformers import AutoTokenizer

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
cv_path = "/datadrive5/huypn16/ICV/controlvector.gguf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def do_sample(engine):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_text = "Solving the following mathematical problem. Problem: Calculate the following expression: (10222 + 23123123 * 4 - 1 ) * 5 - 6. Step 1:" 
    prompt = tokenizer(prompt_text, return_tensors="pt")
    print(len(prompt[0]))
    
    # first prompt with a control vector and second without.
    # prompts = [(prompt_text,
    #             SamplingParams(temperature=0.65,
    #                            max_tokens=100),
    #             ControlVectorRequest("chaotic", 1, cv_path, scale=1.0)),
    prompts = [(prompt_text,
                SamplingParams(temperature=0.0,
                               max_tokens=100), None)]

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
        # if len(results) == 4:
        #     break
    return results

def sampling(engine, text) -> Union[str, torch.Tensor]:
    prompt = tokenizer(text, return_tensors="pt")
    print(len(prompt[0]))
    
    # first prompt with a control vector and second without.
    # prompts = [(prompt_text,
    #             SamplingParams(temperature=0.65,
    #                            max_tokens=100),
    #             ControlVectorRequest("chaotic", 1, cv_path, scale=1.0)),
    prompts = [(prompt_text,
                SamplingParams(temperature=0.0,
                               max_tokens=100), None)]

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
        # if len(results) == 4:
        #     break
    return results
    
    

def test_cv_adapter():
    engine_args = EngineArgs(model=MODEL_PATH, enable_control_vector=True, gpu_memory_utilization=0.95, return_hidden_states=True)
    engine = LLMEngine.from_engine_args(engine_args)
    result = do_sample(engine)
    print(list(result))
    print(len(result))

import gguf
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

SEED = 42

TaskHandler = load_task("reward")
task_agent = TaskHandler("1.0.0")
task_agent.set_seed(SEED)

n_comps = 8
# icv_reward_weighed = task_agent.get_reward(
#         model, tokenize_each_demonstration(
#             math_directions, tokenizer, prefix=("", "")
#             ),rewards=rewards, rank=n_comps
#         )

# icv_unweighted, _ = task_agent.obtain_icv(
#         model, tokenize_each_demonstration(
#             math_directions, tokenizer, prefix=("", "")
#             ), rank=n_comps
#         )
# print(len(icv_reward_weighed))
# print(len(icv_unweighted))
# print(icv_reward_weighed[0].shape)
# print(icv_unweighted[0].shape)

# query = tokenizer(prefix)

if __name__ == "__main__":
    print("I am stupid")
    test_cv_adapter()