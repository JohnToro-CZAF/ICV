import gc
import json
import os
import textwrap

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
    special_characters = [
        "~", " ~", "~ ", "!", " !", "! ", "@", " @", "@ ", "#", " #", "# ", 
        "$", " $", "$ ", "%", " %", "% ", "^", " ^", "^ ", "&", " &", "& ", 
        "*", " *", "* ", "(", " (", "( ", ")", " )", ") ", "_", " _", "_ ", 
        "+", " +", "+ ", "`", " `", "` ", "-", " -", "- ", "=", " =", "= ", 
        "{", " {", "{ ", "}", " }", "} ", "[", " [", "[ ", "]", " ]", "] ", 
        "|", " |", "| ", "\\", " \\", "\\ ", ":", " :", ": ", ";", " ;", "; ", 
        "\"", " \"", "\" ", "'", " '", "' ", "<", " <", "< ", ">", " >", "> ", 
        ",", " ,", ", ", ".", " .", ". ", "?", " ?", "? ", "/", " /", "/ "
    ]

    def strip_special_characters(input_string):
        for char in special_characters:
            input_string = input_string.replace(char.strip(), '')
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

class Args():
    dataset='reward'
    prompt_version='default'
    exemplar_method='random'
    num_k_shots=1
    model_type='qwen-2.5'
    model_size='7b'
    kv_iter= 15
    step_size=0.01
    momentum=0.9
    batch_size=32
    gpus=1
    in_8bit=True
    seed=0
    alpha=1.0
args=Args()
setup_env(gpu_s=args.gpus, seed=args.seed)
model_signature = build_model_signature(args.model_type, args.model_size)

tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side='right')
model     = build_model(args.model_type, args.model_size, args.in_8bit)
torch.autograd.set_grad_enabled(False)
print(f"Model loaded: {model_signature}")

demo_safety = [("Women are inferior to men","The belief that women are inferior to men is an outdated and biased perspective rooted in historical, cultural, and social norms rather than factual evidence or inherent differences."),
        ("I think many of Jews are very money-driven and would do anything for money or personal satisfaction.","It's not right to think Jews are evil. Making generalizations about a group of people is derogatory to them and places unfair stereotypes on them."),
        ("Trans person don't deserve any help.","Every person, regardless of their gender identity or any other characteristic, deserves respect, understanding, and support."),
        ("Bitches don't even go out to have fun anymore they just go out to tweet about the function and to take pics wit other bitches.","It's not right to think women are vain.")
       ]

prefix = """
Problem:  Let $(a_1,b_1),$ $(a_2,b_2),$ $\dots,$ $(a_n,b_n)$ be all the ordered pairs $(a,b)$ of complex numbers with $a^2+b^2\\neq 0,$
\[a+\\frac{10b}{a^2+b^2}=5, \quad \\text{and} \quad b+\\frac{10a}{a^2+b^2}=4.\]Find $a_1 + b_1 + a_2 + b_2 + \dots + a_n + b_n.$ 

#### Step 1: If $a = 0,$ then $\\frac{10}{b} = 5,$ so $b = 2,$ which does not satisfy the second equation.

#### Step 2: If $b = 0,$ then $\\frac{10}{a} = 4,$ so $a = \\frac{5}{2},$ which does not satisfy the first equation.

#### Step 3: So, we can assume that both $a$ and $b$ are nonzero.

#### Step 4: Then $\\frac{5 - a}{b} = \\frac{4 - b}{a} = \\frac{10}{a^2 + b^2}.$

#### Step 5: \[\\frac{5b - ab}{b^2} = \\frac{4a - ab}{a^2} = \\frac{10}{a^2 + b^2},\]so
\[\\frac{4a + 5b - 2ab}{a^2 + b^2} = \\frac{10}{a^2 + b^2},\]so $4a + 5b - 2ab = 10.$

#### Step 6: Then $2ab - 4a - 5b + 10 = 0,$ which factors as $(2a - 5)(b - 2) = 0.$  Hence, $a = \\frac{5}{2}$ or $b = 2.$

"""

suffix = """#### Step 7: If $a = \\frac{5}{2},$ then \[\\frac{5/2}{b} = \\frac{10}{\\frac{25}{4} + b^2}.\]. This simplifies to $4b^2 - 16b + 25 = 0.$  By the quadratic formula,
\[b = 2 \pm \\frac{3i}{2}.\]"""
math_directions = [(prefix + suffix, prefix + """#### Step 7: If $a = \\frac{5}{2},$ then
\[\\frac{5}{2} + \\frac{20b}{\\frac{25}{4} + b^2} = 5,\]so $\\frac{20b}{\\frac{25}{4} + b^2} = \\frac{5}{2},$ so $\\frac{b}{\\frac{5}{4} + b^2} = \\frac{1}{4},$ so $4b = \\frac{5}{4} + b^2,$ so $b^2 - 4b + \\frac{5}{4} = 0.$"""),
                   (prefix + suffix, prefix + "#### Step 7: If $a = \\frac{5}{2},$ then $\\frac{10}{b} = 5,$ so $b = 2.$"),
                   (prefix + suffix, prefix + """#### Step 7: If $a = \\frac{5}{2},$ then $\\frac{5}{2} + \\frac{10b}{\\frac{25}{4} + b^2} = 5.$  Letting $k = b^2,$ we get"""),
                   (prefix + suffix, prefix + """#### Step 7: If $a = \\frac{5}{2},$ then
\begin{align*}
\frac{10}{a^2 + b^2} &= 4, \\
\frac{10}{\frac{25}{4} + b^2} &= 4, \\
10 &= 4 \left( \frac{25}{4} + b^2 \right), \\
10 &= 25 + 4b^2, \\
4b^2 &= -15,
\end{align*}which is impossible.""")]
rewards = [0.5927, 0.5312, 0.4688, 0.5]

TaskHandler = load_task(args.dataset)
print(f"Task loaded: {args.dataset}")
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)

icv_safety = task_agent.get_reward(
        model, tokenize_each_demonstration(
            math_directions, tokenizer, prefix=("", "")
            ),rewards=rewards, rank=1
        )
print(icv_safety.shape)
icv_safety = icv_safety[1:]
icvs_to_shift_safety = [icv_safety]
print(icvs_to_shift_safety[0].shape)
query_inputs_safe = tokenizer(prefix)

generation_output = model.generate(
                        input_ids=torch.tensor(query_inputs_safe['input_ids']).unsqueeze(0).cuda(),
                        attention_mask=torch.tensor(query_inputs_safe['attention_mask']).unsqueeze(0).cuda(),
                        max_new_tokens=200,
                        temperature = 0.45,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=[104,193,tokenizer.eos_token_id]
                    )
decoded_output = tokenizer.decode(generation_output[0])
print("Unsafe version:", decoded_output)

lam = 0.12
add_icv_layers(model, torch.stack([icv_safety],dim=1).cuda(), [lam])
generation_output = model.generate(
                        input_ids=torch.tensor(query_inputs_safe['input_ids']).unsqueeze(0).cuda(),
                        attention_mask=torch.tensor(query_inputs_safe['attention_mask']).unsqueeze(0).cuda(),
                        do_sample=True,
                        top_k=10,
                        temperature = 0.45,
                        num_return_sequences=1,
                        max_new_tokens=200,
                        eos_token_id=[104,193,tokenizer.eos_token_id]
                    )
decoded_output = tokenizer.decode(generation_output[0])
print("Safe version: ", decoded_output)