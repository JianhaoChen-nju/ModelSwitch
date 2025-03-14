from datasets import load_dataset as load_hf_dataset
import os
import json
import re
import time
import requests
from openai import OpenAI
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import torch

device='cuda'
# model="meta-llama/Llama-3.1-8B-Instruct"
access_token = "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"
# login(access_token)

class Hg_model:
    def __init__(
    self,
    model,
    temperature=1.0,
    top_p=1.0
    ):  
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir='../../../Multi-Agent_Collaboration/debatellm/eval/work/pi_chuangg_umass_edu/.cahce',trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16,

                                                                    # pad_token_id=tokenizer.eos_token_id,
                                                                    #load_in_8bit=True,
                                                                    device_map="balanced_low_0",
                                                                    cache_dir='../../../Multi-Agent_Collaboration/debatellm/eval/work/pi_chuangg_umass_edu/.cahce',trust_remote_code=True)
    #generator = pipeline(model=lm_id, device=device, torch_dtype=torch.float16)

    def run_llm(self,prompt):
        # print(f"{lm_ids[index]} is outputing....... ")
        sampling_params = {
        "max_new_tokens": 512,
        "temperature": 1,
        "top_p": 1,
        "num_return_sequences": 1,
        'return_dict_in_generate':True,
        'do_sample':True
        }
        if 'llama' in model:
            messages = [
            {"role": "system", "content": "You are a bot that responds to weather queries. Stop when you get the answer"},
            {"role": "user", "content": f"{prompt}"}
            ]

            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            prompt_len = inputs.shape[-1]
            # print(sampling_params)
            output_dict = self.model.generate(inputs, # max_length=prompt_len + sampling_params['max_new_tokens'],
            **sampling_params,return_legacy_cache=False,pad_token_id=tokenizer.eos_token_id)
            generated_samples =tokenizer.batch_decode(output_dict.sequences[:,prompt_len:], skip_special_tokens=True,)

        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.to(device).shape[-1]
            # print(sampling_params)
            output_dict = self.model.generate(**inputs, # max_length=prompt_len + sampling_params['max_new_tokens'],
            **sampling_params,return_legacy_cache=False,pad_token_id=tokenizer.eos_token_id)
            generated_samples =tokenizer.batch_decode(output_dict.sequences[:,prompt_len:], skip_special_tokens=True,)
        return generated_samples[0]


# Try except decorator
def try_except_decorator(func: Callable) -> Callable:
    def func_wrapper(*args: Any, **kwargs: Any) -> Callable:
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print("API error occurred:", str(e), ". Retrying in 1 second...")
                time.sleep(3)
    return func_wrapper

def set_api_and_proxy():
    # byjunior031@gmail.coms
    # os.environ["OPENAI_API_KEY"] = "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"
    # ivykkg@berlin.com
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

    # os.environ["http_proxy"] = "http://127.0.0.1:7897"
    # os.environ["https_proxy"] = "http://127.0.0.1:7897"

    os.environ["CLAUDESHOP_API_BASE"] = "https://api.claudeshop.top/v1"
    os.environ["CLAUDESHOP_API_KEY"] = "sk-WtIfXtM8Mtoz2DwVDhKYUMWeNNtEhD79Q5LKuKTBi2mGwVs7"#"sk-VxUMGYMoiYWdFRFE5pcp9femWhhinvNUGY7qziHn7VwpB03O"
    os.environ['NO_PROXY'] = 'api.claudeshop.top'

@try_except_decorator
def gpt(
    prompt: str,
    model: str = "gpt-4o-mini",
    stop: list[str] = [],
    temperature: float=1.0,
    top_p: float=1.0,
    num_sampling:int=1
):

    # set_api_and_proxy()

    # gpt-4o-mini
    client = OpenAI(
            api_key="sk-8eHzwMgmpm05NTJ6y26e2SiWbAPf6wLGIiH2zr0u1iHjZd1p",
            base_url="https://api.claudeshop.top/v1"
    )

    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stop=stop,
            top_p=top_p,
            n=1
        ).to_dict()
    return response["choices"][0]["message"]["content"]
@try_except_decorator
def claude(
    prompt: str,
    model: str = "claude-3-haiku-20240307",
    system_prompt: str = "",
    stop: list[str] = [],
    temperature: float=1.0,
    top_p: float=1.0
):
    # set_api_and_proxy()
    os.environ["CLAUDESHOP_API_KEY"] = "sk-WtIfXtM8Mtoz2DwVDhKYUMWeNNtEhD79Q5LKuKTBi2mGwVs7"
    client = OpenAI(
                api_key = os.environ["CLAUDESHOP_API_KEY"],
                base_url = "https://api.claudeshop.top/v1"
            )
    if system_prompt=="":
        response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p
            ).to_dict()
    else:
        response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                    ],
                temperature=temperature,
                top_p=top_p
            ).to_dict()
    result=response["choices"][0]["message"]["content"]
    # print(result)
    for s in stop:
        result=result.split(s)[0]
    
    return result
@try_except_decorator
def gemini(
    prompt: str,

    model: str = "gemini-1.5-flash-latest",
    stop: list[str] = [],
    temperature: float=1.0,
    top_p: float=1.0,
    num_sampling:int=1 
):
    set_api_and_proxy()
    client = OpenAI(
            api_key="sk-jFMH3CyP3TFbKnSdk4V412YmIlBJRaycbniOjoiZobYwvTAo",
            base_url="https://api.claudeshop.top/v1"
            )
    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p
        ).to_dict()

    # print(result)
    result=response["choices"][0]["message"]["content"]
    for s in stop:
        result=result.split(s)[0]

    return result

@try_except_decorator
def gemini(
    prompt: str,
    model: str = "gemini-1.5-flash-latest",
    stop: list[str] = [],
    temperature: float=1.0,
    top_p: float=1.0    
):
    set_api_and_proxy()
    client = OpenAI(
                api_key = os.environ["CLAUDESHOP_API_KEY"],
                base_url = os.environ["CLAUDESHOP_API_BASE"]
            )
    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p
        ).to_dict()
    result=response["choices"][0]["message"]["content"]
    # print(result)
    for s in stop:
        result=result.split(s)[0]
    
    return result

@try_except_decorator
def summarize_text(
    prompt,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 50
):
    set_api_and_proxy()
    # gpt-4o-mini
    client = OpenAI(
            api_key="sk-jFMH3CyP3TFbKnSdk4V412YmIlBJRaycbniOjoiZobYwvTAo",
            base_url="https://api.claudeshop.top/v1"
    )
    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Summarize the following text in one sentence:\n\n"+prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ).to_dict()
    
    return response["choices"][0]["message"]["content"]

@try_except_decorator
def evaluate_topic(
    question,
    statement1,
    statement2,
    model: str = "gpt-4o-mini",
    max_tokens: int = 3,
    temperature: float=0.0
):
    # 构建请求内容
    prompt = (
        f"Question:\n{question}\n\nDetermine whether the following two statements about the Question are discussing the same subproblem. Please ignore their conclusions about this subproblem and only focus on whether they are discussing the same topic:\n"
        f"1. {statement1}\n2. {statement2}\n\nAnswer with one of the words: 'Yes', 'No':\n"
    )
    # print(prompt)
    # 调用 OpenAI 的 ChatGPT API
    set_api_and_proxy()
    client = OpenAI(
            api_key="sk-8eHzwMgmpm05NTJ6y26e2SiWbAPf6wLGIiH2zr0u1iHjZd1p",
            base_url="https://api.claudeshop.top/v1"
        )

    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        ).to_dict()
    result=response["choices"][0]["message"]["content"].lower()
    # print(result)
    return result

@try_except_decorator
def evaluate_conclusion(
    question,
    statement1,
    statement2,
    model: str = "gpt-4o-mini",
    max_tokens: int = 3,
    temperature: float=0.0
):
    # 构建请求内容
    prompt = (
        f"Question:\n{question}\n\nDetermine whether the following two statements about one of the subproblems of the Question lead to the same conclusion. Please pay special attention to the logic and calculation parts:\n"
        f"1. {statement1}\n2. {statement2}\n\nAnswer with one of the words: 'Yes', 'No':\n"
    )
    # print(prompt)
    # 调用 OpenAI 的 ChatGPT API
    set_api_and_proxy()
    client = OpenAI(
            api_key = os.environ["OPENAI_API_KEY"],
            base_url = os.environ["OPENAI_API_BASE"]
        )

    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        ).to_dict()
    result=response["choices"][0]["message"]["content"].lower()
    # print(result)
    return result

