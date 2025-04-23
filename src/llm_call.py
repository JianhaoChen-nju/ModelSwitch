from datasets import load_dataset as load_hf_dataset
import os
import json
import re
import time
import requests
from openai import OpenAI
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import torch

device='cuda'
os.environ["OPENAI_API_KEY"] = "sk-WtIfXtM8Mtoz2DwVDhKYUMWeNNtEhD79Q5LKuKTBi2mGwVs7"
os.environ["OPENAI_API_BASE"] = "https://api.claudeshop.top/v1"
class Hg_model:
    def __init__(
    self,
    model,
    temperature=1.0,
    top_p=1.0
    ):  
        self.tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16,

                                                                    # pad_token_id=tokenizer.eos_token_id,
                                                                    #load_in_8bit=True,
                                                                    device_map="auto",
                                                                    trust_remote_code=True)
    #generator = pipeline(model=lm_id, device=device, torch_dtype=torch.float16)
        self.model_name=model

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
        if 'llama' in self.model_name:
            messages = [
            {"role": "system", "content": "You are a robot completing questions. Stop when you get the answer"},
            {"role": "user", "content": f"{prompt}"}
            ]

            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            prompt_len = inputs.shape[-1]
            # print(sampling_params)
            output_dict = self.model.generate(inputs, # max_length=prompt_len + sampling_params['max_new_tokens'],
            **sampling_params,return_legacy_cache=False,pad_token_id=self.tokenizer.eos_token_id)
            generated_samples =self.tokenizer.batch_decode(output_dict.sequences[:,prompt_len:], skip_special_tokens=True,)

        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.to(device).shape[-1]
            # print(sampling_params)
            output_dict = self.model.generate(**inputs, # max_length=prompt_len + sampling_params['max_new_tokens'],
            **sampling_params,return_legacy_cache=False,pad_token_id=self.tokenizer.eos_token_id)
            generated_samples =self.tokenizer.batch_decode(output_dict.sequences[:,prompt_len:], skip_special_tokens=True,)
        return generated_samples[0]
    def __contains__(self,str):
        if self.model_name.__contains__(str):
            return True
        else:
            return False


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

@try_except_decorator
def gpt(
    prompt: str,
    model: str = "gpt-4o-mini",
    stop: list[str] = [],
    temperature: float=1.0,
    top_p: float=1.0,
    num_sampling:int=1
):
    client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_BASE"]
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
    client = OpenAI(
                api_key = os.environ["OPENAI_API_KEY"],
                base_url = os.environ["OPENAI_API_BASE"]
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
                api_key = os.environ["OPENAI_API_KEY"],
                base_url = os.environ["OPENAI_API_BASE"]
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


