import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *
import copy
import argparse

def compute_correctness(ans_list,s,dataset):
    is_correct=0
    ans_list=[item for item in ans_list if item !="Error" and item!=""]
    ans_list=[item.replace(" ","") for item in ans_list]
    ans= most_frequent(ans_list)
    if dataset in ["GSM8K","MGSM"]:
        a = float(ans.replace(',',''))
        if dataset=="MGSM":
            s=float(s["solution"])
        else:
            s=float(s["solution"].replace(',',''))
        if abs(s-a) < 1e-6:
            is_correct=1
    elif dataset=="MATH":
        if is_equiv(s["solution"].replace(" ",""),ans):
            is_correct=1
    else:
        matches = re.findall(r'\{(.*?)\}', ans)
        if matches:
            ans=matches[0]
        if ans.replace(" ","").lower()==s["solution"].replace(" ","").lower():
            is_correct=1

    
    return is_correct

def extract_via_string(ans):
    pattern = r'\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])/\d{4}\b'
    ans = re.findall(pattern, ans)
    if ans :
        return ans[-1]
    else:
        return ""

lm_ids=["Llama-3.1-8B-Instruct","gemma-2-9b-it"]

def metrics_mad_MLD(dataset):
    correct=0
    with open(f"Results/{dataset}/MAD_MLD/closed_source/results.json","r")as f:
        datas=json.load(f)
    total=len(datas)
    if dataset=="DATE":
        for data in datas:
            answer=data["answer"]
            answer_list=[]
            for index_agnet in range(5):
                answer_list.extend(extract_via_string([answer[f"Agent_{index_agnet}"]["Round_2"]]))
            correct+=compute_correctness(answer_list,data,dataset)
    else:
        for data in datas:
            correct+=compute_correctness([data["answer_letter"]],data,dataset)
    print(f"MAD_MLD Accuracy:{correct*100.0/total}")

def metrics_mad(dataset):
    correct=0
    with open(f"Results/{dataset}/MAD/closed_source/results.json","r")as f:
        datas=json.load(f)
    total=len(datas)
    if dataset=="DATE":
        for data in datas:
            answer=data["answer"]
            answer_list=[]
            for index_agnet in range(5):
                answer_list.extend(extract_via_string([answer[f"Agent_{index_agnet}"]["Round_2"]]))
            correct+=compute_correctness(answer_list,data,dataset)
    else:
        for data in datas:
            correct+=compute_correctness([data["answer_letter"]],data,dataset)
    print(f"MAD Accuracy:{correct*100.0/total}")

def metrics_chateval(dataset):
    with open(f"Results/{dataset}/chateval/closed_source/results.json","r")as f:
        datas=json.load(f)
    correct=0
    total=len(datas)
    agent_list=["General Public","Critic","Scientist"]
    if dataset=="DATE":
        for data in datas:
            answer=data["answer"]
            answer_list=[]
            for index_agnet in range(3):
                answer_list.extend(extract_via_string([answer[f"{agent_list[index_agnet]}"]["Round_4"]]))
            correct+=compute_correctness(answer_list,data,dataset)
    else:
        for data in datas:
            correct+=compute_correctness([data["answer_letter"]],data,dataset)
    print(f"CHATEVAL Accuracy:{correct*100.0/total}")
def metrics_ms(dataset,budget=16):
    correct=0
    lm_ids=["gpt-4o-mini","gemini-1.5-flash-latest"]
    with open (f"Results/{dataset}/MS/closed_source/results.json")as f:
        datas=json.load(f)
        total=len(datas)
        for data in datas:
            ans_list=[]
            for lm_id in lm_ids:
                ans_list.extend(data[f"{lm_id}_ans_list"][:budget//2])
            correct+=compute_correctness(ans_list,data,dataset)
    print(f"Model_Switch_Budget{budget} Accuracy:{correct*100.0/total}")
        
def metric_moa(dataset):
    correct=0
    with open(f"Results/{dataset}/MOA/closed_source/results.json","r")as f:
        datas=json.load(f)
    total=len(datas)
    for data in datas:
        correct+=compute_correctness([data["final_answer"]],data,dataset)
    print(f"MOA Accuracy:{correct*100.0/total}")
def metric_agent_verse(dataset):
    correct=0
    with open(f"Results/{dataset}/AGENTVERSE/results.json")as f:
        datas=json.load(f)
    total=len(datas)
    for data in datas:
        correct+=compute_correctness([data["answer_letter"]],data,dataset)
    print(f"AgentVerse Accuracy:{correct*100.0/total}")
def metric_RM(dataset,budget):
    lm_ids=["gpt-4o-mini","gemini-1.5-flash-latest"]
    with open(f"Results/{dataset}/Qwen2.5-MATH-RM-72B/results.json")as f:
        datas=json.load(f)
    total=len(datas)
    correct_ms=0
    correct_list=[0 for i in lm_ids]
    for data in datas:
        ans_list=[]
        score_list=[]
        for index_lm,lm_id in enumerate(lm_ids):
            max_score=max(data[f"{lm_id}_score"])
            score_list.append(max_score)
            ans_list.append(data[f"{lm_id}_ans_list"][data[f"{lm_id}_score"].index(max_score)])
            correct_list[index_lm]+=compute_correctness([ans_list[index_lm]],data,dataset)
        correct_ms+=compute_correctness([ans_list[score_list.index(max(score_list))]],data,dataset)
    print(f"Model_switch_RM Accuracy:{correct_ms*100.0/total}")
    print(f"Best_Single_RM Accuracy:{max(correct_list)*100.0/total}")
def metric_Sampling(dataset,budget):
    lm_ids=["gpt-4o-mini","gemini-1.5-flash-latest","gpt-4o","gemini-1.5-pro"]
    with open (f"Results/{dataset}/MS/closed_source/results.json","r")as f:
        datas=json.load(f)
        total=len(datas)
        new_data=[]
        sampling_num=budget
        correct_each=[0 for _ in lm_ids]
        for index_data,data in enumerate(datas):
            for index_lm,lm_id in enumerate(lm_ids):
                final_ans_list=data[f"{lm_id}_ans_list"][:sampling_num]
                correct_each[index_lm]+=compute_correctness(final_ans_list,data,dataset)
        for index_lm,lm_id in enumerate(lm_ids):
            print(f"{lm_id}_Budget{budget}_Accuracy: {correct_each[index_lm]*100.0/total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the dataset.")
    parser.add_argument("--Evaluation", type=str, default="MS_SC")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="GSM8K")
    args = parser.parse_args()
    dataset=args.dataset
    budget=args.budget
    if args.Evaluation=="MS_SC":
        metrics_ms(dataset,budget)
        metric_Sampling(dataset,budget)
    elif args.Evaluation=="MS_MAD":
        metrics_ms(dataset,budget=16)
        metrics_mad_MLD(dataset)
        metric_moa(dataset)
        metrics_mad(dataset)
        metrics_chateval(dataset)
        metric_agent_verse(dataset)
    elif args.Evaluation=="RM":
        metrics_ms(dataset,budget)
        metric_Sampling(dataset,budget)
        metric_RM(dataset,budget)

