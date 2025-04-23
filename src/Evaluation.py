import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *
import copy
import argparse
import math
weights_dict={
    "GSM8K":{'1': 1, '2': 1, '3': 1},
    "MATH": {'1': 2, '2': 2, '3': 1},
    "MMLU-Pro":{'1': 1, '2': 1, '3': 1}, 
    "MGSM":{'1': 1, '2': 1, '3': 1},
    "MathBench": {"1":2, "2":2, "3":2},
    "DATE": {"1":1.5, "2":1.5, "3":1}
}
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
def calculate_scores_with_weights(answer_list, weights):
    cleaned_answer_list = []
    for answers in answer_list:
        cleaned_answers = [item.replace(" ", "") for item in answers if item != "Error" and item != ""]
        cleaned_answer_list.append(cleaned_answers)

    answer_scores = {}

    # V2 entropy
    def internal_consistency_score(answers):
        score = {}
        total_answers = len(answers)
        if len(answers) == 0:
            return score
        counts = {}
        for answer in answers:
            counts[answer] = counts.get(answer, 0) + 1

        entropy = 0
        for count in counts.values():
            probability = count / total_answers
            entropy -= probability * math.log2(probability) if probability > 0 else 0

        max_entropy = math.log2(len(counts))

        bias = 1.0 / len(answers)
        weight = bias + (1 - bias) * (1 - (entropy / max_entropy)) if max_entropy > 0 else 1
        # print(weight)
        for answer, count in counts.items():
            score[answer] = count * weight
            # score[answer] = count * probability
            # score[answer] = count

        return score

    def external_weight_score(internal_scores, weight):
        score = {key: value * weight for key, value in internal_scores.items()}
        return score

    weighted_scores_list = []
    for i, answers in enumerate(cleaned_answer_list):
        internal_scores = internal_consistency_score(answers)
        weighted_scores = external_weight_score(internal_scores, weights[str(i + 1)])
        weighted_scores_list.append(weighted_scores)

    all_answers = [answer for answers in cleaned_answer_list for answer in answers]
    for answer in set(all_answers):
        total_score = 0
        for weighted_scores in weighted_scores_list:
            total_score += weighted_scores.get(answer, 0)

        answer_scores[answer] = total_score

    def select_best_answer():
        max_score = -1
        best_answer = None

        for answers in cleaned_answer_list:
            for answer in answers:
                score = answer_scores.get(answer, 0)
                if score > max_score:
                    max_score = score
                    best_answer = answer

        return best_answer

    return select_best_answer()
def vote_algorithmn_performance(dataset,weights):
    file=f"Results/{dataset}/MS/closed_source/results.json"
    with open(file,"r") as f1:
        datas=json.load(f1)
    lm_ids=["gpt-4o-mini","gemini-1.5-flash-latest","claude"]
    first_round_Acc=0
    first_round_Correct=0

    second_round_Acc=0
    second_round_Correct=0

    third_round_Acc=0
    third_round_Correct=0
    mixed_Acc=0
    mixed_Correct=0
    Total=len(datas)
    Correct_answer_before=0
    Correct_answer_after=0
    request_len=0
    correct2wrong=[]
    wrong2correct=[]
    correct2wrong_data=[]
    wrong2correct_data=[]
    CC=0
    CW=0
    WC=0
    WW=0
    CW_data=[]
    WC_data=[]
    
    for i,n in enumerate(datas):
        j=0
        k=0
        if dataset=="GSM8K":
            s = float(datas[i]["solution"].replace(',',''))
        else:
            s = datas[i]["solution"]

        first_round_answers=datas[i][f"{lm_ids[0]}_ans_list"][0:6]
        second_round_answers=datas[i][f"{lm_ids[1]}_ans_list"][0:6]
        third_round_answers=datas[i][f"{lm_ids[2]}_ans_list"][0:4]
        
        request_len+=len(datas[i][f"{lm_ids[0]}_sampling"])
        first_round_Correct+=compute_correctness(first_round_answers,datas[i],dataset)
        second_round_Correct+=compute_correctness(second_round_answers,datas[i],dataset)
        third_round_Correct+=compute_correctness(third_round_answers,datas[i],dataset)

        if dataset in ["MGSM","MATH","MMLU-Pro"]:
            mixed_final_answer=calculate_scores_with_weights([second_round_answers,first_round_answers,third_round_answers],weights)
            mixed_answers=second_round_answers+first_round_answers+third_round_answers
        else: 
            mixed_final_answer=calculate_scores_with_weights([first_round_answers,second_round_answers,third_round_answers],weights)
            mixed_answers=first_round_answers+second_round_answers+third_round_answers
        correctness=compute_correctness([mixed_final_answer],datas[i],dataset)
        mixed_Correct+=correctness


    first_round_Acc=first_round_Correct*1.0/Total
    second_round_Acc=second_round_Correct*1.0/Total
    third_round_Acc=third_round_Correct*1.0/Total
    mixed_Acc=mixed_Correct*100.0/Total
    print(f"Model_Switch Accuracy:{mixed_Acc}")



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
                answer_list.append(extract_via_string(answer[f"Agent_{index_agnet}"]["Round_2"]))
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
                answer_list.append(extract_via_string(answer[f"Agent_{index_agnet}"]["Round_2"]))
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
                answer_list.append(extract_via_string(answer[f"{agent_list[index_agnet]}"]["Round_4"]))
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
        if data["correct"]:
            correct+=1
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
    budgets=[budget,budget,1,1]
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
            print(f"{lm_id}_Budget{budgets[index_lm]}_Accuracy: {correct_each[index_lm]*100.0/total}")

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
        vote_algorithmn_performance(dataset,weights_dict[dataset])
        metrics_mad_MLD(dataset)
        metric_moa(dataset)
        metrics_mad(dataset)
        metrics_chateval(dataset)
        metric_agent_verse(dataset)
    elif args.Evaluation=="RM":
        metrics_ms(dataset,budget)
        metric_Sampling(dataset,budget)
        metric_RM(dataset,budget)

