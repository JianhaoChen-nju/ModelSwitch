import random
import numpy as np
import csv
import json
from datasets import load_dataset as load_hf_dataset

def gsm8k_questions(
    Sampling: bool = False,
    Sampling_Numbers: int = 100,
    random_seed: int = 42
):
    """Load gsm8k dataset and question format function.

    Args:
        Sampling: whether to do sampling
        Sampling_Numbers: how many samples to sample
        random_seed: random seed for reproducibility

    Returns:
        questions: list of questions.
        Sampling_questions: list of sampled questions if Sampling is True.
    """
    random.seed(random_seed)

    questions = []
    Sampling_questions = []

    data = load_hf_dataset("arrow", data_files={"train": "../Datasets/GSM8K/gsm8k-train.arrow", "test": "../Datasets/GSM8K/gsm8k-test.arrow"}, split="test")

    for i, q in enumerate(data):
        question = {
            "question": q["question"],
            "no": int(i),
            "category": "Math",
            "solution": q["answer"].split('#### ')[-1],
        }
        questions.append(question)

    if Sampling:
        if Sampling_Numbers > len(questions):
            raise ValueError("Sampling_Numbers cannot exceed the total number of samples in the dataset.")

        sampled_indices = random.sample(range(len(questions)), Sampling_Numbers)
        for i in sampled_indices:
            sampling_question = questions[i]
            Sampling_questions.append(sampling_question)

        return Sampling_questions
    return questions

def date_question(
    Sampling: bool = False,
    Sampling_Numbers: int = 100,
    random_seed: int = 42
):
    random.seed(random_seed)
    datas=load_hf_dataset("json",data_files="../Datasets/DATE/DATE.json",split="train")
    questions=[]
    for data in datas:
        question={
            'question':data['question'],
            'solution':data['final_answer']
        }
        questions.append(question)
    return questions






def mmlu_pro_question(
    Sampling: bool = False,
    Sampling_Numbers: int = 100,
    random_seed: int = 42
):
    random.seed(random_seed)
    dataset=load_hf_dataset("json",data_files="../Datasets/MMLU_Pro/MMLU-Pro.json",split="train")
    questions=[]
    for index in range(len(dataset)):
        question={
            'question':dataset[index]['question'],
            'no':index,
            'solution':dataset[index]['solution']
        }
        questions.append(question)

    if Sampling:
        Sampling_questions=[]
        if Sampling_Numbers > len(questions):
            raise ValueError("Sampling_Numbers cannot exceed the total number of samples in the dataset.")

        sampled_indices=random.sample(range(len(questions)),Sampling_Numbers)
        for i in sampled_indices:
            sampling_question = questions[i]
            Sampling_questions.append(sampling_question)

        return Sampling_questions
    return questions


def MGSM_questions(
    Sampling: bool = True,
    Sampling_Numbers: int = 100,
    random_seed: int = 42
):
    random.seed(random_seed)
    # 除去英文
    dataset=[]
    subsets=["bn","de","es","fr","ja","ru","sw","te","th","zh"]
    subset_len=250
    dataset = load_hf_dataset("json",data_files="../Datasets/MGSM.json",split="train")
    # print(len(dataset))
    questions=[]
    for index in range(len(dataset)):
        question={
            'question':dataset[index]['question'],
            'no':index,
            'solution':dataset[index]['answer_number']
        }
        questions.append(question)
    
    if Sampling:
        Sampling_questions=[]
        if Sampling_Numbers > len(questions):
            raise ValueError("Sampling_Numbers cannot exceed the total number of samples in the dataset.")

        sampled_indices=random.sample(range(subset_len),Sampling_Numbers)
        for i in sampled_indices:
            for j in range(len(subsets)):
                
                sampling_question = questions[i+(j*subset_len)]
                Sampling_questions.append(sampling_question)

        return Sampling_questions
    return questions




def mathbench_questions(
Sampling: bool = False,
Sampling_Numbers: int = 100,
random_seed: int = 42
):
    dataset=[]
    with open("../Datasets/MathBench.txt","r")as f:
        datas_line=f.readlines()
        for data in datas_line:
            dataset.append(json.loads(data))
    random.seed(random_seed)
    questions=[]
    for index in range(len(dataset)):
        question={
            'question':dataset[index]['question'],
            'no':index,
            'solution':dataset[index]['answer']
        }
        questions.append(question)
    if Sampling:
        Sampling_questions=[]
        if Sampling_Numbers > len(questions):
            raise ValueError("Sampling_Numbers cannot exceed the total number of samples in the dataset.")

        sampled_indices=random.sample(range(len(questions)),Sampling_Numbers)
        for i in sampled_indices:
                sampling_question = questions[i]
                Sampling_questions.append(sampling_question)

        return Sampling_questions
    return questions

def math_questions(
Sampling: bool = False,
Sampling_Numbers: int = 100,
random_seed: int = 42
):
    dataset = load_hf_dataset("../Datasets/MATH.json",split="train")
    random.seed(random_seed)
    questions=[]
    for index in range(len(dataset)):
        question={
            'question':dataset[index]['problem'],
            'no':index,
            'level':dataset[index]['level'],
            'solution':dataset[index]['answer']
        }
        questions.append(question)
    if Sampling:
        Sampling_questions=[]
        if Sampling_Numbers > len(questions):
            raise ValueError("Sampling_Numbers cannot exceed the total number of samples in the dataset.")

        sampled_indices=random.sample(range(len(questions)),Sampling_Numbers)
        for i in sampled_indices:
                sampling_question = questions[i]
                Sampling_questions.append(sampling_question)

        return Sampling_questions
    return questions
def load_dataset(dataset_name,Sampling,Sampling_Numbers):
    if dataset_name=="GSM8K":
        dataset = gsm8k_questions(Sampling=False, Sampling_Numbers=Sampling_Numbers)
    elif dataset_name=="MGSM":
        dataset = MGSM_questions(Sampling=True, Sampling_Numbers=100)
    elif dataset_name=="MATH":
        dataset=math_questions(Sampling=False,Sampling_Numbers=500)
    elif dataset_name=="MMLU_Pro":
        dataset=mmlu_pro_question(Sampling=True,Sampling_Numbers=500)
    elif dataset_name=="DATE":
        dataset=date_question(Sampling=Sampling,Sampling_Numbers=Sampling_Numbers)
    elif dataset_name=="MathBench":
        dataset=mathbench_questions(Sampling=False,Sampling_Numbers=500)
    else:
        print("Not ")
    return dataset