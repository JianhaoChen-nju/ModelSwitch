# 分析代码
import json
from sympy import *
from sympy.parsing.latex import parse_latex
import math
import argparse
import random
random.seed(43)

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        # print(f"ss1: {type(ss1)}")
        # print(f"ss2: {ss2}")

        ss1 = parse_latex(ss1)
        ss2 = parse_latex(ss2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception as e:
        # print(e)
        return str1 == str2

def most_frequent(list):
    if len(list)==0:
        return "-1", 0
    counter = 0
    num = list[0]

    for i in list:
        current_frequency = list.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter

def compute_correctness(ans_list,s,dataset):
    is_correct=0
    ans_list=[item for item in ans_list if item !="Error" and item!=""]
    ans_list=[item.replace(" ","") for item in ans_list]
    ans,_ = most_frequent(ans_list)
    if dataset in ["gsm8k","MGSM"]:
        a = float(ans.replace(',',''))
        if abs(s-a) < 1e-6:
            is_correct=1
    elif dataset=="math":
        if is_equiv(s.replace(" ",""),ans):
            is_correct=1
    elif dataset=="simpleqa":
        if ans.replace(",","").replace(".","").replace(" ","").lower()==s.replace(",","").replace(".","").replace(" ","").lower():
            is_correct=1
    elif dataset=="logiQA":
        ans_dict={0:"A",1:"B",2:"C",3:"D"}
        s=ans_dict[s]
        if ans.replace(" ","").lower()==s.replace(" ","").lower():
            is_correct=1
    else:
    #去空格，去大小写
        if ans.replace(" ","").lower()==s.replace(" ","").lower():
            is_correct=1

    
    return is_correct

def single_file_analysis(dataset):
    file1=f"results/{dataset}/SWN1/gemini-1.5-flash/results.json"

    with open(file1,"r") as f1:
        data1=json.load(f1)
    first_round_Acc=0
    first_round_Correct=0

    second_round_Acc=0
    second_round_Correct=0

    Total=len(data1)
    # 中间结果正确率
    Correct_answer_before=0
    Correct_answer_after=0
    request_len=0
    correct2wrong=[]
    wrong2correct=[]
    correct2wrong_data=[]
    wrong2correct_data=[]
    potential=[]
    for i,n in enumerate(data1):
        no1=data1[i]["no"]
        if dataset=="gsm8k":
            s = float(data1[i]["solution"].replace(',',''))
        else:
            s = data1[i]["solution"]

        first_round_answers=data1[i]["intermediate_results"][0:1]
        second_round_answers=data1[i]["ans_list"]
        final_answers = data1[i]["ans_list"]
        
        request_len+=len(data1[i]["step_sampling"])
        # request_len+=len(data1[i]["outputs"])
        second_round_Correct+=compute_correctness(final_answers,s,dataset)
        first_round_Correct+=compute_correctness(first_round_answers,s,dataset)

        # if compute_correctness(first_round_answers,s)==1 and compute_correctness(final_answers,s)==0:
        #     correct2wrong.append(no1)
        #     correct2wrong_data.append(n)
        # if compute_correctness(first_round_answers,s)==0 and compute_correctness(second_round_answers,s)==1 and compute_correctness(final_answers,s)==0:
        #     potential.append(no1)
        # if compute_correctness(first_round_answers,s)==0 and compute_correctness(final_answers,s)==1:
        #     wrong2correct.append(no1)
        #     wrong2correct_data.append(n)

        # if len(data1[i]["outputs"])!=0:
        #     answers_before=data1[i]["intermediate_results"]
        #     answers_before=[item for item in answers_before if item !="Error"]

        #     answers_after=data1[i]["ans_list"]
        #     answers_after=[item for item in answers_after if item !="Error"]
        #     for ans in answers_before:
        #         # a = float(ans.replace(',',''))
        #         # if abs(s-a) < 1e-6:
        #         if ans.replace(" ","").lower()==s:
        #             Correct_answer_before+=1
        #     for ans in answers_after:
        #         # a = float(ans.replace(',',''))
        #         # if abs(s-a) < 1e-6:
        #         if ans.replace(" ","").lower()==s:
        #             Correct_answer_after+=1
        # if i ==2:
        #     break
    # print(Correct)
    first_round_Acc=first_round_Correct*1.0/Total
    second_round_Acc=second_round_Correct*1.0/Total

    print("dataset:",dataset)
    print("第一轮正确率:",first_round_Acc)
    print("-"*50)
    # print("第二轮正确率:",second_round_Acc)
    # #计算正确答案before和after
    # print("正确改为错误的题号为：",correct2wrong,"数量为：",len(correct2wrong))
    # print("错误改为正确的题号为：",wrong2correct,"数量为：",len(wrong2correct))
    # print("potential:",potential)
    #计算平均调用次数
    # print("平均重复采样次数:",request_len*1.0/Total)

    # file2="results/gsm8k/SWN/dev/wrong.json"
    # with open(file2,"w") as f:
    #     json.dump(correct2wrong_data,f,indent=4)



def random_check():
    
    result = random.randint(0, 99)
    
    if result < 50:
        # print("0")
        return 0
    else:
        # print("1")
        return 1

# output = random_check()
# print(output)

def mix_results_in_1file(dataset):
    # 分析两个文件合成
    file=f"results/{dataset}/result.json"
    CW_file=f"results/{dataset}/SWN1/CW/CW.json"
    WC_file=f"results/{dataset}/SWN1/CW/WC.json"
    with open(file,"r") as f1:
        data1=json.load(f1)


    first_round_Acc=0
    first_round_Correct=0

    second_round_Acc=0
    second_round_Correct=0

    mixed_Acc=0
    mixed_Correct=0
    Total=len(data1)
    # 中间结果正确率
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
    Sampling_Number=8
    for i,n in enumerate(data1):
            
        if dataset in ["gsm8k","MGSM"]:
            s = float(str(data1[i]["solution"]).replace(',',''))
        else:
            s = data1[i]["solution"]


        first_round_answers=data1[i]["gpt-4o-mini_ans_list"][0:Sampling_Number]
        # print(len(first_round_answers))
        second_round_answers=data1[i]["gemini-1.5-flash-latest_ans_list"][0:Sampling_Number]
        first_round_answers=[item.replace(" ","") for item in first_round_answers]
        if len(set(first_round_answers))==1:
            request_len+=Sampling_Number
        else:
            request_len+=2*Sampling_Number
        mixed_answers = data1[i]["gpt-4o-mini_ans_list"][0:1]+data1[i]["gemini-1.5-flash-latest_ans_list"][0:1]

        mixed_answers = [mixed_answers[random_check()]]
        

        first_round_Correct+=compute_correctness(first_round_answers,s,dataset)
        second_round_Correct+=compute_correctness(second_round_answers,s,dataset)
        mixed_Correct+=compute_correctness(mixed_answers,s,dataset)
        # if compute_correctness(mixed_answers,s,dataset)==0:
        #     print(data1[i]["question"])

    first_round_Acc=first_round_Correct*1.0/Total

    second_round_Acc=second_round_Correct*1.0/Total
    mixed_Acc=mixed_Correct*1.0/Total
    
    print("dataset:",dataset)
    print("模型1正确率:",first_round_Acc)
    # print("模型1正确答案总数:",first_round_Correct)
    print("模型2正确率:",second_round_Acc)
    print("混合正确率:",mixed_Acc)
    print("request_len:",request_len/Total)
    print("-"*50)
    

def mix_file_analysis(dataset):
    # 分析两个文件合成
    file1=f"results/{dataset}/ModelFusion/merged/gemini-1.5-flash-latest.json"
    file2=f"results/{dataset}/ModelFusion/merged/gpt-4o-mini.json"

    # file1=f"results/{dataset}/ModelFusion/gemini-1.5-flash-latest/results.json"
    # file1=f"results/{dataset}/ModelFusion/claude-3-haiku-20240307/results.json"
    CW_file=f"results/{dataset}/SWN1/CW/CW.json"
    WC_file=f"results/{dataset}/SWN1/CW/WC.json"
    with open(file1,"r") as f1:
        data1=json.load(f1)

    with open(file2,"r") as f1:
        data2=json.load(f1)

    first_round_Acc=0
    first_round_Correct=0

    second_round_Acc=0
    second_round_Correct=0

    mixed_Acc=0
    mixed_Correct=0
    Total=len(data1)
    # 中间结果正确率
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
    for i,n in enumerate(data1):
        no1=data1[i]["no"]
        j=0
        for j1,m1 in enumerate(data2):
            if data2[j1]["no"]==no1:
                j=j1
            
        if dataset in ["gsm8k","MGSM"]:
            s = float(str(data1[i]["solution"]).replace(',',''))
        else:
            s = data1[i]["solution"]

        Sampling_number=3
        first_round_answers=data1[i]["ans_list"]
        # print(len(first_round_answers))
        second_round_answers=data2[j]["ans_list"]
        mixed_answers = data1[i]["ans_list"][0:Sampling_number]+data2[j]["ans_list"][0:Sampling_number]
        # mixed_answers = [mixed_answers[random_check()]]
        
        request_len+=len(data1[i]["step_sampling"])
        # request_len+=len(data1[i]["outputs"])
        first_round_Correct+=compute_correctness(first_round_answers,s,dataset)
        second_round_Correct+=compute_correctness(second_round_answers,s,dataset)
        mixed_Correct+=compute_correctness(mixed_answers,s,dataset)


    first_round_Acc=first_round_Correct*1.0/Total

    second_round_Acc=second_round_Correct*1.0/Total
    mixed_Acc=mixed_Correct*1.0/Total
    
    print("dataset:",dataset)
    print("模型1正确率:",first_round_Acc)
    # print("模型1正确答案总数:",first_round_Correct)
    print("模型2正确率:",second_round_Acc)
    print("混合正确率:",mixed_Acc)
    print("-"*50)


def synthetic_file(dataset):
    '''math'''
    file2=f"results/{dataset}/ModelFusion/merged/gpt-4o-mini.json"
    file1=f"results/{dataset}/ModelFusion/merged/gemini-1.5-flash-latest.json"
    file3=f"results/{dataset}/ModelFusion/synthetic/results.json"
    dev_file=f"results/{dataset}/ModelFusion/synthetic/dev.json"
    metric_output_path=f"results/{dataset}/ModelFusion/synthetic/metrics.json"

    with open(file1,"r") as f1:
        data1=json.load(f1)

    with open(file2,"r") as f1:
        data2=json.load(f1)

    Correct=0
    Acc=0
    Total=len(data1)
    # 中间结果正确率
    request=0
    data3=[]
    dev_data=[]
    for i,n in enumerate(data1):
        d={}
        d["question"]=data1[i]["question"]
        d["no"]=data1[i]["no"]
        d["solution"]=data1[i]["solution"]
        d["step_sampling"]=data1[i]["step_sampling"]
        d["intermediate_results"]=data1[i]["intermediate_results"]
        no1=data1[i]["no"]
        j=0
        for j1,m1 in enumerate(data2):
            if data2[j1]["no"]==no1:
                j=j1
        
        if dataset in ["gsm8k","MGSM"]:
            s = float(str(data1[i]["solution"]).replace(',',''))
        else:
            s = data1[i]["solution"]

        sampling_number=3
        first_round_answers=data1[i]["ans_list"][0:sampling_number]
        second_round_answers=data2[j]["ans_list"][0:sampling_number]

        # first_round_outputs=[item for item in first_round_answers if item !="Error" and item!=""]
        # first_round_outputs=[item.replace(" ","") for item in first_round_answers]

        outputs_set=set(first_round_answers)
        if len(outputs_set)==1 and first_round_answers[0]!="Error" and first_round_answers[0]!="":
            # print(first_round_answers)
            d["outputs"]=[]
            d["ans_list"]=first_round_answers
            request+=len(d["ans_list"])
            d["final_answer"],_=most_frequent(d["ans_list"])
            correctness=compute_correctness(d["ans_list"],s, dataset)
            if correctness==1:
                Correct+=1
                d["Correctness"]=True
            else:
                d["Correctness"]=False
            data3.append(d)
            continue

        
        mixed_answers = first_round_answers+second_round_answers
        d["outputs"]=data2[j]["step_sampling"]
        d["ans_list"]=mixed_answers
        request+=len(d["ans_list"])
        d["final_answer"],_=most_frequent(d["ans_list"])
        correctness=compute_correctness(d["ans_list"],s, dataset)
        if correctness==1:
            Correct+=1
            d["Correctness"]=True
        else:
            d["Correctness"]=False
        data3.append(d)
        dev_data.append(d)
    
    print(dataset)
    Acc=Correct*1.0/Total
    print("平均请求次数:",request*1.0/Total)
    print("正确率:",Acc)
    print("-"*50)
    # with open(file3,"w") as f:
    #     json.dump(data3,f,indent=4)

    # with open(dev_file,"w") as f:
    #     json.dump(dev_data,f,indent=4)


    # with open(metric_output_path, "w") as f:
    #     f.write(f"total={Total}")
    #     f.write("\n")
    #     f.write(f"acc={Acc}")

def vote_algorithmn_performance_in_1file(dataset,weights):
    '''math'''

    file1=f"results/{dataset}/result.json"

    file4=f"results/{dataset}/ModelFusion/synthetic/results.json"
    metric_output_path=f"results/{dataset}/ModelFusion/synthetic/metrics.json"

    with open(file1,"r") as f1:
        data1=json.load(f1)

    Correct=0
    Acc=0
    Total=len(data1)
    # 中间结果正确率
    mixed_Acc=0
    mixed_Correct=0
    third_round_Acc=0
    third_round_Correct=0
    for i,n in enumerate(data1):
        d={}
        d["question"]=data1[i]["question"]
        d["solution"]=data1[i]["solution"]
        # d["statement"]=data1[i]["statement"]
        j=0
        k=0
        
        s = data1[i]["solution"]

        first_round_answers=data1[i]["gpt-4o-mini_ans_list"][0:6]
        # print(len(first_round_answers))
        second_round_answers=data1[i]["gemini-1.5-flash-latest_ans_list"][0:6]
        third_round_answers=data1[i]["claude_ans_list"]
        
        # first_round_Correct+=compute_correctness(first_round_answers,s,dataset)
        # second_round_Correct+=compute_correctness(second_round_answers,s,dataset)
        third_round_Correct+=compute_correctness(third_round_answers,s,dataset)
        # second_round_answers=[]
        # mixed_Correct+=compute_correctness(mixed_answers,s,dataset)
        if dataset in ["MGSM","math","mmlu_physical","mmlu_pro","AGIEval"]:
            mixed_final_answer=calculate_scores_with_weights(second_round_answers,first_round_answers,third_round_answers,weights)
            mixed_answers=second_round_answers+first_round_answers+third_round_answers
        else: 
            mixed_final_answer=calculate_scores_with_weights(first_round_answers,second_round_answers,third_round_answers,weights)
            mixed_answers=first_round_answers+second_round_answers+third_round_answers
        correctness=compute_correctness([mixed_final_answer],s,dataset)
        mixed_Correct+=correctness
        # if correctness==0:
        #     print(mixed_answers,s)

    # first_round_Acc=first_round_Correct*1.0/Total
    # second_round_Acc=second_round_Correct*1.0/Total
    third_round_Acc=third_round_Correct*1.0/Total
    mixed_Acc=mixed_Correct*1.0/Total

    # print("dataset:",dataset)
    # print("模型1正确率:",first_round_Acc)
    # print("模型2正确率:",second_round_Acc)
    print("模型3正确率:",third_round_Acc)
    # print("混合正确率:",mixed_Acc)
    # print("-"*50)
    return mixed_Acc

def synthetic_3file(dataset,weights):
    '''math'''
    if dataset in ["gsm8k","mmlu_physical"]:
        file1=f"results/{dataset}/ModelFusion/gpt-4o-mini/results.json"
        file2=f"results/{dataset}/ModelFusion/gemini-1.5-flash-latest/results.json"
    else:
        file2=f"results/{dataset}/ModelFusion/gpt-4o-mini/results.json"
        file1=f"results/{dataset}/ModelFusion/gemini-1.5-flash-latest/results.json"
    file3=f"results/{dataset}/ModelFusion/claude-3-haiku-20240307/results.json"
    file4=f"results/{dataset}/ModelFusion/synthetic/results.json"
    # dev_file=f"results/{dataset}/ModelFusion/synthetic/dev.json"
    metric_output_path=f"results/{dataset}/ModelFusion/synthetic/metrics.json"

    with open(file1,"r") as f1:
        data1=json.load(f1)

    with open(file2,"r") as f1:
        data2=json.load(f1)

    with open(file3,"r") as f1:
        data3=json.load(f1)

    Correct=0
    Acc=0
    Total=len(data1)
    # 中间结果正确率
    request=0
    data4=[]
    stage2=0
    stage3=0
    for i,n in enumerate(data1):
        d={}
        d["question"]=data1[i]["question"]
        d["no"]=data1[i]["no"]
        d["solution"]=data1[i]["solution"]
        # d["statement"]=data1[i]["statement"]
        d["step_sampling"]=data1[i]["step_sampling"]
        d["intermediate_results"]=data1[i]["intermediate_results"]
        no1=data1[i]["no"]
        j=0
        k=0
        for j1,m1 in enumerate(data2):
            if data2[j1]["no"]==no1:
                j=j1
                break
        for k1,m1 in enumerate(data3):
            if data3[k1]["no"]==no1:
                k=k1
                break
        
        if dataset in ["gsm8k","MGSM"]:
            s = float(str(data1[i]["solution"]).replace(',',''))
        else:
            s = data1[i]["solution"]

        first_round_answers=data1[i]["intermediate_results"][0:6]
        first_round_outputs=[item.replace(" ","").lower() for item in first_round_answers]
        outputs_set=set(first_round_outputs)
        if len(outputs_set)==1 and first_round_outputs[0]!="" and first_round_outputs[0]!="Error":
            d["outputs"]=[]
            d["ans_list"]=first_round_answers
            request+=len(d["ans_list"])
            d["final_answer"],_=most_frequent(first_round_answers)
            correctness=compute_correctness(d["ans_list"],s, dataset)
            if correctness==1:
                Correct+=1
                d["Correctness"]=True
            else:
                d["Correctness"]=False
            data4.append(d)
            continue

        second_round_answers=data2[j]["intermediate_results"][0:6]
        stage2+=len(second_round_answers)
        second_round_outputs=[item.replace(" ","").lower() for item in second_round_answers]
        outputs_set=set(second_round_outputs)
        if len(outputs_set)==1 and second_round_outputs[0]!="" and second_round_outputs[0]!="Error":
            d["outputs"]=[]
            d["ans_list"]=second_round_answers
            request+=len(first_round_answers+second_round_answers)
            
            d["final_answer"],_=most_frequent(second_round_answers)
            correctness=compute_correctness(d["ans_list"],s, dataset)
            if correctness==1:
                Correct+=1
                d["Correctness"]=True
            else:
                d["Correctness"]=False
            data4.append(d)
            continue

        
        third_round_answers=data3[k]["intermediate_results"][0:4]
        stage3+=len(third_round_answers)
        mixed_answers = first_round_answers+second_round_answers+third_round_answers
        d["outputs"]=data3[k]["step_sampling"]
        d["ans_list"]=mixed_answers
        request+=len(d["ans_list"])
        # d["final_answer"],_=most_frequent(d["ans_list"])
        d["final_answer"]=calculate_scores_with_weights(first_round_answers,second_round_answers,third_round_answers,weights)
        
        correctness=compute_correctness([d["final_answer"]],s, dataset)
        if correctness==1:
            Correct+=1
            d["Correctness"]=True
        else:
            d["Correctness"]=False
        data4.append(d)
        
    Acc=Correct*1.0/Total
    # print("平均请求次数:",request*1.0/Total)
    # print("二阶段请求数:",stage2*1.0/Total)
    # print("三阶段请求数:",stage3*1.0/Total)
    # print("正确率:",Acc)
    # with open(file4,"w") as f:
    #     json.dump(data4,f,indent=4)

    # with open(dev_file,"w") as f:
    #     json.dump(dev_data,f,indent=4)
    # with open(metric_output_path, "w") as f:
    #     f.write(f"total={Total}")
    #     f.write("\n")
    #     f.write(f"acc={Acc}")

    return Acc

def merge_2file(dataset):
    '''math'''
    # llm="gpt-4o-mini"
    # llm="gemini-1.5-flash-latest"
    # llm="claude-3-haiku-20240307"
    # if dataset=="MGSM":
    #     file1=f"results/{dataset}/ModelFusion/{llm}/results.json"
    # else:
    #     file1=f"results/{dataset}/ModelFusion/merged/{llm}.json"
    # file2=f"results/{dataset}/ModelFusion/{llm}/results3.json"
    file1=f"results/{dataset}/ModelFusion/merged/gpt-4o-mini.json"
    file2=f"results/{dataset}/ModelFusion/merged/gemini-1.5-flash-latest.json"
    # merged_file=f"results/{dataset}/ModelFusion/merged/{llm}.json"
    # metric_output_path=f"results/{dataset}/ModelFusion/synthetic/metrics.json"

    with open(file1,"r") as f1:
        data1=json.load(f1)

    with open(file2,"r") as f1:
        data2=json.load(f1)


    Correct=0
    Acc=0
    Total=len(data1)
    # 中间结果正确率
    request=0
    data4=[]
    dev_data=[]
    for i,n in enumerate(data1):
        d={}
        d["question"]=data1[i]["question"]
        d["no"]=data1[i]["no"]
        d["solution"]=data1[i]["solution"]
        d["step_sampling"]=[]
        d["intermediate_results"]=[]
        no1=data1[i]["no"]
        j=0
        k=0
        for j1,m1 in enumerate(data2):
            if data2[j1]["no"]==no1:
                j=j1
                break
        if dataset in ["gsm8k","MGSM"]:
            s = float(str(data1[i]["solution"]).replace(',',''))
        else:
            s = data1[i]["solution"]

        if dataset=="MGSM":
            first_round_answers=data1[i]["intermediate_results"]
            second_round_answers=data2[j]["intermediate_results"]
            d["outputs"]=data1[i]["step_sampling"]+data2[j]["step_sampling"]
        else:
            first_round_answers=data1[i]["ans_list"]
            second_round_answers=data2[j]["intermediate_results"]
            d["outputs"]=data1[i]["outputs"]+data2[j]["step_sampling"]
            
        mixed_answers = first_round_answers+second_round_answers
        d["ans_list"]=mixed_answers
        request+=len(d["ans_list"])
        d["final_answer"],_=most_frequent(d["ans_list"])
        correctness=compute_correctness(d["ans_list"],s, dataset)
        if correctness==1:
            Correct+=1
            d["Correctness"]=True
        else:
            d["Correctness"]=False
        data4.append(d)
        dev_data.append(d)
        
    Acc=Correct*1.0/Total
    print("平均请求次数:",request*1.0/Total)
    print("正确率:",Acc)
    # with open(merged_file,"w") as f:
    #     json.dump(data4,f,indent=4)

def merge_3file(dataset):
    '''math'''
    llm="gpt-4o-mini"
    # llm="gemini-1.5-flash-latest"
    # llm="claude-3-haiku-20240307"
    file1=f"results/{dataset}/ModelFusion/{llm}/results.json"
    file2=f"results/{dataset}/ModelFusion/{llm}/results1.json"
    file3=f"results/{dataset}/ModelFusion/{llm}/results2.json"
    merged_file=f"results/{dataset}/ModelFusion/merged/{llm}.json"
    # metric_output_path=f"results/{dataset}/ModelFusion/synthetic/metrics.json"

    with open(file1,"r") as f1:
        data1=json.load(f1)

    with open(file2,"r") as f1:
        data2=json.load(f1)

    with open(file3,"r") as f1:
        data3=json.load(f1)

    Correct=0
    Acc=0
    Total=len(data1)
    # 中间结果正确率
    request=0
    data4=[]
    dev_data=[]
    for i,n in enumerate(data1):
        d={}
        d["question"]=data1[i]["question"]
        d["no"]=data1[i]["no"]
        d["solution"]=data1[i]["solution"]
        # d["statement"]=data1[i]["statement"]
        d["step_sampling"]=[]
        d["intermediate_results"]=[]
        no1=data1[i]["no"]
        j=0
        k=0
        for j1,m1 in enumerate(data2):
            if data2[j1]["no"]==no1:
                j=j1
                break
        for k1,m2 in enumerate(data3):
            if data3[k1]["no"]==no1:
                k=k1
                break
        if dataset in ["gsm8k","MGSM"]:
            s = float(str(data1[i]["solution"]).replace(',',''))
        else:
            s = data1[i]["solution"]

        first_round_answers=data1[i]["intermediate_results"]
        second_round_answers=data2[j]["intermediate_results"]
        third_round_answers=data3[k]["intermediate_results"]
        mixed_answers = first_round_answers+second_round_answers+third_round_answers
        
        d["outputs"]=data1[i]["step_sampling"]+data2[j]["step_sampling"]+data3[k]["step_sampling"]
        d["ans_list"]=mixed_answers
        request+=len(d["ans_list"])
        d["final_answer"],_=most_frequent(d["ans_list"])
        correctness=compute_correctness(d["ans_list"],s, dataset)
        if correctness==1:
            Correct+=1
            d["Correctness"]=True
        else:
            d["Correctness"]=False
        data4.append(d)
        dev_data.append(d)
        
    Acc=Correct*1.0/Total
    print("平均请求次数:",request*1.0/Total)
    print("正确率:",Acc)
    with open(merged_file,"w") as f:
        json.dump(data4,f,indent=4)



def four_stage_analysis():
    # 分析四阶段正确率
    file1="results/math/SWN1/synthetic/results.json"

    with open(file1,"r") as f1:
        data1=json.load(f1)

    Total=len(data1)
    stage1_answers=[]
    stage1_correctness=0
    stage1_acc=0

    stage2_answers=[]
    stage2_correctness=0
    stage2_acc=0

    stage3_answers=[]
    stage3_correctness=0
    stage3_acc=0

    stage4_answers=[]
    stage4_correctness=0
    stage4_acc=0
    CW=0
    WC=0
    for i,n in enumerate(data1):
        no1=data1[i]["no"]
        s = data1[i]["solution"].replace(" ","").lower()
        answers1=data1[i]["ans_list"][0:5]
        answers1=[item.replace(" ","").lower() for item in answers1]
        if len(set(answers1))==1:
            # stage 1
            stage1_answer=answers1
            stage1_answers.append(stage1_answer)
            stage1_correctness+=compute_correctness(stage1_answer,s)
        
        else:
            # stage 2
            answers2=data1[i]["ans_list"][5:10]
            answers2=[item.replace(" ","").lower() for item in answers2]
            outputs_set=set(answers2)
            if len(outputs_set)==1:
                stage2_answer=answers2
                stage2_answers.append(stage2_answer)
                stage2_correctness+=compute_correctness(stage2_answer,s)
            else:
                A1,_ = most_frequent(answers1)
                A2,_ = most_frequent(answers2)
                # stage 3
                if A1==A2:
                    stage3_answer=answers1+answers2
                    stage3_answers.append(stage3_answer)
                    stage3_correctness+=compute_correctness(stage3_answer,s)
                else:
                    stage4_answer=answers1+answers2
                    stage4_answers.append(stage4_answer)
                    stage4_correctness+=compute_correctness(stage4_answer,s)
                    if compute_correctness(answers1,s)==1 and compute_correctness(answers2,s)==0:
                        CW+=1
                    
                    elif compute_correctness(answers1,s)==0 and compute_correctness(answers2,s)==1:
                        WC+=1
        

    stage1_acc=(len(stage1_answers)-stage1_correctness)*1.0/len(stage1_answers)
    stage2_acc=(len(stage2_answers)-stage2_correctness)*1.0/len(stage2_answers)
    stage3_acc=(len(stage3_answers)-stage3_correctness)*1.0/len(stage3_answers)
    stage4_acc=(len(stage4_answers)-stage4_correctness)*1.0/len(stage4_answers)
    print("阶段1错误率:",stage1_acc,"错误数/总数",len(stage1_answers)-stage1_correctness,len(stage1_answers))
    print("阶段2错误率:",stage2_acc,"错误数/总数",len(stage2_answers)-stage2_correctness,len(stage2_answers))
    print("阶段3错误率:",stage3_acc,"错误数/总数",len(stage3_answers)-stage3_correctness,len(stage3_answers))
    print("阶段4错误率:",stage4_acc,"错误数/总数",len(stage4_answers)-stage4_correctness,len(stage4_answers))
    print("CW:",CW,"WC:",WC)

def mix_3file_analysis(dataset):
    # 分析三个文件合成
    file1=f"results/{dataset}/ModelFusion/merged/gpt-4o-mini.json"
    file2=f"results/{dataset}/ModelFusion/merged/gemini-1.5-flash-latest.json"
    file3=f"results/{dataset}/ModelFusion/merged/claude-3-haiku-20240307.json"
    # file2=f"results/{dataset}/SWN1/gemini-1.5-flash/results1.json"
    # file3=f"results/{dataset}/SWN1/gemini-1.5-flash/results2.json"
    CW_file=f"results/{dataset}/SWN1/CW/CW.json"
    WC_file=f"results/{dataset}/SWN1/CW/WC.json"
    with open(file1,"r") as f1:
        data1=json.load(f1)

    with open(file2,"r") as f1:
        data2=json.load(f1)
    
    with open(file3,"r") as f1:
        data3=json.load(f1)

    first_round_Acc=0
    first_round_Correct=0

    second_round_Acc=0
    second_round_Correct=0

    third_round_Acc=0
    third_round_Correct=0
    mixed_Acc=0
    mixed_Correct=0
    Total=len(data1)
    # 中间结果正确率
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
    for i,n in enumerate(data1):
        no1=data1[i]["no"]
        j=0
        k=0
        for j1,m1 in enumerate(data2):
            if data2[j1]["no"]==no1:
                j=j1
                break    
        for k1,n1 in enumerate(data3):
            if data3[k1]["no"]==no1:
                k=k1
                break
        
        if dataset=="gsm8k":
            s = float(data1[i]["solution"].replace(',',''))
        else:
            s = data1[i]["solution"]

        first_round_answers=data1[i]["ans_list"][0:6]
        # print(len(first_round_answers))
        second_round_answers=data2[j]["ans_list"][0:6]
        third_round_answers=data3[k]["ans_list"][0:6]
        mixed_answers = data1[i]["ans_list"][0:6]+data2[j]["ans_list"][0:6]+data3[k]["ans_list"][0:4]
        
        request_len+=len(data1[i]["step_sampling"])
        # request_len+=len(data1[i]["outputs"])
        first_round_Correct+=compute_correctness(first_round_answers,s,dataset)
        second_round_Correct+=compute_correctness(second_round_answers,s,dataset)
        third_round_Correct+=compute_correctness(third_round_answers,s,dataset)
        mixed_Correct+=compute_correctness(mixed_answers,s,dataset)

        # if compute_correctness(first_round_answers,s)==1 and compute_correctness(second_round_answers,s)==0:
        #     print("no:",no1,"mixed_correctness:",compute_correctness(mixed_answers,s))
    #     if compute_correctness(first_round_answers,s)==1 and compute_correctness(second_round_answers,s)==1:
    #         CC+=1
    #     elif compute_correctness(first_round_answers,s)==1 and compute_correctness(second_round_answers,s)==0:
    #         CW+=1
    #         CW_data.append(data1[i])
    #     elif compute_correctness(first_round_answers,s)==0 and compute_correctness(second_round_answers,s)==1:
    #         WC+=1
    #         WC_data.append(data1[i])
    #     else:
    #         if compute_correctness(mixed_answers,s)==1:
    #             print(no1)
    #         WW+=1
    
    # print("CC:",CC,"CW:",CW,"WC:",WC,"WW:",WW)  
    first_round_Acc=first_round_Correct*1.0/Total
    second_round_Acc=second_round_Correct*1.0/Total
    third_round_Acc=third_round_Correct*1.0/Total
    mixed_Acc=mixed_Correct*1.0/Total

    print("dataset:",dataset)
    print("模型1正确率:",first_round_Acc)
    # print("模型1正确答案总数:",first_round_Correct)
    print("模型2正确率:",second_round_Acc)
    print("模型3正确率:",third_round_Acc)
    print("混合正确率:",mixed_Acc)
    print("-"*50)

def calculate_scores_with_weights(first_round_answers, second_round_answers, third_round_answers,weights):
    first_round_answers=[item for item in first_round_answers if item !="Error" and item!=""]
    first_round_answers=[item.replace(" ","") for item in first_round_answers]
    second_round_answers=[item for item in second_round_answers if item !="Error" and item!=""]
    second_round_answers=[item.replace(" ","") for item in second_round_answers]
    third_round_answers=[item for item in third_round_answers if item !="Error" and item!=""]
    third_round_answers=[item.replace(" ","") for item in third_round_answers]

    
    # 统计每个答案的得分
    answer_scores = {}

    # 计算内部得分V0（最朴素版本）
    # def internal_consistency_score(answers):
    #     score = {}
    #     for answer in answers:
    #         score[answer] = score.get(answer, 0) + 1
    #     return score
    
    # V1频次*频率
    # def internal_consistency_score(answers):
    #     score = {}
    #     total_answers = len(answers)
        
    #     # 计算每个答案的出现次数
    #     counts = {}
    #     for answer in answers:
    #         counts[answer] = counts.get(answer, 0) + 1
        
    #     # 计算得分
    #     for answer, count in counts.items():
    #         score[answer] = count * (count / total_answers)
        
    #     return score
    
    # V2 entropy
    def internal_consistency_score(answers):
        score = {}
        total_answers = len(answers)
        if len(answers)==0:
            return score
        # 计算每个答案的出现次数
        counts = {}
        for answer in answers:
            counts[answer] = counts.get(answer, 0) + 1
        
        # 计算熵
        entropy = 0
        for count in counts.values():
            probability = count / total_answers
            entropy -= probability * math.log2(probability) if probability > 0 else 0
        
        # 最大熵
        max_entropy = math.log2(len(counts))
        
        bias=1.0/len(answers)
        # 计算归一化权重
        weight = bias + (1-bias) * (1 - (entropy / max_entropy)) if max_entropy > 0 else 1
        # print(weight)
        # 计算得分
        for answer, count in counts.items():
            score[answer] = count * weight
        
        return score

    # 计算外部权重得分
    def external_weight_score(internal_scores, weight):
        score = {key: value * weight for key, value in internal_scores.items()}
        return score

    # 处理模型A
    a_internal_scores = internal_consistency_score(first_round_answers)
    a_weighted_scores = external_weight_score(a_internal_scores, weights['A'])
    
    # 处理模型B
    b_internal_scores = internal_consistency_score(second_round_answers)
    b_weighted_scores = external_weight_score(b_internal_scores, weights['B'])

    # 处理模型C
    c_internal_scores = internal_consistency_score(third_round_answers)
    c_weighted_scores = external_weight_score(c_internal_scores, weights['C'])
    # print(c_internal_scores)
    # print(c_external_scores)
    # 合并得分
    for answer in set(first_round_answers + second_round_answers + third_round_answers):
        total_score = 0
        # 加入权重后得分
        total_score += a_weighted_scores.get(answer, 0)
        total_score += b_weighted_scores.get(answer, 0)
        total_score += c_weighted_scores.get(answer, 0)
        
        answer_scores[answer] = total_score

    # 按优先级选择得分最高的答案
    def select_best_answer():
        max_score = -1
        best_answer = None

        # 优先级顺序：first_round > second_round > third_round
        for answer in first_round_answers + second_round_answers + third_round_answers:
            score = answer_scores.get(answer, 0)
            if score > max_score:
                max_score = score
                best_answer = answer
        
        return best_answer
    
    return select_best_answer()

def vote_algorithmn_performance(dataset,weights):
    # 分析三个文件合成
    file1=f"results/{dataset}/ModelFusion/merged/gpt-4o-mini.json"
    file2=f"results/{dataset}/ModelFusion/merged/gemini-1.5-flash-latest.json"
    file3=f"results/{dataset}/ModelFusion/merged/claude-3-haiku-20240307.json"
    CW_file=f"results/{dataset}/SWN1/CW/CW.json"
    WC_file=f"results/{dataset}/SWN1/CW/WC.json"
    with open(file1,"r") as f1:
        data1=json.load(f1)

    with open(file2,"r") as f1:
        data2=json.load(f1)
    
    with open(file3,"r") as f1:
        data3=json.load(f1)

    first_round_Acc=0
    first_round_Correct=0

    second_round_Acc=0
    second_round_Correct=0

    third_round_Acc=0
    third_round_Correct=0
    mixed_Acc=0
    mixed_Correct=0
    Total=len(data1)
    # 中间结果正确率
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
    
    for i,n in enumerate(data1):
        no1=data1[i]["no"]
        j=0
        k=0
        for j1,m1 in enumerate(data2):
            if data2[j1]["no"]==no1:
                j=j1
                break    
        for k1,n1 in enumerate(data3):
            if data3[k1]["no"]==no1:
                k=k1
                break
        
        if dataset=="gsm8k":
            s = float(data1[i]["solution"].replace(',',''))
        else:
            s = data1[i]["solution"]

        first_round_answers=data1[i]["ans_list"][0:6]
        # print(len(first_round_answers))
        second_round_answers=data2[j]["ans_list"][0:6]
        third_round_answers=data3[k]["ans_list"][0:4]
        
        request_len+=len(data1[i]["step_sampling"])
        first_round_Correct+=compute_correctness(first_round_answers,s,dataset)
        second_round_Correct+=compute_correctness(second_round_answers,s,dataset)
        third_round_Correct+=compute_correctness(third_round_answers,s,dataset)
        # second_round_answers=[]
        # mixed_Correct+=compute_correctness(mixed_answers,s,dataset)
        if dataset in ["MGSM","math","mmlu_physical","mmlu_pro","AGIEval"]:
            mixed_final_answer=calculate_scores_with_weights(second_round_answers,first_round_answers,third_round_answers,weights)
            mixed_answers=second_round_answers+first_round_answers+third_round_answers
        else: 
            mixed_final_answer=calculate_scores_with_weights(first_round_answers,second_round_answers,third_round_answers,weights)
            mixed_answers=first_round_answers+second_round_answers+third_round_answers
        correctness=compute_correctness([mixed_final_answer],s,dataset)
        mixed_Correct+=correctness
        # if correctness==0:
        #     print(mixed_answers,s)

    first_round_Acc=first_round_Correct*1.0/Total
    second_round_Acc=second_round_Correct*1.0/Total
    third_round_Acc=third_round_Correct*1.0/Total
    mixed_Acc=mixed_Correct*1.0/Total

    # print("dataset:",dataset)
    # print("模型1正确率:",first_round_Acc)
    # print("模型2正确率:",second_round_Acc)
    # print("模型3正确率:",third_round_Acc)
    # print("混合正确率:",mixed_Acc)
    # print("-"*50)
    return mixed_Acc

def search_optimal_weights(dataset):
        best_weights = None
        best_score = float('-inf')

        for a_weight in range(1, 6):
            for b_weight in range(1, 6):
                for c_weight in range(1, 6):
                    if a_weight==b_weight and b_weight==c_weight and a_weight!=1:
                        continue
                    weights = {'A': a_weight, 'B': b_weight, 'C': c_weight}
                    score = vote_algorithmn_performance_in_1file(dataset,weights)
                    if score > best_score:
                        best_score = score
                        best_weights = weights

        return best_weights, best_score

def correlation(dataset):
    file=f"Results/{dataset}/MS/closed_source/gpt-4o-mini.json"
    # file=f"Results/{dataset}/MS/open_source/results.json"
    with open(file,"r") as f1:
        data1=json.load(f1)


    first_round_Acc=0
    first_round_Correct=0
    w_dict={}
    for i,n in enumerate(data1):
        if dataset in ["gsm8k","MGSM"]:
            s = float(str(data1[i]["solution"]).replace(',',''))
        else:
            s = data1[i]["solution"]

        first_round_answers=data1[i]["ans_list"][0:16]

        def internal_consistency_score(answers):
            total_answers = len(answers)
            if len(answers)==0:
                return 0.1
            # 计算每个答案的出现次数
            counts = {}
            for answer in answers:
                counts[answer] = counts.get(answer, 0) + 1
            
            # 计算熵
            entropy = 0
            for count in counts.values():
                probability = count / total_answers
                entropy -= probability * math.log2(probability) if probability > 0 else 0
            
            # 最大熵
            max_entropy = math.log2(len(counts))
            
            bias=1.0/len(answers)
            # 计算归一化权重
            weight = bias + (1-bias) * (1 - (entropy / max_entropy)) if max_entropy > 0 else 1
            # print(weight)
            # 计算得分
            
            return weight

        w = internal_consistency_score(first_round_answers)
        correctness=compute_correctness(first_round_answers,s,dataset)
        if w not in w_dict:
            w_dict[w]=[correctness,1]
        else:
            w_dict[w][0]+=correctness
            w_dict[w][1]+=1
    print(w_dict)
    # 定义分组区间
    categories = {
        "[0.1-0.2)": [0.1, 0.2],
        "[0.2-0.4)": [0.2, 0.4],
        "[0.4-0.6)": [0.4, 0.6],
        "[0.6-1]": [0.6, 1.0]
    }

    # 初始化结果字典
    result = {key: [0, 0] for key in categories}

    # 遍历原始字典并分类
    for key, value in w_dict.items():
        for category, (start, end) in categories.items():
            if start <= key < end or (category == "[0.6-1]" and start <= key <= end):
                result[category][0] += value[0]
                result[category][1] += value[1]
                break

    # 打印结果
    print(result)
        


def statistic(dataset):
    # llm="gpt-4o-mini"
    llm="claude-3-haiku-20240307"
    # llm="gemini-1.5-flash-latest"
    file=f"results/{dataset}/ModelFusion/merged/{llm}.json"
    with open(file,"r") as f1:
        data1=json.load(f1)

    answer4_num=0
    for i,n in enumerate(data1):
        d={}
        d["question"]=data1[i]["question"]
        d["no"]=data1[i]["no"]
        d["solution"]=data1[i]["solution"]
        d["step_sampling"]=data1[i]["step_sampling"]
        d["intermediate_results"]=data1[i]["intermediate_results"]
        no1=data1[i]["no"]
        j=0
        
        # s = float(data1[i]["solution"].replace(',',''))
        s = data1[i]["solution"]

        answers=data1[i]["ans_list"]
        filtered_answers=[item for item in answers if item!=""]
        ans_set=set(filtered_answers)
        ranked_list=[]
        for a in ans_set:
            num=filtered_answers.count(a)
            ranked_list.append(num)
        if len(ranked_list)<=3:
            answer4_num+=1
        else:
            rl=sorted(ranked_list,reverse=True)
            # print(rl)
            if rl[3]<=1:
                answer4_num+=1
    print(answer4_num)



datasets=[
    # "gsm8k",
    # "MGSM",
    # "math",
    # "mmlu_pro",
    # "DATE",
    "MathBench"
    # "last_letters",
    # "MathBench"
    # "AGIEval",
    # "logiQA"
    # "TruthfulQA"
    # "mmlu_physical",
    # "mmlu_health",
    # "simpleqa"
    ]

# initial weight
weights_dict={
    "gsm8k":{'A': 1, 'B': 1, 'C': 1},
    "math": {'A': 1, 'B': 1, 'C': 1},
    "last_letters":{'A': 1, 'B': 1, 'C': 1},
    "mmlu_physical":{'A': 1, 'B': 1, 'C': 1},
    "mmlu_pro":{'A': 1, 'B': 1, 'C': 1}, 
    "AGIEval":{'A': 1, 'B': 1, 'C': 1},
    "mmlu_health":{'A': 1, 'B': 1, 'C': 1},
    "MGSM":{'A': 1, 'B': 1, 'C': 1},
    "simpleqa": {'A': 1, 'B': 1, 'C': 1},
    "DATE": {"A":1, "B":1, "C":1},
    "MathBench": {"A":1, "B":1, "C":1}
}


# full version
#
# weights_dict={
#     "gsm8k":{'A': 1, 'B': 1, 'C': 1},
#     "last_letters":{'A': 1, 'B': 1, 'C': 1},
#     "math": {'A': 5, 'B': 3, 'C': 2},
#     "mmlu_phsical":{'A': 4, 'B': 2, 'C': 1},
#     "mmlu_pro":{'A': 4, 'B': 3, 'C': 3}, 
#     "AGIEval":{'A': 4, 'B': 3, 'C': 1},
#     "mmlu_health": {'A': 4, 'B': 3, 'C': 1},
#     "MGSM": {'A': 2, 'B': 2, 'C': 1},
#     "DATE": {"A":1, "B":1, "C":1},
#     "MathBench": {"A":1, "B":1, "C":1}
# }

# efficient version
# weights_dict={
#     "gsm8k":{'A': 2, 'B': 2, 'C': 1},
#     "last_letters":{'A': 5, 'B': 5, 'C': 1},
#     "math": {'A': 1, 'B': 1, 'C': 1},
#     "mmlu_physical":{'A': 2, 'B': 5, 'C': 1},
#     "mmlu_pro":{'A': 3, 'B': 2, 'C': 2}, 
#     "AGIEval":{'A': 4, 'B': 3, 'C': 1},
#     "MGSM": {'A': 4, 'B': 3, 'C': 1},
# }




# for dataset in datasets:
#     # merge_2file(dataset)
#     # statistic(dataset)
#     # mix_file_analysis(dataset)
#     # synthetic_file(dataset)
#     # single_file_analysis(dataset)
#     correlation(dataset)
#     # mix_results_in_1file(dataset)
#     # mix_3file_analysis(dataset)
#     # print(vote_algorithmn_performance(dataset,weights_dict[dataset]))
#     # print(search_optimal_weights(dataset))
#     # print(vote_algorithmn_performance_in_1file(dataset,weights_dict[dataset]))
#     # synthetic_3file(dataset,weights_dict[dataset])

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Run specific functions on datasets")
    parser.add_argument("function", type=str, help="The function to run")
    parser.add_argument("dataset", type=str, help="The dataset to process")
    parser.add_argument("--weights", type=str, default=None, help="Weights for specific functions (optional)")

    args = parser.parse_args()

    # 获取命令行参数
    function_name = args.function
    dataset = args.dataset
    weights = args.weights

    # 通过字符串调用对应的函数
    if function_name == "merge_2file":
        merge_2file(dataset)
    elif function_name == "statistic":
        statistic(dataset)
    elif function_name == "mix_file_analysis":
        mix_file_analysis(dataset)
    elif function_name == "synthetic_file":
        synthetic_file(dataset)
    elif function_name == "single_file_analysis":
        single_file_analysis(dataset)
    elif function_name == "correlation":
        correlation(dataset)
    elif function_name == "mix_results_in_1file":
        mix_results_in_1file(dataset)
    elif function_name == "mix_3file_analysis":
        mix_3file_analysis(dataset)
    elif function_name == "vote_algorithmn_performance":
        if weights is not None:
            vote_algorithmn_performance(dataset, weights)
        else:
            print("Error: Weights are required for vote_algorithmn_performance")
    elif function_name == "search_optimal_weights":
        search_optimal_weights(dataset)
    elif function_name == "vote_algorithmn_performance_in_1file":
        if weights is not None:
            vote_algorithmn_performance_in_1file(dataset, weights)
        else:
            print("Error: Weights are required for vote_algorithmn_performance_in_1file")
    elif function_name == "synthetic_3file":
        if weights is not None:
            synthetic_3file(dataset, weights)
        else:
            print("Error: Weights are required for synthetic_3file")
    else:
        print(f"Error: Function {function_name} is not recognized")