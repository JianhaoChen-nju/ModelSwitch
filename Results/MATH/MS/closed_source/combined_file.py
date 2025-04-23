import json
with open("gemini-1.5-pro.json","r")as f:
    gemini=json.load(f)
with open("gpt-4o.json","r")as f:
    gpt=json.load(f)
with open("results.json","r")as f:
    datas=json.load(f)
question_dict={}
for data in datas:
    question_dict[data["question"]]=data
for data in gemini:
    question_dict[data["question"]]["gemini-1.5-pro_sampling"]=data["ans_sampling"]
    question_dict[data["question"]]["gemini-1.5-pro_ans_list"]=data["ans_list"]
    question_dict[data["question"]]["gemini-1.5-pro_final_answer"]=data["final_answer"]
for data in gpt:
    question_dict[data["question"]]["gpt-4o_sampling"]=data["ans_sampling"]
    question_dict[data["question"]]["gpt-4o_ans_list"]=data["ans_list"]
    question_dict[data["question"]]["gpt-4o_final_answer"]=data["final_answer"]
results=[]
for key,value in question_dict.items():
    results.append(value)
with open("results.json","w")as f:
    json.dump(results,f,indent=4)