import json
def combine_files(Type,dataset,trigger):
    if Type=="ModelSwitch":
        if trigger == "closed_source":
            model_list=["gpt-4o-mini","gemini-1.5-flash-latest","claude-3-haiku-20240307","gpt-4o","gemini-1.5-pro"]
            data_list=[]
            dict_question_data_list={}
            for model in model_list:
                with open(f"Results/{dataset}/MS/{trigger}/{model}.json","r") as f:
                    data=json.load(f)
                    data_list.append(data)
            for index_data,datas in enumerate(data_list):
                for data in datas:
                    if index_data>2:
                        data["outputs"]=data["step_sampling"]

                    if data["no"] in dict_question_data_list:
                        dict_question_data_list[data["no"]][f"{model_list[index_data]}_outputs"]=data["outputs"]
                        dict_question_data_list[data["no"]][f"{model_list[index_data]}_ans_list"]=data["ans_list"]
                        dict_question_data_list[data["no"]][f"{model_list[index_data]}_final_answer"]=data["final_answer"]
                        dict_question_data_list[data["no"]][f"{model_list[index_data]}_Correctness"]=data["Correctness"]
                    else:
                        dict_question_data_list[data["no"]]={
                            "question":data["question"],
                            "no":data["no"],
                            "solution":data["solution"],
                            f"{model_list[index_data]}_outputs":data["outputs"],
                            f"{model_list[index_data]}_ans_list":data["ans_list"],
                            f"{model_list[index_data]}_final_answer":data["final_answer"],
                            f"{model_list[index_data]}_Correctness":data["Correctness"]

                        }
            new_data=[]
            for key,value in dict_question_data_list.items():
                new_data.append(value)
            with open(f"Results/{dataset}/MS/{trigger}/results.json","w")as f:
                json.dump(new_data,f,indent=4)
        else:
            model="Llama-3.1-70B-Instruct"
            with open(f"Results/{dataset}/MS/{trigger}/results.json","r")as f:
                data_original=json.load(f)
            with open(f"Results/{dataset}/MS/{trigger}/results_70B.json","r")as f:
                data_new=json.load(f)
            data_dict_question={}
            for data in data_new:
                question =data['question']
                data_dict_question[question]=data
            for data1 in data_original:
                no=data1['question']
                data1[f"{model}_Sampling"]=data_dict_question[no][f"{model}_Sampling"]
                data1[f"{model}_ans_list"]=data_dict_question[no][f"{model}_ans_list"]
                data1[f"{model}_final_answer"]=data_dict_question[no][f"{model}_final_answer"]
            with open(f"Results/{dataset}/MS/{trigger}/results.json","w")as f:
                json.dump(data_original,f,indent=4)

    else:
        datas=[]
        dict_map={}
        new_datas=[]
        for i in range(1,3):
            with open(f"Results/{dataset}/MOA/closed_source/layer{i}.json","r") as f:
                data=json.load(f)
                datas.append(data)
        for data in datas[1]:
            dict_map[data['no']]=data
        for data in datas[0]:
            new_data={
                "question":data["question"],
                "no":data["no"],
                "solution":data["solution"],
                "proposer_outputs":data["proposer_outputs"],
                "aggregator_outputs1":data["aggregator_outputs1"],
                "aggregator_outputs2":dict_map[data['no']]["proposer_outputs"],
                "aggregator_outputs3":dict_map[data['no']]["aggregator_outputs1"],
                "aggregator_outputs4":dict_map[data['no']]["aggregator_outputs2"],
                "final_output":dict_map[data['no']]["final_output"],
                "final_answer":dict_map[data['no']]["final_answer"],
                "Correctness":dict_map[data['no']]["Correctness"]



            }
            new_datas.append(new_data)
        with open(f"Results/{dataset}/MOA/closed_source/results.json","w") as f:
            json.dump(new_datas,f,indent=4)

dataset_list=["MGSM"]
for dataset in dataset_list:
    combine_files("MOA",dataset,"open_source")