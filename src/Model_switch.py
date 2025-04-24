import llm_call
from llm_call import Hg_model
import json
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from load_dataset import load_dataset
import argparse

def reasoning_layer(
    input,
    n_Sampling: int,
    stop: list[str] = [],
    llm: str = "gpt-4o-mini"
):
    outputs=[]
    for i in range(n_Sampling):
        if llm.__contains__("gemini"):
            outputs.append(llm_call.gemini(input, model=llm, stop=stop))
        elif llm.__contains__("claude"):
            outputs.append(llm_call.claude(input, model=llm,stop=stop))
        elif llm.__contains__("gpt"):
            outputs.append(llm_call.gpt(input, model=llm, stop=stop))
        else:
            outputs.append(llm.run_llm(input))


    return outputs
def metric(dataset_name, results):
    total=len(results)
    correct=0
    for data in results:
        correct+=compute_correctness(data["ans_list"],data,dataset_name)
    acc=correct*1.0/total

    
    return total, acc

def log_out(dataset_name, results):
    total, acc= metric(dataset_name,results)
    metric_output_path=f"Results/{dataset_name}/MS/metrics_test.json"
    with open(metric_output_path, "w") as f:
        f.write(f"total={total}")
        f.write("\n")
        f.write(f"acc={acc}")
        # f.write(f"acc2={acc2}")

    output_path = f"Results/{dataset_name}/MS/results_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

def evaluate(num_workers, dataset_name,Sampling,Sampling_Numbers,modellist,results_sampling,ConsistencyThreshold):
    dataset=load_dataset(dataset_name,Sampling,Sampling_Numbers)
    results = []
    def process_data(data):
        # prompt = form_propmt(dataset_name,data)
        # outputs, ans_list, final_result, log , ans_aggregator,final_result_aggregator = stepwise_network.stepwise_network(dataset_name, prompt,data['question'], statement_sampling=statement_layer_sampling, reasoning_step_sampling=reasoning_step_sampling, results_sampling=results_sampling)
        index=0
        data["ans_sampling"]=[]
        data["ans_list"]=[]
        while index<len(modellist):
            llm=modellist[index]
            outputs, ans_list, final_result= generate(
                dataset_name,
                data['question'],
                llm, 
                results_sampling=results_sampling,
                )
            data["ans_sampling"]+=outputs
            data["ans_list"]+= ans_list
            if calculate_num_sampling(ans_list)<ConsistencyThreshold:
                index+=1
            else:
                break
        # data["ans_aggregator"]=ans_aggregator
        # data["final_result_aggregator"]=final_result_aggregator
        data["final_answer"]=most_frequent(data["ans_list"])
        return data
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_data, data): data for data in dataset}
        for future in tqdm(as_completed(futures), total=len(dataset)):
            results.append(future.result())

    log_out(dataset_name, results)
def generate(dataset_name,question,llm,results_sampling):
    prompt_tail="\n\nThink step by step and slove this question. Your final answer should be in the form \"The answer to this question is \\boxed{X}\", X is your answer and at the end of your response:\n\n"
    prompt=f"Question:\n{question}\n"+prompt_tail
    outputs=reasoning_layer(input=prompt,n_Sampling=results_sampling,llm=llm)
    inputs=outputs
    if dataset_name in ["GSM8K","MGSM","MathBench"]:
        outputs=extract_final_answers_gsm8k(inputs)
        filtered_outputs = [item for item in outputs if item != "Error"]
        most_frequent_ans=most_frequent(filtered_outputs)
        output=most_frequent_ans
    elif dataset_name in ["MATH"]:
        outputs=get_boxed_answer(inputs)
        most_frequent_ans=most_frequent(outputs)
        output=most_frequent_ans
    elif dataset_name in ["MMLU-Pro"]:
        outputs=get_boxed_answer(inputs)
        outputs=get_last_uppercase(outputs)
        most_frequent_ans=most_frequent(outputs)
        output=most_frequent_ans
    elif dataset_name=="DATE":
        outputs=get_boxed_answer(inputs)
        most_frequent_ans=most_frequent(outputs)
        output=most_frequent_ans
    else:
        outputs=inputs
        output=inputs[0]
    answer_list=outputs
    final_answer=output
    return inputs,answer_list,final_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the dataset.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for parallel processing.")
    parser.add_argument("--dataset_name", type=str, default="GSM8K", help="Name of the dataset.")
    parser.add_argument("--Sampling",type=bool, default=False, help="Whether to do sampling")
    parser.add_argument("--Sampling_Numbers",type=int, default=10, help="Number of Sampling")
    parser.add_argument("--results_sampling", type=int, default=5, help="Sampling numbers of output.")
    parser.add_argument("--modellist",type=str,default="")
    parser.add_argument("--ConsistencyThreshold", type=int, default=1, help="Consistency threshold to determine if a switch is needed")
    parser.add_argument("--Open_SourceModel", type=bool, default=False)

    
    args = parser.parse_args()
    modellist=args.modellist.split("|")
    if args.Open_SourceModel:
        modellist=[Hg_model(model) for model in modellist]
    ConsistencyThreshold=args.ConsistencyThreshold
    assert 0<ConsistencyThreshold<=1
    
    evaluate(
        args.num_workers,
        dataset_name=args.dataset_name,
        Sampling=args.Sampling,
        Sampling_Numbers=args.Sampling_Numbers,
        modellist=modellist,
        results_sampling=args.results_sampling,
        ConsistencyThreshold=ConsistencyThreshold
    )

