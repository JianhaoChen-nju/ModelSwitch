import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import matplotlib.ticker as ticker
import numpy as np
import json
import math
from Analyze import compute_correctness
# from statsmodels.nonparametric.smoothers_lowess import lowess
def correlation_data(dataset):
    file_list=[
        f"Results/{dataset}/MS/closed_source/gpt-4o-mini.json",
        f"Results/{dataset}/MS/closed_source/gemini-1.5-flash-latest.json",
        f"Results/{dataset}/MS/closed_source/claude-3-haiku-20240307.json"
    ]

    file = f"Results/{dataset}/MS/open_source/results.json"
    model_list = ["","","","gemma-2-9b-it", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]

    # 将 file 复制成与 model_list 长度相同的列表
    file_list1 = [file] * len(model_list)
    file_list+=file_list1
    # 使用 zip 组合成对
    pairs = list(zip(file_list, model_list))

    w_dict_list=[]
    for file,model in pairs:
        # print(file,model)
        with open(file,"r") as f1:
            data1=json.load(f1)

        w_dict={}
        for i,n in enumerate(data1):
            if dataset in ["gsm8k","MGSM"]:
                s = float(str(data1[i]["solution"]).replace(',',''))
            else:
                s = data1[i]["solution"]

            if file.__contains__("closed_source"):
                ans_list=data1[i]["ans_list"]
            else:
                ans_list=data1[i][f"{model}_ans_list"]

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

            w = internal_consistency_score(ans_list)
            correctness=compute_correctness(ans_list,s,dataset)
            if w not in w_dict:
                w_dict[w]=[correctness,1]
            else:
                w_dict[w][0]+=correctness
                w_dict[w][1]+=1
        w_dict_list.append(w_dict)
    return w_dict_list

def correlation():
    # 数据
    data_list=correlation_data("MATH")
    plt.figure(figsize=(12, 9))

    color_list=["green","red","purple","blue","black","yellow"]
    for i,data in enumerate(data_list):
    # 对于合并后的数据，处理 value[1] 小于 10 的情况
        # final_data=data
        final_data = {}
        for key, value in data.items():
            if value[1] < 15:  # 如果 value[1] 小于 10
                # 找到最近的 key
                closest_key = None
                min_distance = float('inf')
                for other_key in final_data.keys():
                    distance = abs(key - other_key)
                    if distance < min_distance:
                        min_distance = distance
                        closest_key = other_key

                if closest_key is not None:
                    # 合并到最近的 key
                    combined_value_1 = final_data[closest_key][0] + value[0]
                    combined_value_2 = final_data[closest_key][1] + value[1]
                    # 更新 key 值（权重加权平均）
                    new_key = (closest_key * final_data[closest_key][1] + key * value[1]) / combined_value_2
                    # 更新合并后的结果
                    final_data[new_key] = [combined_value_1, combined_value_2]
                    del final_data[closest_key]  # 删除旧的 key
                else:
                    # 如果没有最近的 key，直接加入
                    final_data[key] = value
            else:
                # 如果 value[1] >= 10，直接加入
                final_data[key] = value
        print(final_data)

        # 计算 Accuracy（value[0]/value[1]）
        consistency_scores = list(final_data.keys())
        accuracies = [v[0] / v[1] for v in final_data.values()]

        # 创建 DataFrame 以便使用 seaborn
        df = pd.DataFrame({
            "Consistency Score": consistency_scores,
            "Accuracy": accuracies
        })

        # 绘制散点图和拟合曲线
        
        #V1
        sns.regplot(
            x="Consistency Score", 
            y="Accuracy", 
            data=df, 
            scatter_kws={'color': color_list[i], 's': 50}, 
            line_kws={'color': color_list[i]}, 
            ci=None,
            )
    #V1

    #V2
    # plt.scatter(df["Consistency Score"], df["Accuracy"], color='blue', s=50, label="Data Points")

    # # 使用 statsmodels 的 LOWESS 进行平滑
    # smoothed = lowess(df["Accuracy"], df["Consistency Score"], frac=1)  # 调整 frac 参数控制平滑程度

    # # 绘制平滑曲线
    # plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, label="Smooth Curve (LOWESS)")
    #V2

    # 设置标题和标签
    plt.xlabel("Consistency Score", fontsize=28)
    plt.ylabel("Accuracy", fontsize=28)

    # 获取当前的轴对象
    ax = plt.gca()

    # 隐藏上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置刻度格式
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))  # x轴刻度保留1位小数
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))  # y轴刻度保留1位小数

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=24)

    # 设置坐标轴范围
    ax.set_ylim(-0.1, 1.1)


    # 调整原点的显示：将x轴和y轴的0刻度对齐
    ax.spines['left'].set_position(('data', 0))  # y轴移动到x=0的位置
    ax.spines['bottom'].set_position(('data', 0))  # x轴移动到y=0的位置

    # 设置刻度方向
    ax.spines['left'].set_bounds(0, 1.1)  # 限制y轴从0到1
    ax.spines['bottom'].set_bounds(0, 1.1)  # 限制x轴从0到1

    xticks = [0.2,0.4,0.6,0.8,1]
    ax.set_xticks(xticks)
    ax.set_xlim(-0.1, 1.1)
    # 显示图形
    plt.tight_layout()
    plt.show()

correlation()