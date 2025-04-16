import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
from collections import defaultdict
import matplotlib.ticker as ticker
import numpy as np
import json
import math
from data import correlation_list
from Analyze import compute_correctness,is_equiv
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import linregress
import random
# from statsmodels.nonparametric.smoothers_lowess import lowess

# 替换为实际路径
font_path = '/usr/share/fonts/truetype/cmu/cmunrm.ttf'
font_prop = fm.FontProperties(fname=font_path)
font_path_bold_italic = '/usr/share/fonts/truetype/cmu/cmunbi.ttf'  # 假设这个是粗斜体
font_prop_bold_italic = fm.FontProperties(fname=font_path_bold_italic)

# 全局设置字体
plt.rcParams['font.family'] = font_prop.get_name()  # 设置全局字体
plt.rcParams['mathtext.fontset'] = 'custom'  # 设置数学字体
plt.rcParams['mathtext.it'] = font_prop.get_name()  # 设置数学斜体字体


def compute_acc(ans_list,s,dataset):

    acc=0
    is_correct=0
    ans_list=[item for item in ans_list if item !="Error"]
    ans_list=[item.replace(" ","") for item in ans_list]
    total = len(ans_list)
    for ans in ans_list:
        if dataset in ["GSM8K","MGSM"]:
            a = float(ans.replace(',',''))
            if abs(s-a) < 1e-6:
                is_correct+=1
        elif dataset=="math" or dataset=="MATH":
            if is_equiv(s.replace(" ",""),ans):
                is_correct+=1
        elif dataset=="simpleqa":
            if ans.replace(",","").replace(".","").replace(" ","").lower()==s.replace(",","").replace(".","").replace(" ","").lower():
                is_correct+=1
        elif dataset=="logiQA":
            ans_dict={0:"A",1:"B",2:"C",3:"D"}
            s=ans_dict[s]
            if ans.replace(" ","").lower()==s.replace(" ","").lower():
                is_correct+=1
        else:
        #去空格，去大小写
            if ans.replace(" ","").lower()==s.replace(" ","").lower():
                is_correct+=1

    if total==0:
        return 0
    return is_correct*1.0/total

def replace_empty_answers(ans_list):
    # 定义可选的字母
    options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # 遍历ans_list，替换空字符串
    for i in range(len(ans_list)):
        if ans_list[i] == "":
            ans_list[i] = random.choice(options)

    return ans_list

def correlation_data(dataset):
    file_list=[
        f"Results/{dataset}/MS/closed_source/results.json"
    ]*3 + [f"Results/{dataset}/MS/open_source/results.json"] *3
    model_list = ["gpt-4o-mini","gemini-1.5-flash-latest","claude","gemma-2-9b-it", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
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
        
        
        return entropy
    # 使用 zip 组合成对
    pairs = list(zip(file_list, model_list))

    w_dict_list=[]
    for file,model in pairs:
        # print(file,model)
        with open(file,"r") as f1:
            data1=json.load(f1)

        w_dict={}
        for i,n in enumerate(data1):
            if dataset in ["GSM8K","MGSM"]:
                s = float(str(data1[i]["solution"]).replace(',',''))
            else:
                s = data1[i]["solution"]

            
            ans_list=data1[i][f"{model}_ans_list"]
            # ans_list=[item for item in ans_list if item!=""]


            # # 调用函数替换空字符串
            # ans_list = replace_empty_answers(ans_list)

            w = internal_consistency_score(ans_list)
            acc=compute_acc(ans_list,s,dataset)
            w_dict[i]=[w,acc]
        w_dict_list.append(w_dict)
    return w_dict_list


def correlation_v1():
    # 数据
    # 
    data_list = correlation_list
    print(data_list)
    data_list2 = correlation_data("MathBench")
    label_list = ["GPT-4o mini", "Gemini 1.5 Flash", "Claude 3 Haiku", "Gemma-2-9B-it", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
    color_list = ["#FFBE7A", "#82B0D2", "#C76DA2", "#2878B5", "#BEB8DC", "#96CCCB"]

    # 创建 2x3 的子图布局
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))  # 调整图形大小
    axes = axes.flatten()  # 将 2x3 的二维数组展平为一维数组，方便索引

    for i, data in enumerate(data_list):
        # 计算 Accuracy（value[0]/value[1]）
        consistency_scores = [v[0] for v in data.values()]
        accuracies = [v[1]*100 for v in data.values()]

        # 创建 DataFrame 以便使用 seaborn
        df = pd.DataFrame({
            "Entropy": consistency_scores,
            "Accuracy (%)": accuracies
        })

        # 计算每个点的出现次数
        df['count'] = df.groupby(['Entropy', 'Accuracy (%)']).transform('size')

        # 获取当前的子图
        ax = axes[i]

        # 绘制散点图和拟合曲线
        sns.regplot(
            x="Entropy",
            y="Accuracy (%)",
            data=df,
            scatter_kws={'color': color_list[i], 's': 50},  # 设置散点样式
            line_kws={'color': color_list[i]}, 
            ci=None,
            ax=ax,  # 指定当前子图
            scatter=False,  # 不绘制默认的散点
            label=""
        )
        slope, intercept, r_value, p_value, std_err = linregress(df["Entropy"], df["Accuracy (%)"])
        # 截断拟合曲线
        line = ax.get_lines()[-1]
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        mask = (y_data >= 0) & (y_data <= 100)
        line.set_xdata(x_data[mask])
        line.set_ydata(y_data[mask])

        # 使用 plt.scatter 绘制散点
        ax.scatter(
            df["Entropy"],
            df["Accuracy (%)"],
            color=color_list[i],
            s=df['count'] * 12,  # 根据出现次数调整大小
            marker='o',  # 设置散点样式
            alpha=0.2
        )
        # 如果需要进一步清除轴标签
        ax.set(xlabel='', ylabel='')
        # 设置标题
        ax.set_title(label_list[i], fontsize=24)
        # 添加显著性值到图中
        text = f"$r = {r_value:.2f}$\n$p < 0.001$" 
        ax.text(0.63, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
         # 添加水平和垂直虚线
        ax.axhline(y=50, color='grey', linestyle='--', linewidth=2)
        ax.axvline(x=2, color='grey', linestyle='--', linewidth=2)

        # # 添加文本标签
        # ax.text(0.2, 60, "Consistent\n-> correct", fontsize=16, ha='center', va='center',color='#C00000')
        # ax.text(3.6, 40, "Inconsistent\n-> wrong", fontsize=16, ha='center', va='center',color='#C00000')

        # 设置坐标轴标签
        # ax.set_xlabel("Entropy", fontsize=36)
        if i==0:
            ax.set_ylabel("Accuracy (%)", fontsize=36)
        # 设置坐标轴范围
        ax.set_ylim(-10, 110)
        ax.set_xticks([0.0,2.0,4.0])
        ax.set_xlim(-0.4, 4.4) # TODO
        # 设置刻度格式
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # x轴刻度保留1位小数
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # y轴刻度保留1位小数

        # 设置刻度参数
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)

    # 调整子图布局
    plt.tight_layout()
    plt.show()

def correlation_v2():
    # 数据
    data_list = correlation_list
    data_list2 = correlation_data("MathBench")
    label_list = ["GPT-4o mini", "Gemini 1.5 Flash", "Claude 3 Haiku", "Gemma-2-9B-it", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
    color_list = ["#FFBE7A", "#82B0D2", "#C76DA2", "#2878B5", "#BEB8DC", "#96CCCB"]

    # 创建 2x6 的子图布局
    fig, axes = plt.subplots(2, 6, figsize=(24, 9.5))  # 调整图形大小
    axes = axes.flatten()  # 将 2x6 的二维数组展平为一维数组，方便索引

    # 定义绘制单个子图的函数
    def plot_subplot(col, row, ax, data, color, label):
        # 计算 Accuracy（value[0]/value[1]）
        consistency_scores = [v[0] for v in data.values()]
        accuracies = [v[1] * 100 for v in data.values()]

        # 创建 DataFrame 以便使用 seaborn
        df = pd.DataFrame({
            "Entropy": consistency_scores,
            "Accuracy (%)": accuracies
        })

        # 计算每个点的出现次数
        df['count'] = df.groupby(['Entropy', 'Accuracy (%)']).transform('size')

        # 绘制散点图和拟合曲线
        sns.regplot(
            x="Entropy",
            y="Accuracy (%)",
            data=df,
            scatter_kws={'color': color, 's': 50},  # 设置散点样式
            line_kws={'color': color},
            ci=None,
            ax=ax,  # 指定当前子图
            scatter=False,  # 不绘制默认的散点
            label=""
        )
        slope, intercept, r_value, p_value, std_err = linregress(df["Entropy"], df["Accuracy (%)"])
        # 截断拟合曲线
        line = ax.get_lines()[-1]
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        mask = (y_data >= 0) & (y_data <= 100)
        line.set_xdata(x_data[mask])
        line.set_ydata(y_data[mask])

        # 使用 plt.scatter 绘制散点
        ax.scatter(
            df["Entropy"],
            df["Accuracy (%)"],
            color=color,
            s=df['count'] * 12,  # 根据出现次数调整大小
            marker='o',  # 设置散点样式
            alpha=0.2
        )
        # 设置标题
        ax.set_title(label, fontsize=24)
        ax.set(xlabel='', ylabel='')
        # 添加显著性值到图中
        text = f"$r = {r_value:.2f}$\n$p < 0.001$"
        ax.text(0.63, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        # 添加水平和垂直虚线
        ax.axhline(y=50, color='grey', linestyle='--', linewidth=2)
        ax.axvline(x=2, color='grey', linestyle='--', linewidth=2)

        # 设置坐标轴标签
        if col==0:
            ax.set_ylabel("Accuracy (%)", fontsize=36)
        if row==2:
            ax.set_xlabel("Entropy", fontsize=36)
        # 设置坐标轴范围
        ax.set_ylim(-10, 110)
        ax.set_xticks([0.0, 2.0, 4.0])
        ax.set_xlim(-0.4, 4.4)
        # 设置刻度格式
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # x轴刻度保留1位小数
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # y轴刻度保留1位小数

        # 设置刻度参数
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)

    # 绘制第一行（data_list）
    for i, data in enumerate(data_list):
        plot_subplot(i , 1, axes[i], data, color_list[i], label_list[i])

    # 绘制第二行（data_list2）
    for i, data in enumerate(data_list2):
        plot_subplot(i, 2, axes[i + 6], data, color_list[i], label_list[i])

    # 添加大标题
    fig.text(0.5, 0.965, "MATH", fontsize=36, ha='center', va='center')
    fig.text(0.5, 0.495, "MathBench", fontsize=36, ha='center', va='center')

    # 调整子图布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.89, bottom=0.11, hspace=0.5)  # 调整顶部、底部间距和行间距
    plt.show()

def close_source_efficiency():
    # 数据
    datasets = ["GSM8K", "MATH", "MathBench", "MGSM", "Date", "MMLU-Pro"]
    theoretical_sampling = 32  # 理论采样次数
    actual_sampling = np.array([18.08, 21.25, 20.76, 18.51, 20.5, 26.03])  # 实际采样次数

    # 计算节省比例
    savings_percentage = (1 - actual_sampling / theoretical_sampling) * 100

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.bar(datasets, savings_percentage, color='#2CA02C')

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom', fontsize=24)

    # 添加标题和标签
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.set_ylabel("Cost Savings Percentage (%)", fontsize=32)
    ax.set_xticklabels(datasets,rotation=15, fontsize=24)
    ax.set_ylim(0, 50)  # 设置y轴范围

    # 美化图表
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # 获取当前的轴对象
    ax = plt.gca()

    # 隐藏上边框和右边框

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 显示图表
    plt.show()

def open_source_efficiency():
    # 数据
    datasets = ["GSM8K", "MATH", "MathBench", "MGSM", "Date", "MMLU-Pro"]
    theoretical_sampling = 32  # 理论采样次数
    actual_sampling = np.array([24.53, 28.37, 26.92, 25.53, 26.64, 29.75])  # 实际采样次数

    # 计算节省比例
    savings_percentage = (1 - actual_sampling / theoretical_sampling) * 100

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.bar(datasets, savings_percentage, color='#2CA02C')

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom', fontsize=24)

    # 添加标题和标签
    ax.tick_params(axis='both', which='major', labelsize=28)
    # ax.set_title("Computation Savings Across Datasets", fontsize=24)
    ax.set_ylabel("Cost Savings Percentage (%)", fontsize=32)
    ax.set_ylim(0, 30)  # 设置y轴范围
    ax.set_xticklabels(datasets,rotation=15, fontsize=24)
    # 美化图表
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 获取当前的轴对象
    ax = plt.gca()

    # 隐藏上边框和右边框

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    # 显示图表
    plt.show()



def accuracy_and_coverage_500_de():
    # 示例数据
    sampling_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(11, 9))
    # 绘制 Accuracy 曲线
    sns.lineplot(ax=ax, x=sampling_numbers, y=[70.6, 74.8, 75.6, 76.2, 77.8, 77, 77.6, 77.8, 78], marker='o',  color="#FFBE7A", markersize=15,linewidth=3)
    sns.lineplot(ax=ax, x=sampling_numbers, y=[69.6, 74.8, 77.6, 78.8, 79.6, 79.8, 79.5, 80.2, 79.8], marker='s', color="#82B0D2", markersize=15,linewidth=3)

    # 设置坐标轴标签
    ax.set_title("MATH", fontsize=32)
    ax.set_xlabel('Number of Samplings (K)', fontsize=32)
    ax.set_ylabel('Accuracy (%)', fontsize=32)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    # 修改横坐标显示
    ax.tick_params(axis='both', which='major', labelsize=28)
    xticks = [i for i in range(1, 10)]
    xticks_labels = [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$', r'$2^9$']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, fontsize=28)
    ax.set_ylim(69,85)

    modelswitch_point = (math.log2(35), 81)  # ModelSwitch 的点
    gemini_point = (math.log2(512), 79.8)      # Gemini 1.5 Flash 的点
    # 绘制 |------------| 样式的虚线
    # 左边的竖线
    # ax.vlines(x=modelswitch_point[0], ymin=gemini_point[1], ymax=gemini_point[1]+Vertical_Line_len, colors='black',  linewidth=2)
    # ax.arrow(modelswitch_point[0], modelswitch_point[1]-1.5, 0, 1, head_width=0.1, head_length=0.2, fc='black', ec='black', linewidth=1)
    # 右边的竖线
    ax.vlines(x=gemini_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax.vlines(x=modelswitch_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax.annotate(
        '', 
        xy=(modelswitch_point[0]+0.15, modelswitch_point[1]), 
        xytext=(gemini_point[0], modelswitch_point[1]),
        arrowprops=dict(
            arrowstyle='<->', 
            color='black',
            lw=2,
            shrinkA=0,  # 调整起点收缩
            shrinkB=0,  # 调整终点收缩
            mutation_scale=20  # 放大箭头
        )
    )

    # ax.vlines(x=modelswitch_point[0], ymin=69, ymax=modelswitch_point[1], colors='gray', linestyles='dashed', linewidth=2)
    # 在点上方显示 k=35 和 k=256
    ax.text(modelswitch_point[0]-0.7, modelswitch_point[1]-0.8, "K=35", fontsize=28, color='black', ha='center')
    ax.text(gemini_point[0]-0.7, gemini_point[1]-0.8, "K=512", fontsize=28, color='black', ha='center')
    ax.text((modelswitch_point[0] + gemini_point[0]) / 2 , modelswitch_point[1]+2, "better accuracy,", fontsize=29, color='black', ha='center')
    ax.text(
        (modelswitch_point[0] + gemini_point[0]) / 2,
        modelswitch_point[1]+1,
        "save 93% samplings",
        fontsize=29,
        color='red',
        ha='center',
        fontproperties=font_prop_bold_italic
    )

    # 绘制散点图
    scatter_x = [2.5, 5.5, 10.7, 35]
    scatter_y1 = [76.6, 78.6, 79.4, 81]
    normalized_x = [math.log2(x) for x in scatter_x]
    for i in range(len(scatter_x)):
        ax.scatter(normalized_x[i], scatter_y1[i], color='#C00000', marker='*', s=400+400*i,zorder=3)

    # 显示图例
    # ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    handles, labels = [], []
    # Add necessary labels for global legend
    handles.append(plt.scatter([], [], marker='*', s=700,label="ModelSwitch", color="#C00000"))
    handles.append(plt.Line2D([], [], marker='s', label='Gemini 1.5 Flash', color="#82B0D2",markersize=15,linewidth=3))
    handles.append(plt.Line2D([], [], marker='o', label='GPT-4o mini', color="#FFBE7A",markersize=15,linewidth=3))
    
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.93, 0.13), fontsize=24)
    
    # 调整布局
    # plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # 自定义网格线
    # 水平虚线
    # 获取当前 X 轴的范围
    
    xmin, xmax = ax.get_xlim()
    ax.hlines(y=modelswitch_point[1], xmin=-1, xmax=11, colors='gray', linestyles='dashed', linewidth=1)
    ax.set_xlim(0.5,9.5)
    plt.tight_layout()
    plt.show()

def accuracy_and_coverage_500_MathBench():
    # 示例数据
    sampling_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(11, 9))
    # 绘制 Accuracy 曲线
    sns.lineplot(ax=ax, x=sampling_numbers, y=[54.33, 64.00, 68.67, 68.33, 72, 71.67, 72.33, 73.67, 73.67], marker='o',  color="#FFBE7A", markersize=15,linewidth=3)
    sns.lineplot(ax=ax, x=sampling_numbers, y=[46.67, 55.00, 62.00, 67.33, 68.33, 67.33, 70, 69, 69.67], marker='s', color="#82B0D2", markersize=15,linewidth=3)

    # 设置坐标轴标签
    ax.set_title("MathBench", fontsize=32)
    ax.set_xlabel('Number of Samplings (K)', fontsize=32)
    ax.set_ylabel('Accuracy (%)', fontsize=32)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    # 修改横坐标显示
    ax.tick_params(axis='both', which='major', labelsize=28)
    xticks = [i for i in range(1, 10)]
    xticks_labels = [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$', r'$2^9$']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, fontsize=28)
    ax.set_ylim(44, 86)

    modelswitch_point = (math.log2(32), 74)  # ModelSwitch 的点
    gemini_point = (math.log2(512), 73.67)      # Gemini 1.5 Flash 的点
    ax.axhline(y=68.67, color="#82B0D2", linestyle="-.", linewidth=2)
    # 提取坐标
    x1, y1 = modelswitch_point
    x2, y2 = gemini_point

    # 定义折线的路径
    # 从 (x1, y1) 向上画到 (x1, y2 + offset)，再向右画到 (x2, y2 + offset)，最后向下画到 (x2, y2)
    offset = 6  # 控制折线的高度


    # 绘制折线和点
    ax.hlines(y = (y2 + offset), xmin=x1, xmax=x2, colors='black', linewidth=2)
    # ax.plot(path_x, path_y, label="Path", color="black",lw=2)

    # 在路径的末端添加箭头
    # 第一个箭头：从 (x1, y2 + offset) 指向 (x1, y1)（向下）
    ax.annotate('', xy=(x1, y1+1.5), xytext=(x1, y2 + offset),
                arrowprops=dict(
                    arrowstyle='->', 
                    color='black', 
                    lw=2,
                    shrinkA=0,  # 调整起点收缩
                    shrinkB=0,  # 调整终点收缩
                    mutation_scale=20  # 放大箭头
                    ))


    # 第三个箭头：从 (x2, y2 + offset) 指向 (x2, y2)（向下）
    ax.annotate('', xy=(x2, y2+0.5), xytext=(x2, y2 + offset),
                arrowprops=dict(
                    arrowstyle='->', 
                    color='black', 
                    lw=2,
                    shrinkA=0,  # 调整起点收缩
                    shrinkB=0,  # 调整终点收缩
                    mutation_scale=20  # 放大箭头
                    ))

    # 绘制 |------------| 样式的虚线
    # 左边的竖线
    # ax.vlines(x=modelswitch_point[0], ymin=gemini_point[1], ymax=gemini_point[1]+Vertical_Line_len, colors='black',  linewidth=2)
    # ax.arrow(modelswitch_point[0], modelswitch_point[1]-1.5, 0, 1, head_width=0.1, head_length=0.2, fc='black', ec='black', linewidth=1)
    # 右边的竖线
    # ax.vlines(x=gemini_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    # ax.vlines(x=modelswitch_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax.annotate(
        '', 
        xy=(modelswitch_point[0]+0.15, modelswitch_point[1]), 
        xytext=(8, modelswitch_point[1]),
        arrowprops=dict(
            arrowstyle='<->', 
            color='black',
            lw=2,
            shrinkA=0,  # 调整起点收缩
            shrinkB=0,  # 调整终点收缩
            mutation_scale=20  # 放大箭头
        )
    )

    # ax.vlines(x=modelswitch_point[0], ymin=69, ymax=modelswitch_point[1], colors='gray', linestyles='dashed', linewidth=2)
    # 在点上方显示 k=35 和 k=256
    ax.text(modelswitch_point[0] - 0.5, modelswitch_point[1] + 0.8, "K=32", fontsize=24, color='black', ha='center')
    ax.text(8, modelswitch_point[1] + 0.8, "K=256", fontsize=24, color='black', ha='center')
    ax.text(gemini_point[0], gemini_point[1] - 2.5, "K=512", fontsize=24, color='black', ha='center')
    # 绘制散点图
    scatter_x = [2.9, 6.5, 12, 32]
    scatter_y1 = [62.67, 69, 72, 74]
    # ax.text((modelswitch_point[0] + gemini_point[0]) / 2 , modelswitch_point[1]+5, "better accuracy,", fontsize=29, color='black', ha='center')
    ax.text((modelswitch_point[0] + 8) / 2 , modelswitch_point[1]+2, "save 87%", fontsize=29, color='red', ha='center',fontproperties=font_prop_bold_italic)
    ax.text(
        (modelswitch_point[0] + gemini_point[0]) / 2,
        modelswitch_point[1]+7,
        "save 93% samplings",
        fontsize=29,
        color='red',
        ha='center',
        fontproperties=font_prop_bold_italic
    )

    normalized_x = [math.log2(x) for x in scatter_x]
    for i in range(len(scatter_x)):
        ax.scatter(normalized_x[i], scatter_y1[i], color='#C00000', marker='*', s=400+400*i,zorder=3)

    # 显示图例
    # ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    handles, labels = [], []
    # Add necessary labels for global legend
    handles.append(plt.scatter([], [], marker='*', s=700,label="ModelSwitch", color="#C00000"))
    handles.append(plt.Line2D([], [], marker='o', label='Gemma-2-9b-it', color="#FFBE7A",markersize=15,linewidth=3))
    handles.append(plt.Line2D([], [], marker='s', label='Llama-3.1-8B-Instruct', color="#82B0D2",markersize=15,linewidth=3))
    handles.append(plt.Line2D([], [], color="#82B0D2", linestyle="-.", linewidth=2, label="Llama-3.1-70B-Instruct"))
    
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.98, 0.13), fontsize=24)
    
    # 调整布局
    # plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # 自定义网格线
    # 水平虚线
    # 获取当前 X 轴的范围
    
    # xmin, xmax = ax.get_xlim()
    # ax.hlines(y=modelswitch_point[1], xmin=-1, xmax=11, colors='gray', linestyles='dashed', linewidth=1)
    # ax.set_xlim(0.5,9.5)
    plt.tight_layout()
    plt.show()    


def accuracy_and_coverage_combined():
    # 示例数据
    sampling_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))  # 1行2列，整体长度乘以2

    # ------------------------- Math 子图 -------------------------
    # 绘制 Accuracy 曲线
    sns.lineplot(ax=ax1, x=sampling_numbers, y=[70.6, 74.8, 75.6, 76.2, 77.8, 77, 77.6, 77.8, 78], marker='o', color="#96CCCB", markersize=15, linewidth=3)
    sns.lineplot(ax=ax1, x=sampling_numbers, y=[69.6, 74.8, 77.6, 78.8, 79.6, 79.8, 79.5, 80.2, 79.8], marker='s', color="#BEB8DC", markersize=15, linewidth=3)

    # 设置坐标轴标签
    ax1.set_title("Math", fontsize=32)
    ax1.set_xlabel('Number of Samplings (K)', fontsize=32)
    ax1.set_ylabel('Accuracy (%)', fontsize=32)
    ax1.spines['top'].set_linewidth(3)
    ax1.spines['right'].set_linewidth(3)
    ax1.spines['left'].set_linewidth(3)
    ax1.spines['bottom'].set_linewidth(3)

    # 修改横坐标显示
    ax1.tick_params(axis='both', which='major', labelsize=28)
    xticks = [i for i in range(1, 10)]
    xticks_labels = [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$', r'$2^9$']
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks_labels, fontsize=28)
    ax1.set_ylim(69, 85)
    ax1.axhline(y=73.8, color="#96CCCB", linestyle=":", linewidth=2)
    ax1.axhline(y=71.6, color="#BEB8DC", linestyle="-.", linewidth=2)
    # 绘制 ModelSwitch 和 Gemini 的虚线
    modelswitch_point = (math.log2(35), 81)  # ModelSwitch 的点
    gemini_point = (math.log2(512), 79.8)  # Gemini 1.5 Flash 的点
    ax1.vlines(x=gemini_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax1.vlines(x=modelswitch_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax1.annotate(
        '', 
        xy=(modelswitch_point[0] + 0.15, modelswitch_point[1]), 
        xytext=(gemini_point[0], modelswitch_point[1]),
        arrowprops=dict(
            arrowstyle='<->', 
            color='black',
            lw=2,
            shrinkA=0,  # 调整起点收缩
            shrinkB=0,  # 调整终点收缩
            mutation_scale=20  # 放大箭头
        )
    )

    # 标注 K=35 和 K=512
    ax1.text(modelswitch_point[0] - 0.7, modelswitch_point[1] + 0.4, "K=35", fontsize=28, color='black', ha='center')
    ax1.text(gemini_point[0] - 0.7, modelswitch_point[1] + 0.4, "K=512", fontsize=28, color='black', ha='center')
    ax1.text(
        (modelswitch_point[0] + gemini_point[0]) / 2,
        modelswitch_point[1]+1.5,
        "save 93% samplings",
        fontsize=29,
        color='red',
        ha='center',
        fontproperties=font_prop_bold_italic
    )
    # 绘制散点图
    scatter_x = [2.5, 5.5, 10.7, 35]
    scatter_y1 = [76.6, 78.6, 79.4, 81]
    normalized_x = [math.log2(x) for x in scatter_x]
    for i in range(len(scatter_x)):
        ax1.scatter(normalized_x[i], scatter_y1[i], color='#C00000', marker='*', s=400 + 400 * i, zorder=3)
    # ax1.hlines(y=modelswitch_point[1], xmin=-1, xmax=11, colors='gray', linestyles='dashed', linewidth=1)
    # ax1.set_xlim(0.5,9.5)
    # ------------------------- DATE 子图 -------------------------
    # 绘制 Accuracy 曲线					
    sns.lineplot(ax=ax2, x=sampling_numbers, y=[56.10, 64.23, 68.00, 60.43, 66.94, 70.19, 71, 71.54, 71.54], marker='o', color="#82B0D2", markersize=15, linewidth=3)
    sns.lineplot(ax=ax2, x=sampling_numbers, y=[40.65, 53.12, 59.00, 62.33, 64.5, 65.04, 64.77, 64.77, 64.77], marker='s', color="#FFBE7A", markersize=15, linewidth=3)
    ax2.axhline(y=70.46, color="#82B0D2", linestyle="-.", linewidth=2)
    # 设置坐标轴标签
    ax2.set_title("DATE", fontsize=32)
    ax2.set_xlabel('Number of Samplings (K)', fontsize=32)
    # ax2.set_ylabel('Accuracy (%)', fontsize=32)
    ax2.spines['top'].set_linewidth(3)
    ax2.spines['right'].set_linewidth(3)
    ax2.spines['left'].set_linewidth(3)
    ax2.spines['bottom'].set_linewidth(3)

    # 修改横坐标显示
    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks_labels, fontsize=28)
    ax2.set_ylim(39, 76)

    # 绘制 ModelSwitch 和 Gemini 的虚线
    modelswitch_point = (math.log2(12.4), 70.19)  # ModelSwitch 的点
    gemini_point = (math.log2(64), 70.19)  # Gemini 1.5 Flash 的点
    # ax2.vlines(x=gemini_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    # ax2.vlines(x=modelswitch_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax2.annotate(
        '', 
        xy=(modelswitch_point[0] + 0.15, modelswitch_point[1]), 
        xytext=(gemini_point[0], modelswitch_point[1]),
        arrowprops=dict(
            arrowstyle='<->', 
            color='black',
            lw=2,
            shrinkA=0,  # 调整起点收缩
            shrinkB=0,  # 调整终点收缩
            mutation_scale=20  # 放大箭头
        )
    )

    # 标注 K=35 和 K=512
    ax2.text(modelswitch_point[0], modelswitch_point[1] - 2.7, "K=12", fontsize=28, color='black', ha='center')
    ax2.text(gemini_point[0], modelswitch_point[1] - 2.7, "K=64", fontsize=28, color='black', ha='center')
    ax2.text(
        (modelswitch_point[0] + gemini_point[0]) / 2,
        modelswitch_point[1]+1.5,
        "save 81% samplings",
        fontsize=29,
        color='red',
        ha='center',
        fontproperties=font_prop_bold_italic
    )

    # 绘制散点图
    scatter_x = [2.9, 6.4, 12.4]
    scatter_y1 = [62.33, 68.3, 70.19]
    normalized_x = [math.log2(x) for x in scatter_x]
    for i in range(len(scatter_x)):
        ax2.scatter(normalized_x[i], scatter_y1[i], color='#C00000', marker='*', s=400 + 400 * i, zorder=3)
    # ax2.hlines(y=modelswitch_point[1], xmin=-1, xmax=11, colors='gray', linestyles='dashed', linewidth=1)
    # ax2.set_xlim(0.5,9.5)
    # ------------------------- 统一图例 -------------------------
    handles, labels = [], []
    handles.append(plt.scatter([], [], marker='*', s=700, label="ModelSwitch", color="#C00000"))
    handles.append(plt.Line2D([], [], marker='s', label='Gemini 1.5 Flash', color="#BEB8DC", markersize=15, linewidth=3))
    handles.append(plt.Line2D([], [], marker='o', label='GPT-4o mini', color="#96CCCB", markersize=15, linewidth=3))
    handles.append(plt.Line2D([], [], label='Gemini 1.5 Pro', color="#BEB8DC", linestyle="-.", markersize=15, linewidth=3))
    handles.append(plt.Line2D([], [], label='GPT-4o', color="#96CCCB", linestyle=":", markersize=15, linewidth=3))
    handles.append(plt.Line2D([], [], marker='s', label='Gemma-2-9b-it', color="#82B0D2", markersize=15, linewidth=3))
    handles.append(plt.Line2D([], [], marker='o', label='Llama-3.1-8B-Instruct', color="#FFBE7A", markersize=15, linewidth=3))
    handles.append(plt.Line2D([], [], color="#82B0D2", linestyle="-.", linewidth=2, label="Llama-3.1-70B-Instruct"))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=28, ncol=4, frameon=False)

    # 调整布局
    plt.tight_layout()
    plt.show()

def accuracy_and_coverage_combined_v1():
    # 示例数据
    sampling_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1行2列，整体长度乘以2

    # ------------------------- Math 子图 -------------------------
    # 绘制 Accuracy 曲线
    sns.lineplot(ax=ax1, x=sampling_numbers, y=[70.6, 74.8, 75.6, 76.2, 77.8, 77, 77.6, 77.8, 78], marker='o', color="#96CCCB", markersize=10, linewidth=2)
    sns.lineplot(ax=ax1, x=sampling_numbers, y=[69.6, 74.8, 77.6, 78.8, 79.6, 79.8, 79.5, 80.2, 79.8], marker='s', color="#BEB8DC", markersize=10, linewidth=2)

    # 设置坐标轴标签
    ax1.set_title("Math", fontsize=20)
    ax1.set_xlabel('Number of Samplings (K)', fontsize=20)
    ax1.set_ylabel('Accuracy (%)', fontsize=20)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)

    # 修改横坐标显示
    ax1.tick_params(axis='both', which='major', labelsize=16)
    xticks = [i for i in range(1, 10)]
    xticks_labels = [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$', r'$2^9$']
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks_labels, fontsize=16)
    ax1.set_ylim(69, 85)
    ax1.axhline(y=73.8, color="#96CCCB", linestyle=":", linewidth=2)
    ax1.axhline(y=71.6, color="#BEB8DC", linestyle="-.", linewidth=2)
    # 绘制 ModelSwitch 和 Gemini 的虚线
    modelswitch_point = (math.log2(35), 81)  # ModelSwitch 的点
    gemini_point = (math.log2(512), 79.8)  # Gemini 1.5 Flash 的点
    ax1.vlines(x=gemini_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax1.vlines(x=modelswitch_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax1.annotate(
        '', 
        xy=(modelswitch_point[0] + 0.15, modelswitch_point[1]), 
        xytext=(gemini_point[0], modelswitch_point[1]),
        arrowprops=dict(
            arrowstyle='<->', 
            color='black',
            lw=1.5,
            shrinkA=0,  # 调整起点收缩
            shrinkB=0,  # 调整终点收缩
            mutation_scale=20  # 放大箭头
        )
    )

    # 标注 K=35 和 K=512
    ax1.text(modelswitch_point[0] - 0.7, modelswitch_point[1] + 0.4, "K=35", fontsize=14, color='black', ha='center')
    ax1.text(gemini_point[0] - 0.7, modelswitch_point[1] + 0.4, "K=512", fontsize=14, color='black', ha='center')
    ax1.text((modelswitch_point[0] + gemini_point[0]) / 2 , modelswitch_point[1]+2.5, "better accuracy,", fontsize=16, color='black', ha='center',fontproperties=font_prop_bold_italic)
    ax1.text(
        (modelswitch_point[0] + gemini_point[0]) / 2,
        modelswitch_point[1]+1.5,
        "14× efficient",
        fontsize=16,
        color='red',
        ha='center',
        fontproperties=font_prop_bold_italic
    )
    # 绘制散点图
    scatter_x = [2.5, 5.5, 10.7, 35]
    scatter_y1 = [76.6, 78.6, 79.4, 81]
    normalized_x = [math.log2(x) for x in scatter_x]
    for i in range(len(scatter_x)):
        ax1.scatter(normalized_x[i], scatter_y1[i], color='#C00000', marker='*', s=200 + 200 * i, zorder=3)
    # ax1.hlines(y=modelswitch_point[1], xmin=-1, xmax=11, colors='gray', linestyles='dashed', linewidth=1)
    # ax1.set_xlim(0.5,9.5)
    # ------------------------- DATE 子图 -------------------------
    # 绘制 Accuracy 曲线
    sns.lineplot(ax=ax2, x=sampling_numbers, y=[54.33, 64.00, 68.67, 68.33, 72, 71.67, 72.33, 73.67, 73.67], marker='o',  color="#FFBE7A", markersize=10,linewidth=2)
    sns.lineplot(ax=ax2, x=sampling_numbers, y=[46.67, 55.00, 62.00, 67.33, 68.33, 67.33, 70, 69, 69.67], marker='s', color="#82B0D2", markersize=10,linewidth=2)

    # 设置坐标轴标签
    ax2.set_title("MathBench", fontsize=20)
    ax2.set_xlabel('Number of Samplings (K)', fontsize=20)
    # ax2.set_ylabel('Accuracy (%)', fontsize=20)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)

    # 修改横坐标显示
    ax2.tick_params(axis='both', which='major', labelsize=16)
    xticks = [i for i in range(1, 10)]
    xticks_labels = [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$', r'$2^9$']
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks_labels, fontsize=16)
    ax2.set_ylim(44, 86)

    modelswitch_point = (math.log2(48), 75)  # ModelSwitch 的点
    gemini_point = (math.log2(512), 73.67)      # Gemini 1.5 Flash 的点
    ax2.axhline(y=68.67, color="#82B0D2", linestyle="-.", linewidth=2)
    # 提取坐标
    x1, y1 = modelswitch_point
    x2, y2 = gemini_point

    # 定义折线的路径
    # 从 (x1, y1) 向上画到 (x1, y2 + offset)，再向右画到 (x2, y2 + offset)，最后向下画到 (x2, y2)
    offset = 6  # 控制折线的高度


    # 绘制折线和点
    # ax2.hlines(y = (y2 + offset), xmin=x1, xmax=x2, colors='black', linewidth=2)
    # ax.plot(path_x, path_y, label="Path", color="black",lw=2)

    # 在路径的末端添加箭头
    # 第一个箭头：从 (x1, y2 + offset) 指向 (x1, y1)（向下）
    # ax2.annotate('', xy=(x1, y1+1.5), xytext=(x1, y2 + offset),
    #             arrowprops=dict(
    #                 arrowstyle='->', 
    #                 color='black', 
    #                 lw=1.5,
    #                 shrinkA=0,  # 调整起点收缩
    #                 shrinkB=0,  # 调整终点收缩
    #                 mutation_scale=20  # 放大箭头
    #                 ))


    # 第三个箭头：从 (x2, y2 + offset) 指向 (x2, y2)（向下）
    # ax2.annotate('', xy=(x2, y2+0.5), xytext=(x2, y2 + offset),
    #             arrowprops=dict(
    #                 arrowstyle='->', 
    #                 color='black', 
    #                 lw=1.5,
    #                 shrinkA=0,  # 调整起点收缩
    #                 shrinkB=0,  # 调整终点收缩
    #                 mutation_scale=20  # 放大箭头
    #                 ))

    # 绘制 |------------| 样式的虚线
    # 左边的竖线
    # ax.vlines(x=modelswitch_point[0], ymin=gemini_point[1], ymax=gemini_point[1]+Vertical_Line_len, colors='black',  linewidth=2)
    # ax.arrow(modelswitch_point[0], modelswitch_point[1]-1.5, 0, 1, head_width=0.1, head_length=0.2, fc='black', ec='black', linewidth=1)
    # 右边的竖线
    ax2.vlines(x=gemini_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax2.vlines(x=modelswitch_point[0], ymin=0, ymax=100, colors='gray', linestyles='dashed', linewidth=1)
    ax2.annotate(
        '', 
        xy=(modelswitch_point[0]+0.15, modelswitch_point[1]), 
        xytext=(gemini_point[0], modelswitch_point[1]),
        arrowprops=dict(
            arrowstyle='<->', 
            color='black',
            lw=1.5,
            shrinkA=0,  # 调整起点收缩
            shrinkB=0,  # 调整终点收缩
            mutation_scale=20  # 放大箭头
        )
    )

    # ax.vlines(x=modelswitch_point[0], ymin=69, ymax=modelswitch_point[1], colors='gray', linestyles='dashed', linewidth=2)
    # 在点上方显示 k=35 和 k=256
    ax2.text(modelswitch_point[0] - 0.7, modelswitch_point[1] + 0.8, "K=48", fontsize=14, color='black', ha='center')
    ax2.text(gemini_point[0]-0.7, modelswitch_point[1] + 0.8, "K=512", fontsize=14, color='black', ha='center')
    # ax2.text(gemini_point[0], gemini_point[1] - 2.5, "K=512", fontsize=14, color='black', ha='center')
    # 绘制散点图
    scatter_x = [2.9, 6.5, 12, 48]
    scatter_y1 = [62.67, 69, 72, 75]
    ax2.text((modelswitch_point[0] + gemini_point[0]) / 2 , modelswitch_point[1]+5.5, "better accuracy,", fontsize=16, color='black', ha='center',fontproperties=font_prop_bold_italic)
    ax2.text((modelswitch_point[0] + gemini_point[0]) / 2 , modelswitch_point[1]+3, "10× efficient", fontsize=16, color='red', ha='center',fontproperties=font_prop_bold_italic)
    # ax2.text(
    #     (modelswitch_point[0] + gemini_point[0]) / 2,
    #     modelswitch_point[1]+7,
    #     "10 × efficient",
    #     fontsize=14,
    #     color='red',
    #     ha='center',
    #     fontproperties=font_prop_bold_italic
    # )

    normalized_x = [math.log2(x) for x in scatter_x]
    for i in range(len(scatter_x)):
        ax2.scatter(normalized_x[i], scatter_y1[i], color='#C00000', marker='*', s=200+200*i,zorder=3)

    # ax2.hlines(y=modelswitch_point[1], xmin=-1, xmax=11, colors='gray', linestyles='dashed', linewidth=1)
    # ax2.set_xlim(0.5,9.5)
    # ------------------------- 统一图例 -------------------------
    handles, labels = [], []
    handles.append(plt.scatter([], [], marker='*', s=400, label="ModelSwitch", color="#C00000"))
    handles.append(plt.Line2D([], [], marker='s', label='Gemini 1.5 Flash', color="#BEB8DC", markersize=10, linewidth=2))
    handles.append(plt.Line2D([], [], marker='o', label='GPT-4o mini', color="#96CCCB", markersize=10, linewidth=2))
    handles.append(plt.Line2D([], [], label='Gemini 1.5 Pro', color="#BEB8DC", linestyle="-.", markersize=10, linewidth=2))
    handles.append(plt.Line2D([], [], label='GPT-4o', color="#96CCCB", linestyle=":", markersize=10, linewidth=2))
    handles.append(plt.Line2D([], [], marker='o', label='Gemma-2-9b-it', color="#FFBE7A", markersize=10, linewidth=2))
    handles.append(plt.Line2D([], [], marker='s', label='Llama-3.1-8B-Instruct', color="#82B0D2", markersize=10, linewidth=2))
    handles.append(plt.Line2D([], [], color="#82B0D2", linestyle="-.", linewidth=2, label="Llama-3.1-70B-Instruct"))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=16, ncol=4, frameon=False)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.show()



def close_source_performance_gain_v1():
    # 数据
    methods = ['MAD', 'ChatEval', 'MOA', 'ModelSwitch']
    color_list = ["#82B0D2", "#FFBE7A", "#BEB8DC", "#FA7F6F"]
    datasets = ['GSM8K', 'MATH', 'MathBench', 'MGSM', 'DATE', 'MMLU-Pro']
    best_single_scores = [93.56, 73.8, 72.67, 88.8, 73.44, 53]
    scores = [
        [95.30, 77.4, 77, 90.4, 83.20, 50.2],    # MAD
        [93.6, 75.8, 75.3, 87.2, 76.42, 43],  # Chateval(3*5)
        [94.92, 79.8, 79, 89.5, 78.32, 50.8], # MOA
        [96.13, 80.2, 80, 91.4, 78.8, 63.2]  # ModelSwitch
    ]

    # 转置 scores 数据，使每个数据集对应一组方法的分数
    scores = np.array(scores).T

    # 创建 1×6 的子图布局
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6.5))
    axes = axes.flatten()

    # 绘制每个数据集的柱状图
    for i, (dataset, dataset_scores, best_score) in enumerate(zip(datasets, scores, best_single_scores)):
        ax = axes[i]
        x = np.arange(len(methods))  # 横坐标索引
        width = 0.6  # 柱状图宽度

        # 绘制柱状图
        ax.bar(x, dataset_scores, width, color=color_list, edgecolor='black')

        # 绘制基准横线（best_single_scores）
        ax.axhline(best_score, color='black', linestyle='--', linewidth=1.5, label='Best Single')

        # # 设置标题
        # ax.set_title(f'{dataset}', fontsize=24)

        # 隐藏子图的横坐标
        ax.set_xticks([])

        # # 设置纵坐标范围
        # ax.set_ylim(0, 100)
        

        # # 隐藏上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # # 调整Y轴
        if dataset=="GSM8K":
            ax.set_ylim(91, 99)
        elif dataset=="MATH":
            ax.set_ylim(69, 81)
        elif dataset=="MathBench":
            ax.set_ylim(69, 81)
        elif dataset=="MGSM":
            ax.set_ylim(84, 96)
        elif dataset=="DATE":
            ax.set_ylim(72, 86)
        elif dataset=="MMLU-Pro":
            ax.set_ylim(39, 65)
        ax.tick_params(axis='both', which='major', labelsize=16)
        # 仅在第一个子图显示纵轴标签
        if i == 0 or i ==3:
            ax.set_ylabel('Accuracy (%)', fontsize=20)
        ax.set_xlabel(f'{dataset}', fontsize=20)
        # ax.set_yticks([])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # 添加图例
    # handles = [plt.Line2D([0], [0], color=color, lw=10) for color in color_list]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black') for color in color_list]
    handles.append(plt.Line2D([0], [0], color='black', linestyle='--', lw=1.5))
    labels = methods + ['Best Single']
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=16, frameon=False)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # 为图例留出空间
    plt.show()


def open_source_performance_gain_v1():
    # 数据
    methods = ['MOA', 'ModelSwitch']
    color_list = ["#BEB8DC", "#FA7F6F"]
    datasets = ['GSM8K', 'MATH', 'MathBench', 'MGSM', 'DATE', 'MMLU-Pro']
    best_single_scores = [80.29, 47.86, 64.00, 74.00, 56.10, 30.60]
    scores = [
        [89.08,53.43,67.33,72.1,53.12,30],
        [94.24,64.29,76.33,85.80,70.19,36.40]
    ]

    # 转置 scores 数据，使每个数据集对应一组方法的分数
    scores = np.array(scores).T

    # 创建 1×6 的子图布局
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 4.5))
    axes = axes.flatten()

    # 绘制每个数据集的柱状图
    for i, (dataset, dataset_scores, best_score) in enumerate(zip(datasets, scores, best_single_scores)):
        ax = axes[i]
        x = np.arange(len(methods))  # 横坐标索引
        width = 0.6  # 柱状图宽度

        # 绘制柱状图
        ax.bar(x, dataset_scores, width, color=color_list, edgecolor='black')

        # 绘制基准横线（best_single_scores）
        ax.axhline(best_score, color='black', linestyle='--', linewidth=1.5, label='Best Single')

        # # 设置标题
        # ax.set_title(f'{dataset}', fontsize=24)

        # 隐藏子图的横坐标
        ax.set_xticks([])

        # # 设置纵坐标范围
        # ax.set_ylim(0, 100)
        

        # # 隐藏上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)

        # # 调整Y轴
        if dataset=="GSM8K":
            ax.set_ylim(79, 95)
        elif dataset=="MATH":
            ax.set_ylim(45, 66)
        elif dataset=="MathBench":
            ax.set_ylim(60, 80)
        elif dataset=="MGSM":
            ax.set_ylim(70, 90)
        elif dataset=="DATE":
            ax.set_ylim(50, 73)
        elif dataset=="MMLU-Pro":
            ax.set_ylim(28, 39)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # 仅在第一个子图显示纵轴标签
        if i == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=28)
        ax.set_xlabel(f'{dataset}', fontsize=24)
        # ax.set_yticks([])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # 添加图例
    # handles = [plt.Line2D([0], [0], color=color, lw=10) for color in color_list]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black') for color in color_list]
    handles.append(plt.Line2D([0], [0], color='black', linestyle='--', lw=1.5))
    labels = methods + ['Best Single']
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=24, frameon=False)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # 为图例留出空间
    plt.show()


def theoretical_analysis():
    # ======================
    # 第一部分：创建画布和子图
    # ======================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 14英寸宽 × 6英寸高
    # plt.subplots_adjust(wspace=0.3)  # 调整子图间距

    # ======================
    # 通用样式设置函数
    # ======================
    def plot_bars(ax, title, x_data, y_data,flag):
        """ 绘制单个子图的柱状图 """
        labels = ['GPT-4o mini', 'Gemini 1.5 Flash', 'ModelSwitch']
        bar_width = 0.35
        index = np.arange(len(labels))
        
        # 绘制柱状图
        ax.bar(index, x_data, bar_width, 
            color='#5F97D2', edgecolor='black')
        ax.bar(index + bar_width, y_data, bar_width,
            color='none', edgecolor='#5F97D2', linewidth=1.5,
            hatch='////', linestyle='--')
        
        # 通用格式设置
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        if flag:
            ax.set_ylabel('Accurracy (%)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(title, fontsize=20)
        ax.set_xticks(index + bar_width/2)
        ax.set_xticklabels(labels,fontsize=14)
        ax.set_ylim(60, 85)  # 统一y轴范围
        ax.grid(False)
    
        
    # 添加图例
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#5F97D2", edgecolor='black')]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor='#5F97D2',linewidth=1.5, hatch='////', linestyle='--'))
    labels=['Experimental results','Theoretical results']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=16, ncol=2, frameon=False)
    # ======================
    # 填充数据
    # ======================
    # 第一组数据
    x1 = [74.00, 72.67, 78.33]
    y1 = [76.69, 72.17, 79.94]
    
    # 第二组数据
    x2 = [75.33, 72.67, 79.67]
    y2 = [76.82, 72.25, 80.03]

    # ======================
    # 绘制图形
    # ======================
    plot_bars(ax1, 'Sampling Budget=10', x1, y1,True)
    plot_bars(ax2, 'Sampling Budget=16', x2, y2,False)


    plt.show()

# def correlation_v1():

def close_source_vs_sc_v1():
    # Define samples and datasets
    samples = [1, 4, 8, 12, 16]
    datasets = ["GSM8K","MATH", "MathBench", "MGSM", "DATE", "MMLU-Pro"]
    # Accuracy data for each model across datasets
    
    scatter_data={
        "GSM8K": {
            "x": [2.2,4.5,6.8,8.8],
            "y": [95.75,95.98,96.13,96.05]
        },
        "MGSM": {
            "x": [2.2, 4.5, 6.9, 9.4],
            "y": [90.6, 90.9, 91.3, 91.3]
        },
        "MATH": {
            "x": [2.5,5.5,8.5,10.7],
            "y": [76.6,78.6,79,79.4]
        },
        "DATE": {
            "x": [2.2,5,7.7,9.6],
            "y": [75.34, 76.96, 77.51, 78.32]
        },
        "MMLU-Pro": {
            "x": [2.8,6.3,9.9,12.4],
            "y": [55.6,60.4,60.2,60.8]
        },
        "MathBench": {
            "x": [2.4,5.1,7.8,10.1],
            "y": [77.33, 78.67, 78.67, 79.67]
        }
    }

    accuracy_data = {
        "GSM8K": {
            "GPT-4o mini": [93.56, 94.54, 95.07, 95, 94.92],
            "Gemini 1.5 Flash": [93.18, 93.93, 94.23, 94.16, 94.62],
            "GPT-4o": [95.68] * 4,
            "Gemini 1.5 Pro": [95.3] * 4
        },
        "MGSM": {
            "GPT-4o mini": [86.7, 88.5, 88.8, 88.8,89.3],
            "Gemini 1.5 Flash": [88.8, 88.7, 89.1, 89.3,89.4],
            "GPT-4o": [92.1] * 4,
            "Gemini 1.5 Pro": [91.2] * 4
        },
        "MATH": {
            "GPT-4o mini": [70.6, 74.8, 75.6, 76.6, 76.2],
            "Gemini 1.5 Flash": [69.6, 74.8, 77.6, 78, 78.8],
            # "fusion": [76.29, 81.28, 82.57, 83.14],
            "GPT-4o": [73.8] * 4,
            "Gemini 1.5 Pro": [71.6] * 4
        },
        "DATE":
            {
                "GPT-4o mini": [73.44, 76.15, 76.69,76.69, 76.69],
                "Gemini 1.5 Flash": [71, 71, 72.36, 73.44, 74.8],
                "GPT-4o": [72.63] * 4,
                "Gemini 1.5 Pro": [80.76] * 4
            },
        "MMLU-Pro": {
            "GPT-4o mini": [41.6, 46.2, 47, 47.2, 49.6],
            "Gemini 1.5 Flash": [53, 57,58.8, 60.2, 60.4],
            "GPT-4o": [58.4] * 4,
            "Gemini 1.5 Pro": [61] * 4
        },
        "MathBench": {
            "GPT-4o mini": [72.67, 74, 73.67, 74.33,75.33],
            "Gemini 1.5 Flash": [71.67, 71.33, 72.33, 72.33,72.67],
            "GPT-4o": [76.67] * 4,
            "Gemini 1.5 Pro": [78.33] * 4
        }
    }

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):

        ax = axes[idx]
        for model, acc in accuracy_data[dataset].items():
            if model == "GPT-4o mini":
                ax.plot(samples, acc, marker='o', color="#FFBE7A",linewidth=2)
            elif model == "Gemini 1.5 Flash":
                ax.plot(samples, acc, marker='s', color="#82B0D2",linewidth=2)
            elif model == "GPT-4o":
                ax.axhline(y=acc[0], color="#FFBE7A", linestyle=":", linewidth=2)
            elif model == "Gemini 1.5 Pro":
                ax.axhline(y=acc[0], color="#82B0D2", linestyle="-.", linewidth=2)

        ax.set_title(f"{dataset}", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        
        ax.set_xticks(samples)

        # # 绘制散点图
        scatter_x = scatter_data[dataset]["x"]
        scatter_y = scatter_data[dataset]["y"]
        for i in range(len(scatter_x)):
            if i ==3:
                ax.scatter(scatter_x[i], scatter_y[i], facecolor=(0.75, 0, 0, 1), marker='*', edgecolor='#C00000', s=300)
            else:
                ax.scatter(scatter_x[i], scatter_y[i], facecolor=(0.75, 0, 0, 0.25*i), marker='*', edgecolor='#C00000', s=300)
        # # 隐藏上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        if idx==0 or idx==3:
            ax.set_ylabel("Accuracy (%)", fontsize=20)
        if idx >=3:
            ax.set_xlabel("Number of Samplings", fontsize=20)
        if dataset == "GSM8K":
            ax.set_ylim(91.5, 96.5)
            
        elif dataset == "MGSM":
            ax.set_ylim(85, 95)
        elif dataset == "MATH":
            ax.set_ylim(68, 80.5)
        elif dataset == "DATE":
            ax.set_ylim(70, 83)
        elif dataset == "MMLU-Pro":
            ax.set_ylim(40, 65)
        elif dataset == "MathBench":
            ax.set_ylim(70, 82)

        # ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax.grid(True)
    # Collect handles and labels from one plot for global legend
    handles, labels = [], []

    # Add necessary labels for global legend
    handles.append(plt.scatter([], [], label="MS (Budget=4)", color="#C00000", facecolor=(0.75, 0, 0, 0), marker='*', edgecolor='#C00000', s=300))
    handles.append(plt.Line2D([], [], color="#FFBE7A", linestyle=":", linewidth=2, label="GPT-4o"))
    handles.append(plt.scatter([], [], label="MS (Budget=8)", color="#C00000", facecolor=(0.75, 0, 0, 0.25), marker='*', edgecolor='#C00000', s=300))
    handles.append(plt.Line2D([], [], color="#82B0D2", linestyle="-.", linewidth=2, label="Gemini 1.5 Pro"))
    handles.append(plt.scatter([], [], label="MS (Budget=12)", color="#C00000", facecolor=(0.75, 0, 0, 0.5), marker='*', edgecolor='#C00000', s=300))
    handles.append(plt.Line2D([], [], marker='s', color="#FFBE7A", label="GPT-4o mini",linewidth=2))
    handles.append(plt.scatter([], [], label="MS (Budget=16)", color="#C00000", facecolor=(0.75, 0, 0, 1), marker='*', edgecolor='#C00000', s=300))
    handles.append(plt.Line2D([], [], marker='o', label="Gemini-1.5 Flash", color="#82B0D2",linewidth=2))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.525, 1), ncol=4, fontsize=16, frameon=False)


    plt.tight_layout()
    plt.show()

def open_source_vs_sc_v1():
    # Define samples and datasets
    samples = [1, 4, 8, 12, 16]
    datasets = ["GSM8K","MATH", "MathBench", "MGSM", "DATE", "MMLU-Pro"]
    # Accuracy data for each model across datasets
    
    scatter_data={
        "GSM8K": {
            "x": [2.6,5.7,9.3,11.5],
            "y": [88.93,91.96,92.49,92.8]
        },
        "MGSM": {
            "x": [2.6, 6, 9.7, 11.2],
            "y": [80.2,82.9,83.7,84.4]
        },
        "MATH": {
            "x": [3.4,7.3,11.3,14.5],
            "y": [44.6,50.4,53.6,55.4]
        },
        "DATE": {
            "x": [2.8,6.4,10.1,12.4],
            "y": [62.33,68.29,68.29,70.19]
        },
        "MMLU-Pro": {
            "x": [3.2,7.3,11.2,14.4],
            "y": [32.4,34.8,35.6,36.4]
        },
        "MathBench": {
            "x": [2.9,6.5,10.2,12.8],
            "y": [62.67,69,70.67,72]
        }
    }

    accuracy_data = {
        "GSM8K": {
            "Llama-3.1-8B-Instruct": [79.83, 87.41, 90.98, 90.98, 91.81],
            "Gemma-2-9B-it": [75.36, 87.11, 89.84, 91.05, 91.58],
            "Llama-3.1-70B-Instruct": [93.93] * 4,
        },
        "MGSM": {
            "Llama-3.1-8B-Instruct": [52.9, 61.3, 69, 70.7, 71.4],
            "Gemma-2-9B-it": [74, 82.3, 83.5, 83.9, 84.3],
            "Llama-3.1-70B-Instruct": [76.5] * 4,
        },
        "MATH": {
            "Llama-3.1-8B-Instruct": [30.8, 38.6, 47.8, 50.8, 52],
            "Gemma-2-9B-it": [37.8, 46.4, 52, 54.8, 55.8],
            "Llama-3.1-70B-Instruct": [51.2] * 4,
        },
        "DATE":
            {
                "Llama-3.1-8B-Instruct": [40.65, 53.12, 59, 60.98, 62.33],
                "Gemma-2-9B-it": [56.1, 64.23, 68, 65.9, 60.43],
                "Llama-3.1-70B-Instruct": [70.46] * 4,
            },
        "MMLU-Pro": {
            "Llama-3.1-8B-Instruct": [28.4, 32.4, 34.4, 34.4, 33.4],
            "Gemma-2-9B-it": [30.4, 34.6, 36.2, 37.4, 37.4],
            "Llama-3.1-70B-Instruct": [42.8] * 4,
        },
        "MathBench": {
            "Llama-3.1-8B-Instruct": [46.67, 55, 62, 65, 67.33],
            "Gemma-2-9B-it": [54.33, 64, 68.67, 69.33, 68.33],
            "Llama-3.1-70B-Instruct": [68.67] * 4,
        }
    }

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):

        ax = axes[idx]
        for model, acc in accuracy_data[dataset].items():
            if model == "Llama-3.1-8B-Instruct":
                ax.plot(samples, acc, marker='o', color="#05B9E2",linewidth=2)
            elif model == "Gemma-2-9B-it":
                ax.plot(samples, acc, marker='s', color="#BEB8DC",linewidth=2)
            elif model == "Llama-3.1-70B-Instruct":
                ax.axhline(y=acc[0], color="#05B9E2", linestyle=":", linewidth=2)

        ax.set_title(f"{dataset}", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.set_xlabel("Number of Samplings", fontsize=20)
        
        ax.set_xticks(samples)

        # # 绘制散点图
        scatter_x = scatter_data[dataset]["x"]
        scatter_y = scatter_data[dataset]["y"]
        for i in range(len(scatter_x)):
            if i ==3:
                ax.scatter(scatter_x[i], scatter_y[i], facecolor=(0.75, 0, 0, 1), marker='*', edgecolor='#C00000', s=300)
            else:
                ax.scatter(scatter_x[i], scatter_y[i], facecolor=(0.75, 0, 0, 0.25*i), marker='*', edgecolor='#C00000', s=300)
        # # 隐藏上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        if idx==0:
            ax.set_ylabel("Accuracy (%)", fontsize=28)
        if dataset == "GSM8K":
            ax.set_ylim(74, 96)
        elif dataset == "MGSM":
            ax.set_ylim(50, 90)
        elif dataset == "MATH":
            ax.set_ylim(29, 61)
        elif dataset == "DATE":
            ax.set_ylim(38, 75)
        elif dataset == "MMLU-Pro":
            ax.set_ylim(26,46)
        elif dataset == "MathBench":
            ax.set_ylim(44, 76)

        # ax.yaxis.set_major_locator(MaxNLocator(6))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax.grid(True)
    # Collect handles and labels from one plot for global legend
    handles, labels = [], []

    # Add necessary labels for global legend
    handles.append(plt.scatter([], [], label="MS (Budget=4)", color="#C00000", facecolor=(0.75, 0, 0, 0), marker='*', edgecolor='#C00000', s=300))
    handles.append(plt.Line2D([], [], color="#05B9E2", linestyle=":", linewidth=2, label="Llama-3.1-70B-Instruct"))
    handles.append(plt.scatter([], [], label="MS (Budget=8)", color="#C00000", facecolor=(0.75, 0, 0, 0.25), marker='*', edgecolor='#C00000', s=300))
    handles.append(plt.Line2D([], [], marker='o', color="#05B9E2",linewidth=2, label="Llama-3.1-8B-Instruct"))
    handles.append(plt.scatter([], [], label="MS (Budget=12)", color="#C00000", facecolor=(0.75, 0, 0, 0.5), marker='*', edgecolor='#C00000', s=300))
    handles.append(plt.Line2D([], [], marker='s', color="#BEB8DC",linewidth=2, label="Gemma-2-9B-it"))
    handles.append(plt.scatter([], [], label="MS (Budget=16)", color="#C00000", facecolor=(0.75, 0, 0, 1), marker='*', edgecolor='#C00000', s=300))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.525, 1), ncol=4, fontsize=20)


    plt.tight_layout()
    plt.show()

# def scaling_llm_num():
#     # 数据
#     x = [1, 2, 3, 4, 5, 6]
#     mathbench1 = [75.33, 79.67, 80, 80.33, 80.33, 80.66]
#     bar1 = [72.67, 71.67, 70.67, 46.67, 54.33, 64.00]

#     mathbench2 = [68.33, 72.00, 75, 77.67, 79, 79]
#     bar2 = [54.33, 46.67, 64.00, 72.67, 71.67, 70.67]

#     date1 = [76.69, 78.60, 79.13, 78.04, 77.78, 76.69]
#     bar3 = [73.44, 71.00, 57.18, 40.65, 56.10, 49.59]

#     date2 = [60, 70.2, 71, 75.61, 73.98, 73.17]
#     bar4 = [56.10, 40.65, 49.59, 73.44, 71.00, 57.18]


#     colors = ['#FFBE7A', '#82B0D2', '#BEB8DC', '#2878B5', '#C76DA2', '#96CCCB']  # 每个柱子的颜色
#     labels = ['GPT-4o mini', 'Gemini 1.5 Flash', 'Claude 3 Haiku', 'Llama-3.1-8B-Instruct', 'Gemma-2-9b-it', 'Qwen2.5-7B-Instruct']  # 每个柱子的标签

#     # 创建图形和子图
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 13))

#     # 绘制左侧子图
#     ax1.plot(x, mathbench1, color='#FA7F6F', marker='o', label='ModelSwitch', markersize=15, linewidth=3)
#     # 绘制每个柱子
#     for i in range(len(x)):
#         ax1.bar(x[i], bar1[i], color=colors[i], label=labels[i], width=0.6, edgecolor='black')

#     ax1.set_xlabel('Number of LLMs', fontsize=48)
#     ax1.set_ylabel('Accuracy (%)', fontsize=48)
#     ax1.set_ylim(45, 82)
#     ax1.set_xticks([1,2,3,4,5,6])
#     ax1.set_title('MathBench', fontsize=48)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['left'].set_linewidth(3)
#     ax1.spines['bottom'].set_linewidth(3)
#     ax1.tick_params(axis='both', which='major', labelsize=40)

#     # 绘制右侧子图
#     ax2.plot(x, date1, color='#FA7F6F', marker='o', label='ModelSwitch', markersize=15, linewidth=3)
#     # 绘制每个柱子
#     for i in range(len(x)):
#         ax2.bar(x[i], bar3[i], color=colors[i], label=labels[i], width=0.6, edgecolor='black')

#     ax2.set_xlabel('Number of LLMs', fontsize=48)
#     ax2.set_xticks([1,2,3,4,5,6])
#     ax2.set_ylim(39, 81)
#     ax2.set_title('DATE', fontsize=48)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['left'].set_linewidth(3)
#     ax2.spines['bottom'].set_linewidth(3)
#     ax2.tick_params(axis='both', which='major', labelsize=40)

#     # 统一图例
#     handles, labels = ax1.get_legend_handles_labels()  # 获取图例信息
#     # handles[1], handles[2], labels[1], labels[2] = handles[2], handles[1], labels[2], labels[1]
#     # handles[1], handles[4], labels[1], labels[4] = handles[4], handles[1], labels[4], labels[1]
#     # handles[3], handles[6], labels[3], labels[6] = handles[6], handles[3], labels[6], labels[3]
#     # handles[3], handles[5], labels[3], labels[5] = handles[5], handles[3], labels[5], labels[3]
#     fig.legend(handles, labels, loc='upper center', fontsize=36, ncol=3, frameon=False)  # 设置图例位置和样式

#     # 调整布局
#     plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局，留出上方空间给图例
#     plt.show()
    
def scaling_llm_num():
    # 数据
    x = [1, 2, 3, 4, 5, 6]
    mathbench1 = [75.33, 79.67, 80, 80.33, 80.33, 80.66]
    bar1 = [72.67, 71.67, 70.67, 46.67, 54.33, 64.00]

    mathbench2 = [68.33, 72.00, 75, 77.67, 79, 79]
    bar2 = [54.33, 46.67, 64.00, 72.67, 71.67, 70.67]

    date1 = [76.69, 78.60, 79.13, 78.04, 77.78, 76.69]
    bar3 = [73.44, 71.00, 57.18, 40.65, 56.10, 49.59]

    date2 = [60, 70.2, 71, 75.61, 73.98, 73.17]
    bar4 = [56.10, 40.65, 49.59, 73.44, 71.00, 57.18]

    colors1 = ['#FFBE7A', '#82B0D2', '#BEB8DC', '#2878B5', '#C76DA2', '#96CCCB']  # 每个柱子的颜色
    labels1 = ['GPT-4o mini', 'Gemini 1.5 Flash', 'Claude 3 Haiku', 'Llama-3.1-8B-Instruct', 'Gemma-2-9b-it', 'Qwen2.5-7B-Instruct']  # 每个柱子的标签
    
    colors2 = ['#C76DA2', '#2878B5', '#96CCCB', '#FFBE7A', '#82B0D2', '#BEB8DC' ]  # 每个柱子的颜色
    labels2 = ['Gemma-2-9b-it', 'Llama-3.1-8B-Instruct','Qwen2.5-7B-Instruct', 'GPT-4o mini', 'Gemini 1.5 Flash', 'Claude 3 Haiku']  # 每个柱子的标签
    # 创建图形和子图
    fig, axes = plt.subplots(2, 2, figsize=(12,8))  # 2行2列

    # 绘制第一行：MathBench
    ax1 = axes[0, 0]  # 第一行第一列
    ax1.plot(x, mathbench1, color='#FA7F6F', marker='o', label='ModelSwitch', markersize=10, linewidth=2)
    for i in range(len(x)):
        ax1.bar(x[i], bar1[i], color=colors1[i], label=labels1[i] , width=0.6, edgecolor='black')
    # ax1.set_xlabel('Number of LLMs', fontsize=28)
    ax1.set_ylabel('Accuracy (%)', fontsize=20)
    ax1.set_ylim(45, 82)
    ax1.set_xticks([1, 2, 3, 4, 5, 6])
    # ax1.set_title('MathBench (Set 1)', fontsize=28)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    ax2 = axes[0, 1]  # 第一行第二列
    ax2.plot(x, mathbench2, color='#FA7F6F', marker='o', label='ModelSwitch', markersize=10, linewidth=2)
    for i in range(len(x)):
        ax2.bar(x[i], bar2[i], color=colors2[i], label=labels2[i], width=0.6, edgecolor='black')
    # ax2.set_xlabel('Number of LLMs', fontsize=28)
    # ax2.set_ylabel('Accuracy (%)', fontsize=28)
    ax2.set_ylim(45, 82)
    ax2.set_xticks([1, 2, 3, 4, 5, 6])
    # ax2.set_title('MathBench (Set 2)', fontsize=28)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    # 绘制第二行：DATE
    ax3 = axes[1, 0]  # 第二行第一列
    ax3.plot(x, date1, color='#FA7F6F', marker='o', label='ModelSwitch', markersize=10, linewidth=2)
    for i in range(len(x)):
        ax3.bar(x[i], bar3[i], color=colors1[i], label=labels1[i] , width=0.6, edgecolor='black')
    ax3.set_xlabel('Number of LLMs', fontsize=20)
    ax3.set_ylabel('Accuracy (%)', fontsize=20)
    ax3.set_ylim(39, 81)
    ax3.set_xticks([1, 2, 3, 4, 5, 6])
    # ax3.set_title('DATE (Set 1)', fontsize=28)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_linewidth(2)
    ax3.spines['bottom'].set_linewidth(2)
    ax3.tick_params(axis='both', which='major', labelsize=16)

    ax4 = axes[1, 1]  # 第二行第二列
    ax4.plot(x, date2, color='#FA7F6F', marker='o', label='ModelSwitch', markersize=10, linewidth=2)
    for i in range(len(x)):
        ax4.bar(x[i], bar4[i], color=colors2[i], label=labels2[i] , width=0.6, edgecolor='black')
    ax4.set_xlabel('Number of LLMs', fontsize=20)
    # ax4.set_ylabel('Accuracy (%)', fontsize=28)
    ax4.set_ylim(39, 81)
    ax4.set_xticks([1, 2, 3, 4, 5, 6])
    # ax4.set_title('DATE (Set 2)', fontsize=28)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_linewidth(2)
    ax4.spines['bottom'].set_linewidth(2)
    ax4.tick_params(axis='both', which='major', labelsize=16)

    # 标注 mathbench1 最后一个点的 Y 值
    last_x = x[-1]
    last_y = mathbench1[-1]
    ax1.text(last_x, last_y-1.5, f'{last_y:.1f}', fontsize=20, ha='center', va='top')

    # 标注 mathbench1 第三个点的 Y 值
    third_x = x[2]
    third_y = mathbench1[2]
    ax1.text(third_x, third_y-1.5, f'{third_y:.1f}', fontsize=20, ha='center', va='top')

    # 标注 mathbench2 最后一个点的 Y 值
    last_y = mathbench2[-1]
    ax2.text(last_x, last_y-1.5, f'{last_y:.1f}', fontsize=20, ha='center', va='top')

    # 标注 mathbench2 第三个点的 Y 值
    third_y = mathbench2[2]
    ax2.text(third_x, third_y-1.5, f'{third_y:.1f}', fontsize=20, ha='center', va='top')

    # 标注 date1 最后一个点的 Y 值
    last_y = date1[-1]
    ax3.text(last_x, last_y-1.5, f'{last_y:.1f}', fontsize=20, ha='center', va='top')

    # 标注 mathbench2 第三个点的 Y 值
    third_y = date1[2]
    ax3.text(third_x, third_y-1.5, f'{third_y:.1f}', fontsize=20, ha='center', va='top')

    # 标注 date2 最后一个点的 Y 值
    last_y = date2[-1]
    ax4.text(last_x, last_y-1.5, f'{last_y:.1f}', fontsize=20, ha='center', va='top')

    # 标注 mathbench2 第三个点的 Y 值
    third_y = date2[2]
    ax4.text(third_x, third_y-1.5, f'{third_y:.1f}', fontsize=20, ha='center', va='top')


    # 在 ax1 和 ax2 中间上方显示 "MathBench"
    fig.text(0.4, 0.96, 'MathBench', fontsize=20, ha='center', va='center')

    # 在 ax3 和 ax4 中间上方显示 "DATE"
    fig.text(0.4, 0.49, 'DATE', fontsize=20, ha='center', va='center')
    # 统一图例
    handles, labels = ax1.get_legend_handles_labels()  # 获取图例信息
    # fig.legend(handles, labels, loc='upper center', fontsize=24, ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))  # 设置图例位置和样式
    fig.legend(handles, labels, loc='upper right', fontsize=16, ncol=1, frameon=False, bbox_to_anchor=(1, 0.65))  # 设置图例位置和样式
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，留出上方空间给图例
    plt.subplots_adjust(right=0.7,wspace=0.15,hspace=0.25)
    plt.show()    

def RewardModel():
    # 示例数据
    sampling_numbers = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    actual_numbers=[1,2.4,5.1,7.8,10.1,13.2,16]
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 11))
    # 绘制 Accuracy 曲线
    
    sns.lineplot(ax=ax, x=sampling_numbers, y=[72.67,72.67,74,74,73.67,74,74.33,75,75.33], marker='s', color="#82B0D2", markersize=15,linewidth=3)
    sns.lineplot(ax=ax, x=actual_numbers, y=[72.67,77.33,78.67,78.67,79.67,79.67,79], marker='s',  color="#BEB8DC", markersize=15,linewidth=3)
    sns.lineplot(ax=ax, x=sampling_numbers, y=[72.67, 74.33, 76.67,78,79.33,80,81.33, 80.67,81.33], marker='o',  color="#FFBE7A", markersize=15,linewidth=3)
    sns.lineplot(ax=ax, x=actual_numbers, y=[72.67,79,80,80.67,81.33,82.33,83.67], marker='o', color="#96CCCB", markersize=15,linewidth=3)
    # xmin, xmax = ax.get_xlim()
    # ax.hlines(y=81.33, xmin=-1, xmax=20, colors='gray', linestyles='dashed', linewidth=1)
    # ax.set_xlim(0,17)
    # 设置坐标轴标签
    ax.set_title("MathBench", fontsize=40)
    ax.set_xlabel('Number of Samplings (K)', fontsize=40)
    ax.set_ylabel('Accuracy (%)', fontsize=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_linewidth(3)
    # ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    # 修改横坐标显示
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.set_ylim(71,85)


    # 显示图例
    # ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    handles, labels = [], []
    # Add necessary labels for global legend
    handles.append(plt.Line2D([], [], marker='o', label='RM (BoN) of ModelSwitch',  color="#96CCCB",markersize=15,linewidth=3))
    handles.append(plt.Line2D([], [], marker='o', label='RM (BoN) of Best Single', color="#FFBE7A",markersize=15,linewidth=3))
    handles.append(plt.Line2D([], [], marker='s', label='ModelSwitch', color="#BEB8DC",markersize=15,linewidth=3))
    handles.append(plt.Line2D([], [], marker='s', label='SC of Best Single', color="#82B0D2",markersize=15,linewidth=3))
    
    
    fig.legend(handles=handles, loc='upper center',  fontsize=30, ncol=2, frameon=False)
    
    # 调整布局
    # plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # 自定义网格线
    # 水平虚线
    # 获取当前 X 轴的范围
    
    ax.set_xticks(sampling_numbers)
    plt.tight_layout()
    plt.show()

def test():
    # 定义两个点
    modelswitch_point = (math.log2(32), 74)  # ModelSwitch 的点
    gemini_point = (math.log2(512), 73.67)   # Gemini 1.5 Flash 的点

    # 提取坐标
    x1, y1 = modelswitch_point
    x2, y2 = gemini_point

    # 定义折线的路径
    # 从 (x1, y1) 向上画到 (x1, y2 + offset)，再向右画到 (x2, y2 + offset)，最后向下画到 (x2, y2)
    offset = 1  # 控制折线的高度
    path_x = [x1, x1, x2, x2]
    path_y = [y1, y2 + offset, y2 + offset, y2]

    # 绘制折线和点
    plt.figure(figsize=(8, 6))
    plt.plot(path_x, path_y, label="Path", color="blue")

    # 在路径的末端添加箭头
    # 第一个箭头：从 (x1, y2 + offset) 指向 (x1, y1)（向下）
    plt.annotate('', xy=(x1, y1), xytext=(x1, y2 + offset),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))


    # 第三个箭头：从 (x2, y2 + offset) 指向 (x2, y2)（向下）
    plt.annotate('', xy=(x2, y2), xytext=(x2, y2 + offset),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # 绘制原始点
    plt.scatter([x1, x2], [y1, y2], color="red", label="Points")
    plt.text(x1, y1, 'ModelSwitch', fontsize=12, ha='right')
    plt.text(x2, y2, 'Gemini 1.5 Flash', fontsize=12, ha='left')

    # 添加标签和标题
    plt.xlabel("log2(Sampling Budget)")
    plt.ylabel("Performance")
    plt.title("Alignment of ModelSwitch and Gemini 1.5 Flash")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # correlation()
    # Curve_MathBench_v3()
    # close_source_performance_gain()
    # open_source_performance_gain()
    # close_source_efficiency()
    # open_source_efficiency()
    # accuracy_and_coverage_500()
    # accuracy_and_coverage_500_v1()
    # close_source_performance_gain_v1()
    # open_source_performance_gain_v1()
    # close_source_vs_sc_v1()
    # open_source_vs_sc_v1()
    # correlation_v2()
    # theoretical_analysis()
    scaling_llm_num()
    # accuracy_and_coverage_500_MathBench()
    # accuracy_and_coverage_combined()
    # accuracy_and_coverage_combined_v1()
    # RewardModel()
    # close_source_vs_sc_v1()
    # test()

if __name__=="__main__":
    main()