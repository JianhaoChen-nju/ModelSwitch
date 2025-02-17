import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle


def efficieny_and_performance():
    # 数据
    tasks = ['gsm8k', 'math', 'last letters', 'mmlu-physical', 'mmlu-pro', 'AGIEval', 'MGSM']
    cost_savings = [43.70, 34.80, 36.70, 42.00, 19.50, 34.00, 42.70]
    best_sc = [95.07, 80.57, 94.40, 89.22, 58.80, 72.20, 88.70]
    efficient_fusion = [95.98, 82.57, 96.00, 90.31, 61.20, 72.60, 91.00]

    # 计算性能相对提升百分比
    relative_improvement = [(ef - bs) / bs * 100 for ef, bs in zip(efficient_fusion, best_sc)]

    # 创建子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 第一张图：效率提升（Cost Savings）
    ax1 = axes[0]
    ax1.bar(tasks, cost_savings, color='#A8D5BA', alpha=0.7)
    ax1.set_ylabel('Cost Savings (%)')
    ax1.set_title('Efficiency Improvement')
    ax1.set_ylim(0, 50)  # 设置 y 轴范围，让图表更直观
    for i, v in enumerate(cost_savings):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center')  # 显示百分比数值

    # 第二张图：性能提升（Relative Improvement）
    ax2 = axes[1]
    ax2.bar(tasks, relative_improvement, color='#F4A8A8', alpha=0.7)
    ax2.set_ylabel('Relative Improvement (%)')
    ax2.set_title('Performance Improvement')
    ax2.set_ylim(0, max(relative_improvement) + 2)  # 设置 y 轴范围
    for i, v in enumerate(relative_improvement):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center')  # 显示百分比数值

    # 调整布局
    plt.tight_layout()
    plt.show()


def Curve_MathBench():
    accuracy_data = {
        "GPT-4o mini": [72.67, 74, 74, 75.33],
        "Gemini 1.5 Flash": [71.67, 72.67, 72.67, 72.67],
        "ModelSwitch (Close Source)": [72, 78.67, 78.33, 79.67],
        "Gemini 1.5 Pro": [78.33] * 4,
        "GPT-4o": [76.67] * 4,
        "Llama-3.1-8B-Instruct": [46.67, 60, 63.67, 67.33],
        "Gemma-2-9B-it": [54.33, 67.33, 69.33, 68.33],
        "ModelSwitch (Open Source)": [50, 67, 70, 72],
        "Llama-3.1-70B-Instruct": [68.67] * 4,
    }

    # 横坐标
    samples = [1, 4, 7, 11]

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 9))

    # 绘制每个模型的数据
    for model, acc in accuracy_data.items():
        # if model =="GPT-4o mini":
        #     ax.plot(samples, acc, marker='s', color="#2CA02C", label="GPT-4o mini")
        # elif model == "Gemini 1.5 Flash":
        #     ax.plot(samples, acc, marker='o', color="#D6278C", label="Gemini 1.5 Flash")
        if model == "Gemini 1.5 Pro":
            ax.axhline(y=acc[0], color="#D6278C", linestyle="--", linewidth=2, label="Gemini 1.5 Pro")
        elif model == "GPT-4o":
            ax.axhline(y=acc[0], color="#2CA02C", linestyle=":", linewidth=2, label="GPT-4o")
        elif model == "ModelSwitch (Close Source)":
            ax.plot(samples, acc, marker='s', color="#1E90FF", label="ModelSwitch (Close-Source LLM)")
        elif model == "Llama-3.1-70B-Instruct":
            ax.axhline(y=acc[0], color="#E45756", linestyle="-.", linewidth=2, label="Llama-3.1-70B-Instruct")
        elif model == "ModelSwitch (Open Source)":
            ax.plot(samples, acc, marker='o', color="#1E90FF", label="ModelSwitch (Open-Source LLM)")
        # elif model =="Llama-3.1-8B-Instruct":
        #     ax.plot(samples, acc, marker='s', color="#2CA02C")
        # elif model == "Gemma-2-9B-it":
        #     ax.plot(samples, acc, marker='o', color="#D6278C")

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlabel("Sampling Number", fontsize=24)
    ax.set_ylabel("Accuracy (%)", fontsize=24)
    ax.set_xticks(samples)

    # 添加标签和图例
    ax.legend(loc='best', fontsize=18)

    # # 去掉网格线
    # ax.grid(False)

    # 显示图形
    plt.show()

def Curve_MathBench_v2():
    accuracy_data = {
    "ModelSwitch": [78.67, 78.33, 79.67],
    "GPT-4o mini": [74, 74, 75.33],
    "Gemini 1.5 Flash": [72.67, 72.67, 72.67],
    "Gemini 1.5 Pro": [78.33],
    "GPT-4o": [76.67],
    }

    # Define sample numbers for plotting
    samples_common = [6, 10, 16]
    actual_samples_modelswitch = [4, 7, 11]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 9))

    # Bar width
    bar_width = 0.6

    # Plot ModelSwitch data
    ax.bar(np.array(samples_common) - bar_width, accuracy_data["ModelSwitch"], 
        bar_width, label="ModelSwitch", color="#1E90FF")

    # Plot other models' data
    ax.bar(samples_common, accuracy_data["GPT-4o mini"], 
        bar_width, label="GPT-4o mini", color="#2CA02C")
    ax.bar(np.array(samples_common) + bar_width, accuracy_data["Gemini 1.5 Flash"], 
        bar_width, label="Gemini 1.5 Flash", color="#D6278C")

    # Add horizontal lines for fixed models
    ax.axhline(y=accuracy_data["Gemini 1.5 Pro"][0], color="#D6278C", linestyle="--", linewidth=2, label="Gemini 1.5 Pro")
    ax.axhline(y=accuracy_data["GPT-4o"][0], color="#2CA02C", linestyle=":", linewidth=2, label="GPT-4o")

    # Annotate actual sample numbers on top of ModelSwitch bars
    for i, (x, y, sample) in enumerate(zip(np.array(samples_common) - bar_width, accuracy_data["ModelSwitch"], actual_samples_modelswitch)):
        ax.text(x, y + 0.5, f'{sample}', ha='center', va='bottom', fontsize=24, color='black')

    # Set labels and ticks
    ax.set_xlabel("Sampling Number", fontsize=24)
    ax.set_ylabel("Accuracy (%)", fontsize=24)
    ax.set_xticks(samples_common)
    ax.set_xticklabels(samples_common)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlim(5, 17)
    ax.set_ylim(70,82)
    # Add legend
    ax.legend(loc='best', fontsize=15, ncol=3)

    # Show plot
    plt.tight_layout()
    plt.show()

def Curve_MathBench_v3():
    accuracy_data = {
        "ModelSwitch": [78.67, 78.33, 79.67],
        "GPT-4o mini": [74, 74, 75.33],
        "Gemini 1.5 Flash": [72.67, 72.67, 72.67],
        "Gemini 1.5 Pro": [78.33],
        "GPT-4o": [76.67],
    }

    # Define sample numbers for plotting
    positions = np.arange(3)  # Create three positions for the bars
    actual_samples_modelswitch = [4, 7, 11]
    actual_samples = [6, 10, 16]
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 9))

    # Bar width
    bar_width = 0.2

    # Plot ModelSwitch data
    ax.bar(positions - bar_width, accuracy_data["ModelSwitch"], 
           bar_width, label="ModelSwitch", color="#1E90FF")

    # Plot other models' data
    ax.bar(positions, accuracy_data["GPT-4o mini"], 
           bar_width, label="GPT-4o mini", color="#2CA02C")
    ax.bar(positions + bar_width, accuracy_data["Gemini 1.5 Flash"], 
           bar_width, label="Gemini 1.5 Flash", color="#D6278C")

    # Add horizontal lines for fixed models
    ax.axhline(y=accuracy_data["Gemini 1.5 Pro"][0], color="#D6278C", linestyle="--", linewidth=2, label="Gemini 1.5 Pro")
    ax.axhline(y=accuracy_data["GPT-4o"][0], color="#2CA02C", linestyle=":", linewidth=2, label="GPT-4o")

    # Annotate actual sample numbers on top of ModelSwitch bars
    for i, (x, y, sample) in enumerate(zip(positions - bar_width, accuracy_data["ModelSwitch"], actual_samples_modelswitch)):
        ax.text(x, y + 0.5, f'{sample}', ha='center', va='bottom', fontsize=24, color='black')
    # Annotate actual sample numbers on top of ModelSwitch bars
    for i, (x, y, sample) in enumerate(zip(positions, accuracy_data["GPT-4o mini"], actual_samples)):
        ax.text(x, y + 0.5, f'{sample}', ha='center', va='bottom', fontsize=24, color='black')
        # Annotate actual sample numbers on top of ModelSwitch bars
    for i, (x, y, sample) in enumerate(zip(positions + bar_width, accuracy_data["Gemini 1.5 Flash"], actual_samples)):
        ax.text(x, y + 0.5, f'{sample}', ha='center', va='bottom', fontsize=24, color='black')

    # Set labels and ticks
    ax.set_xlabel("Sampling Number", fontsize=24)
    ax.set_ylabel("Accuracy (%)", fontsize=24)
    plt.xticks([])
    # ax.set_xticks(positions)
    # ax.set_xticklabels(['Sample 1', 'Sample 2', 'Sample 3'])  # Label these as needed
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_ylim(70, 82)

    # Add legend
    ax.legend(loc='best', fontsize=15, ncol=3)

    # Show plot
    plt.tight_layout()
    plt.show()

def close_source_vs_sc():
    # Define samples and datasets
    samples = [1, 6, 10, 16]
    datasets = ["GSM8K", "MATH", "MathBench", "MGSM", "DATE", "MMLU-Pro"]

    # Accuracy data for each model across datasets
    accuracy_data = {
        "GSM8K": {
            "GPT-4o mini": [93.56, 95.07, 95, 94.92],
            "Gemini 1.5 Flash": [93.18, 94.24, 94.31, 94.62],
            "fusion": [93.40, 95.91, 95.98, 95.98],
            "GPT-4o": [95.68] * 4,
            "Gemini 1.5 Pro": [95.3] * 4
        },
        "MGSM": {
            "GPT-4o mini": [86.7, 88.7, 88.8, 89.3],
            "Gemini 1.5 Flash": [88.8, 88.7, 88.7, 89.4],
            "fusion": [88.1, 90.9, 91, 91.3],
            "GPT-4o": [92.1] * 4,
            "Gemini 1.5 Pro": [91.2] * 4
        },
        "MATH": {
            "GPT-4o mini": [73.71, 77.29, 78.57, 79.14],
            "Gemini 1.5 Flash": [77.71, 80.71, 81.57, 82],
            "fusion": [76.29, 81.28, 82.57, 83.14],
            "GPT-4o": [78.71] * 4,
            "Gemini 1.5 Pro": [80.86] * 4
        },
        "DATE":
            {
                "GPT-4o mini": [73.44, 76.42, 76.42, 76.69],
                "Gemini 1.5 Flash": [71, 72.63, 74.25, 74.8],
                "fusion": [71.54, 77.51, 77.24, 78.32],
                "GPT-4o": [72.63] * 4,
                "Gemini 1.5 Pro": [80.76] * 4
            },
        "MMLU-Pro": {
            "GPT-4o mini": [41.6, 46.8, 49, 49.6],
            "Gemini 1.5 Flash": [53, 57.4, 58.8, 60.4],
            "fusion": [48.2, 59.6, 60, 60.6],
            "GPT-4o": [58.4] * 4,
            "Gemini 1.5 Pro": [61] * 4
        },
        "MathBench": {
            "GPT-4o mini": [72.67, 74, 74, 75.33],
            "Gemini 1.5 Flash": [71.67, 72.67, 72.67, 72.67],
            "fusion": [71, 78.67, 78.33, 79.67],
            "GPT-4o": [76.67] * 4,
            "Gemini 1.5 Pro": [78.33] * 4
        }
    }

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        for model, acc in accuracy_data[dataset].items():
            if model == "GPT-4o mini":
                ax.plot(samples, acc, marker='s', color="#2CA02C")
            elif model == "Gemini 1.5 Flash":
                ax.plot(samples, acc, marker='o', color="#D6278C")
            elif model == "fusion":
                ax.plot(samples, acc, marker='^', color="#1E90FF")
            elif model == "GPT-4o":
                ax.axhline(y=acc[0], color="#2CA02C", linestyle=":", linewidth=2)
            elif model == "Gemini 1.5 Pro":
                ax.axhline(y=acc[0], color="#D6278C", linestyle="-.", linewidth=2)

        ax.set_title(f"{dataset}", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel("Sampling Number", fontsize=18)
        ax.set_ylabel("Accuracy (%)", fontsize=18)
        ax.set_xticks(samples)
        if dataset == "GSM8K":
            ax.set_ylim(93, 96.5)
        elif dataset == "MGSM":
            ax.set_ylim(86, 94)
        elif dataset == "MATH":
            ax.set_ylim(73, 85)
        elif dataset == "DATE":
            ax.set_ylim(70, 82)
        elif dataset == "MMLU-Pro":
            ax.set_ylim(40, 64)
        elif dataset == "MathBench":
            ax.set_ylim(70, 80)

        ax.yaxis.set_major_locator(MaxNLocator(6))
        # ax.grid(True)
    # Collect handles and labels from one plot for global legend
    handles, labels = [], []

    # Add necessary labels for global legend
    handles.append(plt.Line2D([], [], marker='^', label="ModelSwitch", color="#1E90FF"))
    handles.append(plt.Line2D([], [], color="#2CA02C", linestyle=":", linewidth=2, label="GPT-4o"))
    handles.append(plt.Line2D([], [], color="#D6278C", linestyle="-.", linewidth=2, label="Gemini 1.5 Pro"))
    handles.append(plt.Line2D([], [], marker='s', color="#2CA02C", label="GPT-4o mini"))
    handles.append(plt.Line2D([], [], marker='o', label="Gemini-1.5 Flash", color="#D6278C"))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.525, 1), ncol=5, fontsize=18)

    plt.tight_layout()
    plt.show()


def open_source_vs_sc():
    # Define samples and datasets
    samples = [1, 6, 10, 16]
    datasets = ["GSM8K", "MATH", "MathBench", "MGSM", "DATE", "MMLU-Pro"]






    # Accuracy data for each model across datasets
    accuracy_data = {
        "GSM8K": {
            "Llama-3.1-8B-Instruct": [79.83, 89.92, 91.96, 91.81],
            "Gemma-2-9B-it": [75.36, 88.78, 90.52, 91.58],
            "ModelSwitch": [77.86, 91.13, 93.18, 92.65],
            "Llama-3.1-70B-Instruct": [93.93] * 4,
        },
        "MGSM": {
            "Llama-3.1-8B-Instruct": [52.90, 65.3, 70.3, 71.4],
            "Gemma-2-9B-it": [74, 82.4, 84, 84.3],
            "ModelSwitch": [63.50, 82.6, 83.3, 84.4],
            "Llama-3.1-70B-Instruct": [76.5] * 4,
        },
        "MATH": {
            "Llama-3.1-8B-Instruct": [34.14, 49.14, 53.43, 55.71],
            "Gemma-2-9B-it": [38.57, 56, 57.14, 59],
            "ModelSwitch": [35.29, 53, 57.29, 58.71],
            "Llama-3.1-70B-Instruct": [54.2] * 4,
        },
        "DATE":
            {
                "Llama-3.1-8B-Instruct": [40.65, 57.45, 60.98, 62.33],
                "Gemma-2-9B-it": [56.1, 67.48, 66.94, 60.43],
                "ModelSwitch": [50.68, 65.31, 68.56, 70.19],
                "Llama-3.1-70B-Instruct": [70.46] * 4,
            },
        "MMLU-Pro": {
            "Llama-3.1-8B-Instruct": [28.4, 34.4, 35.2, 33.4],
            "Gemma-2-9B-it": [30.4, 36.6, 37, 37.4],
            "ModelSwitch": [26.20, 34.8, 35, 36.4],
            "Llama-3.1-70B-Instruct": [42.8] * 4,
        },
        "MathBench": {
            "Llama-3.1-8B-Instruct": [46.67, 60, 63.67, 67.33],
            "Gemma-2-9B-it": [54.33, 67.33, 69.33, 68.33],
            "ModelSwitch": [51.00, 67, 70, 72],
            "Llama-3.1-70B-Instruct": [68.67] * 4,
        }
    }

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        for model, acc in accuracy_data[dataset].items():
            if model == "Llama-3.1-8B-Instruct":
                ax.plot(samples, acc, marker='s', color="#2CA02C")
            elif model == "Gemma-2-9B-it":
                ax.plot(samples, acc, marker='o', color="#D6278C")
            elif model == "ModelSwitch":
                ax.plot(samples, acc, marker='^', color="#1E90FF")
            elif model == "Llama-3.1-70B-Instruct":
                ax.axhline(y=acc[0], color="#2CA02C", linestyle=":", linewidth=2)

        ax.set_title(f"{dataset}", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel("Sampling Number", fontsize=18)
        ax.set_ylabel("Accuracy (%)", fontsize=18)
        ax.set_xticks(samples)
        if dataset == "GSM8K":
            ax.set_ylim(75, 95)
        elif dataset == "MGSM":
            ax.set_ylim(50, 85)
        elif dataset == "MATH":
            ax.set_ylim(32, 60)
        elif dataset == "DATE":
            ax.set_ylim(40, 72)
        elif dataset == "MMLU-Pro":
            ax.set_ylim(25, 44)
        elif dataset == "MathBench":
            ax.set_ylim(44, 74)

        ax.yaxis.set_major_locator(MaxNLocator(6))
        # ax.grid(True)
    # Collect handles and labels from one plot for global legend
    handles, labels = [], []

    # Add necessary labels for global legend
    handles.append(plt.Line2D([], [], marker='^', label="ModelSwitch", color="#1E90FF"))
    handles.append(plt.Line2D([], [], color="#2CA02C", linestyle=":", linewidth=2, label="Llama-3.1-70B-Instruct"))
    handles.append(plt.Line2D([], [], marker='s', color="#2CA02C", label="Llama-3.1-8B-Instruct"))
    handles.append(plt.Line2D([], [], marker='o', label="Gemma-2-9B-it", color="#D6278C"))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.525, 1), ncol=4, fontsize=18)

    plt.tight_layout()
    plt.show()


def close_source_vs_ma():
    # Data
    methods = ['MAD', 'MOA', 'ModelSwitch']
    categories = ['GSM8k', 'MGSM', 'MATH', 'LastLetters', 'MMLU-Pro', 'AGIEval']
    values = [
        [95.30, 90.4, 78, 80.4, 50.2, 69.8],  # MAD
        [95, 89.6, 81.14, 94.6, 50.8, 70.4],  # MOA
        [96.13, 91.4, 82.57, 96.8, 63.2, 73.4]  # Fusion
    ]

    # Bar width
    bar_width = 0.15
    index = np.arange(len(categories))

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 9))

    # Colors for each method
    colors = ['#3CB371', '#FF7F50', '#1E90FF']  # MAD: Coral, MOA: MediumSeaGreen, Fusion: DodgerBlue

    # Create bars
    for i, method in enumerate(methods):
        ax.bar(index + i * bar_width, values[i], bar_width, label=method, color=colors[i])

    # Labels and title
    # ax.set_xlabel('Dataset',fontsize=18)
    ax.set_ylabel('Accuracy (%)', fontsize=18)
    ax.set_title('Performance Comparison', fontsize=18)
    ax.set_ylim(40, 100)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(categories)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend()

    # Show plot
    plt.tight_layout()
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
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_ylabel("Cost Savings Percentage (%)", fontsize=24)
    ax.set_ylim(0, 50)  # 设置y轴范围

    # 美化图表
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

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
    ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.set_title("Computation Savings Across Datasets", fontsize=24)
    ax.set_ylabel("Cost Savings Percentage (%)", fontsize=24)
    ax.set_ylim(0, 30)  # 设置y轴范围

    # 美化图表
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 显示图表
    plt.show()


def correlation_gsm8k():
    # 数据
    plt.figure(figsize=(12, 9))  # 设置画布大小
    data = {
        '[0.1-0.2)': [24, 44],
        '[0.2-0.5)': [43, 61],
        '[0.5-1)': [48, 56],
        '1': [1129, 1158]
    }
    # 计算横坐标（区间中间值）
    x_labels = list(data.keys())
    x_values = [0.15, 0.3, 0.6, 1.0]  # 区间中间值，最后一项取1

    # 计算纵坐标（value[0] / value[1] * 100%）
    y_values = [v[0] / v[1] * 100 for v in data.values()]
    # 创建折线图


    # 添加数据点标注
    # for x, y in zip(x_values, y_values):
    #     plt.text(x, y, f'{y:.2f}%', fontsize=18, ha='center', va='bottom')
    data1 = {'[0.1-0.2)': [24, 35], '[0.2-0.4)': [72, 101], '[0.4-0.6)': [66, 72], '[0.6-1]': [1091, 1111]}

    # 计算纵坐标（value[0] / value[1] * 100%）
    y1_values = [v[0] / v[1] * 100 for v in data1.values()]
    # 创建折线图

    plt.plot(x_values, y1_values, marker='s', color='#2CA02C',label="GPT-4o mini")  # 折线图
    plt.plot(x_values, y_values, marker='o', color='#D6278C', label="Gemini 1.5 Flash")  # 折线图
    # 添加数据点标注
    # for x, y in zip(x_values, y1_values):
    #     plt.text(x, y, f'{y:.2f}%', fontsize=18, ha='center', va='bottom')


    # 设置图表标题和标签
    # plt.title('Correlation Between Consistency and Accuracy', fontsize=24)
    plt.xlabel('Consistency Score', fontsize=24)
    plt.ylabel('Accuracy (%)', fontsize=24)

    # 设置横坐标刻度
    plt.yticks(fontsize=24)
    plt.xticks(x_values, x_labels, fontsize=24)  # 使用区间标签作为横坐标
    plt.legend(loc='lower right',fontsize=24)
    # 添加网格
    plt.grid(alpha=0.5, linestyle='--')

    # # 添加图例
    # plt.legend(fontsize=10)

    # 显示图表
    plt.tight_layout()  # 自动调整布局
    plt.show()

def close_source_performance_gain():
    # 数据
    methods = ['MAD', 'Chateval', 'MOA', 'ModelSwitch']
    datasets = ['GSM8K', 'MATH', 'MathBench', 'MGSM', 'DATE', 'MMLU-Pro']
    best_single_scores = [93.56, 77.71, 72.67, 88.8, 73.44, 53]
    scores = [
        [95.30, 78, 77, 90.4, 83.20, 50.2],    # MAD
        [93.6, 78.9, 75.3, 87.2, 76.42, 43],  # Chateval(3*5)
        [94.92, 81.29, 79, 89.5, 78.32, 50.8], # MOA
        [96.13, 82.57, 80, 91.4, 78.8, 63.2]  # Fusion

    ]

    # 计算性能增益
    performance_gains = [[score - best for score, best in zip(method_scores, best_single_scores)] for method_scores in scores]

    # 绘制柱状图
    x = np.arange(len(datasets))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, (method, gains) in enumerate(zip(methods, performance_gains)):
        ax.bar(x + i * width, gains, width, label=method)
    # 添加基准线
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')
    # 添加标签和标题
    # ax.set_xlabel('Dataset', fontsize=24)
    ax.set_ylabel('Performance Gain', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.set_title('Performance Gain Compared to Best Single')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=18)

    plt.show()

def open_source_performance_gain():
    # 数据
    methods = ['MOA', 'ModelSwitch']
    datasets = ['GSM8K', 'MATH', 'MathBench', 'MGSM', 'DATE', 'MMLU-Pro']
    best_single_scores = [80.29, 47.86, 64.00, 74.00, 56.10, 30.60]
    scores = [
        [89.08,53.43,67.33,72.1,53.12,30],
        [94.24,64.29,76.33,85.80,70.19,36.40]
    ]

    # 计算性能增益
    performance_gains = [[score - best for score, best in zip(method_scores, best_single_scores)] for method_scores in scores]

    # 绘制柱状图
    x = np.arange(len(datasets))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, (method, gains) in enumerate(zip(methods, performance_gains)):
        if method == 'MOA':
            ax.bar(x + i * width, gains, width, label=method,color='#2CA02C')
        else:
            ax.bar(x + i * width, gains, width, label=method,color='#D62728')
    # 添加基准线
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')
    # 添加标签和标题
    # ax.set_xlabel('Dataset', fontsize=24)
    ax.set_ylabel('Performance Gain', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.set_title('Performance Gain Compared to Best Single')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=18)

    plt.show()

def main():
    # open_source_performance_gain()
    # close_source_performance_gain()
    # open_source_vs_sc()
    correlation_gsm8k()
    # close_source_efficiency()
    # open_source_efficiency()
    # close_source_vs_sc()
    # close_source_vs_ma()
    # Curve_MathBench()
    # Curve_MathBench_v2()
    Curve_MathBench_v3()

main()
