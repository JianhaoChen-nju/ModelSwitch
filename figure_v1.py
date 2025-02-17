import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# 数据
data = {
    1: [1047, 1077],
    0.23942675831956298: [7, 15],
    0.3473010061326758: [10, 14],
    0.15996415699885244: [6, 13],
    0.6837905625465495: [58, 70],
    0.49040833450037835: [20, 28],
    0.07309430472953364: [5, 7],
    0.17307922394972686: [0, 1],
    0.22430588390457826: [2, 3],
    0.6045463755675999: [5, 7],
    0.48634861513960614: [7, 10],
    0.4001646400668966: [4, 4],
    0.5343471582763926: [3, 3],
    0.3722575510602069: [1, 1],
    0.2413514173018023: [1, 2],
    0.10521812225784527: [3, 15],
    0.0625: [2, 4],
    0.11275503404013795: [1, 1],
    0.27645416725018923: [0, 1],
    0.19182946575151993: [1, 2],
    0.19158420779849994: [0, 1],
    0.16856921205351724: [2, 3],
    0.16947708362509467: [0, 1],
    0.24295136276161375: [0, 1],
    0.25790278943481343: [1, 2],
    0.291273210033801: [1, 4],
    0.13915202120263978: [0, 1],
    0.39571741362721025: [2, 2],
    0.24774508405888623: [0, 2],
    0.3791592375148556: [1, 1],
    0.28664927943588414: [1, 1],
    0.262074665597: [2, 3],
    0.3241897712693562: [1, 1],
    0.3575597646271073: [1, 1],
    0.4439321291597814: [4, 4],
    0.18930949504592193: [0, 1],
    0.2722556924520089: [0, 1],
    0.1702599432198839: [1, 1],
    0.18015661418955597: [0, 1],
    0.2825129395798907: [1, 2],
    0.29052326193291933: [1, 1],
    0.3365532534228848: [0, 1],
    0.1796875: [0, 1],
    0.34323932309539434: [0, 1],
    0.2961182163389773: [1, 1],
    0.20935632577996421: [1, 1],
    0.08543990123347506: [1, 1],
}
# 合并阈值（控制 key 的接近程度）
threshold = 0.05

# 用于存储合并后的结果
merged_data = defaultdict(lambda: [0, 0])  # 初始化为 [0, 0]

# 遍历原始数据并合并
for key, value in data.items():
    found = False
    for merged_key in list(merged_data.keys()):
        if abs(key - merged_key) <= threshold:  # 如果两个 key 的差值小于等于阈值
            merged_data[merged_key][0] += value[0]
            merged_data[merged_key][1] += value[1]
            found = True
            break
    if not found:
        merged_data[key][0] += value[0]
        merged_data[key][1] += value[1]

# 将合并后的结果转换为普通字典
merged_data = dict(merged_data)
print(merged_data)

# 计算 Accuracy（value[0]/value[1]）
consistency_scores = list(merged_data.keys())
accuracies = [v[0] / v[1] for v in merged_data.values()]

# 创建 DataFrame 以便使用 seaborn
df = pd.DataFrame({
    "Consistency Score": consistency_scores,
    "Accuracy": accuracies
})

# 绘制散点图和拟合曲线
plt.figure(figsize=(12, 9))
sns.regplot(x="Consistency Score", y="Accuracy", data=df, scatter_kws={'color': 'blue', 's': 50}, line_kws={'color': 'red'},ci=None)

# 设置标题和标签
# plt.title("Relationship Between Consistency Score and Accuracy", fontsize=14)
plt.xlabel("Consistency Score", fontsize=28)
plt.ylabel("Accuracy", fontsize=28)

ax = plt.gca()  # 获取当前的轴对象
ax.spines['top'].set_visible(False)    # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框
ax.spines['left'].set_visible(True)    # 显示左边框
ax.spines['bottom'].set_visible(True)  # 显示下边框
ax.tick_params(axis='both', which='major', labelsize=24)
ax.set_ylim(-0.1,1.1)
# 显示图形
plt.tight_layout()
plt.show()