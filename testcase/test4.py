import matplotlib.pyplot as plt
import numpy as np

# 数据（各满意度评分的百分比分布）
ratings = ['1', '2', '3', '4', '5']

# 不同方法的评分分布
syncclip = [2, 3, 8, 45, 42]  # 百分比
cnn_rnn = [5, 7, 30, 35, 23]
lstm = [8, 12, 35, 30, 15]
rule_based = [10, 20, 40, 20, 10]

# 创建图像
plt.figure(figsize=(12, 6))

# 设置柱状图的位置
x = np.arange(len(ratings))
width = 0.2

# 绘制柱状图
plt.bar(x - 1.5*width, syncclip, width, label='SyncClip (Ours)', color='darkblue')
plt.bar(x - 0.5*width, cnn_rnn, width, label='CNN+RNN', color='cornflowerblue')
plt.bar(x + 0.5*width, lstm, width, label='LSTM-Based', color='lightblue')
plt.bar(x + 1.5*width, rule_based, width, label='Rule-Based DTW', color='gray')

# 添加标签和标题
plt.xlabel('User Satisfaction Rating', fontsize=12)
plt.ylabel('Percentage of Responses (%)', fontsize=12)
plt.title('Distribution of User Satisfaction Ratings by Method', fontsize=14, fontweight='bold')
plt.xticks(x, ratings)

# 添加图例
plt.legend(loc='upper left')

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for i, method_data in enumerate([syncclip, cnn_rnn, lstm, rule_based]):
    for j, value in enumerate(method_data):
        plt.text(j + (i-1.5)*width, value + 1, str(value) + '%', 
                 ha='center', va='bottom', fontsize=8)

# 添加注释
plt.annotate('87% of SyncClip ratings\nare 4 or higher', 
             xy=(4, 42), xytext=(3.3, 60),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10)

plt.tight_layout()

# 保存图像
plt.savefig('user_ratings.png', dpi=300)
plt.show()