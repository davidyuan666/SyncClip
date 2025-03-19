import matplotlib.pyplot as plt
import numpy as np

# 数据（每分钟场景转换数）
complexity = [5, 8, 10, 12, 15, 20]

# 不同方法在不同复杂度下的同步误差
syncclip_errors = [105, 115, 120, 130, 135, 140]
cnn_rnn_errors = [145, 155, 165, 180, 195, 210]
lstm_errors = [160, 170, 180, 195, 215, 240]
rule_based_errors = [180, 190, 200, 225, 245, 270]

# 创建图像
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(complexity, syncclip_errors, 'o-', label='SyncClip (Ours)', linewidth=2.5, color='darkblue', markersize=8)
plt.plot(complexity, cnn_rnn_errors, 's-', label='CNN+RNN', linewidth=2, color='cornflowerblue')
plt.plot(complexity, lstm_errors, '^-', label='LSTM-Based', linewidth=2, color='lightblue')
plt.plot(complexity, rule_based_errors, 'x-', label='Rule-Based DTW', linewidth=2, color='gray')

# 添加网格和标签
plt.grid(linestyle='--', alpha=0.7)
plt.xlabel('Video Complexity (Scene Transitions per Minute)', fontsize=12)
plt.ylabel('Temporal Synchronization Error (ms)', fontsize=12)
plt.title('Impact of Video Complexity on Synchronization Error', fontsize=14, fontweight='bold')

# 添加图例
plt.legend(loc='upper left', fontsize=11)

# 添加注释指出主要趋势
plt.annotate('SyncClip maintains lower\nerror rates at high complexity', 
             xy=(18, 140), xytext=(15, 90),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10)

plt.tight_layout()

# 保存图像
plt.savefig('complexity_impact.png', dpi=300)
plt.show()