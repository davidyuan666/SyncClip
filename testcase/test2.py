import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['Rule-Based\nDTW', 'LSTM-Based\nModel', 'CNN+RNN\nApproach', 'SyncClip\n(Ours)']
temporal_errors = [200, 180, 165, 120]
semantic_correspondence = [71.5, 75.8, 78.2, 87.3]

# 创建图像
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制时间同步误差（左侧）
ax1.bar(methods, temporal_errors, color=['gray', 'lightblue', 'cornflowerblue', 'darkblue'])
ax1.set_title('Temporal Synchronization Error', fontsize=13, fontweight='bold')
ax1.set_ylabel('Error (ms)', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
# 添加数据标签
for i, v in enumerate(temporal_errors):
    ax1.text(i, v + 5, str(v), ha='center', fontsize=10)

# 绘制语义对应关系（右侧）
ax2.bar(methods, semantic_correspondence, color=['gray', 'lightblue', 'cornflowerblue', 'darkblue'])
ax2.set_title('Semantic Correspondence', fontsize=13, fontweight='bold')
ax2.set_ylabel('Correspondence (%)', fontsize=12)
ax2.set_ylim([65, 95])
ax2.grid(axis='y', linestyle='--', alpha=0.7)
# 添加数据标签
for i, v in enumerate(semantic_correspondence):
    ax2.text(i, v + 1, str(v), ha='center', fontsize=10)

# 标记最佳方法
ax1.bar(methods[3], temporal_errors[3], color='darkblue', edgecolor='red', linewidth=2)
ax2.bar(methods[3], semantic_correspondence[3], color='darkblue', edgecolor='red', linewidth=2)

plt.suptitle('Comparison of Synchronization Performance Across Methods', fontsize=15, fontweight='bold')
plt.tight_layout()

# 保存图像
plt.savefig('sync_method_comparison.png', dpi=300)
plt.show()