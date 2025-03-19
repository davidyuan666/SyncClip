import matplotlib.pyplot as plt
import numpy as np

# 数据
genres = ['Action', 'Documentary', 'Vlog', 'News', 'Sports', 'Music\nVideo', 
          'Short\nFilm', 'Educational', 'Comedy', 'Advertisements']
temporal_errors = [110, 130, 120, 125, 115, 140, 135, 130, 120, 110]
semantic_correspondence = [89.5, 86.2, 88.4, 87.8, 88.9, 84.5, 85.3, 86.7, 87.5, 89.2]

# 创建图像和坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制柱状图（语义对应关系）
bars = ax1.bar(genres, semantic_correspondence, alpha=0.7, color='skyblue', width=0.6)
ax1.set_ylim([80, 92])
ax1.set_ylabel('Semantic Correspondence (%)', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# 创建第二个Y轴
ax2 = ax1.twinx()

# 绘制折线图（时间同步误差）
line = ax2.plot(genres, temporal_errors, 'o-', color='red', linewidth=2, markersize=8)
ax2.set_ylim([100, 150])
ax2.set_ylabel('Temporal Synchronization Error (ms)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.invert_yaxis()  # 倒置y轴，使较低的错误率显示在顶部

# 标题和标签
plt.title('Audio-Visual Synchronization Performance by Genre', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
fig.tight_layout()

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(bars, ['Semantic Correspondence'], loc='upper left')
ax2.legend(line, ['Temporal Sync Error'], loc='upper right')

# 保存图像
plt.savefig('sync_error_by_genre.png', dpi=300)
plt.show()