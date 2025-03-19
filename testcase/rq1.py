import matplotlib.pyplot as plt
import numpy as np

# Figure 1(a): Genre-specific results (Table 1 data)
genres = ['Action', 'Documentary', 'Vlog', 'News', 'Sports', 'Music\nVideo', 
          'Short\nFilm', 'Educational', 'Comedy', 'Advertisements']
precision = [0.94, 0.90, 0.92, 0.91, 0.93, 0.89, 0.90, 0.91, 0.92, 0.93]
recall = [0.91, 0.88, 0.88, 0.87, 0.90, 0.86, 0.87, 0.88, 0.89, 0.90]
f1_score = [0.92, 0.89, 0.90, 0.89, 0.91, 0.87, 0.88, 0.89, 0.90, 0.91]

# Create figure for genre comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Set width of bars
barWidth = 0.25
positions1 = np.arange(len(genres))
positions2 = [x + barWidth for x in positions1]
positions3 = [x + barWidth for x in positions2]

# Create bars
precision_bars = ax.bar(positions1, precision, width=barWidth, label='Precision', 
                        color='#4472C4', edgecolor='black', linewidth=0.5)
recall_bars = ax.bar(positions2, recall, width=barWidth, label='Recall', 
                     color='#ED7D31', edgecolor='black', linewidth=0.5)
f1_bars = ax.bar(positions3, f1_score, width=barWidth, label='F1-Score', 
                 color='#70AD47', edgecolor='black', linewidth=0.5)

# Add text annotations
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

add_labels(precision_bars)
add_labels(recall_bars)
add_labels(f1_bars)

# Customize chart
ax.set_ylim(0.80, 1.00)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('(a) SyncClip Performance Across Different Video Genres', fontsize=14, fontweight='bold')
ax.set_xticks([p + barWidth for p in positions1])
ax.set_xticklabels(genres, rotation=45, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('syncclip_genre_performance.png', dpi=300)
plt.close()  # Close instead of show to continue with next figures

# Figure 1(b): Baseline comparison (Table 2 data)
methods = ['Rule-Based\nApproach', 'SVM-Based\nModel', 'BERT-based\nModel', 'SyncClip\n(Ours)']
baseline_precision = [0.82, 0.85, 0.85, 0.92]
baseline_recall = [0.78, 0.81, 0.81, 0.89]
baseline_f1 = [0.80, 0.83, 0.83, 0.90]

# Create figure for baseline comparison
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Set positions and bar width
x = np.arange(len(methods))
width = 0.25

# Create bars
rects1 = ax2.bar(x - width, baseline_precision, width, label='Precision', 
                color='#4472C4', edgecolor='black', linewidth=0.5)
rects2 = ax2.bar(x, baseline_recall, width, label='Recall', 
                color='#ED7D31', edgecolor='black', linewidth=0.5)
rects3 = ax2.bar(x + width, baseline_f1, width, label='F1-Score', 
                color='#70AD47', edgecolor='black', linewidth=0.5)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Customize chart
ax2.set_ylim(0.75, 0.95)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('(b) SyncClip vs. Baseline Methods', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(loc='lower right')

# Add performance improvement percentage for SyncClip
syncclip_improvement = ((0.90 - 0.83) / 0.83) * 100  # F1-score improvement over best baseline
ax2.annotate(f'+{syncclip_improvement:.1f}% improvement',
            xy=(3, 0.90),
            xytext=(3, 0.92),
            fontsize=10, fontweight='bold', color='green',
            arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8))

plt.tight_layout()
plt.savefig('syncclip_baseline_comparison.png', dpi=300)
plt.close()  # Close instead of show to continue with next figure

# Figure 1(c): Audio-Visual Synchronization Performance by Genre
temporal_errors = [110, 130, 120, 125, 115, 140, 135, 130, 120, 110]
semantic_correspondence = [89.5, 86.2, 88.4, 87.8, 88.9, 84.5, 85.3, 86.7, 87.5, 89.2]

# Create figure and axes for the combined figure 1 with subplots
fig3, ax1 = plt.subplots(figsize=(12, 6))

# Plot line chart for semantic correspondence
line1 = ax1.plot(genres, semantic_correspondence, 'o-', color='blue', linewidth=2, markersize=8, label='Semantic Correspondence')
ax1.set_ylim([80, 92])
ax1.set_ylabel('Semantic Correspondence (%)', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create second Y axis
ax2 = ax1.twinx()

# Plot line chart for temporal synchronization error
line2 = ax2.plot(genres, temporal_errors, 's-', color='red', linewidth=2, markersize=8, label='Temporal Sync Error')
ax2.set_ylim([100, 150])
ax2.set_ylabel('Temporal Synchronization Error (ms)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.invert_yaxis()  # Invert y-axis so lower error shows at top

# Customize chart
plt.title('(c) Temporal Synchronization Error vs. Semantic Correspondence by Genre', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
fig3.tight_layout()

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.savefig('sync_performance_by_genre.png', dpi=300)

# Create a combined figure with all three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))

# Redraw the first plot - Genre comparison
barWidth = 0.25
positions1 = np.arange(len(genres))
positions2 = [x + barWidth for x in positions1]
positions3 = [x + barWidth for x in positions2]

ax1.bar(positions1, precision, width=barWidth, label='Precision', color='#4472C4', edgecolor='black', linewidth=0.5)
ax1.bar(positions2, recall, width=barWidth, label='Recall', color='#ED7D31', edgecolor='black', linewidth=0.5)
ax1.bar(positions3, f1_score, width=barWidth, label='F1-Score', color='#70AD47', edgecolor='black', linewidth=0.5)
ax1.set_ylim(0.80, 1.00)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('(a) SyncClip Performance Across Different Video Genres', fontsize=14, fontweight='bold')
ax1.set_xticks([p + barWidth for p in positions1])
ax1.set_xticklabels(genres, rotation=45, ha='right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(loc='lower left')

# Redraw the second plot - Baseline comparison
x = np.arange(len(methods))
width = 0.25
ax2.bar(x - width, baseline_precision, width, label='Precision', color='#4472C4', edgecolor='black', linewidth=0.5)
ax2.bar(x, baseline_recall, width, label='Recall', color='#ED7D31', edgecolor='black', linewidth=0.5)
ax2.bar(x + width, baseline_f1, width, label='F1-Score', color='#70AD47', edgecolor='black', linewidth=0.5)
ax2.set_ylim(0.75, 0.95)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('(b) SyncClip vs. Baseline Methods', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(loc='lower right')

# Redraw the third plot - Temporal sync vs semantic correspondence
ax3_1 = ax3
ax3_1.plot(genres, semantic_correspondence, 'o-', color='blue', linewidth=2, markersize=8, label='Semantic Correspondence')
ax3_1.set_ylim([80, 92])
ax3_1.set_ylabel('Semantic Correspondence (%)', fontsize=12, color='blue')
ax3_1.tick_params(axis='y', labelcolor='blue')

ax3_2 = ax3.twinx()
ax3_2.plot(genres, temporal_errors, 's-', color='red', linewidth=2, markersize=8, label='Temporal Sync Error')
ax3_2.set_ylim([100, 150])
ax3_2.set_ylabel('Temporal Synchronization Error (ms)', fontsize=12, color='red')
ax3_2.tick_params(axis='y', labelcolor='red')
ax3_2.invert_yaxis()  # Invert y-axis so lower error shows at top

ax3.set_title('(c) Temporal Synchronization Error vs. Semantic Correspondence by Genre', fontsize=14, fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.set_xticks(range(len(genres)))
ax3.set_xticklabels(genres, rotation=45, ha='right')

# Add legend for the third plot
lines1, labels1 = ax3_1.get_legend_handles_labels()
lines2, labels2 = ax3_2.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('rq1_results_combined.png', dpi=300)
plt.show()