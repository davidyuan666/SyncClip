import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Data from the table
metrics = ['Segment F1-Score', 'BLEU Score', 'WER\n(lower is better)', 
           'Sync Error (ms)\n(lower is better)', 'Processing Time (s)\n(lower is better)', 
           'User Satisfaction (1-5)']

# Systems
systems = ['SyncClip', 'Rule-Based', 'SVM-Based', 'Adobe Auto-Editing', 'VideoLLM']

# Data values from the table
# For metrics where lower is better, we'll invert the values for visualization
our_system =    [0.90, 0.85, 1-(0.08), 1-(120/250), 1-(13.3/18.5), 4.6/5]
rule_based =    [0.75, 0.00, 0.00,     1-(250/250), 1-(18.5/18.5), 3.6/5]  # No BLEU/WER
svm_based =     [0.82, 0.00, 0.00,     1-(200/250), 1-(16.2/18.5), 3.9/5]  # No BLEU/WER
adobe =         [0.85, 0.78, 1-(0.12), 1-(150/250), 1-(14.8/18.5), 4.2/5]
videollm =      [0.88, 0.83, 1-(0.10), 1-(130/250), 1-(15.0/18.5), 4.4/5]

data = [our_system, rule_based, svm_based, adobe, videollm]

# Number of metrics
N = len(metrics)

# What will be the angle of each axis in the plot
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Set up the figure - increased size
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

# Set the first axis at the top
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label
# Increase the label distance from the plot
plt.xticks(angles[:-1], metrics, fontsize=11)
text_objs = ax.get_xticklabels()
for i, txt in enumerate(text_objs):
    ang = angles[i]
    # Adjust the label distance from the origin
    if 0 <= ang < np.pi:
        txt.set_horizontalalignment('left')
    else:
        txt.set_horizontalalignment('right')
    txt.set_fontsize(10)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
           color="grey", size=9)
plt.ylim(0, 1.05)  # Slightly increase ylim to make room for labels

# Plot data
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (d, color) in enumerate(zip(data, colors)):
    values = d
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=systems[i])
    ax.fill(angles, values, color=color, alpha=0.1)

# Add legend with better positioning
plt.legend(loc='lower right', bbox_to_anchor=(1.15, 0.95), fontsize=10)

# Add a title with more space
plt.title('Video Editing Systems Performance Comparison', 
          fontsize=15, y=1.05, fontweight='bold')

# # Add subtitle explaining metric normalization
# plt.figtext(0.5, 0.02, 
#            "Note: For WER, Sync Error, and Processing Time, values have been normalized so higher is better.",
#            ha='center', fontsize=9, style='italic')

# Adjusted annotation positions for better readability
highlight_points = [
    (angles[0], our_system[0], "F1=0.90", (-30, 20)),
    (angles[1], our_system[1], "BLEU=0.85", (10, 20)),
    (angles[2], our_system[2], "WER=0.08", (15, 15)),
    (angles[3], our_system[3], "120ms", (-45, 15)),
    (angles[4], our_system[4], "13.3s", (-40, 5)),
    (angles[5], our_system[5], "4.6/5", (15, 15))
]

for angle, value, text, offset in highlight_points:
    ax.annotate(text, 
                xy=(angle, value), 
                xytext=(offset[0], offset[1]),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color='#1f77b4')

plt.tight_layout()

# Save the figure with more padding
plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
plt.show()