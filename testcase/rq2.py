import matplotlib.pyplot as plt
import numpy as np

# Data for visual understanding models (BLEU scores)
visual_models = ['CLIP ViT-B/16', 'Multi-modal\nCLIP ViT-B/16', 'CLIP ViT-L/14\n(Ours)']
bleu_scores = [0.79, 0.81, 0.85]

# Data for audio transcription models (WER scores)
audio_models = ['OpenAI\nWhisper-Small', 'Google\nSpeech-to-Text', 'OpenAI\nWhisper-Medium', 'Whisper-Large-v3\n(Ours)']
wer_scores = [0.14, 0.12, 0.11, 0.08]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Visual Understanding (BLEU scores) - higher is better
bars1 = ax1.bar(visual_models, bleu_scores, color=['lightblue', 'cornflowerblue', 'darkblue'])
ax1.set_title('Visual Understanding Performance', fontsize=14, fontweight='bold')
ax1.set_ylabel('BLEU Score (higher is better)', fontsize=12)
ax1.set_ylim([0.75, 0.90])
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{bleu_scores[i]}', ha='center', fontsize=11)

# Highlight our approach
ax1.bar(visual_models[2], bleu_scores[2], color='darkblue', edgecolor='red', linewidth=2)

# Plot Audio Transcription (WER scores) - lower is better
bars2 = ax2.bar(audio_models, wer_scores, color=['lightblue', 'lightblue', 'cornflowerblue', 'darkblue'])
ax2.set_title('Audio Transcription Performance', fontsize=14, fontweight='bold')
ax2.set_ylabel('Word Error Rate (lower is better)', fontsize=12)
ax2.set_ylim([0, 0.18])
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{wer_scores[i]}', ha='center', fontsize=11)

# Highlight our approach
ax2.bar(audio_models[3], wer_scores[3], color='darkblue', edgecolor='red', linewidth=2)

plt.suptitle('Content Understanding Performance Across Different Models', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save the figure
plt.savefig('content_understanding_comparison.png', dpi=300, bbox_inches='tight')
plt.show()