import matplotlib.pyplot as plt
import numpy as np

# Metrics
metrics = ['Classification Accuracy', 'Detection Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline = [0.89, 0.85, 0.87, 0.83, 0.85]
proposed = [0.96, 0.92, 0.94, 0.91, 0.92]

x = np.arange(len(metrics))  # label locations
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(10,6))

# Attractive color palette
baseline_color = '#1f77b4'   # deep blue
proposed_color = '#ff7f0e'   # vibrant orange

bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', color=baseline_color, edgecolor='black')
bars2 = ax.bar(x + width/2, proposed, width, label='Proposed YOLO + CNN', color=proposed_color, edgecolor='black')

# Add text labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height*100:.0f}%',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0,3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Labels and titles
ax.set_ylabel('Score')
ax.set_title('Performance Metrics Comparison: Baseline vs Proposed System', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=25, ha='right')
ax.set_ylim(0, 1.1)
ax.legend(frameon=True, fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add subtle background color for axes
ax.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.savefig('evaluation_metrics_summary_colored.png', dpi=300)
plt.show()