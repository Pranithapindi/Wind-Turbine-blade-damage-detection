import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Define the steps in sequence
steps = [
    "Blade Image Input\n(Drone / Camera)",
    "Image Preprocessing\nResize, Noise Removal, Normalization",
    "CNN Feature Extraction\nConv Layer, Pooling Layer, Dropout",
    "YOLO Damage Detection\nObject Detection & Localization",
    "Blade Classification\nHealthy vs Faulty"
]

# Positions
x = [0, 1, 2, 3, 4]
y = [0, 0, 0, 0, 0]

fig, ax = plt.subplots(figsize=(16,4))
ax.axis('off')

# Draw boxes with arrows
for i, step in enumerate(steps):
    # Draw rectangle
    bbox = FancyBboxPatch((x[i]-0.45, y[i]-0.25), 0.9, 0.5, boxstyle="round,pad=0.05",
                          fc="#4A90E2", ec="black", lw=1.5)
    ax.add_patch(bbox)
    ax.text(x[i], y[i], step, ha="center", va="center", fontsize=10, color="white", fontweight='bold')
    
    # Draw arrow
    if i < len(steps)-1:
        ax.arrow(x[i]+0.45, y[i], 0.1, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')

# Title
ax.set_title("Wind Turbine Blade Damage Detection Flow", fontsize=14, fontweight='bold')

plt.xlim(-0.5, 4.5)
plt.ylim(-1, 1)
plt.tight_layout()
plt.show()