import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Resolve data path relative to this script's location
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "GestureMidAirD1"

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

unique_labels = np.unique(y_test)
num_classes = len(unique_labels)
cmap = plt.get_cmap("turbo", num_classes)

# 26 subplots in a 5x6 or 6x5 grid
cols = 6
rows = math.ceil(num_classes / cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, 12), sharex=True, sharey=True)
axes = axes.flatten()

for idx, label in enumerate(unique_labels):
    ax = axes[idx]
    color = cmap(idx)
    sample_indices = np.where(y_test == label)[0]

    for sample_idx in sample_indices:
        ax.plot(X_test[sample_idx, 0, :], color=color, alpha=0.7, linewidth=1.2)

    ax.set_title(f"Class {label}", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

# Hide any unused subplot axes
for i in range(num_classes, len(axes)):
    fig.delaxes(axes[i])

fig.suptitle("GestureMidAirD1 - All Classes Visualized Separately", fontsize=15)
plt.tight_layout()
plt.show()
