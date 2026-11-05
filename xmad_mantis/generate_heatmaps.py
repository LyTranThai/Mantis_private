import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    csv_path = r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results\strategy_1_metric_probe\grid_search_3d\grid_search_results.csv"
    out_dir = r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results\strategy_1_metric_probe\grid_search_3d"
    df = pd.read_csv(csv_path)

    m_vals = sorted(df['m'].unique())
    s_vals = sorted(df['s'].unique())

    # Build 2D matrices
    def make_matrix(col_name):
        pivot = df.pivot(index='m', columns='s', values=col_name)
        return pivot.loc[sorted(m_vals, reverse=True), s_vals]

    mat_in_acc = make_matrix('in_domain_acc')
    mat_in_eer = make_matrix('in_domain_eer')
    mat_crd_acc = make_matrix('cross_domain_acc')
    mat_crd_eer = make_matrix('cross_domain_eer')

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    heatmap_configs = [
        (axes[0, 0], mat_in_acc, "In-Domain Accuracy (%)", "YlGnBu", "%", True),
        (axes[0, 1], mat_in_eer, "In-Domain EER (%) [Lower is Better]", "YlOrRd_r", "%", False),
        (axes[1, 0], mat_crd_acc, "Cross-Domain Accuracy (%)", "viridis", "%", True),
        (axes[1, 1], mat_crd_eer, "Cross-Domain EER (%) [Lower is Better]", "coolwarm_r", "%", False)
    ]

    for ax, mat, title, cmap, unit, is_max_better in heatmap_configs:
        im = ax.imshow(mat.values, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(unit, fontsize=10, fontweight='bold')

        # Axis ticks
        ax.set_xticks(range(len(s_vals)))
        ax.set_xticklabels([f"{s:.0f}" for s in s_vals], fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{m:.2f}" for m in mat.index], fontsize=11, fontweight='bold')
        
        ax.set_xlabel("Scale (s)", fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel("Margin (m)", fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)

        # Highlight best cell
        best_val = mat.values.max() if is_max_better else mat.values.min()
        
        # Annotate numbers
        for i in range(len(mat.index)):
            for j in range(len(s_vals)):
                val = mat.values[i, j]
                is_best = (val == best_val)
                text_color = "white" if im.norm(val) > 0.65 or im.norm(val) < 0.25 else "black"
                
                label_str = f"{val:.2f}%"
                if is_best:
                    label_str += "\n(Best)"
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor='gold', lw=3.5))

                ax.text(j, i, label_str, ha="center", va="center", color=text_color,
                        fontsize=11, fontweight='bold' if is_best else 'normal')

    plt.suptitle("Strategy 1: Metric Probe Hyperparameter Heatmaps (Margin $m$ vs Scale $s$)\nXMAD-Benchmark Evaluation (In-Domain vs Cross-Domain)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_img = os.path.join(out_dir, "heatmap_grid_search.png")
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {out_img}")

if __name__ == "__main__":
    main()
