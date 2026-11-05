import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

csv_path = r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results\full_benchmark_suite\exp1_official_benchmark_table.csv"
df = pd.read_csv(csv_path)

models = df['Model'].tolist()
in_acc = df['In-Domain ACC'].tolist()
cross_acc = df['Cross-Domain ACC'].tolist()
cross_eer = df['Cross-Domain EER'].tolist()

x = np.arange(len(models))
width = 0.28

plt.figure(figsize=(10, 5))
plt.bar(x - width, in_acc, width, label='In-Domain ACC (%)', color='royalblue')
plt.bar(x, cross_acc, width, label='Cross-Domain ACC (%)', color='forestgreen')
plt.bar(x + width, cross_eer, width, label='Cross-Domain EER (%, lower is better)', color='crimson')

plt.xticks(x, models, rotation=15, ha='right')
plt.ylabel('Percentage (%)')
plt.title('Official XMAD-Bench Replication: Mantis v2 vs. Published Baselines')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plot_out = r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results\full_benchmark_suite\exp1_official_benchmark_plot.png"
plt.savefig(plot_out, dpi=150)
plt.close()
print("Saved Experiment 1 plot successfully.")
