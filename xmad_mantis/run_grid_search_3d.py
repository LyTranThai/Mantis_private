import os
import sys
import time
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# =========================================================================
# Architecture
# =========================================================================

class MetricProbe(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=64, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        feat = self.net(x)
        return F.normalize(feat, p=2, dim=1)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=64, out_features=2, s=16.0, m=0.30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_feat, label=None):
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(input_feat, w)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        if label is None:
            return cosine * self.s, cosine

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output, cosine

class AntiSpoofMetricModel(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, emb_dim=64, s=16.0, m=0.30):
        super().__init__()
        self.probe = MetricProbe(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=emb_dim)
        self.head = ArcMarginProduct(in_features=emb_dim, out_features=2, s=s, m=m)

    def forward(self, x, label=None):
        emb = self.probe(x)
        logits, cosine = self.head(emb, label)
        return logits, cosine, emb

    def predict_score(self, x):
        self.eval()
        with torch.no_grad():
            emb = self.probe(x)
            w = F.normalize(self.head.weight, p=2, dim=1)
            cosine = F.linear(emb, w)
            probs = F.softmax(cosine * self.head.s, dim=1)
            return probs[:, 1].cpu().numpy()

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])

# =========================================================================
# Train & Eval Single (m, s) Run
# =========================================================================

def run_single_experiment(X_tr, y_tr, X_val, y_val, X_cross, y_cross, m_val, s_val, epochs=20, batch_size=256):
    torch.manual_seed(42)
    np.random.seed(42)

    model = AntiSpoofMetricModel(in_dim=X_tr.shape[1], hidden_dim=128, emb_dim=64, s=s_val, m=m_val)
    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            logits, _, _ = model(bx, by)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()
    train_duration = time.time() - t0

    # In-Domain Evaluation
    val_probs = model.predict_score(torch.tensor(X_val, dtype=torch.float32))
    ind_acc = float(accuracy_score(y_val, (val_probs >= 0.5).astype(int))) * 100.0
    ind_auc = float(roc_auc_score(y_val, val_probs)) * 100.0
    ind_eer, ind_thresh = compute_eer(y_val, val_probs)
    ind_eer *= 100.0

    # Cross-Domain Evaluation
    cross_probs = model.predict_score(torch.tensor(X_cross, dtype=torch.float32))
    crd_acc = float(accuracy_score(y_cross, (cross_probs >= 0.5).astype(int))) * 100.0
    crd_auc = float(roc_auc_score(y_cross, cross_probs)) * 100.0
    crd_eer, crd_thresh = compute_eer(y_cross, cross_probs)
    crd_eer *= 100.0

    return {
        "m": m_val,
        "s": s_val,
        "train_time_sec": round(train_duration, 2),
        "in_domain_acc": round(ind_acc, 2),
        "in_domain_eer": round(ind_eer, 2),
        "in_domain_auc": round(ind_auc, 2),
        "cross_domain_acc": round(crd_acc, 2),
        "cross_domain_eer": round(crd_eer, 2),
        "cross_domain_auc": round(crd_auc, 2),
        "thresh_in": round(ind_thresh, 4),
        "thresh_cross": round(crd_thresh, 4)
    }

# =========================================================================
# Main Grid Search Loop
# =========================================================================

def main():
    out_dir = r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results\strategy_1_metric_probe\grid_search_3d"
    os.makedirs(out_dir, exist_ok=True)
    shards_dir = r"D:\Blank\xmad_store\shards"

    print("=" * 70)
    print("3D HYPERPARAMETER GRID SEARCH: MARGIN (m) & SCALE (s)")
    print("Strategy 1: MetricProbe on Pre-Cached MANTIS V2 Shards")
    print("=" * 70)

    # 1. Load Data Once into Memory
    t_start = time.time()
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "shard_*.parquet")))
    print(f"Loading {len(shard_files)} shards from {shards_dir}...")
    dfs = [pd.read_parquet(f) for f in shard_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} samples in {time.time()-t_start:.2f}s.")

    df_train = df[df['original_split'] == 'train']
    df_val = df[df['original_split'] == 'val']
    df_cross = df[df['original_split'] == '']

    # Sample balanced pools
    n_tr = min(20000, len(df_train[df_train['label']==0]), len(df_train[df_train['label']==1]))
    train_sample = pd.concat([
        df_train[df_train['label']==0].sample(n=n_tr, random_state=42),
        df_train[df_train['label']==1].sample(n=n_tr, random_state=42)
    ])
    X_tr = np.vstack(train_sample['embedding'].values)
    y_tr = train_sample['label'].values

    n_val = min(10000, len(df_val[df_val['label']==0]), len(df_val[df_val['label']==1]))
    val_sample = pd.concat([
        df_val[df_val['label']==0].sample(n=n_val, random_state=42),
        df_val[df_val['label']==1].sample(n=n_val, random_state=42)
    ])
    X_val = np.vstack(val_sample['embedding'].values)
    y_val = val_sample['label'].values

    n_cross = min(10000, len(df_cross[df_cross['label']==0]), len(df_cross[df_cross['label']==1]))
    cross_sample = pd.concat([
        df_cross[df_cross['label']==0].sample(n=n_cross, random_state=42),
        df_cross[df_cross['label']==1].sample(n=n_cross, random_state=42)
    ])
    X_cross = np.vstack(cross_sample['embedding'].values)
    y_cross = cross_sample['label'].values

    print(f"Data Prepared: Train={len(y_tr):,}, In-Domain Val={len(y_val):,}, Cross-Domain Test={len(y_cross):,}")

    # Define 3x3 Grid (9 configurations <= 10 limit)
    m_list = [0.15, 0.30, 0.45]
    s_list = [10.0, 16.0, 24.0]

    grid_results = []
    run_idx = 1
    total_runs = len(m_list) * len(s_list)

    print("\n" + "=" * 80)
    print(f"{'Run':<4} | {'m':<6} | {'s':<6} | {'In-Domain ACC':<14} | {'In-Domain EER':<14} | {'Cross ACC':<12} | {'Cross EER':<12} | {'Time':<6}")
    print("-" * 80)

    for m in m_list:
        for s in s_list:
            res = run_single_experiment(X_tr, y_tr, X_val, y_val, X_cross, y_cross, m_val=m, s_val=s, epochs=18)
            grid_results.append(res)
            print(f"{run_idx:02d}/{total_runs:02d} | {res['m']:<6.2f} | {res['s']:<6.1f} | {res['in_domain_acc']:>6.2f}% ({res['in_domain_auc']:>5.2f}%) | {res['in_domain_eer']:>6.2f}% ({res['thresh_in']:>6.3f}) | {res['cross_domain_acc']:>6.2f}% | {res['cross_domain_eer']:>6.2f}% | {res['train_time_sec']:>4.1f}s")
            run_idx += 1

    print("=" * 80)
    total_elapsed = time.time() - t_start
    print(f"Total Grid Search Execution Time: {total_elapsed:.2f} seconds ({total_elapsed/60.0:.2f} minutes).")

    # Save CSV and JSON
    df_res = pd.DataFrame(grid_results)
    csv_path = os.path.join(out_dir, "grid_search_results.csv")
    df_res.to_csv(csv_path, index=False)
    json_path = os.path.join(out_dir, "grid_search_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(grid_results, f, indent=2)

    # =========================================================================
    # 3D Surface & Scatter Visualization
    # =========================================================================
    print("Generating comprehensive 3D visualization surfaces...")

    # Create meshgrid for smooth interpolation
    M, S = np.meshgrid(m_list, s_list, indexing='ij')

    # Metrics matrices
    Z_in_acc = np.zeros_like(M)
    Z_in_eer = np.zeros_like(M)
    Z_crd_acc = np.zeros_like(M)
    Z_crd_eer = np.zeros_like(M)

    for r in grid_results:
        i = m_list.index(r['m'])
        j = s_list.index(r['s'])
        Z_in_acc[i, j] = r['in_domain_acc']
        Z_in_eer[i, j] = r['in_domain_eer']
        Z_crd_acc[i, j] = r['cross_domain_acc']
        Z_crd_eer[i, j] = r['cross_domain_eer']

    # 4-panel 3D Figure
    fig = plt.figure(figsize=(18, 14), dpi=300)

    panels = [
        (1, "In-Domain Accuracy (%)", Z_in_acc, plt.cm.viridis, "ACC (%)", True),
        (2, "In-Domain EER (%) [Lower is Better]", Z_in_eer, plt.cm.magma_r, "EER (%)", False),
        (3, "Cross-Domain Accuracy (%)", Z_crd_acc, plt.cm.plasma, "ACC (%)", True),
        (4, "Cross-Domain EER (%) [Lower is Better]", Z_crd_eer, plt.cm.coolwarm_r, "EER (%)", False)
    ]

    for idx, title, Z, cmap, z_label, is_max_optimal in panels:
        ax = fig.add_subplot(2, 2, idx, projection='3d')

        # 3D Surface
        surf = ax.plot_surface(M, S, Z, cmap=cmap, edgecolor='k', linewidth=0.6, alpha=0.85)

        # Scatter points
        scatter = ax.scatter(M.flatten(), S.flatten(), Z.flatten(), color='black', s=50, depthshade=False)

        # Highlight optimal point
        opt_idx = np.unravel_index(np.argmax(Z) if is_max_optimal else np.argmin(Z), Z.shape)
        opt_m, opt_s, opt_val = M[opt_idx], S[opt_idx], Z[opt_idx]
        ax.scatter([opt_m], [opt_s], [opt_val], color='red', s=120, edgecolors='white', linewidth=2, zorder=10)
        ax.text(opt_m, opt_s, opt_val + (0.3 if is_max_optimal else -0.3),
                f"Opt: m={opt_m:.2f}, s={opt_s:.0f}\n{opt_val:.2f}%",
                color='red', fontsize=9, fontweight='bold')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=14)
        ax.set_xlabel("Margin (m)", fontsize=10, labelpad=8)
        ax.set_ylabel("Scale (s)", fontsize=10, labelpad=8)
        ax.set_zlabel(z_label, fontsize=10, labelpad=8)
        ax.view_init(elev=28, azim=135)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    plt.suptitle("Strategy 1: 3D Response Surfaces of (m, s) on XMAD-Bench\nEvaluating Accuracy and Equal Error Rate (EER)",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    plot_3d_path = os.path.join(out_dir, "3d_surface_grid_search.png")
    plt.savefig(plot_3d_path, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D Response Surface Figure to: {plot_3d_path}")

    # Single Consolidated 3D Surface (Cross-Domain EER vs In-Domain ACC)
    fig2 = plt.figure(figsize=(10, 8), dpi=300)
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(M, S, Z_crd_acc, cmap=plt.cm.turbo, edgecolor='black', linewidth=0.5, alpha=0.9)
    ax2.scatter(M.flatten(), S.flatten(), Z_crd_acc.flatten(), color='black', s=45)
    best_crd_idx = np.unravel_index(np.argmax(Z_crd_acc), Z_crd_acc.shape)
    ax2.scatter([M[best_crd_idx]], [S[best_crd_idx]], [Z_crd_acc[best_crd_idx]], color='red', s=150, edgecolors='yellow', linewidth=2)
    ax2.text(M[best_crd_idx], S[best_crd_idx], Z_crd_acc[best_crd_idx] + 0.2,
             f"Best Cross ACC:\nm={M[best_crd_idx]:.2f}, s={S[best_crd_idx]:.0f}\n{Z_crd_acc[best_crd_idx]:.2f}%",
             color='black', fontsize=10, fontweight='bold')
    ax2.set_title("3D Cross-Domain Accuracy Optimization Surface across (m, s)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Margin (m)", fontsize=11, labelpad=10)
    ax2.set_ylabel("Scale (s)", fontsize=11, labelpad=10)
    ax2.set_zlabel("Cross-Domain ACC (%)", fontsize=11, labelpad=10)
    ax2.view_init(elev=25, azim=-60)
    fig2.colorbar(surf2, shrink=0.55, aspect=12, pad=0.1)
    plot_single_path = os.path.join(out_dir, "3d_cross_domain_acc_surface.png")
    fig2.savefig(plot_single_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Single 3D Cross-Domain ACC Figure to: {plot_single_path}")

    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETED SUCCESSFULLY!")
    print(f"Results stored in: {out_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
