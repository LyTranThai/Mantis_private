import os
import sys
import time
import glob
import json
import math
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "strategy_1_metric_probe.log")
    
    logger = logging.getLogger("Strategy1_MetricProbe")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger, log_file

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])

# =========================================================================
# Metric Learning Architecture: Non-Linear Embedding + ArcMargin Head
# =========================================================================

class MetricProbe(nn.Module):
    """
    Non-linear projection head that maps MANTIS 256-d representations
    onto a normalized unit hypersphere manifold.
    """
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
    """
    Large Margin Cosine / ArcFace classification head for binary anti-spoofing.
    Enforces angular margin between Bonafide (0) and Spoof (1).
    """
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
        # Normalize weights & feature input
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(input_feat, w)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        if label is None:
            # Inference mode: direct scaled cosine logits
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
            # Probability of spoof (class 1)
            return probs[:, 1].cpu().numpy(), (cosine[:, 1] - cosine[:, 0]).cpu().numpy()

# =========================================================================
# Training Routine
# =========================================================================

def train_metric_probe(X_train, y_train, X_val=None, y_val=None, epochs=25, batch_size=256, lr=2e-3, margin=0.30, scale=16.0, logger=None):
    device = torch.device("cpu")
    model = AntiSpoofMetricModel(in_dim=X_train.shape[1], hidden_dim=128, emb_dim=64, s=scale, m=margin).to(device)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_eer = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            logits, _, _ = model(bx, by)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
        scheduler.step()

        avg_loss = total_loss / len(dataset)
        
        # Periodic validation if available
        if X_val is not None and y_val is not None and (epoch % 5 == 0 or epoch == epochs):
            probs, scores = model.predict_score(torch.tensor(X_val, dtype=torch.float32))
            val_eer, _ = compute_eer(y_val, probs)
            val_acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
            if val_eer < best_eer:
                best_eer = val_eer
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if logger:
                logger.info(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | Val ACC: {val_acc*100:5.2f}% | Val EER: {val_eer*100:5.2f}%")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def evaluate_model_on_test(model, X_test, y_test):
    probs, scores = model.predict_score(torch.tensor(X_test, dtype=torch.float32))
    acc = float(accuracy_score(y_test, (probs >= 0.5).astype(int)))
    auc = float(roc_auc_score(y_test, probs))
    eer, thresh = compute_eer(y_test, probs)
    return acc, auc, eer, thresh, probs

# =========================================================================
# Main Execution Flow
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Strategy 1: Non-Linear Metric Probing & SupCon Evaluation")
    parser.add_argument("--store-dir", type=str, default=r"D:\Blank\xmad_store")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results\strategy_1_metric_probe")
    parser.add_argument("--log-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\logs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.30)
    parser.add_argument("--scale", type=float, default=16.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger, log_file = setup_logger(args.log_dir)

    logger.info("=" * 70)
    logger.info("EXECUTION: STRATEGY 1 — NON-LINEAR METRIC PROBING & ARCMARGIN")
    logger.info("MANTIS V2 ON XMAD-BENCHMARK (ZERO DOWNLOAD, PRE-CACHED EMBEDDINGS)")
    logger.info("=" * 70)

    # 1. Load Parquet Shards
    shards_pattern = os.path.join(args.store_dir, "shards", "shard_*.parquet")
    shard_files = sorted(glob.glob(shards_pattern))
    logger.info(f"Scanning for shards in: {shards_pattern}")
    logger.info(f"Found {len(shard_files)} shards. Loading into memory...")
    t0 = time.time()
    dfs = [pd.read_parquet(f) for f in shard_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} samples in {time.time()-t0:.2f}s.")
    logger.info(f"Class distribution: Bonafide (0) = {len(df[df['label']==0]):,}, Spoof (1) = {len(df[df['label']==1]):,}")

    master_summary = {"strategy": "strategy_1_metric_probing", "hyperparameters": vars(args)}

    # =========================================================================
    # PART 1: Official Standard Train/Test Benchmark
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: OFFICIAL STANDARD TRAIN/TEST BENCHMARK EVALUATION")
    logger.info("=" * 70)

    df_train = df[df['original_split'] == 'train']
    df_val = df[df['original_split'] == 'val']
    df_cross = df[df['original_split'] == '']  # Cross-domain corpora

    n_train_pts = min(20000, len(df_train[df_train['label']==0]), len(df_train[df_train['label']==1]))
    train_sample = pd.concat([
        df_train[df_train['label']==0].sample(n=n_train_pts, random_state=42),
        df_train[df_train['label']==1].sample(n=n_train_pts, random_state=42)
    ])
    X_train_p1 = np.vstack(train_sample['embedding'].values)
    y_train_p1 = train_sample['label'].values

    n_val_pts = min(10000, len(df_val[df_val['label']==0]), len(df_val[df_val['label']==1]))
    val_sample = pd.concat([
        df_val[df_val['label']==0].sample(n=n_val_pts, random_state=42),
        df_val[df_val['label']==1].sample(n=n_val_pts, random_state=42)
    ])
    X_val_p1 = np.vstack(val_sample['embedding'].values)
    y_val_p1 = val_sample['label'].values

    n_cross_pts = min(10000, len(df_cross[df_cross['label']==0]), len(df_cross[df_cross['label']==1]))
    cross_sample = pd.concat([
        df_cross[df_cross['label']==0].sample(n=n_cross_pts, random_state=42),
        df_cross[df_cross['label']==1].sample(n=n_cross_pts, random_state=42)
    ])
    X_cross_p1 = np.vstack(cross_sample['embedding'].values)
    y_cross_p1 = cross_sample['label'].values

    logger.info(f"Training Metric Probe on {len(y_train_p1):,} balanced samples ({args.epochs} epochs)...")
    t_tr0 = time.time()
    metric_model = train_metric_probe(
        X_train_p1, y_train_p1, X_val=X_val_p1, y_val=y_val_p1,
        epochs=args.epochs, batch_size=args.batch_size, margin=args.margin, scale=args.scale, logger=logger
    )
    logger.info(f"Training completed in {time.time()-t_tr0:.2f}s.")

    # Save Checkpoint
    ckpt_path = os.path.join(checkpoints_dir, "best_metric_probe.pt")
    torch.save(metric_model.state_dict(), ckpt_path)
    logger.info(f"Saved best metric probe checkpoint to: {ckpt_path}")

    # Evaluate In-Domain Val
    acc_ind, auc_ind, eer_ind, thresh_ind, probs_ind = evaluate_model_on_test(metric_model, X_val_p1, y_val_p1)
    logger.info(f"[In-Domain Val]     ACC = {acc_ind*100:6.2f}%, AUC = {auc_ind*100:6.2f}%, EER = {eer_ind*100:6.2f}% (Thresh={thresh_ind:.4f})")

    # Evaluate Cross-Domain Test
    acc_crd, auc_crd, eer_crd, thresh_crd, probs_crd = evaluate_model_on_test(metric_model, X_cross_p1, y_cross_p1)
    logger.info(f"[Cross-Domain Test] ACC = {acc_crd*100:6.2f}%, AUC = {auc_crd*100:6.2f}%, EER = {eer_crd*100:6.2f}% (Thresh={thresh_crd:.4f})")

    # Comparison Table
    table_p1 = [
        {"Model": "Mantis v2 + MetricProbe (Strategy 1)", "In-Domain ACC": acc_ind*100, "In-Domain EER": eer_ind*100, "Cross-Domain ACC": acc_crd*100, "Cross-Domain EER": eer_crd*100},
        {"Model": "Mantis v2 Baseline (Linear Probe)", "In-Domain ACC": 89.27, "In-Domain EER": 10.87, "Cross-Domain ACC": 91.42, "Cross-Domain EER": 8.67},
        {"Model": "wav2vec 2.0 (Leader)", "In-Domain ACC": 99.97, "In-Domain EER": 0.01, "Cross-Domain ACC": 89.07, "Cross-Domain EER": 11.82},
        {"Model": "ResNet-18", "In-Domain ACC": 99.98, "In-Domain EER": 0.01, "Cross-Domain ACC": 87.87, "Cross-Domain EER": 12.52},
        {"Model": "AST", "In-Domain ACC": 98.53, "In-Domain EER": 1.35, "Cross-Domain ACC": 81.61, "Cross-Domain EER": 16.91},
        {"Model": "SepTr", "In-Domain ACC": 91.70, "In-Domain EER": 22.05, "Cross-Domain ACC": 72.67, "Cross-Domain EER": 45.69}
    ]
    df_p1 = pd.DataFrame(table_p1)
    df_p1.to_csv(os.path.join(args.output_dir, "benchmark_comparison_table.csv"), index=False)
    master_summary["part1_official_benchmark"] = table_p1

    # =========================================================================
    # PART 2: Per-Language Cross-Domain Disjoint Breakdown
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: PER-LANGUAGE CROSS-DOMAIN DISJOINT BREAKDOWN (7 LANGUAGES)")
    logger.info("=" * 70)

    languages = ["en", "ru", "ar", "ro", "de", "zh", "es"]
    per_lang_res = []

    for lang in languages:
        df_lang = df[df['language'] == lang]
        real_lang = df_lang[df_lang['label'] == 0]
        fake_lang = df_lang[df_lang['label'] == 1]

        real_spks = sorted(real_lang['speaker_id'].dropna().unique().tolist())
        np.random.seed(42)
        np.random.shuffle(real_spks)
        half_spk = max(1, len(real_spks) // 2)
        train_spks = set(real_spks[:half_spk])
        test_spks = set(real_spks[half_spk:])

        lang_attacks = sorted(fake_lang['attack'].unique().tolist())
        tr_att = [lang_attacks[0]] if len(lang_attacks) >= 2 else lang_attacks
        te_att = lang_attacks[1:] if len(lang_attacks) >= 2 else lang_attacks

        real_tr = real_lang[real_lang['speaker_id'].isin(train_spks)]
        real_te = real_lang[real_lang['speaker_id'].isin(test_spks)]
        fake_tr = fake_lang[fake_lang['attack'].isin(tr_att)]
        fake_te = fake_lang[fake_lang['attack'].isin(te_att)]

        n_tr = min(2000, len(real_tr), len(fake_tr))
        n_te = min(1000, len(real_te), len(fake_te))
        if n_tr < 10 or n_te < 10:
            continue

        tr_sub = pd.concat([real_tr.sample(n=n_tr, random_state=42), fake_tr.sample(n=n_tr, random_state=42)])
        te_sub = pd.concat([real_te.sample(n=n_te, random_state=42), fake_te.sample(n=n_te, random_state=42)])

        X_tr = np.vstack(tr_sub['embedding'].values)
        y_tr = tr_sub['label'].values
        X_te = np.vstack(te_sub['embedding'].values)
        y_te = te_sub['label'].values

        lang_model = train_metric_probe(X_tr, y_tr, epochs=15, batch_size=128, margin=args.margin, scale=args.scale)
        acc_l, auc_l, eer_l, _, _ = evaluate_model_on_test(lang_model, X_te, y_te)

        logger.info(f"[{lang.upper():2s}] MetricProbe ACC = {acc_l*100:6.2f}%, AUC = {auc_l*100:6.2f}%, EER = {eer_l*100:6.2f}% (Train={len(y_tr):,}, Test={len(y_te):,})")
        per_lang_res.append({
            "language": lang,
            "train_samples": len(y_tr),
            "test_samples": len(y_te),
            "accuracy": acc_l * 100,
            "roc_auc": auc_l * 100,
            "eer": eer_l * 100
        })

    df_p2 = pd.DataFrame(per_lang_res)
    df_p2.to_csv(os.path.join(args.output_dir, "per_language_results.csv"), index=False)
    master_summary["part2_per_language"] = per_lang_res

    # =========================================================================
    # PART 3: Leave-One-Attack-Out (LOAO) Generalization
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: LEAVE-ONE-ATTACK-OUT (LOAO) ZERO-SHOT GENERALIZATION")
    logger.info("=" * 70)

    top_attacks = ["xtts_v2", "vits+knn", "vits+openvoice", "fairseq_knnvc", "fairseq_freevc", "vits_freevc", "bark+freevc", "vallex"]
    loao_results = []

    for held_out_attack in top_attacks:
        train_pool_fake = df[(df['label'] == 1) & (df['attack'] != held_out_attack)]
        test_pool_fake = df[(df['label'] == 1) & (df['attack'] == held_out_attack)]

        if len(test_pool_fake) < 10:
            continue

        real_spks = sorted(df[df['label'] == 0]['speaker_id'].dropna().unique().tolist())
        np.random.seed(42)
        np.random.shuffle(real_spks)
        half_spk = max(1, len(real_spks) // 2)
        spk_tr = set(real_spks[:half_spk])
        spk_te = set(real_spks[half_spk:])

        real_tr_pool = df[(df['label'] == 0) & (df['speaker_id'].isin(spk_tr))]
        real_te_pool = df[(df['label'] == 0) & (df['speaker_id'].isin(spk_te))]

        n_tr = min(5000, len(real_tr_pool), len(train_pool_fake))
        n_te = min(1000, len(real_te_pool), len(test_pool_fake))

        tr_sub = pd.concat([real_tr_pool.sample(n=n_tr, random_state=42), train_pool_fake.sample(n=n_tr, random_state=42)])
        te_sub = pd.concat([real_te_pool.sample(n=n_te, random_state=42), test_pool_fake.sample(n=n_te, random_state=42)])

        X_tr = np.vstack(tr_sub['embedding'].values)
        y_tr = tr_sub['label'].values
        X_te = np.vstack(te_sub['embedding'].values)
        y_te = te_sub['label'].values

        loao_model = train_metric_probe(X_tr, y_tr, epochs=15, batch_size=256, margin=args.margin, scale=args.scale)
        acc_o, auc_o, eer_o, _, _ = evaluate_model_on_test(loao_model, X_te, y_te)

        logger.info(f"[LOAO Held-Out: {held_out_attack:<18}] MetricProbe Zero-Shot ACC = {acc_o*100:6.2f}%, AUC = {auc_o*100:6.2f}%, EER = {eer_o*100:6.2f}%")
        loao_results.append({
            "held_out_attack": held_out_attack,
            "train_samples": len(y_tr),
            "test_samples": len(y_te),
            "accuracy": acc_o * 100,
            "roc_auc": auc_o * 100,
            "eer": eer_o * 100
        })

    df_p3 = pd.DataFrame(loao_results)
    df_p3.to_csv(os.path.join(args.output_dir, "loao_attack_breakdown.csv"), index=False)
    master_summary["part3_loao_breakdown"] = loao_results

    # Save Master Summary JSON
    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(master_summary, f, indent=2)
    logger.info(f"\nSaved consolidated metrics summary JSON to: {summary_path}")

    # =========================================================================
    # VISUALIZATION SUITE (PNGs)
    # =========================================================================
    logger.info("Generating publication-grade visualization plots...")

    # Plot 1: ROC Curves (In-Domain vs Cross-Domain)
    plt.figure(figsize=(8, 6), dpi=300)
    fpr_ind, tpr_ind, _ = roc_curve(y_val_p1, probs_ind, pos_label=1)
    fpr_crd, tpr_crd, _ = roc_curve(y_cross_p1, probs_crd, pos_label=1)

    plt.plot(fpr_ind, tpr_ind, color='#2b5c8f', lw=2.5, label=f'In-Domain Val (AUC = {auc_ind*100:.2f}%, EER = {eer_ind*100:.2f}%)')
    plt.plot(fpr_crd, tpr_crd, color='#d95f02', lw=2.5, label=f'Cross-Domain Test (AUC = {auc_crd*100:.2f}%, EER = {eer_crd*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Chance Baseline')
    plt.scatter([eer_ind], [1 - eer_ind], color='#2b5c8f', s=60, zorder=5)
    plt.scatter([eer_crd], [1 - eer_crd], color='#d95f02', s=60, zorder=5)

    plt.title("Strategy 1: MetricProbe ROC Curves on XMAD-Bench", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "roc_curves.png"))
    plt.close()

    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    cm_ind = confusion_matrix(y_val_p1, (probs_ind >= 0.5).astype(int), normalize='true')
    cm_crd = confusion_matrix(y_cross_p1, (probs_crd >= 0.5).astype(int), normalize='true')

    im1 = axes[0].imshow(cm_ind, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title(f"In-Domain Confusion Matrix (ACC={acc_ind*100:.2f}%)", fontsize=11, fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Bonafide", "Spoof"])
    axes[0].set_yticklabels(["Bonafide", "Spoof"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm_ind[i, j]*100:.2f}%", ha="center", va="center", color="white" if cm_ind[i, j] > 0.5 else "black", fontweight='bold')

    im2 = axes[1].imshow(cm_crd, interpolation='nearest', cmap=plt.cm.Oranges)
    axes[1].set_title(f"Cross-Domain Confusion Matrix (ACC={acc_crd*100:.2f}%)", fontsize=11, fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Bonafide", "Spoof"])
    axes[1].set_yticklabels(["Bonafide", "Spoof"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm_crd[i, j]*100:.2f}%", ha="center", va="center", color="white" if cm_crd[i, j] > 0.5 else "black", fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    plt.close()

    # Plot 3: LOAO Attack Breakdown
    plt.figure(figsize=(9, 5), dpi=300)
    attacks = [r["held_out_attack"] for r in loao_results]
    accs = [r["accuracy"] for r in loao_results]
    colors = ['#2ca02c' if a >= 80 else ('#ff7f0e' if a >= 60 else '#d62728') for a in accs]

    bars = plt.barh(attacks, accs, color=colors, edgecolor='black', alpha=0.85)
    plt.axvline(x=50, color='gray', linestyle='--', lw=1.2, label='Random Guessing (50%)')
    plt.axvline(x=90, color='blue', linestyle=':', lw=1.2, label='High Reliability (90%)')
    plt.xlim(0, 100)
    plt.title("Strategy 1: Zero-Shot Generalization by Held-Out Attack (LOAO)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Zero-Shot Accuracy (%)", fontsize=11)
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1.2, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%", va='center', fontsize=9, fontweight='bold')
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loao_attack_breakdown.png"))
    plt.close()

    logger.info("=" * 70)
    logger.info("STRATEGY 1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"All artifacts saved in: {args.output_dir}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
