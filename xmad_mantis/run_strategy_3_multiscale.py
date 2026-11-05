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
    log_file = os.path.join(log_dir, "strategy_3_multiscale.log")
    
    logger = logging.getLogger("Strategy3_MultiScale")
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
# Multi-Scale Metric Probe Architecture (768 -> 256 -> 64 -> ArcMargin)
# =========================================================================

class MultiScaleMetricProbe(nn.Module):
    """
    Non-linear projection head mapping multi-scale (768-d) representations
    onto a normalized 64-d unit hypersphere manifold.
    """
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=64, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        feat = self.net(x)
        return F.normalize(feat, p=2, dim=1)

class ArcMarginProduct(nn.Module):
    """
    Large Margin Cosine / ArcFace classification head for binary anti-spoofing.
    Enforces angular margin between Bonafide (0) and Spoof (1).
    """
    def __init__(self, in_features=64, out_features=2, s=24.0, m=0.15):
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
        output = output * self.s

        return output, cosine

class MultiScaleAntiSpoofingSystem(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=64, s=24.0, m=0.15, dropout=0.15):
        super().__init__()
        self.probe = MultiScaleMetricProbe(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
        self.arc_head = ArcMarginProduct(in_features=out_dim, out_features=2, s=s, m=m)

    def forward(self, x, label=None):
        feat = self.probe(x)
        logits, cosine = self.arc_head(feat, label)
        return logits, feat, cosine

    @torch.no_grad()
    def predict_score(self, x):
        self.eval()
        feat = self.probe(x)
        logits, cosine = self.arc_head(feat, label=None)
        probs = F.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy(), cosine[:, 1].cpu().numpy()

# =========================================================================
# Training Routine
# =========================================================================

def train_multiscale_probe(X_train, y_train, X_val=None, y_val=None,
                           epochs=25, batch_size=256, lr=1e-3,
                           weight_decay=1e-4, margin=0.15, scale=24.0,
                           device='cuda' if torch.cuda.is_available() else 'cpu',
                           logger=None):
    model = MultiScaleAntiSpoofingSystem(in_dim=X_train.shape[1], hidden_dim=256, out_dim=64, s=scale, m=margin).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_eer = 1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(bx, by)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
        scheduler.step()

        avg_loss = total_loss / len(dataset)

        if X_val is not None and y_val is not None and (epoch % 5 == 0 or epoch == epochs):
            probs, _ = model.predict_score(torch.tensor(X_val, dtype=torch.float32).to(device))
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

def evaluate_model_on_test(model, X_test, y_test, device='cuda' if torch.cuda.is_available() else 'cpu'):
    probs, scores = model.predict_score(torch.tensor(X_test, dtype=torch.float32).to(device))
    acc = float(accuracy_score(y_test, (probs >= 0.5).astype(int)))
    auc = float(roc_auc_score(y_test, probs))
    eer, thresh = compute_eer(y_test, probs)
    return acc, auc, eer, thresh, probs

# =========================================================================
# Main Benchmark Pipeline
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Strategy 3: Multi-Scale Temporal Chunking Benchmark")
    parser.add_argument("--store-dir", type=str, default=r"D:\Blank\xmad_store", help="Base store dir containing shards_multiscale")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results\strategy_3_multiscale")
    parser.add_argument("--log-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\logs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--scale", type=float, default=24.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger, log_file = setup_logger(args.log_dir)

    logger.info("=" * 75)
    logger.info("STRATEGY 3: MULTI-SCALE TEMPORAL CHUNKING (512 + 1024 + 2048 -> 768-d)")
    logger.info("MANTIS V2 × XMAD-BENCHMARK (RESOLVING VALLE-X NEURAL CODEC BOTTLENECK)")
    logger.info("=" * 75)

    shards_pattern = os.path.join(args.store_dir, "shards_multiscale", "shard_*.parquet")
    shard_files = sorted(glob.glob(shards_pattern))
    logger.info(f"Scanning for multi-scale shards: {shards_pattern}")
    
    if not shard_files:
        logger.error(f"No multi-scale shards found at {shards_pattern}! Run extract_and_cache_multiscale.py first.")
        return

    logger.info(f"Found {len(shard_files)} multi-scale shards. Loading into memory...")
    t0 = time.time()
    dfs = [pd.read_parquet(f) for f in shard_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} samples in {time.time()-t0:.2f}s.")
    logger.info(f"Class distribution: Bonafide (0) = {len(df[df['label']==0]):,}, Spoof (1) = {len(df[df['label']==1]):,}")

    master_summary = {"strategy": "strategy_3_multiscale_temporal_chunking", "hyperparameters": vars(args)}

    # PART 1: Standard In-Domain & Cross-Domain
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: OFFICIAL STANDARD TRAIN/TEST BENCHMARK EVALUATION")
    logger.info("=" * 70)

    df_train = df[df['original_split'] == 'train']
    df_val = df[df['original_split'] == 'val']
    df_cross = df[df['original_split'] == '']

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

    logger.info(f"Training MultiScale MetricProbe on {len(y_train_p1):,} samples (Dim={X_train_p1.shape[1]})...")
    model_p1 = train_multiscale_probe(
        X_train_p1, y_train_p1, X_val=X_val_p1, y_val=y_val_p1,
        epochs=args.epochs, batch_size=args.batch_size, margin=args.margin, scale=args.scale, logger=logger
    )

    acc_val, auc_val, eer_val, th_val, probs_val = evaluate_model_on_test(model_p1, X_val_p1, y_val_p1)
    logger.info(f">> [In-Domain Val]     ACC = {acc_val*100:6.2f}%, AUC = {auc_val*100:6.2f}%, EER = {eer_val*100:6.2f}% (Thresh: {th_val:.4f})")

    if len(df_cross) > 0:
        n_cross_pts = min(19000, len(df_cross[df_cross['label']==0]), len(df_cross[df_cross['label']==1]))
        cross_sample = pd.concat([
            df_cross[df_cross['label']==0].sample(n=n_cross_pts, random_state=42),
            df_cross[df_cross['label']==1].sample(n=n_cross_pts, random_state=42)
        ])
        X_cross_p1 = np.vstack(cross_sample['embedding'].values)
        y_cross_p1 = cross_sample['label'].values
        acc_cross, auc_cross, eer_cross, th_cross, probs_cross = evaluate_model_on_test(model_p1, X_cross_p1, y_cross_p1)
        logger.info(f">> [Cross-Domain Test] ACC = {acc_cross*100:6.2f}%, AUC = {auc_cross*100:6.2f}%, EER = {eer_cross*100:6.2f}% (Thresh: {th_cross:.4f})")
    else:
        acc_cross, auc_cross, eer_cross, th_cross = 0.0, 0.0, 0.0, 0.0

    master_summary["official_benchmark"] = {
        "in_domain_val": {"accuracy": acc_val * 100, "auc": auc_val * 100, "eer": eer_val * 100, "threshold": th_val},
        "cross_domain_test": {"accuracy": acc_cross * 100, "auc": auc_cross * 100, "eer": eer_cross * 100, "threshold": th_cross}
    }

    # PART 3: Leave-One-Attack-Out (LOAO) Generalization
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

        loao_model = train_multiscale_probe(X_tr, y_tr, epochs=15, batch_size=args.batch_size, margin=args.margin, scale=args.scale)
        acc_o, auc_o, eer_o, _, _ = evaluate_model_on_test(loao_model, X_te, y_te)

        logger.info(f"[LOAO Held-Out: {held_out_attack:<18}] MultiScale ACC = {acc_o*100:6.2f}%, AUC = {auc_o*100:6.2f}%, EER = {eer_o*100:6.2f}%")
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

    # Save summary
    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(master_summary, f, indent=2)
    logger.info(f"\nSaved consolidated metrics summary JSON to: {summary_path}")

    # Save checkpoint
    torch.save(model_p1.state_dict(), os.path.join(checkpoints_dir, "best_multiscale_metric_probe.pt"))
    logger.info("Saved best MultiScale model checkpoint.")

if __name__ == "__main__":
    main()
