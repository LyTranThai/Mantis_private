import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)
from datasets import load_dataset, Audio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_audio import parse_wav_bytes
from mantis.architecture import MantisV2
from mantis.trainer import MantisTrainer

def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "multi_scale_run.log")
    
    logger = logging.getLogger("XMAD_Mantis_MultiScale")
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

def evaluate_subset(X, y, scale_name, output_dir, logger):
    logger.info(f"\n{'='*20} EVALUATING SCALE: {scale_name} ({len(X)} samples) {'='*20}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    logger.info(f"Split: Train={len(y_train)} (Real:{np.sum(y_train==0)}, Fake:{np.sum(y_train==1)}), Test={len(y_test)} (Real:{np.sum(y_test==0)}, Fake:{np.sum(y_test==1)})")
    
    # 1. Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    acc_lr = float(accuracy_score(y_test, y_pred_lr))
    auc_lr = float(roc_auc_score(y_test, y_prob_lr))
    eer_lr, thresh_lr = compute_eer(y_test, y_prob_lr)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    acc_rf = float(accuracy_score(y_test, y_pred_rf))
    auc_rf = float(roc_auc_score(y_test, y_prob_rf))
    eer_rf, thresh_rf = compute_eer(y_test, y_prob_rf)
    
    logger.info(f"Logistic Regression: ACC={acc_lr*100:.2f}%, AUC={auc_lr*100:.2f}%, EER={eer_lr*100:.2f}% (Thresh: {thresh_lr:.4f})")
    logger.info(f"Random Forest:       ACC={acc_rf*100:.2f}%, AUC={auc_rf*100:.2f}%, EER={eer_rf*100:.2f}% (Thresh: {thresh_rf:.4f})")
    
    best_name = "Logistic Regression" if auc_lr >= auc_rf else "Random Forest"
    best_y_pred = y_pred_lr if best_name == "Logistic Regression" else y_pred_rf
    best_y_prob = y_prob_lr if best_name == "Logistic Regression" else y_prob_rf
    
    report = classification_report(y_test, best_y_pred, target_names=["Bonafide", "Spoof"], output_dict=True)
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({scale_name} - {best_name})")
    plt.colorbar()
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_file = os.path.join(output_dir, f"cm_{scale_name}.png")
    plt.savefig(cm_file, dpi=120)
    plt.close()
    
    return {
        "scale_name": scale_name,
        "total_samples": len(X),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "logistic_regression": {
            "accuracy": acc_lr,
            "roc_auc": auc_lr,
            "eer": eer_lr,
            "optimal_threshold": thresh_lr
        },
        "random_forest": {
            "accuracy": acc_rf,
            "roc_auc": auc_rf,
            "eer": eer_rf,
            "optimal_threshold": thresh_rf
        },
        "best_model": best_name,
        "best_report": report,
        "confusion_matrix_path": cm_file,
        "roc_data": {
            "fpr_lr": roc_curve(y_test, y_prob_lr)[0].tolist(),
            "tpr_lr": roc_curve(y_test, y_prob_lr)[1].tolist(),
            "fpr_rf": roc_curve(y_test, y_prob_rf)[0].tolist(),
            "tpr_rf": roc_curve(y_test, y_prob_rf)[1].tolist()
        }
    }

def main():
    base_dir = r"C:\Users\Admin\Desktop\SC203\xmad_mantis"
    output_dir = os.path.join(base_dir, "results")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    logger, log_file = setup_logger(log_dir)

    logger.info("=" * 60)
    logger.info("STARTING MULTI-SCALE XMAD-BENCH EVALUATION (250, 400, 1000)")
    logger.info("=" * 60)

    # 1. Load Mantis v2
    device = "cpu"
    logger.info(f"Loading MantisV2 on {device}...")
    network = MantisV2(device=device).from_pretrained("paris-noah/MantisV2")
    trainer = MantisTrainer(device=device, network=network)
    logger.info("MantisV2 ready.")

    # 2. Targeted Shards
    # Shard 54 & 55: Fake English, Shard 81 & 82: Real English
    shards = [
        "data/test-00054-of-00200.parquet",
        "data/test-00055-of-00200.parquet",
        "data/test-00081-of-00200.parquet",
        "data/test-00082-of-00200.parquet",
    ]
    logger.info(f"Streaming from shards: {shards}")
    
    ds = load_dataset(
        "SpeechAntiSpoofingBenchmarks/XMAD",
        data_files={"test": shards},
        split="test",
        streaming=True
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    MAX_PER_CLASS = 1000  # Will acquire 1000 real and 1000 fake
    TARGET_TOTAL = MAX_PER_CLASS * 2

    raw_waveforms = []
    labels = []
    counts = {0: 0, 1: 0}

    logger.info(f"Streaming up to {MAX_PER_CLASS} samples per class ({TARGET_TOTAL} total)...")
    t0 = time.time()
    
    for sample in ds:
        label = int(sample["label"])
        if counts[label] >= MAX_PER_CLASS:
            if counts[0] >= MAX_PER_CLASS and counts[1] >= MAX_PER_CLASS:
                break
            continue

        audio_dict = sample.get("audio")
        if not audio_dict or "bytes" not in audio_dict or audio_dict["bytes"] is None:
            continue

        try:
            wav_arr = parse_wav_bytes(audio_dict["bytes"])
        except Exception:
            continue

        if len(wav_arr) < 512:
            continue

        wav_tensor = torch.tensor(wav_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(wav_tensor, size=512, mode="linear", align_corners=False)
        resized_arr = resized_tensor.squeeze(0).numpy()

        raw_waveforms.append(resized_arr)
        labels.append(label)
        counts[label] += 1

        total = counts[0] + counts[1]
        if total % 100 == 0:
            logger.info(f"Streamed {total}/{TARGET_TOTAL} (Real: {counts[0]}, Fake: {counts[1]}). Elapsed: {time.time()-t0:.1f}s")

    logger.info(f"Finished streaming {len(raw_waveforms)} samples in {time.time()-t0:.2f}s (Real: {counts[0]}, Fake: {counts[1]}).")

    X_all = np.array(raw_waveforms, dtype=np.float32)
    y_all = np.array(labels, dtype=np.int32)

    # 3. Batch Feature Extraction with MantisV2
    logger.info(f"Extracting features with MantisV2 for all {len(X_all)} samples...")
    t_feat = time.time()
    batch_size = 64
    embeddings_list = []
    for b_idx in range(0, len(X_all), batch_size):
        batch_X = X_all[b_idx:b_idx + batch_size]
        batch_emb = trainer.transform(batch_X)
        embeddings_list.append(batch_emb)
        if (b_idx + batch_size) % 256 == 0 or (b_idx + batch_size) >= len(X_all):
            logger.info(f"Transformed {min(b_idx + batch_size, len(X_all))}/{len(X_all)} samples...")

    Z_all = np.concatenate(embeddings_list, axis=0)
    logger.info(f"Feature extraction finished in {time.time() - t_feat:.2f}s. Z shape: {Z_all.shape}")

    # Separate real and fake indices for balanced sub-sampling
    idx_real = np.where(y_all == 0)[0]
    idx_fake = np.where(y_all == 1)[0]
    
    # Shuffle consistently
    np.random.seed(42)
    np.random.shuffle(idx_real)
    np.random.shuffle(idx_fake)

    # 4. Multi-Scale Evaluation
    # Evaluate at 250 per class (500 total), 400 per class (800 total), 1000 per class (2000 total)
    scales = [
        ("scale_250_per_class", 250),
        ("scale_400_per_class", 400),
        ("scale_1000_per_class", min(1000, len(idx_real), len(idx_fake)))
    ]

    all_scale_results = {}

    for scale_id, n_per_class in scales:
        sub_real = idx_real[:n_per_class]
        sub_fake = idx_fake[:n_per_class]
        sub_idx = np.concatenate([sub_real, sub_fake])
        
        X_sub = Z_all[sub_idx]
        y_sub = y_all[sub_idx]
        
        res = evaluate_subset(X_sub, y_sub, scale_id, output_dir, logger)
        all_scale_results[scale_id] = res

    # 5. Combined Multi-Scale Comparison Chart
    plt.figure(figsize=(9, 6))
    for scale_id, res in all_scale_results.items():
        fpr_lr = res["roc_data"]["fpr_lr"]
        tpr_lr = res["roc_data"]["tpr_lr"]
        auc = res["logistic_regression"]["roc_auc"] * 100.0
        eer = res["logistic_regression"]["eer"] * 100.0
        total_s = res["total_samples"]
        plt.plot(fpr_lr, tpr_lr, lw=2, label=f"{scale_id} ({total_s} samples) - AUC: {auc:.1f}%, EER: {eer:.1f}%")

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Mantis v2 Performance Across Sample Scales (XMAD-Bench)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    comp_plot = os.path.join(output_dir, "multi_scale_roc_comparison.png")
    plt.savefig(comp_plot, dpi=150)
    plt.close()
    logger.info(f"Saved multi-scale comparison plot: {comp_plot}")

    # 6. Save Complete Multi-Scale JSON Report
    summary_file = os.path.join(output_dir, "multi_scale_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_scale_results, f, indent=2)
    logger.info(f"Saved complete multi-scale results JSON: {summary_file}")
    
    logger.info("=" * 60)
    logger.info("ALL MULTI-SCALE RUNS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
