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

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_audio import parse_wav_bytes
from mantis.architecture import MantisV2
from mantis.trainer import MantisTrainer

def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "benchmark_run.log")
    
    logger = logging.getLogger("XMAD_Mantis")
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
    """Compute Equal Error Rate (EER) and the optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find threshold where FPR is closest to FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mantis v2 on XMAD-Bench (Common Voice English).")
    parser.add_argument("--samples-per-class", type=int, default=150, help="Number of samples per class (real & fake).")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for Mantis input (multiple of 32).")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results")
    parser.add_argument("--log-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger, log_file = setup_logger(args.log_dir)
    
    logger.info("=" * 60)
    logger.info("STARTING XMAD-BENCH EVALUATION: MANTIS V2 FOUNDATION MODEL")
    logger.info(f"Target Samples per Class: {args.samples_per_class} (Total: {args.samples_per_class * 2})")
    logger.info(f"Target Sequence Length: {args.seq_len} timestamps")
    logger.info(f"Results Directory: {args.output_dir}")
    logger.info(f"Log File: {log_file}")
    logger.info("=" * 60)

    # 1. Load Mantis v2
    start_load = time.time()
    device = "cpu"
    logger.info(f"Loading MantisV2 model checkpoint ('paris-noah/MantisV2') on {device}...")
    network = MantisV2(device=device).from_pretrained("paris-noah/MantisV2")
    trainer = MantisTrainer(device=device, network=network)
    logger.info(f"MantisV2 loaded successfully in {time.time() - start_load:.2f}s.")

    # 2. Targeted English Common Voice shards
    # Shard 54: Fake English, Shard 81: Real English
    shards = [
        "data/test-00054-of-00200.parquet",  # English Spoof
        "data/test-00081-of-00200.parquet",  # English Bonafide
    ]
    logger.info(f"Configuring streaming on targeted shards: {shards}")
    
    ds = load_dataset(
        "SpeechAntiSpoofingBenchmarks/XMAD",
        data_files={"test": shards},
        split="test",
        streaming=True
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    # 3. Stream & Extract Audio Waveforms
    logger.info("Streaming and pre-processing audio samples...")
    raw_waveforms = []
    labels = []
    utterance_metadata = []
    counts = {0: 0, 1: 0}
    
    t0 = time.time()
    for sample in ds:
        label = int(sample["label"])
        if counts[label] >= args.samples_per_class:
            if counts[0] >= args.samples_per_class and counts[1] >= args.samples_per_class:
                break
            continue

        audio_dict = sample.get("audio")
        if not audio_dict or "bytes" not in audio_dict or audio_dict["bytes"] is None:
            continue

        try:
            wav_arr = parse_wav_bytes(audio_dict["bytes"])
        except Exception as e:
            continue

        if len(wav_arr) < 512:
            continue

        # Resize waveform to target sequence length (1, 1, seq_len)
        wav_tensor = torch.tensor(wav_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(wav_tensor, size=args.seq_len, mode="linear", align_corners=False)
        resized_arr = resized_tensor.squeeze(0).numpy() # Shape: (1, seq_len)

        raw_waveforms.append(resized_arr)
        labels.append(label)
        counts[label] += 1
        
        meta = json.loads(sample.get("notes", "{}"))
        utterance_metadata.append({
            "utterance_id": meta.get("utterance_id", ""),
            "language": meta.get("language", ""),
            "corpus": meta.get("corpus", ""),
            "attack": meta.get("attack", "bonafide" if label == 0 else "unknown"),
            "label": label
        })

        total = counts[0] + counts[1]
        if total % 50 == 0:
            logger.info(f"Streamed {total}/{args.samples_per_class * 2} samples (Real: {counts[0]}, Fake: {counts[1]})...")

    logger.info(f"Stream complete in {time.time() - t0:.2f}s. Total acquired: Real={counts[0]}, Fake={counts[1]}.")

    X_all = np.array(raw_waveforms, dtype=np.float32) # Shape: (N, 1, 512)
    y_all = np.array(labels, dtype=np.int32)

    # 4. Extract Latent Embeddings with MantisV2
    logger.info(f"Transforming {len(X_all)} waveforms into deep representations with MantisV2...")
    t_feat = time.time()
    batch_size = 32
    embeddings_list = []
    for b_idx in range(0, len(X_all), batch_size):
        batch_X = X_all[b_idx:b_idx + batch_size]
        batch_emb = trainer.transform(batch_X)
        embeddings_list.append(batch_emb)
        if (b_idx + batch_size) % 96 == 0 or (b_idx + batch_size) >= len(X_all):
            logger.info(f"Extracted features for {min(b_idx + batch_size, len(X_all))}/{len(X_all)} samples...")

    Z_all = np.concatenate(embeddings_list, axis=0) # Shape: (N, 256)
    logger.info(f"Feature extraction complete in {time.time() - t_feat:.2f}s. Embedding matrix shape: {Z_all.shape}")

    # 5. Train / Test Split (Stratified 70% Train, 30% Test)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        Z_all, y_all, utterance_metadata, test_size=0.30, random_state=42, stratify=y_all
    )
    logger.info(f"Data Split: Train={len(y_train)} (Real: {np.sum(y_train == 0)}, Fake: {np.sum(y_train == 1)}), Test={len(y_test)} (Real: {np.sum(y_test == 0)}, Fake: {np.sum(y_test == 1)})")

    # 6. Fit Classifiers & Evaluate
    results = {}
    
    # Model A: Logistic Regression (Linear Probe)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    eer_lr, thresh_lr = compute_eer(y_test, y_prob_lr)
    
    # Model B: Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    eer_rf, thresh_rf = compute_eer(y_test, y_prob_rf)

    logger.info("-" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info(f"Logistic Regression: ACC = {acc_lr*100:.2f}%, AUC = {auc_lr*100:.2f}%, EER = {eer_lr*100:.2f}% (Threshold: {thresh_lr:.4f})")
    logger.info(f"Random Forest:       ACC = {acc_rf*100:.2f}%, AUC = {auc_rf*100:.2f}%, EER = {eer_rf*100:.2f}% (Threshold: {thresh_rf:.4f})")
    logger.info("-" * 50)
    
    # Detailed classification report for best head
    best_name = "Logistic Regression" if auc_lr >= auc_rf else "Random Forest"
    best_y_pred = y_pred_lr if best_name == "Logistic Regression" else y_pred_rf
    best_y_prob = y_prob_lr if best_name == "Logistic Regression" else y_prob_rf
    best_acc = max(acc_lr, acc_rf)
    best_auc = max(auc_lr, auc_rf)
    best_eer = min(eer_lr, eer_rf)

    clf_report = classification_report(y_test, best_y_pred, target_names=["Bonafide (Real)", "Spoof (Fake)"], output_dict=True)
    logger.info(f"\nDetailed Classification Report ({best_name}):\n" + classification_report(y_test, best_y_pred, target_names=["Bonafide (Real)", "Spoof (Fake)"]))

    # 7. Generate Visualizations
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({best_name})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Real", "Fake"])
    plt.yticks(tick_marks, ["Real", "Fake"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot: {cm_path}")

    # ROC Curve Plot
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"Mantis v2 + LogReg (AUC = {auc_lr*100:.2f}%, EER = {eer_lr*100:.2f}%)", lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Mantis v2 + RF (AUC = {auc_rf*100:.2f}%, EER = {eer_rf*100:.2f}%)", lw=2, linestyle='--')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('XMAD-Bench Audio Deepfake Detection - ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curve plot: {roc_path}")

    # 8. Save Metrics Summary JSON
    summary = {
        "benchmark": "XMAD-Bench (Common Voice English)",
        "model": "paris-noah/MantisV2",
        "num_samples_evaluated": len(X_all),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "results": {
            "logistic_regression": {
                "accuracy": float(acc_lr),
                "roc_auc": float(auc_lr),
                "eer": float(eer_lr),
                "optimal_threshold": float(thresh_lr)
            },
            "random_forest": {
                "accuracy": float(acc_rf),
                "roc_auc": float(auc_rf),
                "eer": float(eer_rf),
                "optimal_threshold": float(thresh_rf)
            },
            "best_head": best_name,
            "best_classification_report": clf_report
        },
        "artifacts": {
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "log_file": log_file
        }
    }
    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved metrics summary JSON: {summary_path}")

    logger.info("=" * 60)
    logger.info("BENCHMARK EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
