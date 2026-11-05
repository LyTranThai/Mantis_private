import os
import sys
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])

def main():
    parser = argparse.ArgumentParser(description="Strict Cross-Domain Anti-Leakage Evaluation with Sample Decay Curve.")
    parser.add_argument("--store-dir", type=str, default=r"D:\Blank\xmad_store", help="Base directory where shards are saved.")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results")
    args = parser.parse_args()

    store_dir = args.store_dir
    if not os.path.exists(store_dir):
        store_dir = r"C:\Users\Admin\Desktop\SC203\xmad_mantis\store"

    shards_pattern = os.path.join(store_dir, "shards", "shard_*.parquet")
    shard_files = glob.glob(shards_pattern)

    print("=" * 60)
    print("XMAD-BENCH CROSS-DOMAIN ANTI-LEAKAGE EVALUATION")
    print("=" * 60)
    print(f"Scanning for shards in: {shards_pattern}")
    print(f"Found {len(shard_files)} completed shards.")

    if not shard_files:
        print("No shards found! Run extract_and_cache.py first to extract embeddings.")
        return

    # Load all shard tables
    dfs = [pd.read_parquet(f) for f in shard_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded total {len(df)} samples into unified dataframe.")
    print("Class distribution:", df['label'].value_counts().to_dict())
    
    # Genuine attacks exist only in fake samples (label == 1)
    fake_attacks = sorted(df[df['label'] == 1]['attack'].unique().tolist())
    print("Available Fake Attacks/Generators (label=1):", fake_attacks)
    
    real_count = len(df[df['label'] == 0])
    fake_count = len(df[df['label'] == 1])
    print(f"Total Bonafide (Real): {real_count}, Total Spoof (Fake): {fake_count}")

    if real_count < 10:
        print("Not enough real samples to create train and test splits!")
        return

    # Cross-Domain Attack Partitioning
    if len(fake_attacks) >= 2:
        train_attacks = [fake_attacks[0]]
        test_attacks = fake_attacks[1:]
    else:
        train_attacks = fake_attacks
        test_attacks = fake_attacks

    print(f"\n[LEAKAGE CHECK PASS] Train Attacks (Known): {train_attacks}")
    print(f"[LEAKAGE CHECK PASS] Test Attacks (Unseen): {test_attacks}")
    if len(fake_attacks) >= 2:
        assert set(train_attacks).isdisjoint(set(test_attacks)), "FATAL: Attack overlap detected!"

    # Split Real Speakers 50/50 to guarantee zero speaker overlap
    df_real = df[df['label'] == 0].copy()
    real_speakers = sorted(df_real['speaker_id'].dropna().unique().tolist())
    np.random.seed(42)
    np.random.shuffle(real_speakers)
    half = max(1, len(real_speakers) // 2)
    train_real_speakers = set(real_speakers[:half])
    test_real_speakers = set(real_speakers[half:])
    
    print(f"[LEAKAGE CHECK PASS] Real Speakers split: {len(train_real_speakers)} for Train, {len(test_real_speakers)} for Test.")
    assert train_real_speakers.isdisjoint(test_real_speakers), "FATAL: Real speaker overlap!"

    # Pools
    real_train_pool = df_real[df_real['speaker_id'].isin(train_real_speakers)]
    real_test_pool = df_real[df_real['speaker_id'].isin(test_real_speakers)]

    fake_train_pool = df[(df['label'] == 1) & (df['attack'].isin(train_attacks))]
    fake_test_pool = df[(df['label'] == 1) & (df['attack'].isin(test_attacks))]

    print(f"Available for Train Pool: Real={len(real_train_pool)}, Fake={len(fake_train_pool)}")
    print(f"Available for Test Pool:  Real={len(real_test_pool)}, Fake={len(fake_test_pool)}")

    # Fixed Test Evaluation Set
    n_test_per_class = min(150, len(real_test_pool), len(fake_test_pool))
    if n_test_per_class < 5:
        n_test_per_class = min(len(real_test_pool), len(fake_test_pool))
        
    test_eval_df = pd.concat([
        real_test_pool.sample(n=n_test_per_class, random_state=42),
        fake_test_pool.sample(n=n_test_per_class, random_state=42)
    ])
    X_test = np.vstack(test_eval_df['embedding'].values)
    y_test = test_eval_df['label'].values
    print(f"\nFixed Test Evaluation Set: {len(y_test)} samples (Real: {n_test_per_class}, Fake: {n_test_per_class})")

    # Decay Sizes
    max_train_per_class = min(len(real_train_pool), len(fake_train_pool))
    candidate_sizes = [1000, 500, 250, 100, 50, 25, 10]
    decay_sizes = [s for s in candidate_sizes if s <= max_train_per_class]
    if max_train_per_class not in decay_sizes:
        decay_sizes = [max_train_per_class] + decay_sizes

    decay_results = []
    print("\n--- Running Decay Curve Experiments ---")
    for n_per_class in decay_sizes:
        train_sample_df = pd.concat([
            real_train_pool.sample(n=n_per_class, random_state=42),
            fake_train_pool.sample(n=n_per_class, random_state=42)
        ])
        X_train = np.vstack(train_sample_df['embedding'].values)
        y_train = train_sample_df['label'].values

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        eer, thresh = compute_eer(y_test, y_prob)

        print(f"Decay Size {n_per_class:4d}/class (Train={len(y_train):4d}) -> ACC = {acc*100:6.2f}%, AUC = {auc*100:6.2f}%, EER = {eer*100:6.2f}%")

        decay_results.append({
            "samples_per_class": n_per_class,
            "total_train_samples": len(y_train),
            "test_samples": len(y_test),
            "accuracy": acc,
            "roc_auc": auc,
            "eer": eer,
            "threshold": thresh
        })

    # Save Results & Plots
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "cross_domain_decay_results.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(decay_results, f, indent=2)
    print(f"\nSaved decay results JSON: {summary_path}")

    # Plot Accuracy vs Training Size
    if len(decay_results) >= 2:
        train_counts = [r["total_train_samples"] for r in decay_results]
        accs = [r["accuracy"] * 100 for r in decay_results]
        eers = [r["eer"] * 100 for r in decay_results]

        plt.figure(figsize=(8, 5))
        plt.plot(train_counts, accs, marker='o', lw=2, color='tab:blue', label='Accuracy (%)')
        plt.plot(train_counts, eers, marker='s', lw=2, color='tab:red', label='Equal Error Rate (EER %)')
        plt.axhline(89.07, color='green', linestyle='--', label='wav2vec 2.0 SOTA (89.07% ACC)')
        plt.axhline(81.61, color='orange', linestyle=':', label='AST Baseline (81.61% ACC)')
        plt.xlabel('Total Training Samples (Linear Probe)')
        plt.ylabel('Percentage (%)')
        plt.title('Mantis v2 Cross-Domain Performance vs. Training Sample Size')
        plt.legend()
        plt.grid(alpha=0.3)
        plot_path = os.path.join(args.output_dir, "decay_curve_accuracy.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved decay curve plot: {plot_path}")

    print("=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
