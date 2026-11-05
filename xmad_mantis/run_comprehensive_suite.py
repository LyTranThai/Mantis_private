import os
import sys
import time
import glob
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report

def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "full_benchmark_suite.log")
    
    logger = logging.getLogger("XMAD_Suite")
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

def evaluate_probe(X_train, y_train, X_test, y_test, max_iter=1000):
    clf = LogisticRegression(C=1.0, max_iter=max_iter, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    eer, thresh = compute_eer(y_test, y_prob)
    return acc, auc, eer, thresh, clf

def main():
    parser = argparse.ArgumentParser(description="Comprehensive 5-Part Benchmark Evaluation Suite for Mantis v2 on XMAD-Bench.")
    parser.add_argument("--store-dir", type=str, default=r"D:\Blank\xmad_store", help="Base directory where shards are saved.")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\results\full_benchmark_suite")
    parser.add_argument("--log-dir", type=str, default=r"C:\Users\Admin\Desktop\SC203\xmad_mantis\logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger, log_file = setup_logger(args.log_dir)

    logger.info("=" * 70)
    logger.info("STARTING COMPREHENSIVE 5-PART BENCHMARK EVALUATION SUITE")
    logger.info("MANTIS V2 ON XMAD-BENCH (368,085 SAMPLES, 7 LANGUAGES)")
    logger.info("=" * 70)

    # 1. Load All 200 Shards into Memory
    shards_pattern = os.path.join(args.store_dir, "shards", "shard_*.parquet")
    shard_files = sorted(glob.glob(shards_pattern))
    logger.info(f"Loading {len(shard_files)} shards from: {shards_pattern}")
    t0 = time.time()
    
    dfs = [pd.read_parquet(f) for f in shard_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Successfully loaded all {len(df):,} samples in {time.time()-t0:.2f}s.")
    logger.info(f"Class Distribution: Bonafide (0) = {len(df[df['label']==0]):,}, Spoof (1) = {len(df[df['label']==1]):,}")

    master_summary = {}

    # =========================================================================
    # EXPERIMENT 1: Official Standard Train/Test Benchmark (Paper Replication)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: OFFICIAL STANDARD TRAIN/TEST BENCHMARK (PAPER REPLICATION)")
    logger.info("=" * 70)

    df_train = df[df['original_split'] == 'train']
    df_val = df[df['original_split'] == 'val']
    df_cross = df[df['original_split'] == '']  # External cross-domain test corpora

    logger.info(f"Official Train Split:       {len(df_train):,} samples")
    logger.info(f"Official In-Domain Val:     {len(df_val):,} samples")
    logger.info(f"Official Cross-Domain Test: {len(df_cross):,} samples")

    # Sample manageable balanced pools for linear probe fitting
    n_train_pts = min(20000, len(df_train[df_train['label']==0]), len(df_train[df_train['label']==1]))
    train_sample = pd.concat([
        df_train[df_train['label']==0].sample(n=n_train_pts, random_state=42),
        df_train[df_train['label']==1].sample(n=n_train_pts, random_state=42)
    ])
    X_train_exp1 = np.vstack(train_sample['embedding'].values)
    y_train_exp1 = train_sample['label'].values
    logger.info(f"Fitting Linear Probe on {len(y_train_exp1):,} balanced training samples...")

    # In-Domain Evaluation (Validation split)
    n_val_pts = min(10000, len(df_val[df_val['label']==0]), len(df_val[df_val['label']==1]))
    val_sample = pd.concat([
        df_val[df_val['label']==0].sample(n=n_val_pts, random_state=42),
        df_val[df_val['label']==1].sample(n=n_val_pts, random_state=42)
    ])
    X_val = np.vstack(val_sample['embedding'].values)
    y_val = val_sample['label'].values

    acc_ind, auc_ind, eer_ind, thresh_ind, clf_exp1 = evaluate_probe(X_train_exp1, y_train_exp1, X_val, y_val)
    logger.info(f"[In-Domain Val]     ACC = {acc_ind*100:6.2f}%, AUC = {auc_ind*100:6.2f}%, EER = {eer_ind*100:6.2f}%")

    # Cross-Domain Evaluation (External corpora: M-AILABS, VoxPopuli, MASC, AISHELL-3)
    n_cross_pts = min(10000, len(df_cross[df_cross['label']==0]), len(df_cross[df_cross['label']==1]))
    cross_sample = pd.concat([
        df_cross[df_cross['label']==0].sample(n=n_cross_pts, random_state=42),
        df_cross[df_cross['label']==1].sample(n=n_cross_pts, random_state=42)
    ])
    X_cross = np.vstack(cross_sample['embedding'].values)
    y_cross = cross_sample['label'].values

    acc_crd, auc_crd, eer_crd, thresh_crd, _ = evaluate_probe(X_train_exp1, y_train_exp1, X_cross, y_cross)
    logger.info(f"[Cross-Domain Test] ACC = {acc_crd*100:6.2f}%, AUC = {auc_crd*100:6.2f}%, EER = {eer_crd*100:6.2f}%")

    exp1_table = [
        {"Model": "Mantis v2 (Ours)", "In-Domain ACC": acc_ind*100, "In-Domain EER": eer_ind*100, "Cross-Domain ACC": acc_crd*100, "Cross-Domain EER": eer_crd*100},
        {"Model": "wav2vec 2.0 (Leader)", "In-Domain ACC": 99.97, "In-Domain EER": 0.01, "Cross-Domain ACC": 89.07, "Cross-Domain EER": 11.82},
        {"Model": "ResNet-18", "In-Domain ACC": 99.98, "In-Domain EER": 0.01, "Cross-Domain ACC": 87.87, "Cross-Domain EER": 12.52},
        {"Model": "AST", "In-Domain ACC": 98.53, "In-Domain EER": 1.35, "Cross-Domain ACC": 81.61, "Cross-Domain EER": 16.91},
        {"Model": "SepTr", "In-Domain ACC": 91.70, "In-Domain EER": 22.05, "Cross-Domain ACC": 72.67, "Cross-Domain EER": 45.69}
    ]
    df_exp1_res = pd.DataFrame(exp1_table)
    exp1_csv = os.path.join(args.output_dir, "exp1_official_benchmark_table.csv")
    df_exp1_res.to_csv(exp1_csv, index=False)
    logger.info(f"Saved Table 1 Comparison: {exp1_csv}")
    master_summary["exp1_official_benchmark"] = exp1_table

    # =========================================================================
    # EXPERIMENT 2: Per-Language Cross-Domain Breakdown (7 Languages)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: PER-LANGUAGE CROSS-DOMAIN BREAKDOWN (7 LANGUAGES)")
    logger.info("=" * 70)

    languages = ["en", "ru", "ar", "ro", "de", "zh", "es"]
    exp2_results = []

    for lang in languages:
        df_lang = df[df['language'] == lang]
        real_lang = df_lang[df_lang['label'] == 0]
        fake_lang = df_lang[df_lang['label'] == 1]

        # Disjoint speaker split for real
        real_spks = sorted(real_lang['speaker_id'].dropna().unique().tolist())
        np.random.seed(42)
        np.random.shuffle(real_spks)
        half_spk = max(1, len(real_spks) // 2)
        train_spks = set(real_spks[:half_spk])
        test_spks = set(real_spks[half_spk:])

        # Disjoint generator split for fake
        lang_attacks = sorted(fake_lang['attack'].unique().tolist())
        if len(lang_attacks) >= 2:
            tr_att = [lang_attacks[0]]
            te_att = lang_attacks[1:]
        else:
            tr_att = lang_attacks
            te_att = lang_attacks

        real_tr = real_lang[real_lang['speaker_id'].isin(train_spks)]
        real_te = real_lang[real_lang['speaker_id'].isin(test_spks)]
        fake_tr = fake_lang[fake_lang['attack'].isin(tr_att)]
        fake_te = fake_lang[fake_lang['attack'].isin(te_att)]

        n_tr = min(2000, len(real_tr), len(fake_tr))
        n_te = min(1000, len(real_te), len(fake_te))

        if n_tr < 10 or n_te < 10:
            logger.warning(f"Language {lang}: insufficient samples for disjoint split (tr={n_tr}, te={n_te}).")
            continue

        tr_sub = pd.concat([real_tr.sample(n=n_tr, random_state=42), fake_tr.sample(n=n_tr, random_state=42)])
        te_sub = pd.concat([real_te.sample(n=n_te, random_state=42), fake_te.sample(n=n_te, random_state=42)])

        X_tr = np.vstack(tr_sub['embedding'].values)
        y_tr = tr_sub['label'].values
        X_te = np.vstack(te_sub['embedding'].values)
        y_te = te_sub['label'].values

        acc_l, auc_l, eer_l, _, _ = evaluate_probe(X_tr, y_tr, X_te, y_te)
        logger.info(f"[{lang.upper():2s}] Cross-Domain ACC = {acc_l*100:6.2f}%, AUC = {auc_l*100:6.2f}%, EER = {eer_l*100:6.2f}% (Train={len(y_tr):,}, Test={len(y_te):,})")

        exp2_results.append({
            "language": lang,
            "train_samples": len(y_tr),
            "test_samples": len(y_te),
            "accuracy": acc_l * 100,
            "roc_auc": auc_l * 100,
            "eer": eer_l * 100
        })

    df_exp2 = pd.DataFrame(exp2_results)
    exp2_csv = os.path.join(args.output_dir, "exp2_per_language_results.csv")
    df_exp2.to_csv(exp2_csv, index=False)
    master_summary["exp2_per_language"] = exp2_results

    # =========================================================================
    # EXPERIMENT 3: Leave-One-Attack-Out (LOAO) Zero-Shot Generalization
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: LEAVE-ONE-ATTACK-OUT (LOAO) ZERO-SHOT GENERALIZATION")
    logger.info("=" * 70)

    top_attacks = ["xtts_v2", "vits+knn", "vits+openvoice", "fairseq_knnvc", "fairseq_freevc", "vits_freevc", "bark+freevc", "vallex"]
    real_all = df[df['label'] == 0]
    exp3_results = []

    for held_out_attack in top_attacks:
        df_target_fake = df[(df['label'] == 1) & (df['attack'] == held_out_attack)]
        df_train_fake = df[(df['label'] == 1) & (df['attack'] != held_out_attack)]

        if len(df_target_fake) < 100:
            continue

        n_eval = min(500, len(df_target_fake))
        n_train = 3000

        # Disjoint test
        test_fake = df_target_fake.sample(n=n_eval, random_state=42)
        test_real = real_all.sample(n=n_eval, random_state=42)
        test_df = pd.concat([test_real, test_fake])

        # Train on remaining attacks + remaining real
        train_fake = df_train_fake.sample(n=n_train, random_state=42)
        train_real = real_all.drop(test_real.index).sample(n=n_train, random_state=42)
        train_df = pd.concat([train_real, train_fake])

        X_tr = np.vstack(train_df['embedding'].values)
        y_tr = train_df['label'].values
        X_te = np.vstack(test_df['embedding'].values)
        y_te = test_df['label'].values

        acc_a, auc_a, eer_a, _, _ = evaluate_probe(X_tr, y_tr, X_te, y_te)
        logger.info(f"[LOAO Held-Out: {held_out_attack:18s}] Zero-Shot ACC = {acc_a*100:6.2f}%, AUC = {auc_a*100:6.2f}%, EER = {eer_a*100:6.2f}%")

        exp3_results.append({
            "held_out_attack": held_out_attack,
            "accuracy": acc_a * 100,
            "roc_auc": auc_a * 100,
            "eer": eer_a * 100
        })

    master_summary["exp3_leave_one_attack_out"] = exp3_results

    # Save LOAO Bar Chart
    plt.figure(figsize=(10, 5))
    attacks = [r["held_out_attack"] for r in exp3_results]
    accs = [r["accuracy"] for r in exp3_results]
    eers = [r["eer"] for r in exp3_results]
    x = np.arange(len(attacks))
    width = 0.35
    plt.bar(x - width/2, accs, width, label='Zero-Shot Accuracy (%)', color='tab:blue')
    plt.bar(x + width/2, eers, width, label='Zero-Shot EER (%)', color='tab:red')
    plt.xticks(x, attacks, rotation=35, ha='right')
    plt.ylabel('Percentage (%)')
    plt.title('Mantis v2 Zero-Shot Detection on Held-Out Generative Engines')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    loao_plot = os.path.join(args.output_dir, "exp3_loao_attack_breakdown.png")
    plt.savefig(loao_plot, dpi=150)
    plt.close()

    # =========================================================================
    # EXPERIMENT 4: Cross-Lingual Zero-Shot Transfer Matrix
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4: CROSS-LINGUAL ZERO-SHOT TRANSFER MATRIX")
    logger.info("=" * 70)

    matrix_langs = ["en", "ru", "ar", "de", "es", "ro", "zh"]
    transfer_matrix_acc = np.zeros((len(matrix_langs), len(matrix_langs)))

    for i, tr_lang in enumerate(matrix_langs):
        df_tr = df[df['language'] == tr_lang]
        n_tr = min(2000, len(df_tr[df_tr['label']==0]), len(df_tr[df_tr['label']==1]))
        tr_sample = pd.concat([
            df_tr[df_tr['label']==0].sample(n=n_tr, random_state=42),
            df_tr[df_tr['label']==1].sample(n=n_tr, random_state=42)
        ])
        X_tr = np.vstack(tr_sample['embedding'].values)
        y_tr = tr_sample['label'].values
        
        clf_transfer = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf_transfer.fit(X_tr, y_tr)

        for j, te_lang in enumerate(matrix_langs):
            df_te = df[df['language'] == te_lang]
            n_te = min(500, len(df_te[df_te['label']==0]), len(df_te[df_te['label']==1]))
            te_sample = pd.concat([
                df_te[df_te['label']==0].sample(n=n_te, random_state=42),
                df_te[df_te['label']==1].sample(n=n_te, random_state=42)
            ])
            X_te = np.vstack(te_sample['embedding'].values)
            y_te = te_sample['label'].values
            
            y_pred = clf_transfer.predict(X_te)
            acc_val = float(accuracy_score(y_te, y_pred)) * 100.0
            transfer_matrix_acc[i, j] = acc_val

        logger.info(f"Trained on {tr_lang.upper()}: transfer across {matrix_langs} -> Avg ACC: {transfer_matrix_acc[i].mean():.2f}%")

    master_summary["exp4_cross_lingual_matrix"] = {
        "languages": matrix_langs,
        "matrix": transfer_matrix_acc.tolist()
    }

    # Save Cross-Lingual Heatmap
    plt.figure(figsize=(7, 6))
    plt.imshow(transfer_matrix_acc, cmap='Blues', interpolation='nearest')
    plt.title('Cross-Lingual Deepfake Transfer Matrix (Accuracy %)')
    plt.colorbar()
    plt.xticks(np.arange(len(matrix_langs)), [l.upper() for l in matrix_langs])
    plt.yticks(np.arange(len(matrix_langs)), [l.upper() for l in matrix_langs])
    plt.xlabel('Evaluation Language')
    plt.ylabel('Training Language')
    for ii in range(len(matrix_langs)):
        for jj in range(len(matrix_langs)):
            val = transfer_matrix_acc[ii, jj]
            plt.text(jj, ii, f"{val:.1f}", ha='center', va='center',
                     color='white' if val > 80 else 'black')
    plt.tight_layout()
    heatmap_path = os.path.join(args.output_dir, "exp4_cross_lingual_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    # =========================================================================
    # EXPERIMENT 5: Few-Shot Training Sample Decay Curve
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5: FEW-SHOT TRAINING SAMPLE DECAY CURVE")
    logger.info("=" * 70)

    # 10,000 down to 10 per class (20 to 20,000 total)
    decay_sizes = [10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10]
    fixed_test_size = 5000

    # Fixed Test Set
    df_eval_real = df[df['label'] == 0].sample(n=fixed_test_size, random_state=42)
    df_eval_fake = df[df['label'] == 1].sample(n=fixed_test_size, random_state=42)
    eval_df = pd.concat([df_eval_real, df_eval_fake])
    X_fixed_test = np.vstack(eval_df['embedding'].values)
    y_fixed_test = eval_df['label'].values
    logger.info(f"Fixed Evaluation Set: {len(y_fixed_test):,} samples (5,000 Real, 5,000 Fake)")

    df_train_pool_real = df[df['label'] == 0].drop(df_eval_real.index)
    df_train_pool_fake = df[df['label'] == 1].drop(df_eval_fake.index)

    exp5_results = []
    for n_per_class in decay_sizes:
        sub_real = df_train_pool_real.sample(n=n_per_class, random_state=42)
        sub_fake = df_train_pool_fake.sample(n=n_per_class, random_state=42)
        tr_df = pd.concat([sub_real, sub_fake])

        X_tr = np.vstack(tr_df['embedding'].values)
        y_tr = tr_df['label'].values

        acc_d, auc_d, eer_d, _, _ = evaluate_probe(X_tr, y_tr, X_fixed_test, y_fixed_test)
        total_tr = len(y_tr)
        logger.info(f"Decay: {n_per_class:5d}/class (Train={total_tr:5d}) -> ACC = {acc_d*100:6.2f}%, AUC = {auc_d*100:6.2f}%, EER = {eer_d*100:6.2f}%")

        exp5_results.append({
            "samples_per_class": n_per_class,
            "total_train_samples": total_tr,
            "accuracy": acc_d * 100,
            "roc_auc": auc_d * 100,
            "eer": eer_d * 100
        })

    master_summary["exp5_sample_decay"] = exp5_results

    # Save Decay Curve Plot
    train_pts = [r["total_train_samples"] for r in exp5_results]
    acc_pts = [r["accuracy"] for r in exp5_results]
    eer_pts = [r["eer"] for r in exp5_results]

    plt.figure(figsize=(8, 5))
    plt.plot(train_pts, acc_pts, marker='o', lw=2, color='tab:blue', label='Accuracy (%)')
    plt.plot(train_pts, eer_pts, marker='s', lw=2, color='tab:red', label='Equal Error Rate (EER %)')
    plt.axhline(89.07, color='green', linestyle='--', label='wav2vec 2.0 SOTA (89.07% ACC)')
    plt.axhline(81.61, color='orange', linestyle=':', label='AST Baseline (81.61% ACC)')
    plt.xscale('log')
    plt.xlabel('Total Training Audio Clips (Log Scale)')
    plt.ylabel('Percentage (%)')
    plt.title('Mantis v2 Scaling Law: Performance vs. Training Budget')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    decay_plot = os.path.join(args.output_dir, "exp5_sample_decay_curve.png")
    plt.savefig(decay_plot, dpi=150)
    plt.close()

    # Save Master Summary JSON
    summary_path = os.path.join(args.output_dir, "master_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(master_summary, f, indent=2)
    logger.info(f"\nSaved consolidated Master Summary JSON: {summary_path}")

    logger.info("=" * 70)
    logger.info("ALL 5 BENCHMARK EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info(f"Artifacts and plots saved in: {args.output_dir}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
