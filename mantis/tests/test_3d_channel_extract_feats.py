import os
import sys
import glob

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import pytest
except ImportError:
    pytest = None

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

from mantis.architecture import Mantis8M, MantisV1, MantisV2
from mantis.trainer import MantisTrainer


def ensure_3d_datasets_exist():
    """
    Ensure GestureMidAirD1, D2, and D3 data directories exist.
    If D2 or D3 are missing, generate them by aligning the Leap Motion
    3D trajectories from gesturesCodes/datasetNew with GestureMidAirD1.
    """
    base_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    d1_dir = os.path.join(base_data_dir, 'GestureMidAirD1')
    d2_dir = os.path.join(base_data_dir, 'GestureMidAirD2')
    d3_dir = os.path.join(base_data_dir, 'GestureMidAirD3')

    d2_ready = os.path.exists(d2_dir) and all(
        os.path.exists(os.path.join(d2_dir, f'{v}_{s}.npy'))
        for v in ['X', 'y'] for s in ['train', 'test']
    )
    d3_ready = os.path.exists(d3_dir) and all(
        os.path.exists(os.path.join(d3_dir, f'{v}_{s}.npy'))
        for v in ['X', 'y'] for s in ['train', 'test']
    )

    if d2_ready and d3_ready:
        return

    # Load authentic 3D coordinates from datasetNew
    raw_csv_dir = os.path.join(base_data_dir, 'gesturesCodes', 'datasetNew')
    if not os.path.exists(raw_csv_dir):
        raise FileNotFoundError(f"Raw 3D dataset not found at {raw_csv_dir}")

    subjects = [d for d in os.listdir(raw_csv_dir) if os.path.isdir(os.path.join(raw_csv_dir, d))]
    all_3d = {}
    for sub in subjects:
        csvs = glob.glob(os.path.join(raw_csv_dir, sub, '*.csv'))
        for c in csvs:
            gname = os.path.splitext(os.path.basename(c))[0]
            pts = []
            with open(c, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            all_3d[(sub, gname)] = np.array(pts, dtype=np.float32)

    X_train_d1 = np.load(os.path.join(d1_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(d1_dir, 'y_train.npy'))
    X_test_d1 = np.load(os.path.join(d1_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(d1_dir, 'y_test.npy'))

    def map_samples(X_raw):
        mapping = []
        for i in range(len(X_raw)):
            x_d1 = X_raw[i, 0]
            best_k = None
            best_diff = 1e9
            for k, arr in all_3d.items():
                L = len(arr)
                diff = np.max(np.abs(x_d1[:L] - arr[:, 0]))
                if diff < best_diff:
                    best_diff = diff
                    best_k = k
            mapping.append(all_3d[best_k])
        return mapping

    train_trajectories = map_samples(X_train_d1)
    test_trajectories = map_samples(X_test_d1)

    for dim_idx, target_dir in [(1, d2_dir), (2, d3_dir)]:
        os.makedirs(target_dir, exist_ok=True)
        X_tr = np.zeros_like(X_train_d1)
        for i, arr in enumerate(train_trajectories):
            L = len(arr)
            X_tr[i, 0, :L] = arr[:, dim_idx]
            X_tr[i, 0, L:] = X_train_d1[i, 0, L:]
        X_te = np.zeros_like(X_test_d1)
        for i, arr in enumerate(test_trajectories):
            L = len(arr)
            X_te[i, 0, :L] = arr[:, dim_idx]
            X_te[i, 0, L:] = X_test_d1[i, 0, L:]

        np.save(os.path.join(target_dir, 'X_train.npy'), X_tr)
        np.save(os.path.join(target_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(target_dir, 'X_test.npy'), X_te)
        np.save(os.path.join(target_dir, 'y_test.npy'), y_test)


def load_3d_datasets():
    """
    Loads all 3 dimensions (D1: X, D2: Y, D3: Z) of the GestureMidAir dataset.
    Returns:
        dict: mapping dim (1, 2, 3) to (X_train, X_test, y_train, y_test) resized to 512.
    """
    ensure_3d_datasets_exist()

    def resize(X):
        X_scaled = F.interpolate(
            torch.tensor(X, dtype=torch.float),
            size=512,
            mode='linear',
            align_corners=False
        )
        return X_scaled.numpy()

    datasets = {}
    for dim_num, dim_folder in [(1, 'GestureMidAirD1'), (2, 'GestureMidAirD2'), (3, 'GestureMidAirD3')]:
        path = f'data/{dim_folder}'
        data = [
            np.load(f'{path}/{variable}_{set_name}.npy')
            for variable in ['X', 'y']
            for set_name in ['train', 'test']
        ]
        X_train, X_test, y_train, y_test = data
        X_train_resized = resize(X_train)
        X_test_resized = resize(X_test)

        assert (
            (X_train_resized.shape[2] == 512) and (X_test_resized.shape[2] == 512)
        ), f"After resize, seq length should be 512 but got {X_train_resized.shape[2]} and {X_test_resized.shape[2]} for D{dim_num}"

        datasets[dim_num] = (X_train_resized, X_test_resized, y_train, y_test)

    return datasets


GESTURE_NAMES = {
    0: 'arc3Dleft', 1: 'arc3Dright', 2: 'caret', 3: 'check', 4: 'circle',
    5: 'curly-bracket-left', 6: 'curly-bracket-right', 7: 'delete',
    8: 'left-swipe', 9: 'pigtail', 10: 'poly3Dxyz', 11: 'poly3Dxzy',
    12: 'poly3Dyxz', 13: 'poly3Dyzx', 14: 'poly3Dzxy', 15: 'poly3Dzyx',
    16: 'rectangle', 17: 'right-swipe', 18: 'spiral', 19: 'square-bracket-left',
    20: 'square-bracket-right', 21: 'star', 22: 'triangle', 23: 'v',
    24: 'x', 25: 'zig-zag'
}


def ensemble_3d_predictions(preds_3d, probas_3d, classes, y_true=None):
    """
    Combines predictions from 3 dimensions (D1, D2, D3):
      1. Choose the most common label result (majority vote).
      2. If the results are 3 different labels, choose the one with highest confidence.

    Args:
        preds_3d: list of length 3, each containing 1D array of predicted class labels for test set.
        probas_3d: list of length 3, each containing 2D array (n_samples, n_classes) of class probabilities.
        classes: list of class label arrays recognized by each dimension classifier.
        y_true: optional 1D array of ground truth labels for logging right/wrong status.

    Returns:
        tuple: (final_preds: np.ndarray, log_records: list[dict])
    """
    n_samples = len(preds_3d[0])
    final_preds = []
    log_records = []

    for i in range(n_samples):
        labels = [preds_3d[0][i], preds_3d[1][i], preds_3d[2][i]]

        # Compute confidence for each predicted label
        confidences = []
        for dim in range(3):
            lbl = labels[dim]
            col_idx = np.where(classes[dim] == lbl)[0][0]
            confidences.append(float(probas_3d[dim][i, col_idx]))

        # Check for most common label
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = np.max(counts)

        if max_count >= 2:
            # Majority agreement (2 or 3 votes)
            chosen_label = int(unique_labels[np.argmax(counts)])
            decision_type = f"Majority ({max_count}/3)"
            # Confidence of chosen label from agreeing dims
            agreeing_confs = [confidences[d] for d in range(3) if labels[d] == chosen_label]
            final_conf = max(agreeing_confs)
        else:
            # All 3 are different labels -> pick the one with highest confidence
            highest_conf_idx = int(np.argmax(confidences))
            chosen_label = int(labels[highest_conf_idx])
            final_conf = confidences[highest_conf_idx]
            decision_type = f"Highest Conf (D{highest_conf_idx + 1}: {final_conf:.2f})"

        final_preds.append(chosen_label)

        if y_true is not None:
            true_lbl = int(y_true[i])
            gname = GESTURE_NAMES.get(true_lbl, f"class_{true_lbl}")
            d1_res = "✓" if labels[0] == true_lbl else "✗"
            d2_res = "✓" if labels[1] == true_lbl else "✗"
            d3_res = "✓" if labels[2] == true_lbl else "✗"
            final_res = "✓" if chosen_label == true_lbl else "✗"

            log_records.append({
                "Sample": i,
                "True Label": true_lbl,
                "Gesture": gname,
                "D1 Pred": labels[0],
                "D1 Conf": round(confidences[0], 3),
                "D1 Result": d1_res,
                "D2 Pred": labels[1],
                "D2 Conf": round(confidences[1], 3),
                "D2 Result": d2_res,
                "D3 Pred": labels[2],
                "D3 Conf": round(confidences[2], 3),
                "D3 Result": d3_res,
                "Final Pred": chosen_label,
                "Final Conf": round(final_conf, 3),
                "Final Result": final_res,
                "Decision Logic": decision_type
            })

    return np.array(final_preds), log_records


def save_markdown_log(model_name, dim_accs, ensemble_acc, log_records, out_dir):
    """
    Saves a formatted Markdown table and summary inside the 3D results folder.
    """
    os.makedirs(out_dir, exist_ok=True)
    md_file = os.path.join(out_dir, f"{model_name}_3D_results.md")

    total = len(log_records)
    d1_correct = sum(1 for r in log_records if r["D1 Result"] == "✓")
    d2_correct = sum(1 for r in log_records if r["D2 Result"] == "✓")
    d3_correct = sum(1 for r in log_records if r["D3 Result"] == "✓")
    final_correct = sum(1 for r in log_records if r["Final Result"] == "✓")
    recovered = sum(1 for r in log_records if r["D1 Result"] == "✗" and r["Final Result"] == "✓")

    lines = [
        f"# 3D Feature Extraction & Ensemble Results: {model_name}",
        "",
        "## Summary Performance",
        "",
        f"- **D1 (X-axis) Accuracy**: {dim_accs[1]:.2%} ({d1_correct}/{total})",
        f"- **D2 (Y-axis) Accuracy**: {dim_accs[2]:.2%} ({d2_correct}/{total})",
        f"- **D3 (Z-axis) Accuracy**: {dim_accs[3]:.2%} ({d3_correct}/{total})",
        f"- **3D Ensemble Accuracy**: **{ensemble_acc:.2%} ({final_correct}/{total})**",
        f"- **Improvement over 1D**: **+{ensemble_acc - dim_accs[1]:.2%}** (+{final_correct - d1_correct} cases)",
        f"- **Misclassifications Recovered by 3D Ensemble**: {recovered} cases",
        "",
        "## Decision Rules",
        "1. **Majority Vote**: If 2 or 3 dimensions predict the same label, pick that most common label.",
        "2. **Confidence Tie-Breaker**: If all 3 dimensions predict different labels, choose the prediction with the highest model confidence.",
        "",
        "---",
        "",
        f"## Detailed Per-Sample Log (All {total} Test Cases)",
        "",
        "| Sample | True Label | Gesture | D1 Pred | D1 Conf | D1 Result | D2 Pred | D2 Conf | D2 Result | D3 Pred | D3 Conf | D3 Result | Final Pred | Final Conf | Final Result | Decision Logic |",
        "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for r in log_records:
        lines.append(
            f"| #{r['Sample']:03d} | {r['True Label']} | {r['Gesture']} "
            f"| {r['D1 Pred']} | {r['D1 Conf']:.3f} | {r['D1 Result']} "
            f"| {r['D2 Pred']} | {r['D2 Conf']:.3f} | {r['D2 Result']} "
            f"| {r['D3 Pred']} | {r['D3 Conf']:.3f} | {r['D3 Result']} "
            f"| {r['Final Pred']} | {r['Final Conf']:.3f} | {r['Final Result']} "
            f"| {r['Decision Logic']} |"
        )

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return md_file


def _run_test_3d_channel_extract_feats(device='cpu'):
    datasets_3d = load_3d_datasets()
    _, _, _, y_test = datasets_3d[1]

    for model_name in ["Mantis8M", "MantisV1", "MantisV2"]:
        # Initialize model
        if model_name == "Mantis8M":
            network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        elif model_name == "MantisV1":
            network = MantisV1(device=device).from_pretrained("paris-noah/MantisPlus")
        elif model_name == "MantisV2":
            network = MantisV2(device=device).from_pretrained("paris-noah/MantisV2")

        model = MantisTrainer(device=device, network=network)

        preds_3d = []
        probas_3d = []
        classes_3d = []
        dim_accs = {}

        for dim_num in [1, 2, 3]:
            X_train, X_test, y_train, _ = datasets_3d[dim_num]

            # Extract deep features
            Z_train = model.transform(X_train)
            Z_test = model.transform(X_test)
            assert (
                (Z_train.shape[1] == 256) and (Z_test.shape[1] == 256)
            ), f"Expected 256 deep features, got {Z_train.shape[1]} and {Z_test.shape[1]}"

            # Train classifier
            predictor = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
            predictor.fit(Z_train, y_train)

            # Predict labels and class probabilities
            y_pred = predictor.predict(Z_test)
            y_proba = predictor.predict_proba(Z_test)

            dim_acc = np.mean(y_test == y_pred)
            dim_accs[dim_num] = dim_acc
            print(f"{model_name} (D{dim_num}): Accuracy on the test set is {dim_acc:.4f} ({np.sum(y_test == y_pred)}/{len(y_test)})")

            preds_3d.append(y_pred)
            probas_3d.append(y_proba)
            classes_3d.append(predictor.classes_)

        # Combine predictions using 3D ensemble voting with confidence tie-breaker
        ensemble_pred, log_records = ensemble_3d_predictions(preds_3d, probas_3d, classes_3d, y_true=y_test)
        ensemble_acc = np.mean(y_test == ensemble_pred)
        print(
            f"{model_name} (3D Ensemble): Accuracy on the test set is {ensemble_acc:.4f} "
            f"({np.sum(y_test == ensemble_pred)}/{len(y_test)}) [vs 1D (D1): {dim_accs[1]:.4f}]"
        )

        # Save detailed logs
        out_dir_3d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Result', '3D')
        md_file = save_markdown_log(model_name, dim_accs, ensemble_acc, log_records, out_dir_3d)
        print(f"Logged Markdown report to: {md_file}")

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Result')
        os.makedirs(out_dir, exist_ok=True)
        csv_file = os.path.join(out_dir_3d, f"{model_name}_3D_detailed_log.csv")
        try:
            import pandas as pd
            df_log = pd.DataFrame(log_records)
            df_log.to_csv(csv_file, index=False, encoding='utf-8-sig')
            # Also keep a copy in Result/ root for compatibility
            df_log.to_csv(os.path.join(out_dir, f"{model_name}_3D_detailed_log.csv"), index=False, encoding='utf-8-sig')
            print(f"Logged detailed per-sample breakdown to: {csv_file}")
            print("\nPreview of first 5 samples:")
            preview_cols = ['Sample', 'True Label', 'D1 Conf', 'D1 Result', 'D2 Conf', 'D2 Result', 'D3 Conf', 'D3 Result', 'Final Pred', 'Final Result', 'Decision Logic']
            print(df_log[preview_cols].head(5).to_string(index=False))
            print()
        except Exception as e:
            print(f"Could not save CSV log via pandas: {e}")

        # Verify that ensemble matches or outperforms single-channel D1 accuracy
        assert ensemble_acc >= dim_accs[1], (
            f"{model_name} 3D ensemble accuracy ({ensemble_acc}) should be >= 1D accuracy ({dim_accs[1]})"
        )

    print(
        f"The test of 3D feature extraction and ensembling with device={device} has passed successfully."
    )


if pytest is not None:
    @pytest.mark.parametrize("device", ['cpu'])
    def test_3d_channel_extract_feats(device):
        _run_test_3d_channel_extract_feats(device=device)
else:
    def test_3d_channel_extract_feats(device='cpu'):
        _run_test_3d_channel_extract_feats(device=device)


if __name__ == "__main__":
    test_3d_channel_extract_feats(device="cpu")
