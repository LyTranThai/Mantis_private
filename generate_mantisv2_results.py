import os
import io
import time
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from mantis.architecture import MantisV2
from mantis.trainer import MantisTrainer

GESTURE_NAMES = {
    0: "arc3Dleft",
    1: "arc3Dright",
    2: "caret",
    3: "check",
    4: "circle",
    5: "curly-bracket-left",
    6: "curly-bracket-right",
    7: "delete",
    8: "left-swipe",
    9: "pigtail",
    10: "poly3Dxyz",
    11: "poly3Dxzy",
    12: "poly3Dyxz",
    13: "poly3Dyzx",
    14: "poly3Dzxy",
    15: "poly3Dzyx",
    16: "rectangle",
    17: "right-swipe",
    18: "spiral",
    19: "square-bracket-left",
    20: "square-bracket-right",
    21: "star",
    22: "triangle",
    23: "v",
    24: "x",
    25: "zig-zag",
}

def load_ground_truth_3d():
    data_dir = r"mantis\data\gesturesCodes\datasetNew"
    subjects = ['luca', 'martina', 'pinuccia', 'sabrina', 'stefano']
    trajectories_3d = {}
    for sub in subjects:
        csvs = glob.glob(os.path.join(data_dir, sub, '*.csv'))
        for c in csvs:
            gname = os.path.splitext(os.path.basename(c))[0]
            pts = []
            with open(c, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            trajectories_3d[(sub, gname)] = np.array(pts, dtype=np.float32)
    return trajectories_3d

def match_3d_trajectories(X_test_raw, trajectories_3d):
    mapping = {}
    for i in range(len(X_test_raw)):
        x_d1 = X_test_raw[i, 0]
        best_match_key = None
        best_diff = 1e9
        for key, arr3d in trajectories_3d.items():
            L = min(len(x_d1), len(arr3d[:, 0]))
            diff = np.max(np.abs(x_d1[:L] - arr3d[:L, 0]))
            if diff < best_diff:
                best_diff = diff
                best_match_key = key
        if best_diff < 1e-2:
            mapping[i] = (best_match_key, trajectories_3d[best_match_key])
        else:
            raise ValueError(f"Could not match testcase {i}")
    return mapping

def save_obj_file(arr, filepath, title="gesture"):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# 3D Hand Trajectory: {title}\n")
        f.write(f"# Points: {len(arr)}\n")
        for p in arr:
            f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        f.write("l " + " ".join(str(idx + 1) for idx in range(len(arr))) + "\n")

def main():
    start_time = time.time()
    output_dir = r"C:\Users\Admin\Desktop\SC203\mantis\tests\Result\MantisV2"
    os.makedirs(output_dir, exist_ok=True)

    # Clean old PNG / OBJ files
    for pattern in ["*.png", "*.obj"]:
        for f in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass

    print("Loading datasets...")
    data_dir = r"mantis\data\GestureMidAirD1"
    X_train_raw = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test_raw = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    print("Matching test cases to authentic 3D Leap Motion coordinates...")
    raw_3d_db = load_ground_truth_3d()
    matched_3d = match_3d_trajectories(X_test_raw, raw_3d_db)
    print(f"Successfully matched all {len(matched_3d)} test cases to genuine 3D coordinates.")

    # Resize for MantisV2 inference
    def resize(X):
        return F.interpolate(
            torch.tensor(X, dtype=torch.float), size=512, mode='linear', align_corners=False
        ).numpy()

    X_train = resize(X_train_raw)
    X_test = resize(X_test_raw)

    print("Running MantisV2 inference...")
    device = 'cpu'
    network = MantisV2(device=device).from_pretrained("paris-noah/MantisV2")
    trainer = MantisTrainer(device=device, network=network)

    Z_train = trainer.transform(X_train)
    Z_test = trainer.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    clf.fit(Z_train, y_train)

    y_pred = clf.predict(Z_test)
    y_proba = clf.predict_proba(Z_test)

    num_samples = len(y_test)
    correct_count = int(np.sum(y_test == y_pred))
    accuracy = correct_count / num_samples
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct_count}/{num_samples})")

    # Sort indices by True Label ID (0 to 25), then testcase index
    sorted_indices = sorted(range(num_samples), key=lambda idx: (int(y_test[idx]), idx))

    print("Generating 3D perspective plots, 3D .obj geometry files, and interactive viewer data...")
    total_plot_bytes = 0
    total_obj_bytes = 0
    interactive_data = []

    for i in sorted_indices:
        true_lbl = int(y_test[i])
        pred_lbl = int(y_pred[i])
        is_correct = (true_lbl == pred_lbl)
        conf = float(y_proba[i][pred_lbl])

        true_name = GESTURE_NAMES.get(true_lbl, f"class_{true_lbl}")
        pred_name = GESTURE_NAMES.get(pred_lbl, f"class_{pred_lbl}")
        sub_info, pts_3d = matched_3d[i]

        base_name = f"label_{true_lbl:02d}_testcase_{i:03d}"

        # 1. Save 3D .obj file
        obj_filename = f"{base_name}.obj"
        obj_path = os.path.join(output_dir, obj_filename)
        save_obj_file(pts_3d, obj_path, title=f"Testcase #{i:03d} ({true_name}) - Subject: {sub_info[0]}")
        obj_size = os.path.getsize(obj_path)
        total_obj_bytes += obj_size

        # 2. Render 3D Perspective Plot
        fig = plt.figure(figsize=(4.8, 3.5), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        line_color = '#1b7837' if is_correct else '#d73027'
        ax.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], color=line_color, linewidth=1.6, label='Trajectory')
        ax.scatter([pts_3d[0, 0]], [pts_3d[0, 1]], [pts_3d[0, 2]], color='#009900', s=35, label='Start')
        ax.scatter([pts_3d[-1, 0]], [pts_3d[-1, 1]], [pts_3d[-1, 2]], color='#cc0000', s=35, label='End')

        verdict_str = "CORRECT" if is_correct else "MISMATCH"
        ax.set_title(
            f"Label #{true_lbl:02d}: {true_name} | Case #{i:03d}\n"
            f"Verdict: {pred_name} (ID: {pred_lbl}) [{verdict_str} | Conf: {conf:.1%}]",
            fontsize=8,
            fontweight='bold',
            color='#1b7837' if is_correct else '#b2182b',
            pad=10
        )
        ax.set_xlabel('X (mm)', fontsize=6.5)
        ax.set_ylabel('Y (mm)', fontsize=6.5)
        ax.set_zlabel('Z (mm)', fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.legend(loc='upper right', fontsize=6)
        ax.view_init(elev=28, azim=40)
        fig.tight_layout()

        # Save buffer and quantize
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf)
        opt_buf = io.BytesIO()
        img.convert('P', palette=Image.ADAPTIVE, colors=64).save(opt_buf, format='PNG', optimize=True)
        img_bytes = opt_buf.getvalue()
        total_plot_bytes += len(img_bytes)

        img_filename = f"{base_name}.png"
        img_path = os.path.join(output_dir, img_filename)
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

        # Record for interactive viewer
        interactive_data.append({
            "id": i,
            "label_id": true_lbl,
            "true_name": true_name,
            "pred_name": pred_name,
            "pred_id": pred_lbl,
            "is_correct": bool(is_correct),
            "conf": round(conf * 100, 1),
            "subject": sub_info[0],
            "x": [round(float(v), 2) for v in pts_3d[:, 0]],
            "y": [round(float(v), 2) for v in pts_3d[:, 1]],
            "z": [round(float(v), 2) for v in pts_3d[:, 2]],
        })

    total_png_kb = total_plot_bytes / 1024
    total_obj_kb = total_obj_bytes / 1024
    print(f"Rendered 130 3D plots: {total_png_kb:.1f} KB (Avg {total_png_kb/num_samples:.2f} KB/img)")
    print(f"Generated 130 3D .obj models: {total_obj_kb:.1f} KB (Avg {total_obj_kb/num_samples:.2f} KB/model)")

    # 3. Create Standalone Interactive 3D Web Viewer (HTML)
    print("Building interactive 3D Web Viewer...")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MantisV2 3D Gesture Trajectory Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
    .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); padding: 24px; }}
    h1 {{ margin-top: 0; color: #1a202c; font-size: 24px; }}
    .controls {{ display: flex; gap: 16px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; }}
    select {{ padding: 8px 14px; font-size: 15px; border-radius: 6px; border: 1px solid #cbd5e1; outline: none; }}
    .badge {{ display: inline-block; padding: 6px 12px; border-radius: 6px; font-size: 14px; font-weight: 600; }}
    .badge-success {{ background: #def7ec; color: #03543f; }}
    .badge-danger {{ background: #fde8e8; color: #9b1c1c; }}
    #plot-area {{ width: 100%; height: 650px; border: 1px solid #e2e8f0; border-radius: 8px; }}
    .meta-box {{ margin-top: 16px; font-size: 14px; color: #4b5563; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🖐️ MantisV2 3D Gesture Trajectory Interactive Viewer</h1>
    <div class="controls">
      <label for="caseSelect"><strong>Select Test Case (Sorted by Label ID):</strong></label>
      <select id="caseSelect"></select>
      <span id="verdictBadge" class="badge"></span>
    </div>
    <div id="plot-area"></div>
    <div class="meta-box" id="metaDetails"></div>
  </div>

  <script>
    const data = {json.dumps(interactive_data)};
    const select = document.getElementById('caseSelect');
    const badge = document.getElementById('verdictBadge');
    const meta = document.getElementById('metaDetails');

    data.forEach((item, idx) => {{
      const opt = document.createElement('option');
      opt.value = idx;
      const statusIcon = item.is_correct ? '✅' : '❌';
      opt.textContent = `[Label ${{String(item.label_id).padStart(2, '0')}}] ${{item.true_name}} - Case #${{String(item.id).padStart(3, '0')}} (${{statusIcon}} ${{item.pred_name}})`;
      select.appendChild(opt);
    }});

    function renderPlot(idx) {{
      const item = data[idx];
      badge.textContent = item.is_correct ? `✅ Match (${{item.conf}}%)` : `❌ Mismatch (Predicted: ${{item.pred_name}} ${{item.conf}}%)`;
      badge.className = 'badge ' + (item.is_correct ? 'badge-success' : 'badge-danger');

      meta.innerHTML = `<strong>True Gesture:</strong> ${{item.true_name}} (Label ID: ${{item.label_id}}) &nbsp;|&nbsp; 
                        <strong>Mantis Verdict:</strong> ${{item.pred_name}} (ID: ${{item.pred_id}}) &nbsp;|&nbsp; 
                        <strong>Confidence:</strong> ${{item.conf}}% &nbsp;|&nbsp; 
                        <strong>Subject:</strong> ${{item.subject}} &nbsp;|&nbsp; 
                        <strong>Total 3D Waypoints:</strong> ${{item.x.length}}`;

      const traceLine = {{
        type: 'scatter3d',
        mode: 'lines',
        x: item.x,
        y: item.y,
        z: item.z,
        line: {{ width: 5, color: item.is_correct ? '#1b7837' : '#d73027' }},
        name: 'Trajectory'
      }};

      const traceStart = {{
        type: 'scatter3d',
        mode: 'markers+text',
        x: [item.x[0]],
        y: [item.y[0]],
        z: [item.z[0]],
        marker: {{ size: 7, color: '#00cc44' }},
        text: ['Start'],
        textposition: 'top center',
        name: 'Start'
      }};

      const traceEnd = {{
        type: 'scatter3d',
        mode: 'markers+text',
        x: [item.x[item.x.length - 1]],
        y: [item.y[item.y.length - 1]],
        z: [item.z[item.z.length - 1]],
        marker: {{ size: 7, color: '#ff3333' }},
        text: ['End'],
        textposition: 'top center',
        name: 'End'
      }};

      const layout = {{
        title: `Test Case #${{String(item.id).padStart(3, '0')}}: ${{item.true_name}} in 3D Space`,
        scene: {{
          xaxis: {{ title: 'X (mm)' }},
          yaxis: {{ title: 'Y (mm)' }},
          zaxis: {{ title: 'Z (mm)' }},
          camera: {{ eye: {{ x: 1.4, y: 1.4, z: 1.2 }} }}
        }},
        margin: {{ l: 0, r: 0, b: 0, t: 40 }}
      }};

      Plotly.newPlot('plot-area', [traceLine, traceStart, traceEnd], layout, {{ responsive: true }});
    }}

    select.addEventListener('change', (e) => renderPlot(parseInt(e.target.value)));
    renderPlot(0);
  </script>
</body>
</html>
"""
    viewer_path = os.path.join(output_dir, "interactive_3d_viewer.html")
    with open(viewer_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 4. Generate Updated Results Markdown File with 3D Models and Tables
    print("Updating Markdown report...")
    md_path = os.path.join(output_dir, "results.md")

    # Class stats
    class_stats = {}
    for c_id, c_name in GESTURE_NAMES.items():
        mask = (y_test == c_id)
        total_c = int(np.sum(mask))
        correct_c = int(np.sum((y_test == c_id) & (y_pred == c_id)))
        acc_c = (correct_c / total_c * 100) if total_c > 0 else 0.0
        class_stats[c_id] = {
            "name": c_name,
            "total": total_c,
            "correct": correct_c,
            "acc": acc_c
        }

    md_lines = []
    md_lines.append("# MantisV2 GestureMidAir 3D Test Case Evaluation Results\n")
    md_lines.append("## Executive Summary\n")
    md_lines.append(f"- **Model**: MantisV2 (`paris-noah/MantisV2`) + RandomForest Head")
    md_lines.append(f"- **Dataset**: `GestureMidAir` Authentic 3D Leap Motion Trajectories $(X, Y, Z)$")
    md_lines.append(f"- **Total Test Cases**: {num_samples}")
    md_lines.append(f"- **Overall Test Accuracy**: **{accuracy * 100:.2f}%** ({correct_count}/{num_samples} correct)")
    md_lines.append(f"- **Table Ordering**: **Sorted by True Label ID (00 to 25)**")
    md_lines.append(f"- **3D Interactive Viewer**: [Open Interactive 3D Viewer (`interactive_3d_viewer.html`)](./interactive_3d_viewer.html)")
    md_lines.append(f"- **Storage Efficiency**: {total_png_kb:.1f} KB for all 130 3D plots (avg {total_png_kb/num_samples:.2f} KB/plot), {total_obj_kb:.1f} KB for all 130 `.obj` 3D geometry files (avg {total_obj_kb/num_samples:.2f} KB/model)\n")

    md_lines.append("## Per-Class Accuracy Summary\n")
    md_lines.append("| Class ID | Gesture Name | Total Samples | Correct | Accuracy |")
    md_lines.append("|:---:|:---|:---:|:---:|:---:|")
    for c_id in sorted(class_stats.keys()):
        stat = class_stats[c_id]
        md_lines.append(f"| {c_id:02d} | {stat['name']} | {stat['total']} | {stat['correct']} | {stat['acc']:.1f}% |")

    md_lines.append("\n## Detailed 3D Test Case Results (Sorted by Label ID)\n")
    md_lines.append("| Label ID | Labeled Gesture (True) | Testcase # | 3D Trajectory Plot | 3D Model (.obj) | Mantis Verdict (Predicted) | Confidence | Verdict |")
    md_lines.append("|:---:|:---|:---:|:---:|:---:|:---|:---:|:---:|")

    for i in sorted_indices:
        true_lbl = int(y_test[i])
        pred_lbl = int(y_pred[i])
        is_correct = (true_lbl == pred_lbl)
        conf = float(y_proba[i][pred_lbl])
        true_name = GESTURE_NAMES.get(true_lbl, f"class_{true_lbl}")
        pred_name = GESTURE_NAMES.get(pred_lbl, f"class_{pred_lbl}")
        verdict_icon = "✅ Match" if is_correct else "❌ Mismatch"

        img_filename = f"label_{true_lbl:02d}_testcase_{i:03d}.png"
        obj_filename = f"label_{true_lbl:02d}_testcase_{i:03d}.obj"

        img_tag = f'<img src="./{img_filename}" width="220" alt="3D Testcase {i:03d}" />'
        obj_link = f'[`{obj_filename}`](./{obj_filename})'

        md_lines.append(
            f"| `{true_lbl:02d}` | **{true_name}** | `#{i:03d}` | {img_tag} | {obj_link} | **{pred_name}** (`{pred_lbl}`) | {conf:.1%} | {verdict_icon} |"
        )

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print(f"Results successfully saved to {md_path}, {readme_path}, and {viewer_path}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
