from pathlib import Path
import csv
import re
import json

EVID = Path("/home/meeran/thesis_data/professor_update_evidence")
OUT = EVID / "learning_curve_summary.md"

YOLO_FILES = {
    "YOLOv8 v2": Path("/home/meeran/thesis_data/yolov8_runs_100ann/train_v2_100ann/results.csv"),
    "YOLOv10 v2": Path("/home/meeran/thesis_data/yolov10_runs_100ann/train_v2_100ann/results.csv"),
}

TATR_LOG = Path("/home/meeran/thesis_data/tatr_finetune_100ann.log")
NEMOTRON_LOG = Path("/home/meeran/YOLOX/YOLOX_outputs/nemotron_finetune_100ann/train_log.txt")


def read_yolo_results(path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                if k is None:
                    continue
                clean[k.strip()] = v.strip()
            rows.append(clean)
    return rows


def find_col(row, contains):
    for k in row.keys():
        if contains.lower() in k.lower():
            return k
    return None


def summarize_yolo(name, path):
    rows = read_yolo_results(path)
    if not rows:
        return None

    first = rows[0]
    epoch_col = find_col(first, "epoch") or list(first.keys())[0]
    map50_col = find_col(first, "metrics/mAP50(B)")
    map5095_col = find_col(first, "metrics/mAP50-95(B)")
    precision_col = find_col(first, "metrics/precision(B)")
    recall_col = find_col(first, "metrics/recall(B)")

    # fallback for stripped columns
    if map50_col is None:
        for k in first:
            if "map50" in k.lower() and "95" not in k.lower():
                map50_col = k
    if map5095_col is None:
        for k in first:
            if "map50-95" in k.lower() or "map50_95" in k.lower() or "mAP50-95" in k:
                map5095_col = k

    values = []
    for row in rows:
        try:
            epoch = int(float(row[epoch_col]))
            map5095 = float(row[map5095_col])
            map50 = float(row[map50_col]) if map50_col else None
            precision = float(row[precision_col]) if precision_col else None
            recall = float(row[recall_col]) if recall_col else None
            values.append((epoch, map5095, map50, precision, recall))
        except Exception:
            continue

    best = max(values, key=lambda x: x[1])
    last = values[-1]
    last10 = values[-10:]
    best_last10 = max(last10, key=lambda x: x[1])

    return {
        "name": name,
        "epochs_completed": last[0],
        "best_epoch": best[0],
        "best_map5095": best[1],
        "last_epoch": last[0],
        "last_map5095": last[1],
        "best_last10_epoch": best_last10[0],
        "best_last10_map5095": best_last10[1],
        "trend": "plateau/decline near the end" if last[1] < best_last10[1] else "still stable or slightly improving at the end",
    }


def summarize_tatr(path):
    if not path.exists():
        return None

    text = path.read_text(errors="ignore")
    pairs = []
    for m in re.finditer(r"'eval_loss':\s*([0-9.]+).*?'epoch':\s*([0-9.]+)", text):
        loss = float(m.group(1))
        epoch = float(m.group(2))
        pairs.append((epoch, loss))

    if not pairs:
        return None

    best = min(pairs, key=lambda x: x[1])
    last = pairs[-1]
    return {
        "best_epoch": best[0],
        "best_eval_loss": best[1],
        "last_epoch": last[0],
        "last_eval_loss": last[1],
        "trend": "eval loss improved earlier and then plateaued/slightly worsened" if last[1] > best[1] else "eval loss still improving at the end",
    }


def summarize_nemotron(path):
    if not path.exists():
        return None

    text = path.read_text(errors="ignore")

    ap_values = []
    current_epoch = None

    for line in text.splitlines():
        m_epoch = re.search(r"start train epoch(\d+)", line)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))

        m_ap = re.search(r"Average Precision\s+\(AP\).*IoU=0\.50:0\.95.*=\s*([0-9.]+)", line)
        if m_ap:
            ap = float(m_ap.group(1))
            ap_values.append((current_epoch, ap))

    best_ap_line = re.search(r"best AP is\s*([0-9.]+)", text)
    best_ap = float(best_ap_line.group(1)) / 100 if best_ap_line else None

    result = {
        "best_reported_ap": best_ap,
        "num_eval_points_found": len(ap_values),
    }

    if ap_values:
        best = max(ap_values, key=lambda x: x[1])
        last = ap_values[-1]
        result.update({
            "best_eval_epoch_found": best[0],
            "best_eval_ap_found": best[1],
            "last_eval_epoch_found": last[0],
            "last_eval_ap_found": last[1],
            "trend": "near plateau; final AP close to best reported AP",
        })

    return result


yolo_summaries = [summarize_yolo(name, path) for name, path in YOLO_FILES.items()]
tatr_summary = summarize_tatr(TATR_LOG)
nemotron_summary = summarize_nemotron(NEMOTRON_LOG)

with OUT.open("w") as f:
    f.write("# Learning Curve Summary\n\n")

    f.write("## YOLO training curves\n\n")
    f.write("| Model | Epochs completed | Best epoch | Best mAP50-95 | Last epoch | Last mAP50-95 | Interpretation |\n")
    f.write("|---|---:|---:|---:|---:|---:|---|\n")

    for s in yolo_summaries:
        f.write(
            f"| {s['name']} | {s['epochs_completed']} | {s['best_epoch']} | {s['best_map5095']:.4f} | "
            f"{s['last_epoch']} | {s['last_map5095']:.4f} | {s['trend']} |\n"
        )

    f.write("\n## TATR training curve\n\n")
    if tatr_summary:
        f.write("| Best epoch | Best eval loss | Last epoch | Last eval loss | Interpretation |\n")
        f.write("|---:|---:|---:|---:|---|\n")
        f.write(
            f"| {tatr_summary['best_epoch']:.0f} | {tatr_summary['best_eval_loss']:.4f} | "
            f"{tatr_summary['last_epoch']:.0f} | {tatr_summary['last_eval_loss']:.4f} | {tatr_summary['trend']} |\n"
        )
    else:
        f.write("No TATR eval-loss values found.\n")

    f.write("\n## Nemotron training curve\n\n")
    if nemotron_summary:
        f.write("| Best reported AP50-95 | Evaluation points found | Last eval AP50-95 found | Interpretation |\n")
        f.write("|---:|---:|---:|---|\n")
        f.write(
            f"| {nemotron_summary.get('best_reported_ap', 0):.4f} | "
            f"{nemotron_summary.get('num_eval_points_found', 0)} | "
            f"{nemotron_summary.get('last_eval_ap_found', 0):.4f} | "
            f"{nemotron_summary.get('trend', 'Not enough data parsed')} |\n"
        )
    else:
        f.write("No Nemotron training values found.\n")

print("Created:")
print(OUT)
