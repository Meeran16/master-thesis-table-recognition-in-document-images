from pathlib import Path
import csv
import statistics

GT_DIR = Path("/home/meeran/thesis_data/dataset_100ann_5class/val/labels")

PRED_DIRS = {
    "YOLOv8 v2": Path("/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels"),
    "YOLOv10 v2": Path("/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels"),
    "TATR v2 @ 0.40": Path("/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4"),
    "Nemotron v2 @ 0.25": Path("/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels"),
}

CLASS_NAMES = {
    0: "Row",
    1: "Column",
    2: "Cell",
    3: "Header-Row",
    4: "Spanning-Cell",
}

OUT_DIR = Path("/home/meeran/thesis_data/iou_annotation_quality")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_yolo_labels(path):
    boxes = []
    if not path.exists():
        return boxes

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        conf = float(parts[5]) if len(parts) > 5 else None

        boxes.append({
            "cls": cls,
            "xyxy": (x1, y1, x2, y2),
            "conf": conf,
        })

    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def greedy_match(gt_boxes, pred_boxes, threshold=0.5, class_aware=True):
    candidates = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if class_aware and gt["cls"] != pred["cls"]:
                continue
            score = iou(gt["xyxy"], pred["xyxy"])
            if score >= threshold:
                candidates.append((score, gi, pi))

    candidates.sort(reverse=True)

    used_gt = set()
    used_pred = set()
    matched_ious = []
    matched_classes = []

    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched_ious.append(score)
        matched_classes.append(gt_boxes[gi]["cls"])

    tp = len(matched_ious)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_ious": matched_ious,
        "matched_classes": matched_classes,
    }


def best_iou_per_gt(gt_boxes, pred_boxes, class_aware=False):
    values = []

    for gt in gt_boxes:
        best = 0.0
        for pred in pred_boxes:
            if class_aware and gt["cls"] != pred["cls"]:
                continue
            best = max(best, iou(gt["xyxy"], pred["xyxy"]))
        values.append(best)

    return values


def summarize(values):
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def fmt(x):
    return f"{x:.3f}"


gt_files = sorted(GT_DIR.glob("*.txt"))

summary_rows = []
per_class_rows = []

for model_name, pred_dir in PRED_DIRS.items():
    all_gt = []
    all_pred = []

    for gt_file in gt_files:
        stem = gt_file.stem
        pred_file = pred_dir / f"{stem}.txt"

        gt_boxes = read_yolo_labels(gt_file)
        pred_boxes = read_yolo_labels(pred_file)

        all_gt.extend(gt_boxes)
        all_pred.extend(pred_boxes)

    result_ca_050 = greedy_match(all_gt, all_pred, threshold=0.50, class_aware=True)
    result_ca_075 = greedy_match(all_gt, all_pred, threshold=0.75, class_aware=True)
    result_cg_050 = greedy_match(all_gt, all_pred, threshold=0.50, class_aware=False)
    result_cg_075 = greedy_match(all_gt, all_pred, threshold=0.75, class_aware=False)

    matched_iou_summary = summarize(result_cg_050["matched_ious"])
    best_iou_values = best_iou_per_gt(all_gt, all_pred, class_aware=False)
    best_iou_summary = summarize(best_iou_values)

    summary_rows.append({
        "Model": model_name,
        "GT boxes": len(all_gt),
        "Pred boxes": len(all_pred),
        "Class-aware P@0.50": fmt(result_ca_050["precision"]),
        "Class-aware R@0.50": fmt(result_ca_050["recall"]),
        "Class-aware F1@0.50": fmt(result_ca_050["f1"]),
        "Class-aware F1@0.75": fmt(result_ca_075["f1"]),
        "Class-agnostic P@0.50": fmt(result_cg_050["precision"]),
        "Class-agnostic R@0.50": fmt(result_cg_050["recall"]),
        "Class-agnostic F1@0.50": fmt(result_cg_050["f1"]),
        "Class-agnostic F1@0.75": fmt(result_cg_075["f1"]),
        "Mean matched IoU": fmt(matched_iou_summary["mean"]),
        "Median matched IoU": fmt(matched_iou_summary["median"]),
        "Mean best IoU per GT": fmt(best_iou_summary["mean"]),
        "Median best IoU per GT": fmt(best_iou_summary["median"]),
    })

    for cls_id, cls_name in CLASS_NAMES.items():
        gt_cls = [b for b in all_gt if b["cls"] == cls_id]
        pred_cls = [b for b in all_pred if b["cls"] == cls_id]

        cls_result = greedy_match(gt_cls, pred_cls, threshold=0.50, class_aware=True)
        cls_iou_summary = summarize(cls_result["matched_ious"])

        per_class_rows.append({
            "Model": model_name,
            "Class": cls_name,
            "GT boxes": len(gt_cls),
            "Pred boxes": len(pred_cls),
            "Precision@0.50": fmt(cls_result["precision"]),
            "Recall@0.50": fmt(cls_result["recall"]),
            "F1@0.50": fmt(cls_result["f1"]),
            "Mean matched IoU": fmt(cls_iou_summary["mean"]),
            "Median matched IoU": fmt(cls_iou_summary["median"]),
        })


summary_csv = OUT_DIR / "iou_annotation_quality_summary.csv"
per_class_csv = OUT_DIR / "iou_annotation_quality_per_class.csv"
summary_md = OUT_DIR / "iou_annotation_quality_summary.md"

with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

with per_class_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
    writer.writeheader()
    writer.writerows(per_class_rows)

with summary_md.open("w") as f:
    f.write("# IoU Annotation Quality Summary\n\n")
    f.write("This table evaluates localization quality for annotation usefulness. Class-aware matching requires the correct class label. Class-agnostic matching checks box overlap regardless of label, which is useful for estimating pre-annotation effort in Label Studio.\n\n")

    headers = list(summary_rows[0].keys())
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in summary_rows:
        f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")

    f.write("\n\n## Per-class IoU summary\n\n")
    headers = list(per_class_rows[0].keys())
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in per_class_rows:
        f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")

print("Saved:")
print(summary_csv)
print(per_class_csv)
print(summary_md)
