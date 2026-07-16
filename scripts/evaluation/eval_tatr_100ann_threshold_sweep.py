#!/usr/bin/env python3
import os
import json
from pathlib import Path
from collections import defaultdict, Counter

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_DIR = Path("/home/meeran/thesis_data/tatr_finetuned_100ann/final")
IMG_DIR = Path("/home/meeran/thesis_data/dataset_100ann_5class/val/images")
GT_DIR = Path("/home/meeran/thesis_data/dataset_100ann_5class/val/labels")
OUT_ROOT = Path("/home/meeran/thesis_data/tatr_finetuned_100ann_predictions")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MIN_THRESHOLD = min(THRESHOLDS)
IOU_THRESHOLD = 0.50

# Unified evaluation class IDs:
# 0 Row
# 1 Column
# 2 Cell
# 3 Header-Row
# 4 Spanning-Cell
CLASS_NAMES = {
    0: "Row",
    1: "Column",
    2: "Cell",
    3: "Header-Row",
    4: "Spanning-Cell",
}

# TATR native labels -> unified labels
# 0 table: skipped
# 1 table column -> Column
# 2 table row -> Row
# 3 table column header -> Header-Row
# 4 table projected row header: skipped
# 5 table spanning cell -> Spanning-Cell
TATR_TO_UNIFIED = {
    1: 1,
    2: 0,
    3: 3,
    5: 4,
}

SUPPORTED_CLASSES = [0, 1, 3, 4]  # Row, Column, Header-Row, Spanning-Cell


def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    cx = x1 + bw / 2
    cy = y1 + bh / 2

    return [
        cx / img_w,
        cy / img_h,
        bw / img_w,
        bh / img_h,
    ]


def yolo_to_xyxy(yolo_box):
    cx, cy, w, h = yolo_box
    return [
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    ]


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

    return inter / union if union > 0 else 0.0


def load_yolo_labels(path):
    out = defaultdict(list)
    if not path.exists():
        return out

    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cid = int(float(parts[0]))
        if cid not in CLASS_NAMES:
            continue

        box = list(map(float, parts[1:]))
        out[cid].append(box)

    return out


def match_class(gt_boxes, pred_boxes):
    gt_xyxy = [yolo_to_xyxy(b) for b in gt_boxes]
    pred_xyxy = [yolo_to_xyxy(b) for b in pred_boxes]

    pairs = []
    for pi, pb in enumerate(pred_xyxy):
        for gi, gb in enumerate(gt_xyxy):
            pairs.append((iou(pb, gb), pi, gi))

    pairs.sort(reverse=True, key=lambda x: x[0])

    used_p = set()
    used_g = set()
    tp = 0
    matched_ious = []

    for score, pi, gi in pairs:
        if score < IOU_THRESHOLD:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1
        matched_ious.append(score)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn, matched_ious


def calculate_metrics(pred_dir):
    gt_files = set(p.name for p in GT_DIR.glob("*.txt"))
    pred_files = set(p.name for p in pred_dir.glob("*.txt"))
    common = sorted(gt_files & pred_files)

    totals = {cid: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0} for cid in CLASS_NAMES}
    all_ious = []

    for name in common:
        gt = load_yolo_labels(GT_DIR / name)
        pred = load_yolo_labels(pred_dir / name)

        for cid in CLASS_NAMES:
            gt_boxes = gt.get(cid, [])
            pred_boxes = pred.get(cid, [])

            tp, fp, fn, matched_ious = match_class(gt_boxes, pred_boxes)

            totals[cid]["gt"] += len(gt_boxes)
            totals[cid]["pred"] += len(pred_boxes)
            totals[cid]["tp"] += tp
            totals[cid]["fp"] += fp
            totals[cid]["fn"] += fn
            all_ious.extend(matched_ious)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    result = {
        "common_images": len(common),
        "classes": {},
    }

    overall = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    supported = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}

    for cid, name in CLASS_NAMES.items():
        row = totals[cid]
        p, r, f = prf(row["tp"], row["fp"], row["fn"])

        result["classes"][cid] = {
            "name": name,
            **row,
            "precision": p,
            "recall": r,
            "f1": f,
        }

        for k in overall:
            overall[k] += row[k]

        if cid in SUPPORTED_CLASSES:
            for k in supported:
                supported[k] += row[k]

    op, or_, of = prf(overall["tp"], overall["fp"], overall["fn"])
    sp, sr, sf = prf(supported["tp"], supported["fp"], supported["fn"])

    result["overall"] = {**overall, "precision": op, "recall": or_, "f1": of}
    result["supported"] = {**supported, "precision": sp, "recall": sr, "f1": sf}

    if all_ious:
        result["avg_matched_iou"] = sum(all_ious) / len(all_ious)
        result["matched_pairs"] = len(all_ious)
    else:
        result["avg_matched_iou"] = 0.0
        result["matched_pairs"] = 0

    return result


def write_predictions(all_predictions):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    threshold_results = {}

    for threshold in THRESHOLDS:
        tag = str(threshold).replace(".", "p")
        pred_dir = OUT_ROOT / f"thr_{tag}"
        pred_dir.mkdir(parents=True, exist_ok=True)

        total_boxes = 0
        class_counts = Counter()

        for img_name, preds in all_predictions.items():
            lines = []
            for pred in preds:
                if pred["score"] < threshold:
                    continue

                cid = pred["unified_class"]
                cx, cy, w, h = pred["yolo_box"]

                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                total_boxes += 1
                class_counts[CLASS_NAMES[cid]] += 1

            out_path = pred_dir / f"{Path(img_name).stem}.txt"
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        metrics = calculate_metrics(pred_dir)
        metrics["threshold"] = threshold
        metrics["total_boxes"] = total_boxes
        metrics["class_counts"] = dict(class_counts)
        metrics["pred_dir"] = str(pred_dir)

        threshold_results[threshold] = metrics

    return threshold_results


def print_summary(threshold_results):
    print("\n" + "=" * 90)
    print("TATR 100-ann threshold sweep")
    print("=" * 90)
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"GT dir: {GT_DIR}")
    print(f"Prediction root: {OUT_ROOT}")

    print("\nThreshold summary:")
    print(f"{'thr':>6} {'boxes':>7} {'overall_P':>10} {'overall_R':>10} {'overall_F1':>10} {'supported_P':>12} {'supported_R':>12} {'supported_F1':>12}")
    print("-" * 90)

    best_supported = None
    best_overall = None

    for threshold in THRESHOLDS:
        m = threshold_results[threshold]
        o = m["overall"]
        s = m["supported"]

        print(
            f"{threshold:>6.2f} "
            f"{m['total_boxes']:>7} "
            f"{o['precision']:>10.3f} {o['recall']:>10.3f} {o['f1']:>10.3f} "
            f"{s['precision']:>12.3f} {s['recall']:>12.3f} {s['f1']:>12.3f}"
        )

        if best_supported is None or s["f1"] > threshold_results[best_supported]["supported"]["f1"]:
            best_supported = threshold

        if best_overall is None or o["f1"] > threshold_results[best_overall]["overall"]["f1"]:
            best_overall = threshold

    print("\nBest supported-class threshold:", best_supported)
    print("Best overall threshold:", best_overall)

    for title, threshold in [("BEST SUPPORTED", best_supported), ("BEST OVERALL", best_overall)]:
        m = threshold_results[threshold]
        print("\n" + "=" * 90)
        print(f"{title} RESULT @ threshold {threshold}")
        print("=" * 90)
        print("Class counts:", m["class_counts"])
        print(f"Common images: {m['common_images']}")
        print(f"Matched pairs: {m['matched_pairs']}")
        print(f"Average matched IoU: {m['avg_matched_iou']:.4f}")

        print("\nPer-class:")
        print(f"{'Class':<15} {'GT':>5} {'Pred':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("-" * 80)

        for cid in CLASS_NAMES:
            c = m["classes"][cid]
            print(
                f"{c['name']:<15} {c['gt']:>5} {c['pred']:>5} {c['tp']:>5} "
                f"{c['fp']:>5} {c['fn']:>5} "
                f"{c['precision']:>7.3f} {c['recall']:>7.3f} {c['f1']:>7.3f}"
            )

        o = m["overall"]
        s = m["supported"]

        print("-" * 80)
        print(
            f"{'OVERALL':<15} {o['gt']:>5} {o['pred']:>5} {o['tp']:>5} "
            f"{o['fp']:>5} {o['fn']:>5} "
            f"{o['precision']:>7.3f} {o['recall']:>7.3f} {o['f1']:>7.3f}"
        )
        print(
            f"{'SUPPORTED':<15} {s['gt']:>5} {s['pred']:>5} {s['tp']:>5} "
            f"{s['fp']:>5} {s['fn']:>5} "
            f"{s['precision']:>7.3f} {s['recall']:>7.3f} {s['f1']:>7.3f}"
        )


def main():
    print("=== TATR 100-ann inference + threshold sweep ===")
    print("Model:", MODEL_DIR)
    print("Images:", IMG_DIR)
    print("GT:", GT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("Model id2label:", model.config.id2label)

    image_files = sorted(
        list(IMG_DIR.glob("*.png")) +
        list(IMG_DIR.glob("*.jpg")) +
        list(IMG_DIR.glob("*.jpeg"))
    )

    print("Images found:", len(image_files))

    all_predictions = {}

    with torch.no_grad():
        for i, img_path in enumerate(image_files, 1):
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size

            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)

            target_sizes = torch.tensor([[img_h, img_w]], device=device)
            results = processor.post_process_object_detection(
                outputs,
                threshold=MIN_THRESHOLD,
                target_sizes=target_sizes,
            )[0]

            preds = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                tatr_cls = int(label.item())
                if tatr_cls not in TATR_TO_UNIFIED:
                    continue

                unified_cls = TATR_TO_UNIFIED[tatr_cls]
                yolo_box = xyxy_to_yolo(box.tolist(), img_w, img_h)

                if yolo_box[2] <= 0 or yolo_box[3] <= 0:
                    continue

                preds.append({
                    "score": float(score.item()),
                    "tatr_class": tatr_cls,
                    "unified_class": unified_cls,
                    "yolo_box": yolo_box,
                })

            all_predictions[img_path.name] = preds

            if i % 10 == 0 or i == len(image_files):
                print(f"[{i}/{len(image_files)}] {img_path.name}: {len(preds)} raw boxes above {MIN_THRESHOLD}")

    threshold_results = write_predictions(all_predictions)

    summary_path = OUT_ROOT / "threshold_sweep_metrics.json"
    summary_path.write_text(json.dumps(threshold_results, indent=2))

    print_summary(threshold_results)
    print("\nSaved JSON summary:", summary_path)


if __name__ == "__main__":
    main()
