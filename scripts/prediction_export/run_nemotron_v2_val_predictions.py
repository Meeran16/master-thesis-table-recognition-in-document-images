#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import cv2
import torch

YOLOX_ROOT = "/home/meeran/YOLOX"
sys.path.insert(0, YOLOX_ROOT)

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

EXP_FILE = "/home/meeran/YOLOX/exps/nemotron_finetune_100ann.py"
CKPT = "/home/meeran/YOLOX/YOLOX_outputs/nemotron_finetune_100ann/best_ckpt.pth"

IMG_DIR = Path("/home/meeran/thesis_data/dataset_100ann_5class/val/images")
OUT_DIR = Path("/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF = 0.25
NMS = 0.65

# Nemotron internal class order:
# 0 = Cell
# 1 = Row
# 2 = Column
#
# Unified 5-class overlay/evaluation IDs:
# 0 Row
# 1 Column
# 2 Cell
# 3 Header-Row
# 4 Spanning-Cell
NEMOTRON_TO_UNIFIED = {
    0: 2,  # Cell
    1: 0,  # Row
    2: 1,  # Column
}

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(float(x1), w))
    y1 = max(0.0, min(float(y1), h))
    x2 = max(0.0, min(float(x2), w))
    y2 = max(0.0, min(float(y2), h))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    cx = x1 + bw / 2
    cy = y1 + bh / 2

    return cx / w, cy / h, bw / w, bh / h

def main():
    print("=== Nemotron v2 validation prediction export ===")
    print("Exp:", EXP_FILE)
    print("Checkpoint:", CKPT)
    print("Images:", IMG_DIR)
    print("Output:", OUT_DIR)
    print("Conf:", CONF)
    print("NMS:", NMS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    preproc = ValTransform(legacy=False)

    image_files = sorted(
        list(IMG_DIR.glob("*.png")) +
        list(IMG_DIR.glob("*.jpg")) +
        list(IMG_DIR.glob("*.jpeg"))
    )

    print("Images found:", len(image_files))

    total_boxes = 0
    per_image_counts = []

    with torch.no_grad():
        for idx, img_path in enumerate(image_files, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                print("ERROR: cannot read", img_path)
                continue

            h, w = img.shape[:2]
            ratio = min(exp.test_size[0] / h, exp.test_size[1] / w)

            img_input, _ = preproc(img, None, exp.test_size)
            img_input = torch.from_numpy(img_input).unsqueeze(0).float().to(device)

            outputs = model(img_input)
            outputs = postprocess(outputs, exp.num_classes, CONF, NMS)

            lines = []

            if outputs[0] is not None:
                pred = outputs[0].cpu()

                boxes = pred[:, 0:4] / ratio
                obj_conf = pred[:, 4]
                cls_conf = pred[:, 5]
                cls_ids = pred[:, 6].int()

                scores = obj_conf * cls_conf

                for box, score, cls_id in zip(boxes, scores, cls_ids):
                    cls_id = int(cls_id.item())
                    if cls_id not in NEMOTRON_TO_UNIFIED:
                        continue

                    unified_cls = NEMOTRON_TO_UNIFIED[cls_id]

                    x1, y1, x2, y2 = box.tolist()
                    cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

                    if bw <= 0 or bh <= 0:
                        continue

                    lines.append(
                        f"{unified_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(score):.6f}"
                    )

            out_path = OUT_DIR / f"{img_path.stem}.txt"
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""))

            total_boxes += len(lines)
            per_image_counts.append(len(lines))

            if idx % 5 == 0 or idx == len(image_files):
                print(f"[{idx}/{len(image_files)}] {img_path.name}: {len(lines)} boxes")

    print("\nDone.")
    print("Images:", len(image_files))
    print("Total boxes:", total_boxes)
    print("Labels saved to:", OUT_DIR)
    print("Empty prediction files:", sum(1 for c in per_image_counts if c == 0))

if __name__ == "__main__":
    main()
