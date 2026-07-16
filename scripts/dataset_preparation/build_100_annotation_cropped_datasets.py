from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
import random
import shutil
import json

# -----------------------------
# Input / Output paths
# -----------------------------
SRC = Path("/home/meeran/thesis_data/ls_yolo_100_annotations_with_images")

IMG_DIR = SRC / "images"
LBL_DIR = SRC / "labels"

OUT5 = Path("/home/meeran/thesis_data/dataset_100ann_5class")
OUT3 = Path("/home/meeran/thesis_data/dataset_100ann_3class")

SEED = 42
TRAIN_RATIO = 0.8

# Original Label Studio YOLO class IDs
ORIG = {
    1: "Cell",
    2: "Column",
    4: "Header-Row",
    12: "Row",
    14: "Spanning-Cell",
    15: "Table",
}

TABLE_ID = 15

# 5-class mapping
MAP5 = {
    12: 0,  # Row
    2: 1,   # Column
    1: 2,   # Cell
    4: 3,   # Header-Row
    14: 4,  # Spanning-Cell
}

NAMES5 = ["Row", "Column", "Cell", "Header-Row", "Spanning-Cell"]

# 3-class mapping for Nemotron-style training
MAP3 = {
    12: 0,  # Row
    4: 0,   # Header-Row -> Row
    2: 1,   # Column
    1: 2,   # Cell
    14: 2,  # Spanning-Cell -> Cell
}

NAMES3 = ["Row", "Column", "Cell"]


def reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    for split in ["train", "val"]:
        (p / split / "images").mkdir(parents=True, exist_ok=True)
        (p / split / "labels").mkdir(parents=True, exist_ok=True)


def yolo_to_xyxy(line, w, h):
    parts = line.strip().split()
    cls = int(parts[0])
    x, y, bw, bh = map(float, parts[1:5])

    cx = x * w
    cy = y * h
    box_w = bw * w
    box_h = bh * h

    x1 = cx - box_w / 2
    y1 = cy - box_h / 2
    x2 = cx + box_w / 2
    y2 = cy + box_h / 2

    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, w, h):
    # clip
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 1 or bh <= 1:
        return None

    cx = x1 + bw / 2
    cy = y1 + bh / 2

    return f"{cls} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def find_image_for_label(label_path: Path):
    stem = label_path.stem

    candidates = list(IMG_DIR.glob(stem + ".*"))
    if candidates:
        return candidates[0]

    # Some Label Studio exports use files without extensions
    direct = IMG_DIR / stem
    if direct.exists():
        return direct

    # Fallback: match stem
    for p in IMG_DIR.iterdir():
        if p.stem == stem or p.name == stem:
            return p

    return None


def box_center_inside(box, table):
    _, x1, y1, x2, y2 = box
    _, tx1, ty1, tx2, ty2 = table

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return tx1 <= cx <= tx2 and ty1 <= cy <= ty2


def intersection_to_crop(box, table):
    cls, x1, y1, x2, y2 = box
    _, tx1, ty1, tx2, ty2 = table

    ix1 = max(x1, tx1)
    iy1 = max(y1, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    # shift to crop coordinates
    return cls, ix1 - tx1, iy1 - ty1, ix2 - tx1, iy2 - ty1


def write_yaml(out_dir: Path, names):
    yaml_text = f"""path: {out_dir}
train: train/images
val: val/images

names:
"""
    for i, name in enumerate(names):
        yaml_text += f"  {i}: {name}\n"

    (out_dir / "data.yaml").write_text(yaml_text)


def main():
    reset_dir(OUT5)
    reset_dir(OUT3)

    label_files = sorted(LBL_DIR.glob("*.txt"))

    # Split by original page, not by table crop.
    # This prevents tables from the same page appearing in both train and val.
    stems = [p.stem for p in label_files]
    random.seed(SEED)
    random.shuffle(stems)

    train_n = int(len(stems) * TRAIN_RATIO)
    train_stems = set(stems[:train_n])

    stats = Counter()
    cls5_counter = Counter()
    cls3_counter = Counter()
    table_counts = Counter()
    skipped_empty_crops = []

    manifest = []

    for label_path in label_files:
        img_path = find_image_for_label(label_path)

        if img_path is None:
            stats["missing_image"] += 1
            print("Missing image for:", label_path.name)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            stats["bad_image"] += 1
            print("Bad image:", img_path, e)
            continue

        w, h = img.size

        lines = [x for x in label_path.read_text().splitlines() if x.strip()]
        boxes = [yolo_to_xyxy(line, w, h) for line in lines]

        table_boxes = [b for b in boxes if b[0] == TABLE_ID]
        structure_boxes = [b for b in boxes if b[0] in MAP5]

        table_counts[len(table_boxes)] += 1
        stats["source_pages"] += 1
        stats["table_boxes_total"] += len(table_boxes)

        split = "train" if label_path.stem in train_stems else "val"

        for t_idx, table in enumerate(table_boxes):
            _, tx1, ty1, tx2, ty2 = table

            # clip table box to image
            tx1 = max(0, min(w, tx1))
            ty1 = max(0, min(h, ty1))
            tx2 = max(0, min(w, tx2))
            ty2 = max(0, min(h, ty2))

            crop_w = tx2 - tx1
            crop_h = ty2 - ty1

            if crop_w <= 5 or crop_h <= 5:
                stats["bad_table_box"] += 1
                continue

            crop = img.crop((tx1, ty1, tx2, ty2))

            crop_name = f"{label_path.stem}_table_{t_idx}.png"
            label_name = f"{label_path.stem}_table_{t_idx}.txt"

            labels5 = []
            labels3 = []

            for b in structure_boxes:
                if not box_center_inside(b, (TABLE_ID, tx1, ty1, tx2, ty2)):
                    continue

                remapped = intersection_to_crop(b, (TABLE_ID, tx1, ty1, tx2, ty2))
                if remapped is None:
                    continue

                orig_cls, cx1, cy1, cx2, cy2 = remapped

                out5_cls = MAP5[orig_cls]
                line5 = xyxy_to_yolo(out5_cls, cx1, cy1, cx2, cy2, crop_w, crop_h)

                out3_cls = MAP3[orig_cls]
                line3 = xyxy_to_yolo(out3_cls, cx1, cy1, cx2, cy2, crop_w, crop_h)

                if line5:
                    labels5.append(line5)
                    cls5_counter[out5_cls] += 1

                if line3:
                    labels3.append(line3)
                    cls3_counter[out3_cls] += 1

            if not labels5:
                skipped_empty_crops.append(crop_name)
                stats["empty_table_crops_skipped"] += 1
                continue

            # Save crop and labels for 5-class dataset
            crop.save(OUT5 / split / "images" / crop_name)
            (OUT5 / split / "labels" / label_name).write_text("\n".join(labels5) + "\n")

            # Save crop and labels for 3-class dataset
            crop.save(OUT3 / split / "images" / crop_name)
            (OUT3 / split / "labels" / label_name).write_text("\n".join(labels3) + "\n")

            stats[f"{split}_crops"] += 1
            stats["kept_crops_total"] += 1

            manifest.append({
                "source_label": label_path.name,
                "source_image": img_path.name,
                "crop": crop_name,
                "split": split,
                "table_index": t_idx,
                "table_box_xyxy": [round(tx1, 2), round(ty1, 2), round(tx2, 2), round(ty2, 2)],
                "num_5class_labels": len(labels5),
                "num_3class_labels": len(labels3),
            })

    write_yaml(OUT5, NAMES5)
    write_yaml(OUT3, NAMES3)

    (OUT5 / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT3 / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("=== DATASET BUILD COMPLETE ===")
    print()
    print("Input:")
    print("  Source pages:", stats["source_pages"])
    print("  Total Table boxes:", stats["table_boxes_total"])
    print()
    print("Table boxes per page:")
    for k, v in sorted(table_counts.items()):
        print(f"  {k} table boxes: {v} pages")

    print()
    print("Output crops:")
    print("  Kept crops total:", stats["kept_crops_total"])
    print("  Train crops:", stats["train_crops"])
    print("  Val crops:", stats["val_crops"])
    print("  Empty table crops skipped:", stats["empty_table_crops_skipped"])
    print("  Bad table boxes:", stats["bad_table_box"])
    print("  Missing images:", stats["missing_image"])
    print("  Bad images:", stats["bad_image"])

    print()
    print("5-class counts:")
    for i, name in enumerate(NAMES5):
        print(f"  {i} {name}: {cls5_counter[i]}")

    print()
    print("3-class counts:")
    for i, name in enumerate(NAMES3):
        print(f"  {i} {name}: {cls3_counter[i]}")

    if skipped_empty_crops:
        print()
        print("First skipped empty crops:")
        for x in skipped_empty_crops[:20]:
            print(" ", x)

    print()
    print("5-class YAML:", OUT5 / "data.yaml")
    print("3-class YAML:", OUT3 / "data.yaml")


if __name__ == "__main__":
    main()
