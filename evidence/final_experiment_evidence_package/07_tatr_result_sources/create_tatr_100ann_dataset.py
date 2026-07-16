import json
from pathlib import Path
from PIL import Image
from collections import Counter

ROOT = Path("/home/meeran/thesis_data")
SRC = ROOT / "dataset_100ann_5class"
OUT_DIR = ROOT / "tatr_finetune_100ann"
ANN_DIR = OUT_DIR / "annotations"
ANN_DIR.mkdir(parents=True, exist_ok=True)

# Source YOLO class IDs:
# 0 Row
# 1 Column
# 2 Cell
# 3 Header-Row
# 4 Spanning-Cell
#
# TATR native category IDs:
# 1 table column
# 2 table row
# 3 table column header
# 5 table spanning cell
YOLO_TO_TATR = {
    0: 2,  # Row -> table row
    1: 1,  # Column -> table column
    3: 3,  # Header-Row -> table column header
    4: 5,  # Spanning-Cell -> table spanning cell
}

TATR_CATEGORIES = [
    {"id": 0, "name": "table", "supercategory": "table"},
    {"id": 1, "name": "table column", "supercategory": "table"},
    {"id": 2, "name": "table row", "supercategory": "table"},
    {"id": 3, "name": "table column header", "supercategory": "table"},
    {"id": 4, "name": "table projected row header", "supercategory": "table"},
    {"id": 5, "name": "table spanning cell", "supercategory": "table"},
]

NAME = {
    1: "table column",
    2: "table row",
    3: "table column header",
    5: "table spanning cell",
}

def find_split_dirs(split):
    candidates = []

    # Layout A: images/train, labels/train
    candidates.append((SRC / "images" / split, SRC / "labels" / split))

    # Layout B: train/images, train/labels
    candidates.append((SRC / split / "images", SRC / split / "labels"))

    # Layout C: train/images, labels/train
    candidates.append((SRC / split / "images", SRC / "labels" / split))

    # Layout D: images/valid, labels/valid if split is val
    if split == "val":
        candidates.append((SRC / "images" / "valid", SRC / "labels" / "valid"))
        candidates.append((SRC / "valid" / "images", SRC / "valid" / "labels"))
        candidates.append((SRC / "valid" / "images", SRC / "labels" / "valid"))

    for img_dir, lbl_dir in candidates:
        imgs = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
        if img_dir.exists() and lbl_dir.exists() and len(imgs) > 0:
            return img_dir, lbl_dir

    print(f"\nERROR: Could not find valid {split} image/label folders.")
    print("Tried:")
    for img_dir, lbl_dir in candidates:
        print("  IMG:", img_dir, "exists=", img_dir.exists())
        print("  LBL:", lbl_dir, "exists=", lbl_dir.exists())
    raise SystemExit(1)

def yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h):
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    bw = w * img_w
    bh = h * img_h

    x = max(0.0, min(x, img_w))
    y = max(0.0, min(y, img_h))
    bw = max(0.0, min(bw, img_w - x))
    bh = max(0.0, min(bh, img_h - y))

    return [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)]

def build_split(split):
    img_dir, lbl_dir = find_split_dirs(split)

    print(f"\nUsing {split}:")
    print("  image dir:", img_dir)
    print("  label dir:", lbl_dir)

    image_files = sorted(
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.jpeg"))
    )

    images = []
    annotations = []
    counts = Counter()
    skipped_cells = 0
    missing_labels = 0
    bad_lines = 0
    ann_id = 1
    img_id = 1

    for img_path in image_files:
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            img_id += 1
            continue

        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                bad_lines += 1
                continue

            yolo_cls = int(float(parts[0]))

            # Cell is intentionally not used for TATR training.
            if yolo_cls == 2:
                skipped_cells += 1
                continue

            if yolo_cls not in YOLO_TO_TATR:
                continue

            tatr_cls = YOLO_TO_TATR[yolo_cls]
            cx, cy, w, h = map(float, parts[1:])
            bbox = yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h)
            area = round(bbox[2] * bbox[3], 2)

            if area <= 1:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": tatr_cls,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })

            counts[NAME[tatr_cls]] += 1
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": TATR_CATEGORIES,
    }

    out_path = ANN_DIR / f"{split}.json"
    out_path.write_text(json.dumps(coco, indent=2))

    print(f"\n=== {split} ===")
    print("images:", len(images))
    print("annotations used:", len(annotations))
    print("class counts:", dict(counts))
    print("skipped Cell annotations:", skipped_cells)
    print("missing labels:", missing_labels)
    print("bad lines:", bad_lines)
    print("saved:", out_path)

build_split("train")
build_split("val")

print("\nDone.")
print("Output:", OUT_DIR)
