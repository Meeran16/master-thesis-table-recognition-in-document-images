from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IMG_DIR = Path("/home/meeran/thesis_data/dataset_100ann_5class/val/images")

SOURCES = {
    "ground_truth": Path("/home/meeran/thesis_data/dataset_100ann_5class/val/labels"),
    "yolov8_v2": Path("/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels"),
    "yolov10_v2": Path("/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels"),
    "tatr_v2_thr_0p4": Path("/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4"),
    "nemotron_v2": Path("/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels"),
}

DISPLAY_NAMES = {
    "ground_truth": "Ground Truth",
    "yolov8_v2": "YOLOv8 v2",
    "yolov10_v2": "YOLOv10 v2",
    "tatr_v2_thr_0p4": "TATR v2 @ 0.40",
    "nemotron_v2": "Nemotron v2 @ 0.25",
}

OUT_ROOT = Path("/home/meeran/thesis_data/final_overlays")
GRID_DIR = OUT_ROOT / "comparison_grid_5models"
GRID_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Row",
    1: "Column",
    2: "Cell",
    3: "Header-Row",
    4: "Spanning-Cell",
}

COLORS = {
    0: (255, 80, 80),
    1: (80, 160, 255),
    2: (80, 220, 120),
    3: (255, 180, 60),
    4: (190, 100, 255),
}

try:
    FONT_SMALL = ImageFont.truetype("DejaVuSans.ttf", 18)
except:
    FONT_SMALL = ImageFont.load_default()

def load_labels(label_path):
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cid = int(float(parts[0]))
        if cid not in CLASS_NAMES:
            continue

        cx, cy, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else None
        boxes.append((cid, cx, cy, w, h, conf))

    return boxes

def draw_overlay(img_path, label_path, title):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    W, H = img.size
    boxes = load_labels(label_path)

    draw.rectangle([0, 0, W, 42], fill=(255, 255, 255))
    draw.text((10, 8), f"{title} | boxes: {len(boxes)}", fill=(0, 0, 0), font=FONT_SMALL)

    for cid, cx, cy, w, h, conf in boxes:
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H

        color = COLORS.get(cid, (255, 255, 255))
        width = 3 if cid != 2 else 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label = CLASS_NAMES[cid]
        if conf is not None:
            label = f"{label} {conf:.2f}"

        text_y = max(45, y1 - 22)
        bbox = draw.textbbox((x1, text_y), label, font=FONT_SMALL)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, text_y), label, fill=(0, 0, 0), font=FONT_SMALL)

    return img

def resize_panel(img, width=760):
    W, H = img.size
    scale = width / W
    return img.resize((width, int(H * scale)))

def make_grid(panels):
    resized = [(title, resize_panel(img)) for title, img in panels]

    panel_w = max(img.size[0] for _, img in resized)
    panel_h = max(img.size[1] for _, img in resized)

    grid_w = panel_w * 3
    grid_h = panel_h * 2

    grid = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    positions = [
        (0, 0),
        (panel_w, 0),
        (panel_w * 2, 0),
        (0, panel_h),
        (panel_w, panel_h),
    ]

    for (title, img), (x, y) in zip(resized, positions):
        canvas = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        canvas.paste(img, (0, 0))
        grid.paste(canvas, (x, y))

    return grid

def main():
    image_files = sorted(
        list(IMG_DIR.glob("*.png")) +
        list(IMG_DIR.glob("*.jpg")) +
        list(IMG_DIR.glob("*.jpeg"))
    )

    print("Images:", len(image_files))

    for source_name in SOURCES:
        (OUT_ROOT / source_name).mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_files, 1):
        stem = img_path.stem
        overlays = {}

        for source_name, label_dir in SOURCES.items():
            label_path = label_dir / f"{stem}.txt"
            title = DISPLAY_NAMES[source_name]
            overlay = draw_overlay(img_path, label_path, title)

            out_path = OUT_ROOT / source_name / f"{stem}_{source_name}.png"
            overlay.save(out_path)
            overlays[source_name] = overlay

        grid = make_grid([
            ("Ground Truth", overlays["ground_truth"]),
            ("YOLOv8 v2", overlays["yolov8_v2"]),
            ("YOLOv10 v2", overlays["yolov10_v2"]),
            ("TATR v2 @ 0.40", overlays["tatr_v2_thr_0p4"]),
            ("Nemotron v2 @ 0.25", overlays["nemotron_v2"]),
        ])

        grid.save(GRID_DIR / f"{stem}_5model_comparison.png")

        if i % 5 == 0 or i == len(image_files):
            print(f"[{i}/{len(image_files)}] {img_path.name}")

    print("\nDone.")
    print("5-model comparison grids:", GRID_DIR)

if __name__ == "__main__":
    main()
