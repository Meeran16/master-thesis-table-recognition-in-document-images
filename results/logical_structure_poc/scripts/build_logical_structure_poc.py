from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import csv
import math
import argparse

CLASS_NAMES = {
    0: "Row",
    1: "Column",
    2: "Cell",
    3: "Header-Row",
    4: "Spanning-Cell",
}

IMAGE_DIR = Path("/home/meeran/thesis_data/dataset_100ann_5class/val/images")

SOURCES = {
    "ground_truth": {
        "label_dir": Path("/home/meeran/thesis_data/dataset_100ann_5class/val/labels"),
        "conf_threshold": 0.0,
        "apply_nms": False,
    },
    "yolov10_v2": {
        "label_dir": Path("/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels"),
        "conf_threshold": 0.25,
        "apply_nms": True,
    },
    "yolov8_v2": {
        "label_dir": Path("/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels"),
        "conf_threshold": 0.25,
        "apply_nms": True,
    },
}

OUT_DIR = Path("/home/meeran/thesis_data/logical_structure_poc")
JSON_DIR = OUT_DIR / "json"
HTML_DIR = OUT_DIR / "html"
OVERLAY_DIR = OUT_DIR / "overlays"

for d in [JSON_DIR, HTML_DIR, OVERLAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


def area(b):
    return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])


def iou(a, b):
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def nms_classwise(boxes, threshold=0.75):
    kept = []
    for cls in sorted(set(b["class_id"] for b in boxes)):
        cls_boxes = [b for b in boxes if b["class_id"] == cls]
        cls_boxes.sort(key=lambda b: b["conf"], reverse=True)

        while cls_boxes:
            current = cls_boxes.pop(0)
            kept.append(current)
            cls_boxes = [b for b in cls_boxes if iou(current, b) < threshold]

    return kept


def read_yolo_labels(label_file, image_file, conf_threshold=0.0, apply_nms=False):
    img = Image.open(image_file)
    img_w, img_h = img.size

    boxes = []
    if not label_file.exists():
        return boxes, img_w, img_h

    for line_no, line in enumerate(label_file.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) not in (5, 6):
            print(f"Skipping malformed line {line_no} in {label_file}: {line}")
            continue

        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) == 6 else 1.0

        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)

        boxes.append({
            "class_id": cls,
            "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "xc": (x1 + x2) / 2.0,
            "yc": (y1 + y2) / 2.0,
            "w": x2 - x1,
            "h": y2 - y1,
        })

    if apply_nms:
        boxes = nms_classwise(boxes, threshold=0.75)

    return boxes, img_w, img_h


def overlap_1d(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def derive_row_axes_from_cells(cells, img_w):
    if not cells:
        return []

    cells_sorted = sorted(cells, key=lambda b: b["yc"])
    heights = sorted([b["h"] for b in cells_sorted if b["h"] > 0])
    median_h = heights[len(heights) // 2] if heights else 20.0
    tol = max(10.0, median_h * 0.55)

    clusters = []
    for b in cells_sorted:
        placed = False
        for c in clusters:
            if abs(b["yc"] - c["yc_mean"]) <= tol:
                c["members"].append(b)
                c["yc_mean"] = sum(m["yc"] for m in c["members"]) / len(c["members"])
                c["y1"] = min(c["y1"], b["y1"])
                c["y2"] = max(c["y2"], b["y2"])
                placed = True
                break
        if not placed:
            clusters.append({
                "members": [b],
                "yc_mean": b["yc"],
                "y1": b["y1"],
                "y2": b["y2"],
            })

    rows = []
    for idx, c in enumerate(sorted(clusters, key=lambda x: x["yc_mean"]), start=1):
        rows.append({
            "axis_type": "derived_row",
            "index": idx,
            "x1": 0,
            "y1": c["y1"],
            "x2": img_w,
            "y2": c["y2"],
            "xc": img_w / 2,
            "yc": (c["y1"] + c["y2"]) / 2,
            "w": img_w,
            "h": c["y2"] - c["y1"],
            "source": "derived_from_cells",
        })
    return rows


def derive_col_axes_from_cells(cells, img_h):
    if not cells:
        return []

    cells_sorted = sorted(cells, key=lambda b: b["xc"])
    widths = sorted([b["w"] for b in cells_sorted if b["w"] > 0])
    median_w = widths[len(widths) // 2] if widths else 40.0
    tol = max(10.0, median_w * 0.55)

    clusters = []
    for b in cells_sorted:
        placed = False
        for c in clusters:
            if abs(b["xc"] - c["xc_mean"]) <= tol:
                c["members"].append(b)
                c["xc_mean"] = sum(m["xc"] for m in c["members"]) / len(c["members"])
                c["x1"] = min(c["x1"], b["x1"])
                c["x2"] = max(c["x2"], b["x2"])
                placed = True
                break
        if not placed:
            clusters.append({
                "members": [b],
                "xc_mean": b["xc"],
                "x1": b["x1"],
                "x2": b["x2"],
            })

    cols = []
    for idx, c in enumerate(sorted(clusters, key=lambda x: x["xc_mean"]), start=1):
        cols.append({
            "axis_type": "derived_column",
            "index": idx,
            "x1": c["x1"],
            "y1": 0,
            "x2": c["x2"],
            "y2": img_h,
            "xc": (c["x1"] + c["x2"]) / 2,
            "yc": img_h / 2,
            "w": c["x2"] - c["x1"],
            "h": img_h,
            "source": "derived_from_cells",
        })
    return cols


def axis_matches_for_cell(cell, axes, orientation):
    matches = []

    for axis in axes:
        if orientation == "row":
            ov = overlap_1d(cell["y1"], cell["y2"], axis["y1"], axis["y2"])
            denom = max(1.0, min(cell["h"], axis["h"]))
        else:
            ov = overlap_1d(cell["x1"], cell["x2"], axis["x1"], axis["x2"])
            denom = max(1.0, min(cell["w"], axis["w"]))

        ratio = ov / denom
        if ratio >= 0.35:
            matches.append((axis["index"], ratio))

    if not matches and axes:
        best = None
        best_score = -1
        for axis in axes:
            if orientation == "row":
                ov = overlap_1d(cell["y1"], cell["y2"], axis["y1"], axis["y2"])
            else:
                ov = overlap_1d(cell["x1"], cell["x2"], axis["x1"], axis["x2"])

            if ov > best_score:
                best_score = ov
                best = axis["index"]

        if best is not None:
            matches.append((best, 0.0))

    return sorted(set(idx for idx, _ in matches))


def build_logical_structure(example_id, source_name, boxes, img_w, img_h):
    row_boxes = [b for b in boxes if b["class_id"] in (0, 3)]
    col_boxes = [b for b in boxes if b["class_id"] == 1]
    cell_boxes = [b for b in boxes if b["class_id"] in (2, 4)]

    row_boxes = sorted(row_boxes, key=lambda b: b["yc"])
    col_boxes = sorted(col_boxes, key=lambda b: b["xc"])
    cell_boxes = sorted(cell_boxes, key=lambda b: (b["yc"], b["xc"]))

    rows = []
    for idx, b in enumerate(row_boxes, start=1):
        r = dict(b)
        r["index"] = idx
        r["axis_type"] = "detected_row"
        rows.append(r)

    cols = []
    for idx, b in enumerate(col_boxes, start=1):
        c = dict(b)
        c["index"] = idx
        c["axis_type"] = "detected_column"
        cols.append(c)

    if not rows:
        rows = derive_row_axes_from_cells(cell_boxes, img_w)

    if not cols:
        cols = derive_col_axes_from_cells(cell_boxes, img_h)

    logical_cells = []
    for cell_idx, b in enumerate(cell_boxes, start=1):
        row_ids = axis_matches_for_cell(b, rows, "row")
        col_ids = axis_matches_for_cell(b, cols, "column")

        if row_ids:
            row_start = min(row_ids)
            row_end = max(row_ids)
        else:
            row_start = row_end = None

        if col_ids:
            col_start = min(col_ids)
            col_end = max(col_ids)
        else:
            col_start = col_end = None

        logical_cells.append({
            "cell_id": cell_idx,
            "detected_class": b["class_name"],
            "confidence": round(b["conf"], 4),
            "row_start": row_start,
            "row_end": row_end,
            "column_start": col_start,
            "column_end": col_end,
            "rowspan": (row_end - row_start + 1) if row_start is not None and row_end is not None else None,
            "colspan": (col_end - col_start + 1) if col_start is not None and col_end is not None else None,
            "bbox_xyxy": [round(b["x1"], 1), round(b["y1"], 1), round(b["x2"], 1), round(b["y2"], 1)],
        })

    result = {
        "example_id": example_id,
        "source": source_name,
        "image_width": img_w,
        "image_height": img_h,
        "note": "Proof-of-concept heuristic. This is not a fully validated logical table reconstruction result.",
        "class_mapping": CLASS_NAMES,
        "row_axes_count": len(rows),
        "column_axes_count": len(cols),
        "cell_count": len(cell_boxes),
        "row_axes_source": rows[0].get("source", "detected_boxes") if rows else "none",
        "column_axes_source": cols[0].get("source", "detected_boxes") if cols else "none",
        "logical_cells": logical_cells,
    }

    return result, rows, cols, cell_boxes


def write_html(result, out_file):
    n_rows = max(1, result["row_axes_count"])
    n_cols = max(1, result["column_axes_count"])

    occupied = [[False for _ in range(n_cols)] for _ in range(n_rows)]
    placed = []
    unplaced = []

    cells = sorted(
        result["logical_cells"],
        key=lambda c: (
            999 if c["row_start"] is None else c["row_start"],
            999 if c["column_start"] is None else c["column_start"],
            c["cell_id"],
        )
    )

    for c in cells:
        if c["row_start"] is None or c["column_start"] is None:
            unplaced.append(c)
            continue

        r = c["row_start"] - 1
        col = c["column_start"] - 1
        rowspan = max(1, c["rowspan"] or 1)
        colspan = max(1, c["colspan"] or 1)

        if r >= n_rows or col >= n_cols or occupied[r][col]:
            unplaced.append(c)
            continue

        for rr in range(r, min(n_rows, r + rowspan)):
            for cc in range(col, min(n_cols, col + colspan)):
                occupied[rr][cc] = True

        placed.append((r, col, rowspan, colspan, c))

    cell_at = {(r, c): item for r, c, _, _, item in placed}

    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append("<style>")
    html.append("body{font-family:Arial,sans-serif;margin:24px;}")
    html.append("table{border-collapse:collapse;margin-top:16px;}")
    html.append("td,th{border:1px solid #333;padding:10px;min-width:90px;text-align:center;vertical-align:middle;}")
    html.append(".meta{font-size:14px;color:#333;}")
    html.append(".warn{color:#8a4b00;font-weight:bold;}")
    html.append("</style></head><body>")
    html.append(f"<h2>Logical table proof-of-concept: {result['example_id']} / {result['source']}</h2>")
    html.append("<p class='warn'>Proof-of-concept only. Not a fully validated logical table reconstruction result.</p>")
    html.append(f"<p class='meta'>Rows: {n_rows}, Columns: {n_cols}, Cells: {result['cell_count']}</p>")
    html.append("<table>")

    for r in range(n_rows):
        html.append("<tr>")
        for c in range(n_cols):
            if (r, c) in cell_at:
                cell = cell_at[(r, c)]
                rs = max(1, cell["rowspan"] or 1)
                cs = max(1, cell["colspan"] or 1)
                html.append(
                    f"<td rowspan='{rs}' colspan='{cs}'>"
                    f"Cell {cell['cell_id']}<br>"
                    f"{cell['detected_class']}<br>"
                    f"r{cell['row_start']}-r{cell['row_end']}, "
                    f"c{cell['column_start']}-c{cell['column_end']}"
                    f"</td>"
                )
            elif occupied[r][c]:
                continue
            else:
                html.append("<td>&nbsp;</td>")
        html.append("</tr>")

    html.append("</table>")

    if unplaced:
        html.append("<h3>Unplaced or conflicting detections</h3><ul>")
        for c in unplaced:
            html.append(
                f"<li>Cell {c['cell_id']}: {c['detected_class']}, "
                f"row={c['row_start']}-{c['row_end']}, "
                f"column={c['column_start']}-{c['column_end']}, "
                f"bbox={c['bbox_xyxy']}</li>"
            )
        html.append("</ul>")

    html.append("</body></html>")
    out_file.write_text("\n".join(html), encoding="utf-8")


def draw_overlay(image_file, out_file, rows, cols, cells, result):
    img = Image.open(image_file).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
        small_font = None

    # Draw row axes.
    for r in rows:
        draw.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=(255, 0, 0), width=3)
        draw.text((r["x1"] + 4, r["y1"] + 4), f"R{r['index']}", fill=(255, 0, 0), font=font)

    # Draw column axes.
    for c in cols:
        draw.rectangle([c["x1"], c["y1"], c["x2"], c["y2"]], outline=(0, 80, 255), width=3)
        draw.text((c["x1"] + 4, c["y1"] + 24), f"C{c['index']}", fill=(0, 80, 255), font=font)

    # Draw assigned cells.
    assigned = {c["cell_id"]: c for c in result["logical_cells"]}
    for idx, b in enumerate(cells, start=1):
        rec = assigned.get(idx, {})
        label = f"{idx}:r{rec.get('row_start')}-c{rec.get('column_start')}"
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=(0, 160, 0), width=2)
        draw.text((b["x1"] + 3, b["y1"] + 3), label, fill=(0, 120, 0), font=small_font)

    draw.rectangle([10, 10, 560, 105], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.text((20, 18), "Red: row axes | Blue: column axes | Green: cells", fill=(0, 0, 0), font=small_font)
    draw.text((20, 48), "Labels show heuristic row/column assignment", fill=(0, 0, 0), font=small_font)
    draw.text((20, 78), "Proof-of-concept only, not fully validated", fill=(120, 60, 0), font=small_font)

    img.save(out_file)


def process_example(example_id):
    image_file = IMAGE_DIR / f"{example_id}.png"
    if not image_file.exists():
        print(f"Missing image: {image_file}")
        return []

    summaries = []

    for source_name, cfg in SOURCES.items():
        label_file = cfg["label_dir"] / f"{example_id}.txt"
        if not label_file.exists():
            print(f"Skipping {source_name}/{example_id}: missing {label_file}")
            continue

        boxes, img_w, img_h = read_yolo_labels(
            label_file,
            image_file,
            conf_threshold=cfg["conf_threshold"],
            apply_nms=cfg["apply_nms"],
        )

        result, rows, cols, cells = build_logical_structure(example_id, source_name, boxes, img_w, img_h)

        json_file = JSON_DIR / f"{example_id}_{source_name}_logical_structure.json"
        html_file = HTML_DIR / f"{example_id}_{source_name}_logical_grid.html"
        overlay_file = OVERLAY_DIR / f"{example_id}_{source_name}_logical_overlay.png"

        json_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_html(result, html_file)
        draw_overlay(image_file, overlay_file, rows, cols, cells, result)

        summaries.append({
            "example_id": example_id,
            "source": source_name,
            "row_axes": result["row_axes_count"],
            "column_axes": result["column_axes_count"],
            "cells": result["cell_count"],
            "json": str(json_file),
            "html": str(html_file),
            "overlay": str(overlay_file),
        })

        print(f"Created: {source_name} / {example_id}")
        print(f"  JSON:    {json_file}")
        print(f"  HTML:    {html_file}")
        print(f"  Overlay: {overlay_file}")

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples",
        nargs="+",
        default=[
            "3e4cd12a___table_0",
            "f2b794ad___table_0",
            "080f9182___table_0",
            "14abeab8___table_0",
        ],
        help="Example IDs without .png or .txt extension",
    )
    args = parser.parse_args()

    all_summaries = []
    for ex in args.examples:
        all_summaries.extend(process_example(ex))

    summary_file = OUT_DIR / "logical_structure_poc_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["example_id", "source", "row_axes", "column_axes", "cells", "json", "html", "overlay"],
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    print("\nSummary written to:")
    print(summary_file)


if __name__ == "__main__":
    main()
