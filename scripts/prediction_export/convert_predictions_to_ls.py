#!/usr/bin/env python3
"""
Convert model predictions to Label Studio prediction import format.

For each of the 4 models, reads predictions (YOLO txt format or Nemotron JSON),
converts coordinates from cropped-image space to full-page space using the
mapping file, and outputs Label Studio prediction JSON files.

Two output formats per model:
  1. *_ls_import.json  — Professor's format (data + predictions) for UI import
  2. *_ls_api.json     — API format (task_id + predictions) for programmatic import
"""

import json
import os
from pathlib import Path
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────
MAPPING_FILE  = Path("/home/meeran/thesis_data/cropped_to_ls_mapping_resolved.json")
CROPPED_DIR   = Path("/home/meeran/thesis_data/cropped_images")
OUTPUT_DIR    = Path("/home/meeran/thesis_data/ls_predictions")

# ── Model configs ────────────────────────────────────────────────────────
MODELS = {
    "yolov8_finetuned": {
        "dir": Path("/home/meeran/thesis_data/yolov8_baseline"),
        "class_map": {0: "Row", 1: "Column", 2: "Cell", 3: "Header-Row"},
    },
    "yolov10_finetuned": {
        "dir": Path("/home/meeran/thesis_data/yolov10_baseline"),
        "class_map": {0: "Row", 1: "Column", 2: "Cell", 3: "Header-Row"},
    },
    "nemotron_pretrained": {
        "dir": Path("/home/meeran/thesis_data/nemotron_baseline"),
        "class_map": {1: "Cell", 2: "Row", 3: "Column"},
        "predictions_json": "predictions.json",   # has per-box scores
    },
    "nemotron_finetuned": {
        "dir": Path("/home/meeran/thesis_data/nemotron_finetuned"),
        "class_map": {1: "Cell", 2: "Row", 3: "Column"},
        "predictions_json": "predictions.json",   # has per-box scores
    },
    "tabletransformer_pretrained": {
        "dir": Path("/home/meeran/thesis_data/tabletransformer_baseline"),
        "class_map": {0: "Row", 1: "Column", 3: "Header-Row", 4: "Spanning-Cell"},
    },
}

# ── Coordinate math ──────────────────────────────────────────────────────

def compute_crop_region_pct(orig_w, orig_h, crop_px_w, crop_px_h, table_pct):
    """
    Derive the exact full-page crop region (in LS percentage coords)
    from the Table annotation box plus the actual cropped-image pixel size.
    """
    # Table box in pixels on the full page
    tbl_x = table_pct["x"]   / 100.0 * orig_w
    tbl_y = table_pct["y"]   / 100.0 * orig_h
    tbl_w = table_pct["width"]  / 100.0 * orig_w
    tbl_h = table_pct["height"] / 100.0 * orig_h

    # Extra pixels = padding around the table
    pad_w = crop_px_w - tbl_w
    pad_h = crop_px_h - tbl_h

    # Symmetric padding (may be clamped at image edges)
    cx = max(0.0, tbl_x - pad_w / 2.0)
    cy = max(0.0, tbl_y - pad_h / 2.0)
    cw = min(crop_px_w, orig_w - cx)
    ch = min(crop_px_h, orig_h - cy)

    return {
        "x":      cx / orig_w * 100.0,
        "y":      cy / orig_h * 100.0,
        "width":  cw / orig_w * 100.0,
        "height": ch / orig_h * 100.0,
    }


def cropped_yolo_to_ls_pct(xc, yc, w, h, crop_pct):
    """
    YOLO normalised (0-1, centre format, relative to *cropped* image)
    → Label Studio percentage (0-100, top-left format, relative to *full page*).
    """
    # left-top in normalised cropped space
    xl = xc - w / 2.0
    yt = yc - h / 2.0

    # map to full-page percentage
    x_ls  = crop_pct["x"]      + xl * crop_pct["width"]
    y_ls  = crop_pct["y"]      + yt * crop_pct["height"]
    w_ls  = w * crop_pct["width"]
    h_ls  = h * crop_pct["height"]

    return round(x_ls, 2), round(y_ls, 2), round(w_ls, 2), round(h_ls, 2)


# ── Main conversion ─────────────────────────────────────────────────────

def main():
    # Load mapping
    with open(MAPPING_FILE) as f:
        mapping_data = json.load(f)
    mapping = mapping_data["mapping"]

    print(f"Loaded mapping: {len(mapping)} cropped images")

    # Pre-compute crop regions from actual image dimensions
    crop_regions = {}
    for stem, info in mapping.items():
        img_path = CROPPED_DIR / f"{stem}.png"
        if not img_path.exists():
            print(f"  WARNING: image missing {img_path}")
            continue
        cw, ch = Image.open(img_path).size
        crop_regions[stem] = compute_crop_region_pct(
            info["original_width"], info["original_height"],
            cw, ch,
            info["table_crop_region"],
        )

    print(f"Crop regions computed: {len(crop_regions)}")

    # Verify one entry
    sample_stem = list(crop_regions.keys())[0]
    sample_info = mapping[sample_stem]
    sample_crop = crop_regions[sample_stem]
    print(f"\nVerification ({sample_stem}):")
    print(f"  Original: {sample_info['original_width']}x{sample_info['original_height']}")
    print(f"  Image:    {CROPPED_DIR / (sample_stem + '.png')}")
    cw, ch = Image.open(CROPPED_DIR / f"{sample_stem}.png").size
    print(f"  Cropped:  {cw}x{ch}")
    print(f"  Table:    {sample_info['table_crop_region']}")
    print(f"  Crop%:    x={sample_crop['x']:.2f} y={sample_crop['y']:.2f} "
          f"w={sample_crop['width']:.2f} h={sample_crop['height']:.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Convert each model ──────────────────────────────────────────────
    for model_name, mcfg in MODELS.items():
        model_dir  = mcfg["dir"]
        class_map  = mcfg["class_map"]

        print(f"\n{'='*60}")
        print(f"Converting: {model_name}")
        print(f"{'='*60}")

        # Optional: Nemotron predictions.json (has per-box scores)
        nemotron_json = None
        if "predictions_json" in mcfg:
            pj = model_dir / mcfg["predictions_json"]
            if pj.exists():
                with open(pj) as f:
                    nemotron_json = json.load(f)
                print(f"  Loaded {pj.name}: {len(nemotron_json)} images")

        ls_import_list = []   # professor's format (for UI)
        api_pred_list  = []   # API format (for POST /api/predictions/)
        total_boxes = 0
        skipped = 0

        for stem in sorted(mapping.keys()):
            if stem not in crop_regions:
                skipped += 1
                continue

            crop_pct = crop_regions[stem]
            info     = mapping[stem]

            # ── Read predictions ─────────────────────────────────────
            if nemotron_json is not None:
                png_key = f"{stem}.png"
                if png_key not in nemotron_json:
                    skipped += 1
                    continue
                raw = nemotron_json[png_key]
                preds = []
                for r in raw:
                    label = r["label"]
                    xmin, ymin = r["xmin"], r["ymin"]
                    xmax, ymax = r["xmax"], r["ymax"]
                    score = r.get("score", 0.5)
                    w = xmax - xmin
                    h = ymax - ymin
                    xc = xmin + w / 2.0
                    yc = ymin + h / 2.0
                    preds.append((label, xc, yc, w, h, score))
            else:
                pred_file = model_dir / f"{stem}.txt"
                if not pred_file.exists():
                    skipped += 1
                    continue
                preds = []
                with open(pred_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(float(parts[0]))
                            xc, yc, w, h = map(float, parts[1:5])
                            cname = class_map.get(cid, f"Unknown-{cid}")
                            preds.append((cname, xc, yc, w, h, 0.5))

            if not preds:
                skipped += 1
                continue

            # ── Build result array ───────────────────────────────────
            results = []
            for i, (cname, xc, yc, w, h, score) in enumerate(preds):
                x_ls, y_ls, w_ls, h_ls = cropped_yolo_to_ls_pct(
                    xc, yc, w, h, crop_pct
                )
                # skip degenerate boxes
                if w_ls <= 0 or h_ls <= 0:
                    continue
                # clamp negative coords
                x_ls = max(0.0, x_ls)
                y_ls = max(0.0, y_ls)

                results.append({
                    "id": f"result_{i}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": float(info["original_width"]),
                    "original_height": float(info["original_height"]),
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": x_ls,
                        "y": y_ls,
                        "width": w_ls,
                        "height": h_ls,
                        "rectanglelabels": [cname],
                    },
                    "score": round(float(score), 4),
                })

            if not results:
                skipped += 1
                continue

            total_boxes += len(results)
            avg_score = sum(r["score"] for r in results) / len(results)

            # Professor's import format (data + predictions)
            ls_import_list.append({
                "data": {
                    "image": info["image_url"],
                    "category": info["category"],
                },
                "predictions": [{
                    "model_version": model_name,
                    "score": round(avg_score, 4),
                    "result": results,
                }],
            })

            # API format (task + result + score + model_version)
            api_pred_list.append({
                "task": info["task_id"],
                "result": results,
                "score": round(avg_score, 4),
                "model_version": model_name,
            })

        # ── Save both formats ────────────────────────────────────────
        ui_file = OUTPUT_DIR / f"{model_name}_ls_import.json"
        api_file = OUTPUT_DIR / f"{model_name}_ls_api.json"

        with open(ui_file, "w") as f:
            json.dump(ls_import_list, f, indent=2)

        with open(api_file, "w") as f:
            json.dump(api_pred_list, f, indent=2)

        print(f"  Tasks: {len(ls_import_list)}  Boxes: {total_boxes}  Skipped: {skipped}")
        print(f"  UI import:  {ui_file}")
        print(f"  API import: {api_file}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Verify a few predictions visually in Label Studio")
    print("  2. For UI import: use *_ls_import.json files")
    print("  3. For API import: use *_ls_api.json files with the script")
    print("     (we will create the API import script next)")


if __name__ == "__main__":
    main()
