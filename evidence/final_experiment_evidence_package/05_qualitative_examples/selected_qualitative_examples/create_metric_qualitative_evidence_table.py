from pathlib import Path

ROOT = Path("/home/meeran/thesis_data")
SEL = ROOT / "qualitative_analysis" / "selected_qualitative_examples"

selected = [
    {
        "no": 1,
        "base": "3e4cd12a___table_0",
        "file": "main/01_overall_model_comparison_3e4cd12a.png",
        "category": "Overall model comparison",
        "note": "YOLOv8, YOLOv10, and Nemotron have box counts close to the ground truth, while TATR predicts fewer boxes and lacks full cell-level detail."
    },
    {
        "no": 2,
        "base": "1de7d54e___table_0",
        "file": "main/02_yolov10_clean_preannotation_1de7d54e.png",
        "category": "YOLOv10 clean pre-annotation",
        "note": "YOLOv10 predicts a box count close to the ground truth and gives visually clean structure."
    },
    {
        "no": 3,
        "base": "38260563___table_4",
        "file": "main/03_nemotron_fine_granular_38260563_table4.png",
        "category": "Nemotron fine-granular prediction",
        "note": "Nemotron predicts many more boxes than the ground truth, showing fine-granular behavior."
    },
    {
        "no": 4,
        "base": "f2b794ad___table_0",
        "file": "main/04_precision_oversegmentation_f2b794ad.png",
        "category": "Precision / over-segmentation issue",
        "note": "YOLOv8 and Nemotron produce many extra boxes, explaining why precision can be lower than expected."
    },
    {
        "no": 5,
        "base": "080f9182___table_0",
        "file": "main/05_spanning_cell_rare_class_080f9182.png",
        "category": "Spanning-Cell / rare-class issue",
        "note": "Ground truth contains Spanning-Cell annotations; model behavior differs across architectures."
    },
    {
        "no": 6,
        "base": "d15fdefc___table_0",
        "file": "main/06_header_row_structural_ambiguity_d15fdefc.png",
        "category": "Header-Row / structural ambiguity",
        "note": "Header-like and section-like rows create ambiguity between Header-Row, Row, and Cell predictions."
    },
    {
        "no": 7,
        "base": "14abeab8___table_0",
        "file": "main/07_dense_global_layout_14abeab8.png",
        "category": "Dense/global-layout difficulty",
        "note": "Large landscape table with many rows and columns; global table structure and orientation remain difficult."
    },
]

sources = {
    "GT": [
        ROOT / "dataset_100ann_5class" / "val" / "labels",
    ],
    "YOLOv8": [
        ROOT / "final_overlay_predictions" / "yolov8_v2" / "labels",
    ],
    "YOLOv10": [
        ROOT / "final_overlay_predictions" / "yolov10_v2" / "labels",
    ],
    "TATR": [
        ROOT / "final_overlay_predictions" / "tatr_v2_thr_0p4" / "labels",
        ROOT / "final_overlay_predictions" / "tatr_v2" / "labels",
        ROOT / "tatr_finetuned_100ann_predictions" / "thr_0p4" / "labels",
        ROOT / "tatr_finetuned_100ann_predictions" / "thr_0p4",
    ],
    "Nemotron": [
        ROOT / "final_overlay_predictions" / "nemotron_v2" / "labels",
    ],
}

def count_boxes(base, dirs):
    for d in dirs:
        path = d / f"{base}.txt"
        if path.exists():
            text = path.read_text(errors="ignore").strip()
            if not text:
                return 0, str(path)
            return len(text.splitlines()), str(path)
    return 0, "missing prediction label file"

rows = []
for item in selected:
    counts = {}
    paths = {}
    for model, dirs in sources.items():
        count, src = count_boxes(item["base"], dirs)
        counts[model] = count
        paths[model] = src

    rows.append((item, counts, paths))

out = SEL / "metric_qualitative_evidence_table.md"

with out.open("w", encoding="utf-8") as f:
    f.write("# Metric + Qualitative Evidence Table\n\n")
    f.write("This table connects the selected qualitative examples with exact box counts from the label/prediction files.\n\n")

    f.write("## Main selected examples\n\n")
    f.write("| No. | Image | Category | GT boxes | YOLOv8 boxes | YOLOv10 boxes | TATR boxes | Nemotron boxes | Qualitative evidence |\n")
    f.write("|---:|---|---|---:|---:|---:|---:|---:|---|\n")

    for item, counts, paths in rows:
        f.write(
            f"| {item['no']} | `{item['file']}` | {item['category']} | "
            f"{counts['GT']} | {counts['YOLOv8']} | {counts['YOLOv10']} | {counts['TATR']} | {counts['Nemotron']} | "
            f"{item['note']} |\n"
        )

    f.write("\n## Source label files used for counts\n\n")
    f.write("| No. | Base name | GT source | YOLOv8 source | YOLOv10 source | TATR source | Nemotron source |\n")
    f.write("|---:|---|---|---|---|---|---|\n")

    for item, counts, paths in rows:
        f.write(
            f"| {item['no']} | `{item['base']}` | `{paths['GT']}` | `{paths['YOLOv8']}` | "
            f"`{paths['YOLOv10']}` | `{paths['TATR']}` | `{paths['Nemotron']}` |\n"
        )

    f.write("\n## Interpretation notes\n\n")
    f.write("1. A high predicted-box count can reduce precision because extra or duplicate boxes become false positives.\n")
    f.write("2. A low predicted-box count can indicate conservative behavior and may reduce recall.\n")
    f.write("3. Nemotron often has strong localization for matched boxes but can produce more fine-granular predictions than the annotation schema.\n")
    f.write("4. TATR is useful for row/column-style structure comparison but does not directly provide complete cell-level outputs.\n")
    f.write("5. The selected qualitative examples should be used to explain the quantitative metrics, not as standalone performance claims.\n")

print("Created:", out)
print("")
print(out.read_text())
