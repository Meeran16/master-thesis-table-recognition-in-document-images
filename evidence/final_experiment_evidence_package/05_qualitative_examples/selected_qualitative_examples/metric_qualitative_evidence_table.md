# Metric + Qualitative Evidence Table

This table connects the selected qualitative examples with exact box counts from the label/prediction files.

## Main selected examples

| No. | Image | Category | GT boxes | YOLOv8 boxes | YOLOv10 boxes | TATR boxes | Nemotron boxes | Qualitative evidence |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `main/01_overall_model_comparison_3e4cd12a.png` | Overall model comparison | 28 | 30 | 31 | 15 | 28 | YOLOv8, YOLOv10, and Nemotron have box counts close to the ground truth, while TATR predicts fewer boxes and lacks full cell-level detail. |
| 2 | `main/02_yolov10_clean_preannotation_1de7d54e.png` | YOLOv10 clean pre-annotation | 17 | 18 | 17 | 7 | 18 | YOLOv10 predicts a box count close to the ground truth and gives visually clean structure. |
| 3 | `main/03_nemotron_fine_granular_38260563_table4.png` | Nemotron fine-granular prediction | 6 | 1 | 3 | 4 | 31 | Nemotron predicts many more boxes than the ground truth, showing fine-granular behavior. |
| 4 | `main/04_precision_oversegmentation_f2b794ad.png` | Precision / over-segmentation issue | 24 | 65 | 29 | 14 | 44 | YOLOv8 and Nemotron produce many extra boxes, explaining why precision can be lower than expected. |
| 5 | `main/05_spanning_cell_rare_class_080f9182.png` | Spanning-Cell / rare-class issue | 17 | 21 | 20 | 9 | 24 | Ground truth contains Spanning-Cell annotations; model behavior differs across architectures. |
| 6 | `main/06_header_row_structural_ambiguity_d15fdefc.png` | Header-Row / structural ambiguity | 27 | 33 | 30 | 4 | 28 | Header-like and section-like rows create ambiguity between Header-Row, Row, and Cell predictions. |
| 7 | `main/07_dense_global_layout_14abeab8.png` | Dense/global-layout difficulty | 59 | 91 | 49 | 14 | 59 | Large landscape table with many rows and columns; global table structure and orientation remain difficult. |

## Source label files used for counts

| No. | Base name | GT source | YOLOv8 source | YOLOv10 source | TATR source | Nemotron source |
|---:|---|---|---|---|---|---|
| 1 | `3e4cd12a___table_0` | `/home/meeran/thesis_data/dataset_100ann_5class/val/labels/3e4cd12a___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels/3e4cd12a___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels/3e4cd12a___table_0.txt` | `/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4/3e4cd12a___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels/3e4cd12a___table_0.txt` |
| 2 | `1de7d54e___table_0` | `/home/meeran/thesis_data/dataset_100ann_5class/val/labels/1de7d54e___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels/1de7d54e___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels/1de7d54e___table_0.txt` | `/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4/1de7d54e___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels/1de7d54e___table_0.txt` |
| 3 | `38260563___table_4` | `/home/meeran/thesis_data/dataset_100ann_5class/val/labels/38260563___table_4.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels/38260563___table_4.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels/38260563___table_4.txt` | `/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4/38260563___table_4.txt` | `/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels/38260563___table_4.txt` |
| 4 | `f2b794ad___table_0` | `/home/meeran/thesis_data/dataset_100ann_5class/val/labels/f2b794ad___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels/f2b794ad___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels/f2b794ad___table_0.txt` | `/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4/f2b794ad___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels/f2b794ad___table_0.txt` |
| 5 | `080f9182___table_0` | `/home/meeran/thesis_data/dataset_100ann_5class/val/labels/080f9182___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels/080f9182___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels/080f9182___table_0.txt` | `/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4/080f9182___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels/080f9182___table_0.txt` |
| 6 | `d15fdefc___table_0` | `/home/meeran/thesis_data/dataset_100ann_5class/val/labels/d15fdefc___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels/d15fdefc___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels/d15fdefc___table_0.txt` | `/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4/d15fdefc___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels/d15fdefc___table_0.txt` |
| 7 | `14abeab8___table_0` | `/home/meeran/thesis_data/dataset_100ann_5class/val/labels/14abeab8___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov8_v2/labels/14abeab8___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/yolov10_v2/labels/14abeab8___table_0.txt` | `/home/meeran/thesis_data/tatr_finetuned_100ann_predictions/thr_0p4/14abeab8___table_0.txt` | `/home/meeran/thesis_data/final_overlay_predictions/nemotron_v2/labels/14abeab8___table_0.txt` |

## Interpretation notes

1. A high predicted-box count can reduce precision because extra or duplicate boxes become false positives.
2. A low predicted-box count can indicate conservative behavior and may reduce recall.
3. Nemotron often has strong localization for matched boxes but can produce more fine-granular predictions than the annotation schema.
4. TATR is useful for row/column-style structure comparison but does not directly provide complete cell-level outputs.
5. The selected qualitative examples should be used to explain the quantitative metrics, not as standalone performance claims.
