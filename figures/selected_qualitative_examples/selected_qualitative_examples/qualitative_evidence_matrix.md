# Qualitative Evidence Matrix

This file documents the selected qualitative examples for later use in the report, thesis discussion, or appendix. The goal is to keep each selected image tied to a clear model behavior or error category.

## Main qualitative examples

| No. | Image file | Evidence category | Main models involved | What the image shows | Why this matters |
|---:|---|---|---|---|---|
| 1 | `main/01_overall_model_comparison_3e4cd12a.png` | Overall model comparison | YOLOv8, YOLOv10, TATR, Nemotron | YOLOv8, YOLOv10, and Nemotron have box counts close to the ground truth. TATR captures coarse structure but lacks full cell-level detail. | This is the best general comparison figure because it shows the overall model behavior without being dominated by one extreme failure. |
| 2 | `main/02_yolov10_clean_preannotation_1de7d54e.png` | YOLOv10 clean pre-annotation | YOLOv10, YOLOv8 | YOLOv10 predicts a box count close to the ground truth and gives visually clean structure. | This supports the IoU-based finding that YOLOv10 gives the cleanest pre-annotation balance, even if YOLOv8 has stronger full 5-class mAP. |
| 3 | `main/03_nemotron_fine_granular_38260563_table4.png` | Nemotron fine-granular prediction | Nemotron | Ground truth has 6 boxes, while Nemotron predicts 31 boxes. The prediction is more fine-granular than the annotation schema. | This directly supports Professor Reiser’s observation that Nemotron produces more fine-granular bounding boxes in Label Studio. |
| 4 | `main/04_precision_oversegmentation_f2b794ad.png` | Precision / over-segmentation issue | YOLOv8, Nemotron | Ground truth has 24 boxes, while YOLOv8 predicts 65 boxes and Nemotron predicts 44 boxes. | This explains why precision can be lower than expected. Extra or duplicate boxes reduce precision even when the table structure looks visually simple. |
| 5 | `main/05_spanning_cell_rare_class_080f9182.png` | Spanning-Cell / rare-class issue | YOLOv8, YOLOv10, Nemotron, TATR | The ground truth contains Spanning-Cell annotations. The models differ in how well they capture this rare structural element. | This supports the class-imbalance discussion. Rare classes such as Spanning-Cell remain difficult because they appear much less often than Cell, Row, and Column. |
| 6 | `main/06_header_row_structural_ambiguity_d15fdefc.png` | Header-Row / structural ambiguity | YOLOv8, YOLOv10, TATR, Nemotron | Header-like and section-like rows create ambiguity between Header-Row, Row, and Cell predictions. | This supports the explanation that classification is not trivial because table regions are nested and structurally dependent. |
| 7 | `main/07_dense_global_layout_14abeab8.png` | Dense/global-layout difficulty | YOLOv8, YOLOv10, TATR, Nemotron | The table is wide and dense, with many rows and columns. Predictions show that global table structure and orientation remain difficult. | This supports the point that page-orientation and global-layout information are not explicitly modeled by YOLO-style detectors. |

## Backup examples

| No. | Image file | Possible use | Reason for backup status |
|---:|---|---|---|
| B1 | `backup/backup_01_simple_success_4631f8d1.png` | Simple clean success case | Useful if a minimal success example is needed, but it is less informative than image 1 or 2. |
| B2 | `backup/backup_02_yolov10_missing_boxes_9cf24198.png` | YOLOv10 conservative prediction / missing boxes | Useful if we need to discuss recall, but not essential for the main qualitative set. |
| B3 | `backup/backup_03_rare_structure_b4980366.png` | Alternative rare/spanning-style structure | Useful if image 5 is not clear enough later. |

## Error categories covered

| Error / behavior category | Covered by image no. |
|---|---|
| General model comparison | 1 |
| Clean pre-annotation behavior | 2 |
| Nemotron fine-granular prediction | 3 |
| Over-segmentation and low precision | 4 |
| Rare Spanning-Cell behavior | 5 |
| Header-Row / structural ambiguity | 6 |
| Dense/global-layout difficulty | 7 |
| TATR missing full cell-level output | 1, 5, 6, 7 |

## Key interpretation notes

1. The qualitative examples should not be used as separate experimental results. They support and explain the quantitative metrics.
2. YOLOv8 is useful for full 5-class table-structure detection, but some examples show over-segmentation.
3. YOLOv10 often gives cleaner pre-annotation behavior because it predicts fewer extra boxes.
4. Nemotron can produce well-localized boxes, but its predictions are often more fine-granular than the ground-truth annotation schema.
5. TATR is useful as a transformer-based comparison, but it does not directly produce individual Cell boxes.
6. Header-Row and Spanning-Cell remain difficult due to class imbalance and structural ambiguity.
7. Row, Column, and Cell are not independent visual objects. They are nested structural concepts, which explains why classification can remain difficult.
