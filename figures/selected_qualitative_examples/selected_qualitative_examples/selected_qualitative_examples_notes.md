# Selected Qualitative Examples

This folder contains the selected qualitative examples for analyzing model behavior. The examples were selected after reviewing all 26 validation overlay grids.

## Main selected examples

| No. | File | Category | Main observation | Why selected |
|---:|---|---|---|---|
| 1 | `01_overall_model_comparison_3e4cd12a.png` | Overall model comparison | YOLOv8, YOLOv10, and Nemotron have box counts close to the ground truth, while TATR predicts fewer boxes and lacks full cell-level detail. | Best general comparison figure because it shows normal model behavior without an extreme failure case. |
| 2 | `02_yolov10_clean_preannotation_1de7d54e.png` | YOLOv10 clean pre-annotation | Ground truth has 17 boxes and YOLOv10 also predicts 17 boxes with visually clean structure. | Supports the argument that YOLOv10 gives clean pre-annotation balance. |
| 3 | `03_nemotron_fine_granular_38260563_table4.png` | Nemotron fine-granular prediction | Ground truth has 6 boxes, while Nemotron predicts 31 boxes. | Directly supports the observation that Nemotron produces more fine-granular boxes. |
| 4 | `04_precision_oversegmentation_f2b794ad.png` | Precision / over-segmentation issue | Ground truth has 24 boxes, while YOLOv8 predicts 65 boxes and Nemotron predicts 44 boxes. | Explains why precision is lower than expected despite visually simple table structures. |
| 5 | `05_spanning_cell_rare_class_080f9182.png` | Spanning-Cell / rare-class issue | Ground truth contains Spanning-Cell annotations; model behavior differs across architectures. | Useful for discussing rare structural classes. |
| 6 | `06_header_row_structural_ambiguity_d15fdefc.png` | Header-Row / structural ambiguity | Header-like and section-like rows create ambiguity between Header-Row, Row, and Cell predictions. | Supports the explanation that classification is not trivial. |
| 7 | `07_dense_global_layout_14abeab8.png` | Dense/global-layout difficulty | Large landscape table with many rows and columns; model predictions show that global table structure and orientation remain difficult. | Supports the point about lost page-orientation/global-layout information. |

## Backup examples

| No. | File | Possible use |
|---:|---|---|
| B1 | `backup_01_simple_success_4631f8d1.png` | Simple clean success case. |
| B2 | `backup_02_yolov10_missing_boxes_9cf24198.png` | YOLOv10 conservative prediction / missing boxes. |
| B3 | `backup_03_rare_structure_b4980366.png` | Alternative rare/spanning-style structure. |
