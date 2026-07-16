# Revised Report Tables

## 1. YOLO 50 vs. 100 Annotation Comparison

| Model | Dataset stage | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8 v1 | Initial annotated set, evaluated on expanded validation split | 0.807 | 0.529 | 0.633 | 0.417 |
| YOLOv8 v2 | Fine-tuned with expanded 100-annotation dataset | 0.821 | 0.588 | 0.717 | 0.501 |
| YOLOv10 v1 | Initial annotated set, evaluated on expanded validation split | 0.643 | 0.466 | 0.510 | 0.353 |
| YOLOv10 v2 | Fine-tuned with expanded 100-annotation dataset | 0.863 | 0.493 | 0.581 | 0.417 |

Interpretation:
Both YOLO models improved after using the expanded annotation dataset. YOLOv8 v2 achieved the strongest full 5-class result based on mAP50-95. YOLOv10 v2 achieved higher precision but lower recall.

## 2. Nemotron Old vs. New Comparison

| Model | Dataset stage | Classes | AP50-95 | AP50 | AP75 | AR50-95 |
|---|---|---:|---:|---:|---:|---:|
| Nemotron fine-tuned v1 | Earlier 51-image diagnostic evaluation | 3 | 0.747 | 0.929 | - | - |
| Nemotron fine-tuned v2 | Expanded 100-annotation validation split | 3 | 0.718 | 0.911 | 0.809 | 0.791 |

Interpretation:
Nemotron remains strong for the core table-structure classes: Cell, Row, and Column. The old and new results should be interpreted as a development comparison, not a strict identical-test-set comparison, because the evaluation split changed.

## 3. TATR Old vs. New Comparison

| Model | Dataset stage | Evaluation setting | Best threshold | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| TATR fine-tuned v1 | Earlier annotated set | Full diagnostic | threshold sweep | - | - | ~0.211 |
| TATR fine-tuned v2 | Expanded 100-annotation set | Supported classes | 0.50 | 0.664 | 0.440 | 0.529 |
| TATR fine-tuned v2 | Expanded 100-annotation set | Full diagnostic | 0.40 | 0.556 | 0.219 | 0.314 |

Interpretation:
TATR improved after the expanded annotation set when evaluated on its supported structure classes. However, it is less suitable for direct full-schema Label Studio pre-annotation because it does not directly output Cell boxes.

## 4. IoU-Based Annotation Quality

| Model | GT boxes | Pred boxes | Class-aware F1@0.50 | Class-agnostic F1@0.50 | Mean matched IoU | Median matched IoU |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8 v2 | 475 | 601 | 0.792 | 0.825 | 0.862 | 0.909 |
| YOLOv10 v2 | 475 | 477 | 0.824 | 0.842 | 0.866 | 0.919 |
| TATR v2 @ 0.40 | 475 | 187 | 0.378 | 0.459 | 0.809 | 0.864 |
| Nemotron v2 @ 0.25 | 475 | 637 | 0.754 | 0.788 | 0.905 | 0.932 |

Interpretation:
YOLOv10 v2 gives the cleanest pre-annotation balance because it predicts almost the same number of boxes as the ground truth and has the highest class-aware and class-agnostic F1 at IoU 0.50. YOLOv8 v2 is also useful, but produces more extra boxes. Nemotron v2 has very high matched IoU, but produces more predictions and only covers the core 3-class setup. TATR v2 is not suitable for full cell-level pre-annotation because it does not directly predict Cell boxes.
