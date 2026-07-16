# Metric Interpretation Notes

## Why IoU was added

The IoU analysis was added because the professor asked about annotation usefulness in Label Studio. For pre-annotation, box quality is very important because correcting a class label is easier than manually resizing a bounding box.

## Class-aware vs. class-agnostic matching

Class-aware matching requires:
- good box overlap
- correct class label

Class-agnostic matching requires:
- good box overlap only

Class-agnostic matching is useful for estimating annotation effort. If the box is already correct but the class is wrong, the annotator only needs to change the label.

## Difference between mAP and the IoU annotation-quality table

The mAP values summarize detection performance across confidence rankings and IoU thresholds. The IoU annotation-quality table evaluates predictions at a fixed confidence setting using greedy matching. Therefore, it is more directly related to practical pre-annotation usefulness in Label Studio.

## Main interpretation

YOLOv8 v2 remains the strongest full 5-class model based on mAP50-95.

YOLOv10 v2 gives the cleanest pre-annotation balance based on IoU analysis because it predicts nearly the same number of boxes as the ground truth and achieves the highest class-aware and class-agnostic F1 at IoU 0.50.

Nemotron v2 gives the highest matched IoU, meaning its matched boxes are very well localized. However, it predicts more boxes and only covers the core 3 classes: Cell, Row, and Column.

TATR v2 is useful as a transformer-based comparison, especially for Column and Header-Row, but it is not suitable for direct full cell-level annotation because it does not directly output Cell boxes.
