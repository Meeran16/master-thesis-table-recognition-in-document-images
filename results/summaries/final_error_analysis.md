# Final Error Analysis

## Overview

The final experiments show clear differences between object-detection-based models and the transformer-based TableTransformer model. YOLOv8 v2 achieved the strongest full 5-class performance, while Nemotron v2 achieved strong performance on the reduced 3-class core structure task. TableTransformer provided a useful transformer-based comparison, but its output design was less suitable for direct cell-level detection.

## YOLOv8 v2

YOLOv8 v2 achieved the best overall 5-class result, with the highest recall, mAP50, and mAP50-95 among the YOLO-family models. This indicates that YOLOv8 adapted well to the expanded historical table dataset.

Main strengths:
- Strong Cell detection.
- Improved Column detection after dataset expansion.
- Best overall full 5-class performance.

Main weaknesses:
- Header-Row performance is still affected by ambiguity between normal rows and header rows.
- Spanning-Cell remains difficult because the class has very few training examples.
- Some complex tables still show over-segmentation or missed row boundaries.

## YOLOv10 v2

YOLOv10 v2 achieved the highest precision, but lower recall than YOLOv8 v2. This means YOLOv10 was more conservative: when it predicted boxes, they were often reliable, but it missed more structures.

Main strengths:
- Highest precision among YOLO models.
- Cleaner predictions in some images.
- Good Cell and Column localization.

Main weaknesses:
- Lower recall than YOLOv8 v2.
- Missed detections on some small or narrow cropped tables.
- Weaker overall mAP than YOLOv8 v2.

## Nemotron v2

Nemotron v2 performed strongly on the reduced 3-class core table-structure task: Cell, Row, and Column. It achieved high AP50 and strong AP50-95, showing that the YOLOX/Nemotron setup adapted well to the core structure classes.

Main strengths:
- Strong Cell, Row, and Column AP.
- High AP50, indicating good localization at IoU 0.5.
- Useful comparison against YOLOv8 and YOLOv10 for core structure detection.

Main weaknesses:
- It was evaluated only on three classes: Cell, Row, and Column.
- Header-Row and Spanning-Cell were not included in the Nemotron setup.
- Therefore, Nemotron is not directly comparable to the full 5-class YOLO evaluation.

## TableTransformer / TATR v2

TATR v2 was fine-tuned on the expanded 100-annotation dataset using its native structure classes: Row, Column, Header-Row, and Spanning-Cell. Cell was retained in the full diagnostic evaluation to show that TATR does not directly output individual cell boxes.

Main strengths:
- Strong Header-Row result.
- Good Column detection.
- Useful transformer-based comparison.

Main weaknesses:
- No direct Cell detection.
- Weak Row recall.
- Spanning-Cell remained undetected or poorly detected due to very low class frequency.
- Overall diagnostic F1 is reduced because Cell is unsupported as a direct output class.

## Dataset-related limitations

The dataset contains historical regulatory document images with varied table layouts, scan quality, and structural complexity. These factors make table structure recognition more difficult than modern clean table datasets.

Important limitations:
- Spanning-Cell has very few examples.
- Header-Row is visually ambiguous in many tables.
- Some tables are narrow, fragmented, or split across page regions.
- The validation split is small, with 26 cropped table images.
- Models trained on modern/general table data may not transfer directly to historical documents.

## Main conclusion from error analysis

The errors show that object-detection models are more suitable for direct cell-level table-structure detection in this dataset. YOLOv8 v2 is the strongest full 5-class model. Nemotron v2 is strong on the core 3-class structure task, while TableTransformer is useful for row-column structure comparison but less suitable for direct cell-level detection.
