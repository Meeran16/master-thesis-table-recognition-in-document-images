# Final Experimental Results Summary

## Dataset

The final expanded dataset contains 100 annotated pages and 112 cropped table instances. The final validation split contains 26 cropped table images with 475 total annotations.

## YOLO 5-class comparison

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8 v1 | 0.807 | 0.529 | 0.633 | 0.417 |
| YOLOv8 v2 | 0.821 | 0.588 | 0.717 | 0.501 |
| YOLOv10 v1 | 0.643 | 0.466 | 0.510 | 0.353 |
| YOLOv10 v2 | 0.863 | 0.493 | 0.581 | 0.417 |

YOLOv8 v2 achieved the best overall 5-class performance. YOLOv10 v2 achieved the highest precision but lower recall.

## Nemotron v2 3-class result

| Metric | Value |
|---|---:|
| AP50-95 | 0.718 |
| AP50 | 0.911 |
| AP75 | 0.809 |
| AR50-95 | 0.791 |

| Class | AP |
|---|---:|
| Cell | 0.738 |
| Row | 0.714 |
| Column | 0.702 |

Nemotron v2 achieved strong results on the core 3-class table-structure task.

## TableTransformer / TATR v2 result

| Setting | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Best supported-class result | 0.50 | 0.664 | 0.440 | 0.529 |
| Best full diagnostic result | 0.40 | 0.556 | 0.219 | 0.314 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Row | 0.434 | 0.295 | 0.351 |
| Column | 0.602 | 0.787 | 0.682 |
| Cell | 0.000 | 0.000 | 0.000 |
| Header-Row | 1.000 | 0.857 | 0.923 |
| Spanning-Cell | 0.000 | 0.000 | 0.000 |

TATR performs well for column and header-row detection, but it does not directly predict individual cell boxes and remains weak on spanning-cell detection.

## Main conclusion

YOLOv8 v2 is the best practical full 5-class table-structure detector for this dataset. Nemotron v2 performs strongly on the reduced 3-class core structure task. TATR provides a useful transformer-based comparison, but its output design is less suitable for direct cell-level detection.
