# Training Setup Table

| Model | Maximum epochs | Early stopping | Actual run behavior | Notes |
|---|---:|---|---|---|
| YOLOv8 v2 | 100 | Enabled, patience = 20 | Completed full 100 epochs | Early stopping was configured but did not terminate the final expanded-data run early. |
| YOLOv10 v2 | 100 | Enabled, patience = 20 | Completed full 100 epochs | Early stopping was configured but did not terminate the final expanded-data run early. |
| Nemotron v2 / YOLOX | 50 | Not used in the same way as YOLO | Fixed 50 epochs | Best checkpoint was used for evaluation. |
| TATR v2 | 50 | Not used | Fixed 50 epochs | Threshold sweep was performed after training. |

Important wording for the professor:
YOLOv8 and YOLOv10 were trained with early stopping enabled using patience = 20 and a maximum of 100 epochs. In the final expanded-data runs, early stopping did not terminate training early; both runs completed the full 100 epochs.
