# Learning Curve Summary

## YOLO training curves

| Model | Epochs completed | Best epoch | Best mAP50-95 | Last epoch | Last mAP50-95 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8 v2 | 100 | 93 | 0.5089 | 100 | 0.4963 | plateau/decline near the end |
| YOLOv10 v2 | 100 | 91 | 0.4202 | 100 | 0.4019 | plateau/decline near the end |

## TATR training curve

| Best epoch | Best eval loss | Last epoch | Last eval loss | Interpretation |
|---:|---:|---:|---:|---|
| 38 | 1.5533 | 50 | 1.5686 | eval loss improved earlier and then plateaued/slightly worsened |

## Nemotron training curve

| Best reported AP50-95 | Evaluation points found | Last eval AP50-95 found | Interpretation |
|---:|---:|---:|---|
| 0.7189 | 76 | 0.7200 | near plateau; final AP close to best reported AP |
