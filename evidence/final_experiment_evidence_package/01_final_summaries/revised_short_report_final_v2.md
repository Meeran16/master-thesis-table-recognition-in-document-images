# Updated Short Report: Additional Data Experiments and IoU-Based Annotation Quality

**Project:** Table Recognition in Document Images  
**Student:** Meeran Mydeen Syed Ibrahim  
**Focus:** Table structure recognition in historical regulatory document images

## 1. Purpose of this update

This update extends the previous short report by adding the missing comparison for Nemotron and TableTransformer/TATR on the expanded annotation dataset. It also adds an IoU-based annotation-quality analysis, since box localization quality is important for pre-annotation in Label Studio.

The main reason for adding the IoU analysis is that correcting a class label in Label Studio is much faster than manually correcting a bounding box. Therefore, a model can still be useful for annotation if the predicted boxes are well localized, even when some class labels need correction.

## 2. Dataset and evaluation setup

The expanded dataset contains 100 annotated pages and 112 cropped table instances. The final validation split contains 26 cropped table images with 475 ground-truth structure annotations.

The full 5-class table-structure setup contains:

- Row
- Column
- Cell
- Header-Row
- Spanning-Cell

YOLOv8 and YOLOv10 were evaluated in this full 5-class setup. Nemotron/YOLOX was evaluated on the core 3-class setup: Cell, Row, and Column. TATR was evaluated on its supported structure classes: Row, Column, Header-Row, and Spanning-Cell. Cell was retained in the diagnostic evaluation to show that TATR does not directly output individual cell boxes.

## 3. YOLO comparison before and after the expanded annotation dataset

| Model | Dataset stage | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8 v1 | Initial annotated set, evaluated on expanded validation split | 0.807 | 0.529 | 0.633 | 0.417 |
| YOLOv8 v2 | Fine-tuned with expanded dataset of 100 annotated pages / 112 cropped table instances | 0.821 | 0.588 | 0.717 | 0.501 |
| YOLOv10 v1 | Initial annotated set, evaluated on expanded validation split | 0.643 | 0.466 | 0.510 | 0.353 |
| YOLOv10 v2 | Fine-tuned with expanded dataset of 100 annotated pages / 112 cropped table instances | 0.863 | 0.493 | 0.581 | 0.417 |

Both YOLO models improved after using the expanded annotation dataset. YOLOv8 v2 achieved the strongest full 5-class result based on mAP50-95. YOLOv10 v2 achieved higher precision, but lower recall.

## 4. Nemotron comparison

| Model | Dataset stage | Classes | AP50-95 | AP50 | AP75 | AR50-95 |
|---|---|---:|---:|---:|---:|---:|
| Nemotron fine-tuned v1 | Earlier 51-image diagnostic evaluation | 3 | 0.747 | 0.929 | - | - |
| Nemotron fine-tuned v2 | Expanded validation split from 100 annotated pages / 112 cropped table instances | 3 | 0.718 | 0.911 | 0.809 | 0.791 |

Nemotron remains strong for the core table-structure classes: Cell, Row, and Column. The old and new results should be interpreted as a development comparison, not as a strict identical-test-set comparison, because the evaluation split changed.

The new Nemotron v2 result also gives the following per-class AP values:

| Class | AP |
|---|---:|
| Cell | 0.738 |
| Row | 0.714 |
| Column | 0.702 |

## 5. TableTransformer/TATR comparison

| Model | Dataset stage | Evaluation setting | Best threshold | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| TATR fine-tuned v1 | Earlier annotated set | Full diagnostic | threshold sweep | - | - | ~0.211 |
| TATR fine-tuned v2 | Expanded dataset of 100 annotated pages / 112 cropped table instances | Supported classes | 0.50 | 0.664 | 0.440 | 0.529 |
| TATR fine-tuned v2 | Expanded dataset of 100 annotated pages / 112 cropped table instances | Full diagnostic | 0.40 | 0.556 | 0.219 | 0.314 |

TATR improved after the expanded annotation set when evaluated on its supported structure classes. However, it is less suitable for direct full-schema Label Studio pre-annotation because it does not directly output Cell boxes. This explains why the full diagnostic score remains lower.

## 6. IoU-based annotation-quality analysis

The IoU analysis evaluates how useful the predictions are for pre-annotation. Two matching settings were used:

- **Class-aware matching:** the predicted box must overlap with the ground truth and have the correct class.
- **Class-agnostic matching:** the predicted box only needs to overlap with the ground truth, regardless of class.

Class-agnostic matching is useful for estimating annotation effort. If the box is correct but the label is wrong, the annotator only needs to change the class label. However, the number of predicted boxes must also be considered, because too many predicted boxes can increase manual correction effort.

| Model | GT boxes | Pred boxes | Class-aware F1@0.50 | Class-agnostic F1@0.50 | Mean matched IoU | Median matched IoU | Practical interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLOv8 v2 | 475 | 601 | 0.792 | 0.825 | 0.862 | 0.909 | Good localization and recall, but more extra boxes than YOLOv10. |
| YOLOv10 v2 | 475 | 477 | 0.824 | 0.842 | 0.866 | 0.919 | Best overall pre-annotation balance: highest F1 values and box count closest to ground truth. |
| TATR v2 @ 0.40 | 475 | 187 | 0.378 | 0.459 | 0.809 | 0.864 | Too few predictions for full-schema pre-annotation and no direct Cell output. |
| Nemotron v2 @ 0.25 | 475 | 637 | 0.754 | 0.788 | 0.905 | 0.932 | Highest matched IoU, but many more and more fine-granular predicted boxes. |

YOLOv10 v2 gives the cleanest overall pre-annotation balance because it predicts almost the same number of boxes as the ground truth and has the highest class-aware and class-agnostic F1 at IoU 0.50. YOLOv8 v2 is also useful, but it produces more extra boxes.

Nemotron v2 has the highest mean and median matched IoU, which means that the boxes that can be matched to ground truth are very well localized. However, it also predicts many more bounding boxes than the ground truth. This suggests that the Nemotron predictions are more fine-granular, which matches the visual observation in Label Studio. Therefore, Nemotron should be interpreted as having strong localization for matched boxes, but not as the cleanest pre-annotation output.

TATR v2 is not suitable for full cell-level pre-annotation because it does not directly predict Cell boxes.

## 7. Training setup

| Model | Maximum epochs | Early stopping | Actual run behavior | Notes |
|---|---:|---|---|---|
| YOLOv8 v2 | 100 | Enabled, patience = 20 | Completed full 100 epochs | Early stopping was configured but did not terminate the final expanded-data run early. |
| YOLOv10 v2 | 100 | Enabled, patience = 20 | Completed full 100 epochs | Early stopping was configured but did not terminate the final expanded-data run early. |
| Nemotron v2 / YOLOX | 50 | Fixed-epoch training | Fixed 50 epochs | Best checkpoint was used for evaluation. |
| TATR v2 | 50 | No early-stopping criterion applied | Fixed 50 epochs | Threshold sweep was performed after training. |

YOLOv8 and YOLOv10 were trained with early stopping enabled using patience = 20 and a maximum of 100 epochs. In the final expanded-data runs, early stopping did not terminate training early; both runs completed the full 100 epochs.

## 8. Learning-curve observations

The learning curves were checked to see whether the models were still improving at the end of training.

| Model | Best point | Final point | Interpretation |
|---|---|---|---|
| YOLOv8 v2 | Best mAP50-95 at epoch 93: 0.5089 | Epoch 100: 0.4963 | Plateaued or slightly declined near the end. |
| YOLOv10 v2 | Best mAP50-95 at epoch 91: 0.4202 | Epoch 100: 0.4019 | Plateaued or slightly declined near the end. |
| TATR v2 | Best eval loss at epoch 38: 1.5533 | Epoch 50 eval loss: 1.5686 | Eval loss improved earlier and then slightly worsened. |
| Nemotron v2 | Best reported AP50-95 around 0.7189 | Final parsed AP50-95 around 0.7200 | Close to plateau; final AP was close to the best reported AP. |

The learning curves do not suggest that the models were still strongly improving at the end. YOLOv8 and YOLOv10 reached their best mAP50-95 before the final epoch. TATR reached its best evaluation loss around epoch 38 and then slightly worsened by epoch 50. Nemotron was close to plateau, with the final AP close to the best reported AP.

## 9. Conclusion

The expanded annotation dataset improved the YOLO models and also improved TATR when evaluated on its supported structure classes. YOLOv8 v2 remains the strongest full 5-class model based on mAP50-95. YOLOv10 v2 gives the cleanest pre-annotation balance based on IoU analysis, while Nemotron v2 gives the highest matched IoU but with more fine-granular predicted boxes.

Nemotron v2 provides very strong box localization for the core classes Cell, Row, and Column, but it is not directly comparable to the full 5-class YOLO setup. TATR remains useful as a transformer-based comparison, especially for Column and Header-Row, but it is not suitable for direct full cell-level annotation because it does not directly output Cell boxes.

Overall, the additional analysis shows that YOLO-based models are the most practical for Label Studio pre-annotation in this project, while Nemotron and TATR provide useful comparison points for core structure recognition and transformer-based table structure modeling.
