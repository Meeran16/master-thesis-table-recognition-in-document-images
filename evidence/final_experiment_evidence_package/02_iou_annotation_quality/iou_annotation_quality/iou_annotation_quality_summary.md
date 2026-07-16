# IoU Annotation Quality Summary

This table evaluates localization quality for annotation usefulness. Class-aware matching requires the correct class label. Class-agnostic matching checks box overlap regardless of label, which is useful for estimating pre-annotation effort in Label Studio.

| Model | GT boxes | Pred boxes | Class-aware P@0.50 | Class-aware R@0.50 | Class-aware F1@0.50 | Class-aware F1@0.75 | Class-agnostic P@0.50 | Class-agnostic R@0.50 | Class-agnostic F1@0.50 | Class-agnostic F1@0.75 | Mean matched IoU | Median matched IoU | Mean best IoU per GT | Median best IoU per GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8 v2 | 475 | 601 | 0.709 | 0.897 | 0.792 | 0.673 | 0.739 | 0.935 | 0.825 | 0.691 | 0.862 | 0.909 | 0.863 | 0.906 |
| YOLOv10 v2 | 475 | 477 | 0.822 | 0.825 | 0.824 | 0.697 | 0.841 | 0.844 | 0.842 | 0.714 | 0.866 | 0.919 | 0.835 | 0.886 |
| TATR v2 @ 0.40 | 475 | 187 | 0.668 | 0.263 | 0.378 | 0.284 | 0.813 | 0.320 | 0.459 | 0.320 | 0.809 | 0.864 | 0.492 | 0.448 |
| Nemotron v2 @ 0.25 | 475 | 637 | 0.658 | 0.882 | 0.754 | 0.705 | 0.688 | 0.922 | 0.788 | 0.743 | 0.905 | 0.932 | 0.892 | 0.931 |


## Per-class IoU summary

| Model | Class | GT boxes | Pred boxes | Precision@0.50 | Recall@0.50 | F1@0.50 | Mean matched IoU | Median matched IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8 v2 | Row | 112 | 169 | 0.627 | 0.946 | 0.754 | 0.794 | 0.787 |
| YOLOv8 v2 | Column | 75 | 121 | 0.537 | 0.867 | 0.663 | 0.838 | 0.832 |
| YOLOv8 v2 | Cell | 268 | 288 | 0.840 | 0.903 | 0.871 | 0.906 | 0.942 |
| YOLOv8 v2 | Header-Row | 14 | 23 | 0.565 | 0.929 | 0.703 | 0.819 | 0.809 |
| YOLOv8 v2 | Spanning-Cell | 6 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| YOLOv10 v2 | Row | 112 | 98 | 0.878 | 0.768 | 0.819 | 0.801 | 0.791 |
| YOLOv10 v2 | Column | 75 | 69 | 0.870 | 0.800 | 0.833 | 0.829 | 0.837 |
| YOLOv10 v2 | Cell | 268 | 294 | 0.796 | 0.873 | 0.833 | 0.899 | 0.946 |
| YOLOv10 v2 | Header-Row | 14 | 16 | 0.750 | 0.857 | 0.800 | 0.841 | 0.873 |
| YOLOv10 v2 | Spanning-Cell | 6 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| TATR v2 @ 0.40 | Row | 112 | 76 | 0.671 | 0.455 | 0.543 | 0.823 | 0.863 |
| TATR v2 @ 0.40 | Column | 75 | 98 | 0.633 | 0.827 | 0.717 | 0.816 | 0.880 |
| TATR v2 @ 0.40 | Cell | 268 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| TATR v2 @ 0.40 | Header-Row | 14 | 12 | 1.000 | 0.857 | 0.923 | 0.884 | 0.915 |
| TATR v2 @ 0.40 | Spanning-Cell | 6 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Nemotron v2 @ 0.25 | Row | 112 | 125 | 0.856 | 0.955 | 0.903 | 0.879 | 0.917 |
| Nemotron v2 @ 0.25 | Column | 75 | 126 | 0.548 | 0.920 | 0.687 | 0.910 | 0.931 |
| Nemotron v2 @ 0.25 | Cell | 268 | 386 | 0.630 | 0.907 | 0.743 | 0.907 | 0.934 |
| Nemotron v2 @ 0.25 | Header-Row | 14 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Nemotron v2 @ 0.25 | Spanning-Cell | 6 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
