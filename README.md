# Table Recognition in Document Images

This repository contains the implementation scripts, experiment configurations, evaluation summaries, selected qualitative examples, and reproducibility notes for the master thesis project **Table Recognition in Document Images**.

## Project focus

The thesis investigates table detection and table structure recognition in historical regulatory document images. The experimental focus is on historical BIBB/VET-CVET document images, where tables are affected by scan artifacts, historical typography, inconsistent borders, merged cells, and heterogeneous layouts.

## Models evaluated

- YOLOv8
- YOLOv10
- Nemotron Table Structure model / YOLOX-based setup
- Table Transformer / TATR

## Main table-structure classes

The full 5-class schema used for YOLO-based experiments is:

1. Row
2. Column
3. Cell
4. Header-Row
5. Spanning-Cell

Nemotron was evaluated on the core 3-class setup:

1. Cell
2. Row
3. Column

TATR was evaluated on its supported structure classes and used diagnostically for comparison.

## Repository contents

| Folder | Purpose |
|---|---|
| `scripts/` | Dataset preparation, training, evaluation, prediction export, and visualization scripts |
| `configs/` | Dataset and class-mapping configuration notes |
| `results/` | Final result summaries, metric outputs, and logs |
| `figures/` | Selected qualitative visual examples |
| `evidence/` | Final experiment evidence package |
| `data/` | Dataset documentation only; raw/full data is not committed |
| `models/` | Model checkpoint documentation only; large checkpoint files are not committed |
| `appendix/` | Notes for thesis appendix and reproducibility |

## Important note

Large raw datasets, full training runs, model checkpoints, and archive files are intentionally excluded from this repository. The repository is intended to document the reproducible workflow and preserve thesis-relevant scripts, summaries, and selected evidence.
