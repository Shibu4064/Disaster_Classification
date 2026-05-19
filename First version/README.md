# LADI-v2 Multi-label Disaster Imagery: Backbone + Static GCN + Dynamic GCN

This project is a Kaggle P100-ready pipeline for **LADI-v2: Multi-label Dataset and Classifiers for Low-Altitude Disaster Imagery**.

## What is included

- `notebooks/00_environment_and_dataset_cache.ipynb` — creates a local LADI-v2a resized cache from Hugging Face.
- `notebooks/01_eda_and_label_graphs.ipynb` — EDA, label prevalence, co-occurrence, and static label graph construction.
- `notebooks/02_baseline_backbone_training.ipynb` — ResNet/EfficientNet multi-label baseline.
- `notebooks/03_static_gcn_training.ipynb` — static train-label co-occurrence GCN classifier.
- `notebooks/04_dynamic_gcn_training.ipynb` — image-conditioned Dynamic GCN classifier.
- `notebooks/05_evaluate_compare_and_inference.ipynb` — aggregate metrics, per-label table, plots.
- `src/` — reusable PyTorch modules for data, graphs, models, training, and metrics.
- `docs/LADIv2_GCN_DynamicGCN_Project_Report.docx` — full pipeline report.

## Kaggle running order

1. Upload or unzip this whole folder into a Kaggle notebook environment.
2. Enable internet in Kaggle if you want notebook 00 to download from Hugging Face.
3. Run notebooks in order: `00 -> 01 -> 02 -> 03 -> 04 -> 05`.
4. Outputs are written under `/kaggle/working/ladi_v2_outputs`.

## Recommended P100 settings

- Development run: `BACKBONE='resnet18'`, `IMG_SIZE=320`, `BATCH_SIZE=16` for baseline/static GCN, `BATCH_SIZE=8-12` for Dynamic GCN.
- Final run: raise epochs to 15-20 and optionally use `efficientnet_b0`.
- If CUDA OOM occurs, lower `BATCH_SIZE` first, then `IMG_SIZE`.

## Main research design

1. CNN backbone baseline.
2. Static graph built only from training-label co-occurrence.
3. Static GCN classifier using the label graph to produce classifier weights.
4. Dynamic GCN where the graph is partly image-conditioned and mixed with the static prior.

The dynamic stage is the main methodological contribution because it handles the fact that label dependencies can change across disaster type, geography, and event year.
