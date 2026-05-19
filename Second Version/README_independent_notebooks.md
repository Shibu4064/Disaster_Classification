# LADI-v2 Independent Kaggle Model Notebooks

This package contains four fully self-contained Kaggle notebooks. None of the notebooks depends on the output of notebook 00, notebook 01, or any other notebook.

## Notebooks

1. `01_resnet18_baseline_independent.ipynb`  
   A standard ResNet18 multi-label baseline.

2. `02_efficientnet_b0_baseline_independent.ipynb`  
   A stronger EfficientNet-B0 multi-label baseline.

3. `03_static_gcn_resnet18_independent.ipynb`  
   ResNet18 backbone plus a train-label co-occurrence static GCN. The graph is built inside the notebook.

4. `04_dynamic_gcn_resnet18_independent.ipynb`  
   ResNet18 backbone plus static PMI graph and image-conditioned dynamic graph. The graph is built inside the notebook.

## Kaggle settings

- Accelerator: GPU P100
- Internet: ON
- Run each notebook separately

## Output location

Each notebook saves its own files under:

```text
/kaggle/working/ladi_independent_runs/<run_name>/
```

Typical outputs:

```text
best_model.pt
history.csv
metrics.json
config.json
val_predictions.csv
test_predictions.csv
val_per_label_metrics.csv
test_per_label_metrics.csv
```

## Smoke test

At the top of each notebook, set:

```python
LIMIT_TRAIN = 512
LIMIT_VAL = 128
LIMIT_TEST = 128
NUM_EPOCHS = 2
```

For the final run, set the limits back to `None` and use 12-30 epochs.

## Important

Each notebook downloads/loads LADI-v2 directly from Hugging Face, prepares dataloaders, builds any required label graph internally, trains, evaluates, and saves outputs. Therefore, they are independent but may repeat dataset loading work.
