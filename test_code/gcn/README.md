# Retina Label-Graph GCN

Multi-label retinal disease classification with a CNN backbone and a label-graph
GCN refiner on top of the per-label logits. The codebase runs a fixed set of
experiments per seed (baseline, full fine-tune with different graphs, and a
refiner-only ablation) and aggregates mean ± std across seeds.

## Idea

A standard CNN produces one logit per label and treats labels as independent.
For retinal images this is wasteful — diseases co-occur in structured ways
(e.g. DR and DME, or media opacity and cataract). The model here adds a small
GCN on top of the logit vector that propagates information between labels along
an adjacency built from the **training label co-occurrence statistics**:

```
logits_refined = logits + alpha * GCN(logits, A)
```

`alpha` is a learnable scalar gated through a sigmoid and bounded by
`alpha_max`, so the refiner starts as a near-identity map and can only nudge
the baseline logits.

Two graphs are built from `Y_train`:

- **PMI** — positive pointwise mutual information, top-k sparsified, symmetric-normalized
- **CondProb** — symmetric conditional-probability graph, top-k sparsified, symmetric-normalized

To check whether the *structure* of the graph actually matters (and the refiner
isn't just adding extra capacity), the script also runs:

- **Identity** — graph = `I`, the refiner can only act per-label
- **Permuted** — `A_perm = P A P^T`, same density and weights but labels shuffled

If permuted graphs match the real graph, the gains come from extra parameters,
not semantics. The refiner-only protocol (CNN + head frozen, only the GCN
trained) is the cleanest version of this test, because a frozen CNN cannot
"absorb" wrong adjacency by re-learning its features.

## Project layout

```
retina-label-gcn/
├── README.md
├── requirements.txt
├── run.py                       # entry point (argparse + orchestration)
└── retina_gcn/
    ├── __init__.py
    ├── data.py                  # Dataset, label-column detection
    ├── graphs.py                # adjacency builders, sparsify, normalize, permute
    ├── models.py                # backbones, GCN refiner, combined model
    ├── metrics.py               # AUC / AP / F1, macro & micro, prevalence buckets
    ├── train.py                 # baseline / full / refiner-only training loops
    ├── reporting.py             # per-label tables, summary rows, mean±std
    └── utils.py                 # seeding
```

## Dataset format

```
<data_dir>/
├── train_data.csv
├── val_data.csv                 # used as TEST when --use_val_as_test
├── test_data.csv                # optional
└── images/
    └── images/
        ├── <ID>.jpg
        ├── <ID>.png
        └── ...
```

Each CSV must have an `ID` column matching the image filename (without
extension) plus one column per label containing `0` / `1`. Label columns are
auto-detected (any numeric column with values in `{0, 1}`), or you can pass
them explicitly via `--label_cols`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch ships with CUDA wheels for most platforms — if you need a specific CUDA
version, install torch / torchvision from
[pytorch.org](https://pytorch.org/get-started/locally/) before the
`pip install -r requirements.txt` step.

## Run

Full sweep (baseline + full GCN + refiner-only ablations) across three seeds:

```bash
python run.py \
    --data_dir ./dataset \
    --backbone densenet121 \
    --img_size 512 \
    --batch_size 16 \
    --epochs 15 \
    --warmup_epochs 3 \
    --refonly_epochs 8 \
    --seeds 42,43,44 \
    --out_dir ./runs_label_gcn_final
```

Just the refiner-only ablation (faster, no CNN fine-tuning):

```bash
python run.py --data_dir ./dataset --run_refonly --seeds 42,43,44
```

If you don't have a separate test CSV, use `val_data.csv` as the held-out test
set and carve a fresh internal validation split out of train each seed:

```bash
python run.py --data_dir ./dataset --use_val_as_test --internal_val_frac 0.2
```

### Useful flags

| Flag                  | Default              | Notes                                                    |
| --------------------- | -------------------- | -------------------------------------------------------- |
| `--backbone`          | `densenet121`        | `resnet18` or `densenet121`                              |
| `--img_size`          | `512`                | Square resize                                            |
| `--batch_size`        | `16`                 | Lower if you OOM at 512px                                |
| `--epochs`            | `15`                 | Total budget (full mode = `warmup_epochs` + finetune)    |
| `--lr` / `--lr_refiner` | `3e-4` / `1e-3`    | Backbone lr vs refiner lr                                |
| `--graph_topk`        | `6`                  | Top-k sparsification per row (`0` disables)              |
| `--perm_seed`         | `123`                | RNG for the permutation control                          |
| `--no_amp`            | off                  | Disable mixed precision                                  |
| `--label_cols`        | auto                 | Comma-separated explicit label columns                   |

## Outputs

Each run writes a timestamped folder under `--out_dir`:

```
runs_label_gcn_final/densenet121_<unix_ts>/
├── all_seeds_raw.csv                    # one row per (seed, experiment)
├── aggregate_summary_mean_std.csv       # mean ± std across seeds, all exps
├── aggregate_summary_full.csv           # subset: full fine-tune only
├── aggregate_summary_refonly.csv        # subset: refiner-only ablation
└── seed_<S>/
    ├── config.json
    ├── adjacency_matrices.pt            # PMI / CondProb / I / permuted
    ├── summary.csv
    ├── summary.json
    └── <experiment>/
        ├── <experiment>.pt              # best checkpoint (by val macroAP)
        ├── <experiment>_metrics.json    # val + test metrics
        └── per_label_test.csv           # per-label AUC / AP / F1 + prevalence
```

The aggregate CSVs are the tables to read first.

## Metrics

- **macro AP** — primary metric (good for imbalanced multi-label)
- **macro / micro AUC**, **macro / micro F1 at 0.5**
- **Prevalence buckets** — labels are split into rare / medium / frequent thirds
  by training prevalence; per-bucket mean AP makes it easy to see whether gains
  come from rare or frequent classes.

Model selection across epochs is by **val macro AP** for every experiment.

## Experiments per seed

| Name                         | Mode      | Adjacency                |
| ---------------------------- | --------- | ------------------------ |
| `baseline_cnn`               | baseline  | —                        |
| `gcn_pmi_full`               | full      | PMI                      |
| `gcn_condprob_full`          | full      | Conditional probability  |
| `gcn_identity_full`          | full      | Identity                 |
| `gcn_perm_pmi_full`          | full      | Permuted PMI             |
| `gcn_perm_condprob_full`     | full      | Permuted CondProb        |
| `gcn_pmi_refonly`            | refonly   | PMI                      |
| `gcn_condprob_refonly`       | refonly   | Conditional probability  |
| `gcn_identity_refonly`       | refonly   | Identity                 |
| `gcn_perm_pmi_refonly`       | refonly   | Permuted PMI             |
| `gcn_perm_condprob_refonly`  | refonly   | Permuted CondProb        |

**Full mode**: load baseline backbone + head, warm up the refiner only for
`warmup_epochs`, then fine-tune everything for the remaining epochs.

**Refiner-only mode**: load baseline backbone + head, freeze them, train *only*
the refiner. This is the cleaner control — if PMI/CondProb beats identity and
permuted graphs here, the structure of the label graph is doing real work.

## Reproducibility notes

- `seed_everything` covers Python, NumPy, and PyTorch (CPU + CUDA); cuDNN is
  set to deterministic.
- The output of `pred_all` is the model's `sigmoid(logits)`, computed in eval
  mode with no grad.
- Adjacency matrices are saved per seed (`adjacency_matrices.pt`) so the exact
  graphs used can be reloaded.
- Each experiment's checkpoint is saved at its best validation macroAP and
  reloaded for the test pass.
