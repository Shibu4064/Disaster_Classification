# RetinaX-GATN

Multi-label retinal disease classification with a **Graph Attention Transformer Network (GATN)**. The model combines an EfficientNet-B3 image backbone with a label correlation graph built from BERT embeddings, refined through multi-head self-attention and two GCN layers.

## Idea

Fundus images often show more than one condition at once (e.g. diabetic retinopathy and macular edema co-occur frequently). A plain CNN classifier ignores these label dependencies. GATN models them explicitly:

1. **BERT** turns each label name into a semantic vector.
2. The cosine similarity of these vectors gives an initial correlation matrix `A0`.
3. A **multi-head attention layer** refines `A0` into a learned, non-negative, symmetric adjacency.
4. Two **graph convolutions** propagate label embeddings over this graph, producing one classifier weight vector per label.
5. The image features from the **EfficientNet-B3** backbone are dot-producted with these weights to give logits.

A sparsity term pulls the learned adjacency back toward the identity, so the graph only adds structure where it really helps.

## Project structure

```
retinax-gatn/
├── README.md
├── requirements.txt
├── config.py              # paths and hyperparameters
├── train.py               # training entry point
└── src/
    ├── data.py            # MultiLabelDataset + CLAHE transform
    ├── embeddings.py      # BERT label embeddings (cached to .npy)
    ├── graph.py           # GCN layer + graph-attention layer + adj. normalisation
    ├── model.py           # GATNResnet (backbone + graph head)
    ├── metrics.py         # mAP, CP/CR/CF1, OP/OR/OF1
    └── engine.py          # train / eval / prediction loops
```

## Installation

```bash
git clone <repo-url>
cd retinax-gatn

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

A CUDA-capable GPU is recommended — training with `train_backbone=True` is slow on CPU.

## Data

Place the dataset under `data/RetinaX/` so the layout matches `config.py`:

```
data/RetinaX/
├── train_data.csv
├── test_data.csv
└── images/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

Each CSV must have an `ID` column (image filename without extension) and one binary column per label, for example:

```
ID,DR,ME,Glaucoma,Cataract,AMD,...
0001,1,0,0,0,1,...
0002,0,1,0,0,0,...
```

Label embeddings are computed on first run from the column names and cached to `data/RetinaX/label_embeddings.npy`. Delete that file to recompute them.

## Usage

```bash
python train.py
```

This will:

- load and cache BERT label embeddings,
- build train/val dataloaders with CLAHE preprocessing,
- train for `NUM_EPOCHS` (default 20),
- log mAP and per-class / overall precision-recall-F1 each epoch,
- save the best checkpoint (lowest val loss) to `checkpoints/best_gatn_model.pt`.

All paths and hyperparameters live in `config.py`. Adjust there rather than touching `train.py`.

## Results

Training on the RetinaX dataset for 20 epochs (A100, batch size 32):

| Epoch | train_loss | val_loss | mAP   | CF1   | OF1   |
|-------|-----------:|---------:|------:|------:|------:|
| 1     | 0.4414     | 0.2579   | 0.073 | 0.009 | 0.009 |
| 5     | 0.1166     | 0.1092   | 0.406 | 0.246 | 0.531 |
| 10    | 0.0736     | 0.0878   | 0.569 | 0.452 | 0.638 |
| 15    | 0.0440     | 0.0828   | 0.623 | 0.531 | 0.687 |
| 20    | 0.0222     | 0.0896   | **0.652** | **0.565** | **0.700** |

Val loss plateaus around epoch 15. Best checkpoint is the one saved at the lowest val loss (epoch 15 in this run).

## Metric notation

Following the original GATN paper:

- **mAP** — macro mean average precision
- **CP / CR / CF1** — per-class (macro) precision, recall, F1 at threshold 0.5
- **OP / OR / OF1** — overall (micro) precision, recall, F1 at threshold 0.5

## Key hyperparameters (`config.py`)

| Name              | Default            | Meaning                                 |
|-------------------|--------------------|-----------------------------------------|
| `BATCH_SIZE`      | 32                 | training batch size                     |
| `NUM_EPOCHS`      | 20                 | training epochs                         |
| `BASE_LR`         | 1e-4               | Adam learning rate                      |
| `GAT_HEADS`       | 4                  | attention heads in the GAT layer        |
| `GCN_HIDDEN`      | 1024               | hidden dim between the two GCN layers   |
| `ALPHA`           | 1.0                | weight of the adjacency sparsity term   |
| `IMG_RESIZE`      | 320                | resize short side before center crop    |
| `IMG_CROP`        | 300                | EfficientNet-B3 input size              |

## References

- Chen et al., *Multi-Label Image Recognition with Graph Convolutional Networks*, CVPR 2019.
- Ye et al., *Attention-driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition*, ECCV 2020.
- Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.
