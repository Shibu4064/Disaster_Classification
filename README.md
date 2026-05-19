# Disaster_Classification
# LADI-v2 Multi-Label Disaster Imagery Classification  
## Backbone CNNs, Static Label Graphs, GCN, and Dynamic GCN on Low-Altitude Disaster Imagery

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20P100-20BEFF)
![Task](https://img.shields.io/badge/Task-Multi--Label%20Image%20Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-LADI--v2-purple)

---

## Project Overview

This repository presents a complete deep learning pipeline for **multi-label disaster scene understanding** using the **LADI-v2: Low-Altitude Disaster Imagery dataset**. The project focuses on recognising multiple disaster-relevant visual attributes from aerial imagery, such as flooding, debris, infrastructure, roads, water, buildings, trees, and related environmental signals.

The central research idea is to go beyond a standard CNN classifier by explicitly modelling **relationships between disaster labels**. In real aerial disaster imagery, labels are rarely independent. For example, flooding may co-occur with water, roads, buildings, and debris, while infrastructure-related labels may appear together in different combinations depending on disaster type and scene context.

To capture this structure, the project implements four progressively stronger model families:

1. **ResNet18 Baseline**
2. **EfficientNet-B0 Baseline**
3. **Static GCN with Training Label Co-occurrence Graph**
4. **Dynamic GCN with Image-Conditioned Label Graphs**

The strongest contribution is the **Dynamic GCN**, where the label graph is not fixed for all images. Instead, the model combines a static co-occurrence prior with a sample-specific dynamic graph generated from each image feature representation.

---

## Why This Project Matters

Low-altitude disaster imagery is highly valuable for emergency response, damage assessment, humanitarian mapping, and situational awareness. However, disaster images are complex because one image can contain several relevant conditions at the same time. A single aerial image may show flooding, roads, buildings, trees, water bodies, debris, and infrastructure damage simultaneously.

Traditional single-label classification is therefore unsuitable. This project treats the task as a **multi-label image classification problem** and investigates whether graph-based label dependency modelling can improve recognition performance over conventional CNN baselines.

The project is designed to be:

- **Research-oriented**: includes baselines, graph-based models, ablation logic, and per-label evaluation.
- **Kaggle-ready**: notebooks are prepared for GPU P100 environments.
- **Self-contained**: each independent notebook contains its own setup, dataset loading, model, training loop, and evaluation.
- **Extensible**: the pipeline can be expanded with stronger backbones, attention modules, calibration, or additional graph regularisation.

---

## Dataset

The project uses:

**Dataset:** `MITLL/LADI-v2-dataset`  
**Platform:** Hugging Face Datasets  
**Task:** Multi-label low-altitude disaster image classification  
**Recommended configuration:** resized v2a-style label set  
**Label type:** binary multi-label targets  

Typical labels include disaster and scene indicators such as:

- `bridges_any`
- `buildings_any`
- `debris_any`
- `flooding_any`
- `roads_any`
- `trees_any`
- `water_any`
- and other LADI-v2a multi-label categories

The dataset is particularly suitable for this project because it naturally contains label co-occurrence patterns, which can be converted into graph structures for GCN-based learning.

---

## Core Research Question

> Can graph-based label dependency modelling improve multi-label classification performance on low-altitude disaster imagery compared with standard CNN backbones?

The project explores this question through a staged methodology:

| Stage | Model | Main Idea |
|---|---|---|
| 1 | ResNet18 Baseline | Standard CNN multi-label classifier |
| 2 | EfficientNet-B0 Baseline | Stronger lightweight CNN baseline |
| 3 | Static GCN | Uses training label co-occurrence graph |
| 4 | Dynamic GCN | Generates image-conditioned label graphs |

---

## Methodology

### 1. ResNet18 Baseline

The first model establishes a reliable baseline using a pretrained ResNet18 backbone.

```text
Image → ResNet18 Feature Extractor → Linear Multi-label Head → Label Predictions
```

This model treats each label as an independent output, although the loss is computed jointly across all labels.

**Purpose:**

- Provide a simple and stable benchmark.
- Verify dataset loading and evaluation.
- Establish a reference point for graph-based models.

---

### 2. EfficientNet-B0 Baseline

The second model uses EfficientNet-B0, which is often stronger than ResNet18 for image classification while remaining practical on Kaggle P100.

```text
Image → EfficientNet-B0 Backbone → Linear Multi-label Head → Label Predictions
```

**Purpose:**

- Test whether a stronger CNN backbone improves results.
- Compare graph improvements against a more competitive baseline.
- Maintain GPU-friendly training cost.

---

### 3. Static GCN with Label Co-occurrence Graph

The third model introduces a static label graph built from the **training set only**. This avoids validation/test leakage.

The graph is constructed from label co-occurrence statistics, especially using a PMI-style adjacency matrix:

```text
Training Labels → Co-occurrence Matrix → PMI Graph → Normalised Adjacency Matrix
```

The GCN uses this graph to propagate information between label embeddings. The final label embeddings are transformed into classifier weights.

```text
Image → CNN Feature Extractor
Labels → Static Graph → GCN → Label Classifier Weights
Image Features × Label Weights → Multi-label Predictions
```

**Purpose:**

- Model stable label relationships.
- Encourage semantically related labels to share information.
- Test whether label dependency improves over independent classification heads.

---

### 4. Dynamic GCN with Image-Conditioned Graphs

The Dynamic GCN is the main methodological contribution of the project.

Instead of using only one fixed graph for every image, the model creates an image-specific graph from each image feature vector. This allows label relationships to change depending on the visual context.

```text
Static Graph Prior + Image-Conditioned Dynamic Graph → Mixed Graph → Batch GCN → Predictions
```

The model includes:

- A static graph from training label co-occurrence.
- Image-conditioned label features.
- Dynamic adjacency generation.
- A learnable gate to combine static and dynamic graphs.
- Batch-wise GCN propagation.

**Why this is important:**

In disaster imagery, the relationship between labels is not always constant. For example:

- Flood scenes may link water, roads, buildings, and debris.
- Wind-damage scenes may link trees, debris, and buildings.
- Infrastructure-heavy scenes may emphasise roads, bridges, and buildings.

Dynamic GCN allows the model to adapt label relationships per image rather than forcing one universal label graph.

---

## Repository Structure

```text
LADIv2_4_Independent_Model_Notebooks/
│
├── 01_resnet18_baseline_independent.ipynb
├── 02_efficientnet_b0_baseline_independent.ipynb
├── 03_static_gcn_resnet18_independent.ipynb
├── 04_dynamic_gcn_resnet18_independent.ipynb
│
├── README.md
└── outputs/
    └── generated after running notebooks
```

Each notebook is fully independent. It includes:

- environment setup
- package installation
- dataset loading
- label preparation
- dataloaders
- model definition
- training loop
- validation
- threshold tuning
- test evaluation
- output saving

---

## Notebook Descriptions

### `01_resnet18_baseline_independent.ipynb`

A complete ResNet18 baseline notebook.

**Main features:**

- Downloads/loads LADI-v2 directly.
- Uses torchvision ResNet18.
- Applies multi-label BCE loss.
- Tunes thresholds on validation data.
- Saves test metrics and predictions.

**Use this first** to confirm the full pipeline works.

---

### `02_efficientnet_b0_baseline_independent.ipynb`

A stronger CNN baseline using EfficientNet-B0.

**Main features:**

- Lightweight but powerful image backbone.
- Suitable for Kaggle P100.
- Useful comparison against ResNet18 and GCN variants.

---

### `03_static_gcn_resnet18_independent.ipynb`

A graph-based multi-label classifier using a static label graph.

**Main features:**

- Builds label co-occurrence graph inside the notebook.
- Uses PMI-based adjacency.
- Applies GCN over trainable label embeddings.
- Produces graph-aware classifier weights.

---

### `04_dynamic_gcn_resnet18_independent.ipynb`

The main proposed model.

**Main features:**

- Builds static graph internally.
- Generates image-conditioned dynamic graph.
- Mixes static and dynamic graphs using a learnable gate.
- Performs batch-wise GCN propagation.
- Designed as the strongest methodology component.

---

## Kaggle Setup

Recommended Kaggle settings:

```text
Accelerator: GPU P100
Internet: ON
Persistence: ON
```

Because Kaggle environments can change, the notebooks use a stable package setup designed for P100 compatibility:

```text
torch==2.4.1
torchvision==0.19.1
numpy==1.26.4
scipy==1.13.1
pandas==2.2.2
scikit-learn==1.5.2
pillow==10.4.0
```

The notebooks are designed to avoid common Kaggle P100 issues such as:

- CUDA kernel incompatibility
- PyTorch build mismatch
- Pillow/torchvision import mismatch
- NumPy binary mismatch

---

## How to Run

### Step 1: Upload the Project to Kaggle

Upload the ZIP file as a Kaggle dataset or directly into a notebook.

### Step 2: Open Any Notebook

Each notebook is independent. You can run any of the following without running another notebook first:

```text
01_resnet18_baseline_independent.ipynb
02_efficientnet_b0_baseline_independent.ipynb
03_static_gcn_resnet18_independent.ipynb
04_dynamic_gcn_resnet18_independent.ipynb
```

### Step 3: Enable Internet and GPU

The first run needs internet access to download packages, pretrained weights, and the dataset.

### Step 4: Run From the First Cell

Run the notebook from the top. If the setup cell restarts the runtime, run the notebook again from the top after restart.

---

## Quick Smoke Test

For a quick test, set:

```python
LIMIT_TRAIN = 512
LIMIT_VAL = 128
LIMIT_TEST = 128
NUM_EPOCHS = 2
```

This confirms that:

- dataset loading works,
- the model trains,
- validation runs,
- metrics are saved,
- predictions are generated.

---

## Full Training Recommendation

For a stronger final run:

```python
LIMIT_TRAIN = None
LIMIT_VAL = None
LIMIT_TEST = None
NUM_EPOCHS = 15
```

For the Dynamic GCN:

```python
NUM_EPOCHS = 20
```

If GPU memory is limited, reduce:

```python
BATCH_SIZE = 8
IMAGE_SIZE = 288
```

---

## Evaluation Metrics

The project reports multi-label metrics suitable for imbalanced disaster labels:

| Metric | Purpose |
|---|---|
| Macro Average Precision | Main metric for imbalanced multi-label ranking |
| Micro F1 | Overall label-level performance |
| Macro F1 | Balanced per-label classification performance |
| Per-label AP | Shows which labels benefit most |
| Per-label F1 | Identifies weak and strong classes |
| Validation-tuned Thresholds | Improves F1 without test leakage |

Accuracy is not used as the main metric because it can be misleading in imbalanced multi-label classification.

---

## Output Files

Each notebook saves its outputs under:

```text
/kaggle/working/ladi_independent_runs/<model_name>/
```

Typical outputs include:

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

These files allow later comparison, reporting, and dissertation-level result analysis.

---

## Suggested Experiment Plan

For a strong research report or dissertation chapter, run the following experiments:

| Experiment | Purpose |
|---|---|
| ResNet18 Baseline | Simple reference model |
| EfficientNet-B0 Baseline | Stronger CNN baseline |
| Static GCN | Tests fixed label dependency modelling |
| Dynamic GCN | Tests image-conditioned label dependency modelling |

Optional extensions:

- compare PMI vs conditional probability graphs,
- add identity graph as a control,
- run three random seeds,
- report mean ± standard deviation,
- visualise label co-occurrence heatmaps,
- analyse per-label improvements.

---

## Expected Contribution

This project contributes a practical and research-ready framework for disaster-image multi-label classification by combining:

- CNN-based visual feature extraction,
- training-label co-occurrence graph construction,
- graph convolution over label embeddings,
- image-conditioned dynamic label dependency modelling,
- Kaggle P100-compatible implementation.

The Dynamic GCN component is especially important because it addresses a key limitation of static label graphs: disaster-label relationships can vary from image to image.

---

## Troubleshooting

### CUDA error: no kernel image is available for execution on the device

This usually means the installed PyTorch build does not support the P100 GPU architecture. Use the provided stable environment setup and restart the runtime after installation.

---

### Pillow import error with torchvision

If you see an error related to `PIL._typing`, restart the runtime and make sure the notebook pins:

```text
pillow==10.4.0
torchvision==0.19.1
```

---

### NumPy import error

If you see an error related to `numpy._core.umath`, restart the runtime and ensure:

```text
numpy==1.26.4
```

Do not mix different NumPy versions in the same Kaggle session.

---

### Out-of-memory error

Reduce:

```python
BATCH_SIZE = 8
IMAGE_SIZE = 288
```

For Dynamic GCN, use smaller batch sizes than the baseline models.

---

### Dataset download is slow

Use smoke-test limits first:

```python
LIMIT_TRAIN = 512
LIMIT_VAL = 128
LIMIT_TEST = 128
```

After confirming the notebook works, run the full dataset.

---

## Future Improvements

Potential improvements include:

- replacing ResNet18 with ConvNeXt or Swin Transformer,
- using EfficientNet features inside the GCN models,
- adding graph regularisation loss,
- testing graph attention networks,
- calibrating thresholds with temperature scaling,
- adding test-time augmentation,
- performing multi-seed statistical analysis,
- visualising dynamic graph changes across disaster types,
- explaining predictions using Grad-CAM or attention visualisation.

---

## Citation

If using the dataset or building on this work, cite the LADI-v2 dataset and relevant graph-based multi-label learning literature.

```bibtex
@article{scheele2024ladi,
  title={LADI v2: Multi-label Dataset and Classifiers for Low-Altitude Disaster Imagery},
  author={Scheele, Samuel and Picchione, Katelyn and Liu, Justin},
  journal={arXiv preprint arXiv:2406.02780},
  year={2024}
}

@inproceedings{chen2019mlgcn,
  title={Multi-Label Image Recognition with Graph Convolutional Networks},
  author={Chen, Zhao-Min and Wei, Xiu-Shen and Wang, Peng and Guo, Yanwen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@inproceedings{ye2020dynamicgcn,
  title={Attention-Driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition},
  author={Ye, Jin and He, Jun and Peng, Xiaojiang and Wu, Wenhao and Qiao, Yu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

---

## Project Summary

This project is a complete, reproducible, Kaggle-ready research pipeline for multi-label disaster imagery classification. It starts with strong CNN baselines and progresses toward graph-based and dynamic graph-based models that better reflect the real structure of disaster imagery.

The final goal is not only to classify images, but to investigate whether modelling relationships between disaster labels can lead to more robust, interpretable, and context-aware disaster scene understanding.

---

## Author

**Hrithik Majumdar Shibu**  
MSc Artificial Intelligence  
Aspiring Machine Learning Engineer / Early-Career Researcher  

GitHub: [Shibu4064](https://github.com/Shibu4064)
