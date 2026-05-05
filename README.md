# MRI-GNP

MRI-GNP is a lightweight research codebase for hierarchical multitask learning on brain MRI. It is designed for noninvasive neuropathological profiling of adult-type diffuse glioma from preoperative MRI, with support for partially labeled cohorts, task-specific heads, and cross-task relation modeling.

This public release contains the core training framework used for the MRI-based glioma neuropathology prediction study: model definition, multitask optimization, configuration, and reproducible experiment structure.

## Overview

Adult-type diffuse glioma requires integrated histological and molecular characterization for diagnosis, grading, and treatment planning. MRI-GNP is built around the idea that routine preoperative MRI can support this process through a unified deep learning model rather than a collection of isolated single-marker predictors.

In the accompanying manuscript, MRI-GNP was developed and evaluated on a large multicenter cohort comprising 35,616 MR images from 8,844 patients across 22 datasets in three countries. The study focuses on predicting multiple neuropathology-related targets from MRI, including tasks such as IDH mutation, 1p/19q codeletion, CDKN2A/B homozygous deletion, +7/-10 alteration, TERT promoter mutation, EGFR amplification, ATRX mutation, TP53 mutation, MGMT promoter methylation, Ki-67 expression, and WHO-related grading tasks.

## Highlights

- Hierarchical multitask classifier with an IDH-rooted relation graph
- ViT-first image backbone with fallback CNN backbones
- Joint learning from imaging features and optional tabular metadata
- Native support for incomplete multitask labels through task masking
- YAML-based experiment configuration for cleaner reproducibility
- Simple manifest-driven data interface for public or private datasets
- Public code structure aligned with the MRI-GNP manuscript rather than an internal lab workspace

## Repository Layout

```text
main/
|-- train.py
|-- mri_gnp/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- metrics.py
|   |-- model.py
|   |-- trainer.py
|   `-- utils.py
|-- configs/
|   |-- global.example.yaml
|   `-- experiments/
|       `-- hierarchical_multitask.example.yaml
|-- templates/
|   `-- manifest_template.csv
|-- docs/
|   |-- data_format.md
|   `-- release_checklist.md
|-- requirements.txt
`-- .gitignore
```

## Data Format

The trainer expects preprocessed `.npz` files rather than raw clinical spreadsheets or full hospital pipelines.

Each sample file should contain:

- `image`: `float32`, shape `[C, H, W]`
- `meta`: optional `float32`, shape `[M]`

The manifest is a CSV file with sample paths, split assignment, and task labels. A blank template is provided in [templates/manifest_template.csv](templates/manifest_template.csv), and the full format is described in [docs/data_format.md](docs/data_format.md).

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your runtime configs from the templates:

- `configs/global.example.yaml` -> `configs/global.yaml`
- `configs/experiments/hierarchical_multitask.example.yaml` -> `configs/experiments/run.yaml`

Run a dry check before training:

```bash
python train.py \
  --global-config configs/global.yaml \
  --experiment-config configs/experiments/run.yaml \
  --dry-run
```

Launch training:

```bash
python train.py \
  --global-config configs/global.yaml \
  --experiment-config configs/experiments/run.yaml
```

## Training Design

MRI-GNP keeps the core modeling ideas of the original hierarchical multitask pipeline while presenting them in a cleaner public structure:

- A shared image encoder produces base visual representations
- Optional metadata is fused with image features before task heads
- Each task has its own neck and prediction head
- A hierarchical router refines task embeddings through branch-aware relation blocks
- Loss is computed only where labels are present, enabling mixed-annotation cohorts

The default example uses a hierarchy centered on `IDH`, with optional downstream branches such as oligodendroglial, astrocytic, and GBM-related tasks.

## Study Context

The manuscript frames MRI-GNP as a practical and generalizable system for preoperative neuropathology prediction in adult-type diffuse glioma. The full study also evaluates broader clinical use cases, including neuroradiologist assistance and robustness analysis under missing T1CE conditions. This repository focuses on the hierarchical multitask training core that underpins those experiments.

## Outputs

Each run writes results to the configured output directory, including:

- latest and best model checkpoints
- training history and final metrics
- per-task validation or test prediction CSV files
- relation summary JSON for the hierarchical routing module
- resolved config snapshots for experiment tracking

## Scope

This repository is intended as a research training framework, not a full end-to-end medical product. It does not include raw data preprocessing, institution-specific split rules, or clinical deployment logic.

## Release Notes

Before publishing a public repository or model weights, it is still a good idea to:

- add a real open-source license
- include a citation entry if this supports a paper or preprint
- provide a model card if checkpoints are released
- confirm that no sensitive sample metadata is present in manifests or outputs

See [docs/release_checklist.md](docs/release_checklist.md) for a short checklist.
