# MRI-GNP



## What is included

- A compact hierarchical multitask trainer centered on the original `909_hierarchical_multitask_train_vit.py` idea
- A ViT-first training path with fallback backbones when ViT dependencies are unavailable
- Task-masked multitask learning for partially labeled cohorts
- Hierarchical relation routing with an IDH-rooted branch design
- YAML-based configs instead of `eval`-style experiment files
- Minimal templates for manifest and experiment configuration

## What is intentionally not included

- Private raw MRI preprocessing pipelines
- Internal Excel databases and split rules
- Historical experiment outputs, caches, and notebooks
- Institution-specific path conventions

## Repository layout

```text
main/
├─ train.py
├─ mri_gnp/
│  ├─ config.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ model.py
│  ├─ trainer.py
│  └─ utils.py
├─ configs/
│  ├─ global.example.yaml
│  └─ experiments/
│     └─ hierarchical_multitask.example.yaml
├─ templates/
│  └─ manifest_template.csv
├─ docs/
│  ├─ data_format.md
│  └─ release_checklist.md
└─ requirements.txt
```

## Data interface

The public version trains from prepared `.npz` samples instead of raw hospital data tables.

Each `.npz` sample should contain:

- `image`: `float32`, shape `[C, H, W]`
- `meta`: optional `float32`, shape `[M]`

The manifest is a CSV file that points to sample files and provides split labels plus task labels. See [docs/data_format.md](docs/data_format.md).

## Quick start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Duplicate the example configs and fill in your own paths.

- `configs/global.example.yaml` -> `configs/global.yaml`
- `configs/experiments/hierarchical_multitask.example.yaml` -> `configs/experiments/run.yaml`

3. Prepare your manifest CSV and `.npz` samples.

4. Run a dry check before full training.

```bash
python train.py \
  --global-config configs/global.yaml \
  --experiment-config configs/experiments/run.yaml \
  --dry-run
```

5. Start training.

```bash
python train.py \
  --global-config configs/global.yaml \
  --experiment-config configs/experiments/run.yaml
```

## Core training design

The trainer keeps the key ideas from the original internal script:

- Shared image backbone plus task-specific heads
- Task-specific fusion of image features and structured metadata
- Hierarchical relation routing between root and branch tasks
- Label masking so tasks can train on incomplete annotation matrices
- Weighted sampling to reduce imbalance across multitask labels

## Outputs

For each run, the trainer writes:

- latest and best checkpoints
- training history and final metrics
- optional per-task prediction CSV files for validation and test sets
- a relation summary JSON describing branch structure and learned residual scales
- snapshots of the resolved configs used for the run

## Notes for public release

- This repository is a training core, not a full medical data pipeline.
- Add a real license before publishing.
- If you release checkpoints, add a model card and dataset usage statement.
- Review [docs/release_checklist.md](docs/release_checklist.md) before the first GitHub push.
