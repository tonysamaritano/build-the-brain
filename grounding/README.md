# Grounding Experiments (MNIST)

This folder contains the experiment pipeline for learning MNIST latent spaces, from naive autoencoders to low-data and self-supervised setups.

## Prerequisites

- Python 3.11
- Dependencies from `pyproject.toml`

## Environment Setup

From repo root:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Folder Layout

- `grounding/configs`: Experiment YAMLs (the experiments are defined here)
- `grounding/train_naive_autoencoder.py`: Trainer for `model.architecture: vanilla_fc`
- `grounding/train_naive_covnet_autoencoder.py`: Trainer for `model.architecture: naive_conv` (baseline)
- `grounding/train_covnet.py`: Trainer for conv-based runs with dropout and optional self-supervised augmentations (`naive_conv` and `asymmetric_resnet`)
- `grounding/analyze_autoencoder.py`: kNN latent-separation analysis (+ parameter counts)
- `grounding/analyze_manifold.py`: manifold quality metrics + combined manifold score
- `grounding/generate_latent_points.py`: exports latent points JSON for visualization (UMAP 3D)
- `grounding/latent_space_viewer.html`: Three.js viewer for latent JSON

## Config Schema (Common)

Each config includes these top-level sections:

- `experiment`: name/seed
- `data`: dataset path and split sizes (`train_samples`, `val_samples`, `test_samples`)
- `model`: architecture and architecture-specific params
- `training`: batch size, epochs, LR, weight decay, device (`auto` => CUDA > MPS > CPU)
- `regularization`: currently dropout (`enabled`, `p`)
- `self_supervised`: augmentation controls and expansion factor (`augmentations_per_sample`)
- `output`: checkpoint and report output paths
- `analysis`: kNN settings (`k`, `metric`)

Supported `model.architecture` values:

- `vanilla_fc`
- `naive_conv`
- `asymmetric_resnet` (ResNet-N encoder with shared latent/decoder style)

## Run Experiments

Run from repo root.

### 0) Naive fully-connected autoencoder

```bash
python grounding/train_naive_autoencoder.py --config grounding/configs/0_naive_autoencoder.yaml
python grounding/analyze_autoencoder.py --config grounding/configs/0_naive_autoencoder.yaml
```

### 1) Naive conv autoencoder (full/large-data baseline)

```bash
python grounding/train_naive_covnet_autoencoder.py --config grounding/configs/1_naive_covnet_autoencoder.yaml
python grounding/analyze_autoencoder.py --config grounding/configs/1_naive_covnet_autoencoder.yaml
```

### 2) Naive conv autoencoder (low samples)

```bash
python grounding/train_naive_covnet_autoencoder.py --config grounding/configs/2_naive_covnet_autoencoder_low_samples.yaml
python grounding/analyze_autoencoder.py --config grounding/configs/2_naive_covnet_autoencoder_low_samples.yaml
```

### 3) Regularized conv autoencoder (dropout)

```bash
python grounding/train_covnet.py --config grounding/configs/3_regularized_covnet.yaml
python grounding/analyze_autoencoder.py --config grounding/configs/3_regularized_covnet.yaml
```

### 4) Self-supervised conv autoencoder (augmentations)

```bash
python grounding/train_covnet.py --config grounding/configs/4_self_supervised_covnet_aug100.yaml
python grounding/analyze_autoencoder.py --config grounding/configs/4_self_supervised_covnet_aug100.yaml
```

### 5) Asymmetric ResNet-8 encoder autoencoder

```bash
python grounding/train_covnet.py --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml
python grounding/analyze_autoencoder.py --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml
```

## Manifold Quality Scoring

Compute manifold metrics and aggregate score:

```bash
python grounding/analyze_manifold.py --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml --split test
```

Outputs JSON to `output.manifold_analysis_path` if set, otherwise:

- `artifacts/<experiment_name>_manifold_metrics.json`

## Latent Visualization (UMAP + Three.js)

1. Generate latent points JSON:

```bash
python grounding/generate_latent_points.py \
  --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml \
  --split test \
  --max-points 5000 \
  --l2-normalize \
  --umap-n-neighbors 30 \
  --umap-min-dist 0.1
```

Default output:

- `artifacts/<experiment_name>_<split>_latent_points.json`

2. Serve files from repo root (required for browser `fetch`):

```bash
python -m http.server 8000
```

3. Open viewer:

- `http://localhost:8000/grounding/latent_space_viewer.html?data=/artifacts/5_asymmetric_resnet8_autoencoder_test_latent_points.json`

## Notes

- Checkpoints are saved at `output.checkpoint_path` when validation improves (best model, not just final epoch).
- `num_workers` is automatically forced to `0` if `torch_shm_manager` is not executable in your environment.
- If a checkpoint path is omitted for analysis scripts, they use `output.checkpoint_path` from config.
