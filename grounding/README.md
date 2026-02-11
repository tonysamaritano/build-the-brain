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
- `grounding/export_encoder_onnx.py`: exports encoder-only ONNX model for browser inference
- `grounding/export_decoder_onnx.py`: exports decoder-only ONNX model for browser-side latent interpolation rendering
- `grounding/export_interactive_assets.py`: exports UMAP manifold + reference latents + one MNIST sample per digit
- `grounding/interactive_latent_playground.html`: interactive Three.js + ONNX latent projection demo

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

## Interactive Browser Demo (ONNX + UMAP)

This demo lets you pick a digit sample and interactively apply augmentations in-browser:
- translation
- rotation
- scale
- intensity
- noise
- masking

Then it runs:
1. `sample -> encoder (ONNX in browser)`
2. `latent -> projection on established UMAP manifold (kNN barycentric projection)`
3. renders the projected point in 3D with Three.js

### 1) Export encoder ONNX

```bash
python grounding/export_encoder_onnx.py \
  --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml
```

Default output:
- `artifacts/<experiment_name>_encoder.onnx`

### 1b) Export decoder ONNX

```bash
python grounding/export_decoder_onnx.py \
  --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml
```

Default output:
- `artifacts/<experiment_name>_decoder.onnx`

### 2) Export interactive manifold assets

```bash
python grounding/export_interactive_assets.py \
  --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml \
  --split test \
  --max-points 3000 \
  --l2-normalize \
  --umap-n-neighbors 30 \
  --umap-min-dist 0.1 \
  --projection-k 12
```

Default output:
- `artifacts/<experiment_name>_<split>_interactive_assets.json`

### 3) Serve and open

```bash
python -m http.server 8000
```

Open:
- `http://localhost:8000/grounding/interactive_latent_playground.html?assets=/artifacts/asymmetric_resnet8_autoencoder_test_interactive_assets.json&encoder=/artifacts/asymmetric_resnet8_autoencoder_encoder.onnx&decoder=/artifacts/asymmetric_resnet8_autoencoder_decoder.onnx`
- `http://localhost:8000/grounding/interactive_latent_playground.html?assets=/artifacts/regularized_covnet_test_interactive_assets.json&encoder=/artifacts/regularized_covnet_encoder.onnx&decoder=/artifacts/regularized_covnet_decoder.onnx`

## Notes

- `output.save_every_n_epochs` controls periodic epoch checkpoints (`..._epochXXX.pt`).
- `output.checkpoint_path` is always the final epoch model and is the default checkpoint used by analysis scripts.
- `num_workers` is automatically forced to `0` if `torch_shm_manager` is not executable in your environment.
- If a checkpoint path is omitted for analysis scripts, they use `output.checkpoint_path` from config.

## Agent Guide: Interactive Visualization Assets

Use this section as a handoff spec for coding agents that need to consume and render the interactive latent-space demo.

### Goal

Given a trained config/checkpoint, produce browser-loadable assets and open:

- `grounding/interactive_latent_playground.html`

with query params pointing to:

- interactive manifold JSON
- encoder ONNX
- decoder ONNX

### Required output files

All three are required for full functionality (probe + interpolation mode):

1. `artifacts/<experiment_name>_test_interactive_assets.json`
2. `artifacts/<experiment_name>_encoder.onnx`
3. `artifacts/<experiment_name>_decoder.onnx`

### Generate assets (canonical commands)

From repo root:

```bash
python grounding/export_interactive_assets.py \
  --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml \
  --split test \
  --max-points 3000 \
  --l2-normalize \
  --projection-k 12
```

```bash
python grounding/export_encoder_onnx.py \
  --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml
```

```bash
python grounding/export_decoder_onnx.py \
  --config grounding/configs/5_asymmetric_resnet8_autoencoder.yaml
```

If `output.checkpoint_path` does not exist yet, pass `--checkpoint <path-to-existing-pt>` explicitly.

### JSON asset contract

The interactive page expects these keys in `*_interactive_assets.json`:

- `experiment` (string)
- `num_points` (int)
- `latent_dim` (int)
- `l2_normalized` (bool)
- `projection_k` (int)
- `reference_points` (`N x 3` float array; UMAP coordinates)
- `reference_latents` (`N x latent_dim`; latent vectors used for projection distance)
- `reference_latents_raw` (`N x latent_dim`; raw latent vectors used for interpolation)
- `reference_labels` (`N` ints; point coloring)
- `digit_samples` (`10 x 784` float array; one sample per digit)

Do not rename these keys unless the frontend is updated in lockstep.

### Run locally

```bash
python -m http.server 8000
```

Open:

- `http://localhost:8000/grounding/interactive_latent_playground.html?assets=/artifacts/asymmetric_resnet8_autoencoder_test_interactive_assets.json&encoder=/artifacts/asymmetric_resnet8_autoencoder_encoder.onnx&decoder=/artifacts/asymmetric_resnet8_autoencoder_decoder.onnx`

### Runtime behavior (for integration/testing)

- `probe` mode:
  - input sample/custom draw -> augment -> `E(x)` -> project on fixed manifold
  - decoded preview shows `D(E(x))`
- `interpolate` mode:
  - pick two manifold points -> linear latent path `z_t`
  - regrounded signal is used: `D(E(D(z_t)))`
  - marker position comes from projecting `E(D(z_t))`
  - decoded preview shows `D(E(D(z_t)))`

### Integration checklist for agents

1. Verify all 3 artifact files exist before launching page.
2. Keep `assets`, `encoder`, and `decoder` query params absolute-from-host-root (for example `/artifacts/...`).
3. If embedding in blog/CMS, prefer iframe to hosted page; if blocked, provide link-out.
4. Reduce `--max-points` if interaction feels heavy.
