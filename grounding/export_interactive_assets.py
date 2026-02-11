from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import umap
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from autoencoder_common import (
    build_mnist_splits,
    build_model_from_config,
    resolve_device,
    resolve_num_workers,
    set_seed,
)


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def encode_dataset(
    model: torch.nn.Module,
    dataset: Subset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    latents: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, y in tqdm(loader, desc=f"encode {split_name}", leave=False):
            images = images.to(device)
            z = model.encode(images).cpu().numpy()
            latents.append(z)
            labels.append(y.numpy())
    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)


def maybe_l2_normalize(x: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def sample_indices(n_items: int, max_points: int, seed: int) -> np.ndarray:
    if n_items <= max_points:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n_items, size=max_points, replace=False)


def get_one_sample_per_digit(data_dir: str) -> dict[int, list[float]]:
    ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    found: dict[int, list[float]] = {}
    for image, label in ds:
        digit = int(label)
        if digit in found:
            continue
        found[digit] = image.squeeze(0).numpy().astype(float).reshape(-1).tolist()
        if len(found) == 10:
            break
    missing = [d for d in range(10) if d not in found]
    if missing:
        raise RuntimeError(f"Could not collect all digits from MNIST test set. Missing: {missing}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export assets for interactive latent-space browser demo."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--projection-k", type=int, default=12)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    device = resolve_device(cfg["training"]["device"])
    checkpoint_path = args.checkpoint or cfg["output"]["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    splits = build_mnist_splits(cfg)
    dataset = splits.train if args.split == "train" else splits.test
    model = load_model(cfg, checkpoint_path, device)

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = resolve_num_workers(int(cfg["data"]["num_workers"]))
    latents_raw, labels = encode_dataset(model, dataset, batch_size, num_workers, device, args.split)
    sampled_idx = sample_indices(len(latents_raw), args.max_points, seed)

    latents_raw = latents_raw[sampled_idx]
    labels = labels[sampled_idx]
    latents = maybe_l2_normalize(latents_raw, args.l2_normalize)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=seed,
    )
    points_3d = reducer.fit_transform(latents).astype(float)

    digit_samples = get_one_sample_per_digit(str(cfg["data"]["data_dir"]))
    ordered_samples = [digit_samples[d] for d in range(10)]

    experiment_name = str(cfg["experiment"]["name"])
    default_out = f"./artifacts/{experiment_name}_{args.split}_interactive_assets.json"
    out_path = Path(args.output or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": experiment_name,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "num_points": int(len(points_3d)),
        "latent_dim": int(latents.shape[1]),
        "projection": "umap_3d",
        "umap_n_neighbors": int(args.umap_n_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "l2_normalized": bool(args.l2_normalize),
        "projection_method": "knn_barycentric",
        "projection_k": int(max(1, args.projection_k)),
        "reference_points": points_3d.tolist(),
        "reference_latents": latents.astype(float).tolist(),
        "reference_latents_raw": latents_raw.astype(float).tolist(),
        "reference_labels": labels.astype(int).tolist(),
        "digit_samples": ordered_samples,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Saved interactive assets to {out_path}")


if __name__ == "__main__":
    main()
