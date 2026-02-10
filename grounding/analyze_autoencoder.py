from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from autoencoder_common import (
    build_model_from_config,
    build_mnist_splits,
    resolve_device,
    resolve_num_workers,
    set_seed,
)


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


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

    all_latents: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"encode {split_name}", leave=False):
            images = images.to(device)
            latents = model.encode(images).cpu().numpy()
            all_latents.append(latents)
            all_labels.append(labels.numpy())

    return np.concatenate(all_latents, axis=0), np.concatenate(all_labels, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latent class separation using kNN.")
    parser.add_argument("--config", type=str, default="0_naive_autoencoder.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path. Defaults to output.checkpoint_path from config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    device = resolve_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint or cfg["output"]["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    splits = build_mnist_splits(cfg)
    model = load_model(cfg, checkpoint_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    batch_size = int(cfg["training"]["batch_size"])
    requested_num_workers = int(cfg["data"]["num_workers"])
    num_workers = resolve_num_workers(requested_num_workers)
    print(f"DataLoader workers: requested={requested_num_workers}, effective={num_workers}")

    train_latents, train_labels = encode_dataset(
        model, splits.train, batch_size, num_workers, device, split_name="train"
    )
    test_latents, test_labels = encode_dataset(
        model, splits.test, batch_size, num_workers, device, split_name="test"
    )

    k = int(cfg["analysis"]["k"])
    metric = str(cfg["analysis"]["metric"])
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(train_latents, train_labels)
    acc = knn.score(test_latents, test_labels)

    report = classification_report(test_labels, knn.predict(test_latents), digits=4)
    print(f"kNN latent-space test accuracy: {acc:.4f}")
    print(report)

    experiment_name = str(cfg["experiment"]["name"])
    output_cfg = cfg.get("output", {})
    analysis_path = output_cfg.get("analysis_path", f"./artifacts/{experiment_name}_analysis.txt")
    analysis_path = Path(analysis_path)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(f"experiment: {experiment_name}\n")
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"parameters_total: {total_params}\n")
        f.write(f"parameters_trainable: {trainable_params}\n")
        f.write(f"kNN latent-space test accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Saved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
