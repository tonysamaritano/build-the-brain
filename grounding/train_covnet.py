from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from autoencoder_common import (
    apply_ssl_augmentations,
    build_mnist_splits,
    build_model_from_config,
    resolve_device,
    resolve_num_workers,
    set_seed,
)


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def reconstruction_target(images: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    if recon.ndim == 2:
        return images.view(images.size(0), -1)
    return images


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, cfg: dict[str, Any]) -> float:
    model.eval()
    running = 0.0
    total = 0
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            recon = model(images)
            target = reconstruction_target(images, recon)
            loss = criterion(recon, target)
            batch_size = images.size(0)
            running += loss.item() * batch_size
            total += batch_size
    return running / max(total, 1)


def build_self_supervised_train_dataset(train_subset: Any, cfg: dict[str, Any]) -> TensorDataset:
    ssl_cfg = cfg.get("self_supervised", {})
    n_aug = int(ssl_cfg.get("augmentations_per_sample", 1))
    if n_aug < 1:
        raise ValueError("self_supervised.augmentations_per_sample must be >= 1")

    precompute = bool(ssl_cfg.get("precompute", True))
    cache_path_str = ssl_cfg.get("cache_path", "")
    cache_path = Path(cache_path_str) if cache_path_str else None

    if precompute and cache_path and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        print(f"Loaded precomputed augmented dataset from {cache_path}")
        return TensorDataset(payload["inputs"], payload["targets"])

    aug_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    for idx in tqdm(range(len(train_subset)), desc="precompute ssl", leave=False):
        image, _ = train_subset[idx]
        clean = image.unsqueeze(0)
        clean_batch = clean.repeat(n_aug, 1, 1, 1)
        augmented_batch = apply_ssl_augmentations(clean_batch, cfg)
        aug_batches.append(augmented_batch)
        target_batches.append(clean_batch)

    inputs = torch.cat(aug_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    print(
        f"Precomputed SSL samples: base={len(train_subset)} "
        f"augmentations_per_sample={n_aug} total={inputs.size(0)}"
    )

    if precompute and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"inputs": inputs, "targets": targets}, cache_path)
        print(f"Saved precomputed augmented dataset to {cache_path}")

    return TensorDataset(inputs, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a convolutional MNIST autoencoder.")
    parser.add_argument(
        "--config",
        type=str,
        default="grounding/configs/1_naive_covnet_autoencoder.yaml",
        help="Path to YAML experiment config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))

    device = resolve_device(cfg["training"]["device"])
    splits = build_mnist_splits(cfg)
    print(f"Using device: {device}")

    ssl_cfg = cfg.get("self_supervised", {})
    print(
        "Self-supervised augmentations: "
        f"enabled={bool(ssl_cfg.get('enabled', False))} "
        f"augmentations_per_sample={int(ssl_cfg.get('augmentations_per_sample', 1))}"
    )

    batch_size = int(cfg["training"]["batch_size"])
    requested_num_workers = int(cfg["data"]["num_workers"])
    num_workers = resolve_num_workers(requested_num_workers)
    print(f"DataLoader workers: requested={requested_num_workers}, effective={num_workers}")

    ssl_enabled = bool(ssl_cfg.get("enabled", False))
    if ssl_enabled:
        train_dataset = build_self_supervised_train_dataset(splits.train, cfg)
    else:
        train_dataset = splits.train

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model_from_config(cfg).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    epochs = int(cfg["training"]["epochs"])
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        model.train()
        running = 0.0
        total = 0

        for batch_a, batch_b in tqdm(train_loader, desc=f"train e{epoch:03d}", leave=False):
            if ssl_enabled:
                inputs = batch_a.to(device)
                clean_targets = batch_b.to(device)
            else:
                inputs = batch_a.to(device)
                clean_targets = inputs

            optimizer.zero_grad()
            recon = model(inputs)
            target = reconstruction_target(clean_targets, recon)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()

            batch_size_now = inputs.size(0)
            running += loss.item() * batch_size_now
            total += batch_size_now

        train_loss = running / max(total, 1)
        val_loss = evaluate(model, val_loader, device, criterion, cfg)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    output_cfg = cfg["output"]
    ckpt_path = Path(output_cfg["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
