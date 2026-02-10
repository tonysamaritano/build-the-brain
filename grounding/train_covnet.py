from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from autoencoder_common import (
    apply_denoising_noise,
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
    denoise_in_eval = bool(cfg.get("regularization", {}).get("denoising", {}).get("apply_in_eval", False))
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            inputs = apply_denoising_noise(images, cfg) if denoise_in_eval else images
            recon = model(inputs)
            target = reconstruction_target(images, recon)
            loss = criterion(recon, target)
            batch_size = images.size(0)
            running += loss.item() * batch_size
            total += batch_size
    return running / max(total, 1)


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

    denoise_cfg = cfg.get("regularization", {}).get("denoising", {})
    print(
        "Denoising: "
        f"enabled={bool(denoise_cfg.get('enabled', False))} "
        f"gaussian_std={float(denoise_cfg.get('gaussian_std', 0.0))} "
        f"masking_prob={float(denoise_cfg.get('masking_prob', 0.0))}"
    )

    batch_size = int(cfg["training"]["batch_size"])
    requested_num_workers = int(cfg["data"]["num_workers"])
    num_workers = resolve_num_workers(requested_num_workers)
    print(f"DataLoader workers: requested={requested_num_workers}, effective={num_workers}")

    train_loader = DataLoader(
        splits.train,
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

        for images, _ in tqdm(train_loader, desc=f"train e{epoch:03d}", leave=False):
            images = images.to(device)
            noisy_inputs = apply_denoising_noise(images, cfg)

            optimizer.zero_grad()
            recon = model(noisy_inputs)
            target = reconstruction_target(images, recon)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()

            batch_size_now = images.size(0)
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
