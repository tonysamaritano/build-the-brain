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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a naive convolutional MNIST autoencoder.")
    parser.add_argument(
        "--config",
        type=str,
        default="1_naive_covnet_autoencoder.yaml",
        help="Path to YAML experiment config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))

    device = resolve_device(cfg["training"]["device"])
    splits = build_mnist_splits(cfg)
    print(f"Using device: {device}")

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
    output_cfg = cfg["output"]
    save_every_n_epochs = max(1, int(output_cfg.get("save_every_n_epochs", 1)))
    ckpt_path = Path(output_cfg["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_train_loss = float("nan")
    last_val_loss = float("nan")

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        model.train()
        running = 0.0
        total = 0

        for images, _ in tqdm(train_loader, desc=f"train e{epoch:03d}", leave=False):
            images = images.to(device)
            optimizer.zero_grad()
            recon = model(images)
            target = reconstruction_target(images, recon)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()

            batch_size_now = images.size(0)
            running += loss.item() * batch_size_now
            total += batch_size_now

        train_loss = running / max(total, 1)
        val_loss = evaluate(model, val_loader, device, criterion)
        last_train_loss = train_loss
        last_val_loss = val_loss

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
            flush=True,
        )

        if epoch % save_every_n_epochs == 0:
            epoch_ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_epoch{epoch:03d}{ckpt_path.suffix}")
            payload = {
                "state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(payload, epoch_ckpt_path)
            print(f"Saved periodic checkpoint: {epoch_ckpt_path}")

    final_payload = {
        "state_dict": model.state_dict(),
        "config": cfg,
        "epoch": epochs,
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
    }
    torch.save(final_payload, ckpt_path)
    print(f"Saved final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
