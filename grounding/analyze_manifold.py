from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.manifold import trustworthiness
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    latents: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    inputs_flat: list[np.ndarray] = []

    with torch.no_grad():
        for images, y in tqdm(loader, desc=f"encode {split_name}", leave=False):
            images = images.to(device)
            z = model.encode(images).cpu().numpy()
            latents.append(z)
            labels.append(y.numpy())
            inputs_flat.append(images.view(images.size(0), -1).cpu().numpy())

    return (
        np.concatenate(latents, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(inputs_flat, axis=0),
    )


def label_neighbor_agreement(latents: np.ndarray, labels: np.ndarray, k: int) -> float:
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(latents)
    idx = nbrs.kneighbors(latents, return_distance=False)[:, 1:]
    neigh_labels = labels[idx]
    return float((neigh_labels == labels[:, None]).mean())


def fisher_ratio(latents: np.ndarray, labels: np.ndarray) -> float:
    overall_mean = latents.mean(axis=0)
    between = 0.0
    within = 0.0
    for cls in np.unique(labels):
        cls_points = latents[labels == cls]
        cls_mean = cls_points.mean(axis=0)
        n = cls_points.shape[0]
        between += float(n * np.sum((cls_mean - overall_mean) ** 2))
        within += float(np.sum((cls_points - cls_mean) ** 2))
    if within <= 0.0:
        return 0.0
    return between / within


def bounded_fisher_score(fr: float) -> float:
    return fr / (1.0 + fr)


def manifold_score(metrics: dict[str, float]) -> float:
    s = (metrics["silhouette"] + 1.0) / 2.0
    db = 1.0 / (1.0 + metrics["davies_bouldin"])
    parts = [
        metrics["knn_test_accuracy"],
        metrics["trustworthiness"],
        metrics["neighbor_label_agreement"],
        s,
        db,
        bounded_fisher_score(metrics["fisher_ratio"]),
    ]
    return float(np.mean(parts) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify latent manifold quality.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Split for manifold metrics. kNN is always train->test.",
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

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = resolve_num_workers(int(cfg["data"]["num_workers"]))

    z_train, y_train, x_train = encode_dataset(model, splits.train, batch_size, num_workers, device, "train")
    z_test, y_test, x_test = encode_dataset(model, splits.test, batch_size, num_workers, device, "test")

    k = int(cfg.get("analysis", {}).get("k", 5))
    k = max(1, k)

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(z_train, y_train)
    knn_acc = float(knn.score(z_test, y_test))

    if args.split == "train":
        z_eval, y_eval, x_eval = z_train, y_train, x_train
        split_name = "train"
    else:
        z_eval, y_eval, x_eval = z_test, y_test, x_test
        split_name = "test"

    trust_k = int(min(max(5, k), max(2, len(z_eval) - 1)))
    metrics = {
        "knn_test_accuracy": knn_acc,
        "silhouette": float(silhouette_score(z_eval, y_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(z_eval, y_eval)),
        "davies_bouldin": float(davies_bouldin_score(z_eval, y_eval)),
        "trustworthiness": float(trustworthiness(x_eval, z_eval, n_neighbors=trust_k)),
        "neighbor_label_agreement": float(label_neighbor_agreement(z_eval, y_eval, k=k)),
        "fisher_ratio": float(fisher_ratio(z_eval, y_eval)),
    }
    score = manifold_score(metrics)

    print(f"Manifold split: {split_name}")
    print(f"kNN test accuracy: {metrics['knn_test_accuracy']:.4f}")
    print(f"silhouette: {metrics['silhouette']:.4f}")
    print(f"calinski_harabasz: {metrics['calinski_harabasz']:.2f}")
    print(f"davies_bouldin: {metrics['davies_bouldin']:.4f}")
    print(f"trustworthiness@{trust_k}: {metrics['trustworthiness']:.4f}")
    print(f"neighbor_label_agreement@{k}: {metrics['neighbor_label_agreement']:.4f}")
    print(f"fisher_ratio: {metrics['fisher_ratio']:.4f}")
    print(f"manifold_score: {score:.2f}/100")

    experiment_name = str(cfg["experiment"]["name"])
    output_cfg = cfg.get("output", {})
    out_path = output_cfg.get("manifold_analysis_path", f"./artifacts/{experiment_name}_manifold_metrics.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": experiment_name,
        "checkpoint": str(checkpoint_path),
        "manifold_split": split_name,
        "k_neighbors": k,
        "trustworthiness_neighbors": trust_k,
        "metrics": metrics,
        "manifold_score": score,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved manifold metrics to {out_path}")


if __name__ == "__main__":
    main()
