from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms


@dataclass
class DatasetSplits:
    train: Subset
    val: Subset
    test: Subset


class VanillaAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation_name: str,
    ) -> None:
        super().__init__()

        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        if activation_name not in activation_map:
            raise ValueError(
                f"Unsupported activation '{activation_name}'. "
                f"Expected one of: {sorted(activation_map)}"
            )
        activation = activation_map[activation_name]

        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, dim), activation()])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, dim), activation()])
            prev_dim = dim
        decoder_layers.extend([nn.Linear(prev_dim, input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    # Explicit priority for auto mode: CUDA -> MPS -> CPU.
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def resolve_num_workers(requested_num_workers: int) -> int:
    if requested_num_workers <= 0:
        return 0

    shm_manager = os.path.join(os.path.dirname(torch.__file__), "bin", "torch_shm_manager")
    if os.path.isfile(shm_manager) and not os.access(shm_manager, os.X_OK):
        print(
            f"torch_shm_manager is not executable at '{shm_manager}'. "
            "Falling back to num_workers=0."
        )
        return 0
    return requested_num_workers


def build_mnist_splits(cfg: dict[str, Any]) -> DatasetSplits:
    data_cfg = cfg["data"]
    transform = transforms.ToTensor()

    root = data_cfg["data_dir"]
    full_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    requested_train = int(data_cfg["train_samples"])
    requested_val = int(data_cfg["val_samples"])
    requested_test = int(data_cfg["test_samples"])

    available_train = len(full_train)
    available_test = len(full_test)

    if requested_train + requested_val > available_train:
        raise ValueError(
            f"train_samples + val_samples exceeds available MNIST train set "
            f"({requested_train} + {requested_val} > {available_train})"
        )
    if requested_test > available_test:
        raise ValueError(
            f"test_samples exceeds available MNIST test set "
            f"({requested_test} > {available_test})"
        )

    gen = torch.Generator().manual_seed(int(cfg["experiment"]["seed"]))
    train_subset, val_subset, _ = random_split(
        full_train,
        [requested_train, requested_val, available_train - requested_train - requested_val],
        generator=gen,
    )

    if requested_test == available_test:
        test_subset = Subset(full_test, list(range(available_test)))
    else:
        test_subset = Subset(full_test, list(range(requested_test)))

    return DatasetSplits(train=train_subset, val=val_subset, test=test_subset)
