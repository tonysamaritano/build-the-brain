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
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


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
        dropout_p: float = 0.0,
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
            if dropout_p > 0.0:
                encoder_layers.append(nn.Dropout(p=dropout_p))
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, dim), activation()])
            if dropout_p > 0.0:
                decoder_layers.append(nn.Dropout(p=dropout_p))
            prev_dim = dim
        decoder_layers.extend([nn.Linear(prev_dim, input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.encoder(x)


class NaiveConvAutoencoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation_name: str,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("model.hidden_dims must contain at least one channel size for naive_conv.")

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
        in_ch = input_channels
        for ch in hidden_dims:
            encoder_layers.append(
                nn.Conv2d(in_ch, ch, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            encoder_layers.append(activation())
            if dropout_p > 0.0:
                encoder_layers.append(nn.Dropout2d(p=dropout_p))
            in_ch = ch
        self.features = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            feature_map = self.features(dummy)
        self._feature_shape = feature_map.shape[1:]
        flat_dim = int(np.prod(self._feature_shape))

        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)

        decoder_layers: list[nn.Module] = []
        decoder_channels = list(reversed(hidden_dims))
        for i in range(len(decoder_channels) - 1):
            decoder_layers.append(
                nn.ConvTranspose2d(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            decoder_layers.append(activation())
            if dropout_p > 0.0:
                decoder_layers.append(nn.Dropout2d(p=dropout_p))
        decoder_layers.append(
            nn.ConvTranspose2d(
                decoder_channels[-1],
                input_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        )
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.to_latent(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        h = self.from_latent(z)
        h = h.view(h.size(0), *self._feature_shape)
        return self.decoder(h)


def build_model_from_config(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    reg_cfg = cfg.get("regularization", {})
    dropout_cfg = reg_cfg.get("dropout", {})
    dropout_p = float(dropout_cfg.get("p", 0.0)) if bool(dropout_cfg.get("enabled", False)) else 0.0
    architecture = str(model_cfg.get("architecture", "vanilla_fc")).lower()
    if architecture == "naive_conv":
        return NaiveConvAutoencoder(
            input_channels=int(model_cfg.get("input_channels", 1)),
            input_size=int(model_cfg.get("input_size", 28)),
            hidden_dims=[int(v) for v in model_cfg["hidden_dims"]],
            latent_dim=int(model_cfg["latent_dim"]),
            activation_name=str(model_cfg.get("activation", "relu")).lower(),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            stride=int(model_cfg.get("stride", 2)),
            padding=int(model_cfg.get("padding", 1)),
            output_padding=int(model_cfg.get("output_padding", 1)),
            dropout_p=dropout_p,
        )
    if architecture == "vanilla_fc":
        return VanillaAutoencoder(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dims=[int(v) for v in model_cfg["hidden_dims"]],
            latent_dim=int(model_cfg["latent_dim"]),
            activation_name=str(model_cfg["activation"]).lower(),
            dropout_p=dropout_p,
        )
    raise ValueError(f"Unsupported model.architecture='{architecture}'")


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


def apply_ssl_augmentations(images: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    ssl_cfg = cfg.get("self_supervised", {})
    if not bool(ssl_cfg.get("enabled", False)):
        return images

    aug_cfg = ssl_cfg.get("augmentations", {})
    out = images.clone()

    trans_cfg = aug_cfg.get("translation", {})
    rot_cfg = aug_cfg.get("rotation", {})
    inten_cfg = aug_cfg.get("intensity", {})

    for i in range(out.size(0)):
        img = out[i]

        if bool(trans_cfg.get("enabled", False)):
            max_shift = int(trans_cfg.get("max_shift", 0))
            if max_shift > 0:
                dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                img = TF.affine(
                    img,
                    angle=0.0,
                    translate=[dx, dy],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )

        if bool(rot_cfg.get("enabled", False)):
            max_degrees = float(rot_cfg.get("max_degrees", 0.0))
            if max_degrees > 0.0:
                angle = float((torch.rand(1).item() * 2.0 - 1.0) * max_degrees)
                img = TF.rotate(
                    img,
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )

        if bool(inten_cfg.get("enabled", False)):
            min_scale = float(inten_cfg.get("min_scale", 1.0))
            max_scale = float(inten_cfg.get("max_scale", 1.0))
            if max_scale > 0 and max_scale >= min_scale:
                scale = float(torch.empty(1).uniform_(min_scale, max_scale).item())
                img = img * scale

        out[i] = img

    noise_cfg = aug_cfg.get("noise", {})
    noise_std = float(noise_cfg.get("std", 0.0))
    if bool(noise_cfg.get("enabled", False)) and noise_std > 0.0:
        out = out + noise_std * torch.randn_like(out)

    mask_cfg = aug_cfg.get("masking", {})
    masking_prob = float(mask_cfg.get("prob", 0.0))
    if bool(mask_cfg.get("enabled", False)) and masking_prob > 0.0:
        keep_mask = torch.rand_like(out).gt(masking_prob)
        out = out * keep_mask

    clamp_min = float(aug_cfg.get("clamp_min", 0.0))
    clamp_max = float(aug_cfg.get("clamp_max", 1.0))
    out = torch.clamp(out, min=clamp_min, max=clamp_max)
    return out
