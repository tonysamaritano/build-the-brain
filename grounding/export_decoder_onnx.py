from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from autoencoder_common import build_model_from_config, resolve_device, set_seed


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export decoder to ONNX for browser inference.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))

    device = resolve_device(cfg["training"]["device"])
    checkpoint_path = args.checkpoint or cfg["output"]["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(cfg, checkpoint_path, device)
    decoder = DecoderWrapper(model).to(device)
    decoder.eval()

    latent_dim = int(cfg["model"]["latent_dim"])
    dummy = torch.zeros(1, latent_dim, device=device)

    experiment_name = str(cfg["experiment"]["name"])
    default_out = f"./artifacts/{experiment_name}_decoder.onnx"
    output_path = Path(args.output or default_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        decoder,
        dummy,
        output_path,
        input_names=["latent"],
        output_names=["reconstruction"],
        dynamic_axes={"latent": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
    )

    print(f"Saved ONNX decoder to {output_path}")


if __name__ == "__main__":
    main()
