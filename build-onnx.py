#!/usr/bin/env python3
"""
Convert ISC21 PyTorch model to ONNX format.

Usage:
    python build-onnx.py --output isc_model.onnx
    python build-onnx.py --weight isc_ft_v107 --output isc_ft_v107.onnx
    python build-onnx.py --weight isc_selfsup_v98 --output isc_selfsup_v98.onnx
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ISCNetONNX(nn.Module):
    """
    ONNX-compatible ISCNet model.

    Differences from original:
    - eval_p is baked in (no training mode switching)
    - Explicit input shape handling for ONNX export
    """

    def __init__(
        self,
        backbone: nn.Module,
        fc_dim: int = 256,
        eval_p: float = 1.0,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(
            self.backbone.feature_info.info[-1]["num_chs"], fc_dim, bias=False
        )
        self.bn = nn.BatchNorm1d(fc_dim)
        self.eval_p = eval_p
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone(x)[-1]
        # GEM pooling with fixed eval_p
        x = self._gem(x, self.eval_p)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

    def _gem(self, x: torch.Tensor, p: float, eps: float = 1e-6) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))
        ).pow(1.0 / p)


def load_pytorch_model(
    weight_name: str = "isc_ft_v107",
    fc_dim: int = 256,
    eval_p: float = 1.0,
    l2_normalize: bool = True,
) -> tuple:
    """Load PyTorch model and return model, input_size, and normalization params."""

    AVAILABLE_MODELS = {
        "isc_selfsup_v98": "https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_selfsup_v98.pth.tar",
        "isc_ft_v107": "https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar",
    }

    if weight_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid weight name: {weight_name}, "
            f"available weights are: {list(AVAILABLE_MODELS.keys())}"
        )

    print(f"Loading checkpoint: {weight_name}")
    ckpt = torch.hub.load_state_dict_from_url(
        AVAILABLE_MODELS[weight_name],
        map_location="cpu",
    )

    arch = ckpt["arch"]
    input_size = ckpt["args"].input_size

    if arch == "tf_efficientnetv2_m_in21ft1k":
        arch = "timm/tf_efficientnetv2_m.in21k_ft_in1k"

    print(f"Creating backbone: {arch}")
    backbone = timm.create_model(arch, features_only=True)

    # Get normalization parameters from backbone
    mean = list(backbone.default_cfg["mean"])
    std = list(backbone.default_cfg["std"])

    model = ISCNetONNX(
        backbone=backbone,
        fc_dim=fc_dim,
        eval_p=eval_p,
        l2_normalize=l2_normalize,
    )

    # Load state dict (remove 'module.' prefix if present)
    state_dict = {}
    for key, value in ckpt["state_dict"].items():
        new_key = key.replace("module.", "")
        state_dict[new_key] = value

    # Handle fc_dim interpolation if needed
    if fc_dim != 256:
        print(f"Interpolating fc_dim from 256 to {fc_dim}")
        state_dict["fc.weight"] = F.interpolate(
            state_dict["fc.weight"].permute(1, 0).unsqueeze(0),
            size=fc_dim, mode="linear", align_corners=False,
        ).squeeze(0).permute(1, 0)
        for bn_param in ["bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]:
            state_dict[bn_param] = F.interpolate(
                state_dict[bn_param].unsqueeze(0).unsqueeze(0),
                size=fc_dim, mode="linear", align_corners=False,
            ).squeeze(0).squeeze(0)

    model.load_state_dict(state_dict)
    model.eval()

    return model, input_size, mean, std


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: int,
    dynamic_batch: bool = True,
):
    """Export PyTorch model to ONNX format using PyTorch 2.5+ dynamo exporter."""

    # Create example input as tuple (required for dynamo export)
    example_inputs = (torch.randn(1, 3, input_size, input_size),)

    # Define dynamic shapes for batch size
    dynamic_shapes = None
    if dynamic_batch:
        batch_dim = torch.export.Dim("batch_size", min=1, max=128)
        dynamic_shapes = {"x": {0: batch_dim}}

    print(f"Exporting to ONNX using dynamo=True: {output_path}")
    onnx_program = torch.onnx.export(
        model,
        example_inputs,
        dynamo=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_shapes=dynamic_shapes,
    )
    onnx_program.save(output_path)
    print(f"ONNX model saved to: {output_path}")


def save_config(
    config_path: str,
    input_size: int,
    mean: list,
    std: list,
    fc_dim: int,
    eval_p: float,
    l2_normalize: bool,
    weight_name: str,
):
    """Save model configuration for inference."""
    config = {
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "fc_dim": fc_dim,
        "eval_p": eval_p,
        "l2_normalize": l2_normalize,
        "weight_name": weight_name,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ISC21 PyTorch model to ONNX format"
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="isc_ft_v107",
        choices=["isc_ft_v107", "isc_selfsup_v98"],
        help="Weight name to use (default: isc_ft_v107)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="isc_model.onnx",
        help="Output ONNX file path (default: isc_model.onnx)",
    )
    parser.add_argument(
        "--fc-dim",
        type=int,
        default=256,
        help="Feature dimension (default: 256)",
    )
    parser.add_argument(
        "--eval-p",
        type=float,
        default=1.0,
        help="GEM pooling power for evaluation (default: 1.0)",
    )
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Disable L2 normalization of output embeddings",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Use static batch size instead of dynamic",
    )
    args = parser.parse_args()

    l2_normalize = not args.no_l2_normalize

    # Load PyTorch model
    model, input_size, mean, std = load_pytorch_model(
        weight_name=args.weight,
        fc_dim=args.fc_dim,
        eval_p=args.eval_p,
        l2_normalize=l2_normalize,
    )

    print(f"Model input size: {input_size}x{input_size}")
    print(f"Normalization mean: {mean}")
    print(f"Normalization std: {std}")
    print(f"Feature dimension: {args.fc_dim}")
    print(f"GEM eval_p: {args.eval_p}")
    print(f"L2 normalize: {l2_normalize}")

    # Export to ONNX using PyTorch 2.5+ dynamo exporter
    export_to_onnx(
        model=model,
        output_path=args.output,
        input_size=input_size,
        dynamic_batch=not args.static_batch,
    )

    # Save config file alongside ONNX model
    config_path = str(Path(args.output).with_suffix(".json"))
    save_config(
        config_path=config_path,
        input_size=input_size,
        mean=mean,
        std=std,
        fc_dim=args.fc_dim,
        eval_p=args.eval_p,
        l2_normalize=l2_normalize,
        weight_name=args.weight,
    )

    # Verify the ONNX model
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Test inference comparison
    print("\nComparing PyTorch vs ONNX outputs...")
    import onnxruntime as ort
    import numpy as np

    dummy_input = torch.randn(1, 3, input_size, input_size)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    # ONNX inference
    ort_session = ort.InferenceSession(args.output)
    onnx_output = ort_session.run(None, {"input": dummy_input.numpy()})[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"Max difference between PyTorch and ONNX: {max_diff:.6e}")

    if max_diff < 1e-5:
        print("Outputs match within tolerance!")
    else:
        print("Warning: Outputs differ more than expected")

    print(f"\nDone! Files created:")
    print(f"  - {args.output}")
    print(f"  - {config_path}")


if __name__ == "__main__":
    main()
