from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch


def parse_shape(value: str) -> tuple[int, ...]:
    parts = value.lower().replace(",", "x").split("x")
    try:
        shape = tuple(int(part) for part in parts if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Shape must look like 1x3x224x224") from exc
    if not shape:
        raise argparse.ArgumentTypeError("Shape cannot be empty")
    return shape


def import_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("Use module:object format, for example my_model:MyModel")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_state_dict(checkpoint: Any, state_key: str | None) -> dict[str, torch.Tensor]:
    if state_key:
        for key in state_key.split("."):
            checkpoint = checkpoint[key]
        return checkpoint

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "network_weights"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint

    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def clean_state_dict(state_dict: dict[str, Any], strip_prefix: str) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        if strip_prefix and key.startswith(strip_prefix):
            key = key[len(strip_prefix) :]
        cleaned[key] = value
    return cleaned


def build_model(model_spec: str | None, kwargs_json: str, checkpoint: Any) -> torch.nn.Module:
    if model_spec:
        factory = import_object(model_spec)
        kwargs = json.loads(kwargs_json) if kwargs_json else {}
        model = factory(**kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"{model_spec} did not return a torch.nn.Module")
        return model

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    raise ValueError("Checkpoint is not a full torch.nn.Module. Please pass --model module:Class")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic PyTorch .pth/.pt to ONNX exporter.")
    parser.add_argument("--weights", required=True, help="Input .pth/.pt checkpoint path")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument(
        "--model",
        default=None,
        help="Import path for model class/factory, for example my_model:MyModel",
    )
    parser.add_argument(
        "--model-kwargs-json",
        default="{}",
        help='JSON kwargs for model constructor, for example \'{"num_classes": 2}\'',
    )
    parser.add_argument("--state-key", default=None, help="Optional checkpoint key for state_dict")
    parser.add_argument("--strip-prefix", default="module.", help="Prefix to remove from state_dict keys")
    parser.add_argument("--input-shape", required=True, type=parse_shape, help="Example: 1x3x224x224")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--no-strict", action="store_true", help="Load state_dict with strict=False")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location="cpu")
    model = build_model(args.model, args.model_kwargs_json, checkpoint)

    if not isinstance(checkpoint, torch.nn.Module):
        state_dict = clean_state_dict(get_state_dict(checkpoint, args.state_key), args.strip_prefix)
        model.load_state_dict(state_dict, strict=not args.no_strict)

    model.to(device)
    model.eval()

    dummy_input = torch.randn(*args.input_shape, dtype=torch.float32, device=device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        dummy_output = model(dummy_input)
        if isinstance(dummy_output, (tuple, list)):
            output_names = [f"{args.output_name}_{i}" for i in range(len(dummy_output))]
            output_shapes = [tuple(out.shape) for out in dummy_output if hasattr(out, "shape")]
        else:
            output_names = [args.output_name]
            output_shapes = [tuple(dummy_output.shape)] if hasattr(dummy_output, "shape") else [type(dummy_output)]

        print(f"Input shape: {tuple(dummy_input.shape)}")
        print(f"Output shapes: {output_shapes}")

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=[args.input_name],
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Exported ONNX: {output_path}")


if __name__ == "__main__":
    main()
