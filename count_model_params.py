import argparse
from pathlib import Path


def count_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto

    float_types = {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.DOUBLE: "float64",
        TensorProto.BFLOAT16: "bfloat16",
    }

    model = onnx.load(str(path), load_external_data=False)
    total_all = 0
    total_float = 0
    by_dtype = {}
    tensors = []

    for initializer in model.graph.initializer:
        count = 1
        for dim in initializer.dims:
            count *= int(dim)

        total_all += count
        by_dtype[initializer.data_type] = by_dtype.get(initializer.data_type, 0) + count
        if initializer.data_type in float_types:
            total_float += count
        tensors.append((count, initializer.name, list(initializer.dims), initializer.data_type))

    print(f"file: {path}")
    print(f"initializers: {len(model.graph.initializer):,}")
    print(f"all initializer elements: {total_all:,}")
    print(f"float parameter elements: {total_float:,}")
    print("by dtype:")
    for dtype, count in sorted(by_dtype.items(), key=lambda item: item[0]):
        print(f"  {float_types.get(dtype, str(dtype))}: {count:,}")

    print("largest tensors:")
    for count, name, dims, dtype in sorted(tensors, reverse=True)[:8]:
        print(f"  {count:,}\t{float_types.get(dtype, dtype)}\t{dims}\t{name}")


def count_pth(path: Path) -> None:
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("network_weights") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint.state_dict()

    tensors = []
    total = 0
    trainable_like = 0
    for name, tensor in state_dict.items():
        if not hasattr(tensor, "numel"):
            continue
        count = int(tensor.numel())
        total += count
        if getattr(tensor, "is_floating_point", lambda: False)():
            trainable_like += count
        tensors.append((count, name, tuple(tensor.shape), str(tensor.dtype)))

    print(f"file: {path}")
    print(f"state tensors: {len(tensors):,}")
    print(f"all tensor elements: {total:,}")
    print(f"floating tensor elements: {trainable_like:,}")
    print("largest tensors:")
    for count, name, shape, dtype in sorted(tensors, reverse=True)[:8]:
        print(f"  {count:,}\t{dtype}\t{shape}\t{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count model parameters in ONNX or PyTorch checkpoint files.")
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".onnx"):
        count_onnx(path)
    elif suffix.endswith(".pth") or suffix.endswith(".pt"):
        count_pth(path)
    else:
        raise SystemExit(f"Unsupported file type: {path}")


if __name__ == "__main__":
    main()
