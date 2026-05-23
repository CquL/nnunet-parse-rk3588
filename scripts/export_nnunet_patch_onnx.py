import os
import argparse
from pathlib import Path

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


ROOT = Path(__file__).resolve().parents[1]
MODEL_FOLDER = ROOT / "nnunetv2_PARSE_model_minimal" / "Dataset501_PARSE" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
OUTPUT_DIR = ROOT / "deployment"


def parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.lower().replace(",", "x").split("x")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Patch size must look like 96x160x160")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the nnUNet PARSE network patch model to ONNX.")
    parser.add_argument("--patch-size", type=parse_patch_size, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["nnUNet_results"] = str(ROOT / "nnunetv2_PARSE_model_minimal")
    os.environ["nnUNet_raw"] = str(ROOT / "nnUNet_raw_placeholder")
    os.environ["nnUNet_preprocessed"] = str(ROOT / "nnUNet_preprocessed_placeholder")

    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = nnUNetPredictor(
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_FOLDER),
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )

    network = predictor.network.to(device)
    network.eval()

    patch_size = args.patch_size or tuple(int(x) for x in predictor.configuration_manager.patch_size)
    output_onnx = Path(args.output) if args.output else OUTPUT_DIR / f"parse_3d_fullres_patch_{patch_size[0]}x{patch_size[1]}x{patch_size[2]}.onnx"
    dummy_input = torch.randn((1, 1, *patch_size), dtype=torch.float32, device=device)

    with torch.no_grad():
        dummy_output = network(dummy_input)
        print(f"Input shape:  {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(dummy_output.shape)}")

        torch.onnx.export(
            network,
            dummy_input,
            str(output_onnx),
            input_names=["image_patch"],
            output_names=["logits"],
            opset_version=19,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"Exported ONNX: {output_onnx}")


if __name__ == "__main__":
    main()
