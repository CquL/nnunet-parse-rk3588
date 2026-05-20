import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONDA_ENV = "d2l_gpu"
DEVICE = "cuda"

MODEL_DIR = ROOT / "nnunetv2_PARSE_model_minimal"
INPUT_DIR = ROOT / "nnunetv2_PARSE_test_images"
OUTPUT_DIR = ROOT / "predictions"
RAW_DIR = ROOT / "nnUNet_raw_placeholder"
PREPROCESSED_DIR = ROOT / "nnUNet_preprocessed_placeholder"


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input image directory not found: {INPUT_DIR}")

    RAW_DIR.mkdir(exist_ok=True)
    PREPROCESSED_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_results"] = str(MODEL_DIR)
    env["nnUNet_raw"] = str(RAW_DIR)
    env["nnUNet_preprocessed"] = str(PREPROCESSED_DIR)

    command = [
        "cmd",
        "/c",
        "conda",
        "run",
        "-n",
        CONDA_ENV,
        "nnUNetv2_predict",
        "-i",
        str(INPUT_DIR),
        "-o",
        str(OUTPUT_DIR),
        "-d",
        "501",
        "-c",
        "3d_fullres",
        "-f",
        "0",
        "-tr",
        "nnUNetTrainer",
        "-p",
        "nnUNetPlans",
        "-chk",
        "checkpoint_best.pth",
        "-device",
        DEVICE,
        "--disable_tta",
        "--disable_progress_bar",
        "-npp",
        "1",
        "-nps",
        "1",
    ]

    subprocess.run(command, env=env, check=True)
    print(f"Done. Predictions are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
