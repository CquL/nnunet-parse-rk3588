import argparse
from pathlib import Path

from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_MODEL = ROOT / "deployment" / "parse_3d_fullres_patch_96x160x160.onnx"


def check(ret: int, step: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{step} failed, ret={ret}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the exported nnUNet PARSE ONNX patch model to RKNN.")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX_MODEL))
    parser.add_argument("--rknn", default=None)
    args = parser.parse_args()

    onnx_model = Path(args.onnx)
    rknn_model = Path(args.rknn) if args.rknn else onnx_model.with_suffix(".rknn")

    rknn = RKNN(verbose=False)

    # The nnUNet patch model expects preprocessed CT values, so do not add
    # image-style RGB mean/std normalization here.
    ret = rknn.config(target_platform="rk3588")
    check(ret, "config")

    ret = rknn.load_onnx(model=str(onnx_model))
    check(ret, "load_onnx")

    # First try FP mode. INT8 needs calibration data and should be a second step
    # only after the FP model can compile and run.
    ret = rknn.build(do_quantization=False)
    check(ret, "build")

    ret = rknn.export_rknn(str(rknn_model))
    check(ret, "export_rknn")

    rknn.release()
    print(f"Exported RKNN: {rknn_model}")


if __name__ == "__main__":
    main()
