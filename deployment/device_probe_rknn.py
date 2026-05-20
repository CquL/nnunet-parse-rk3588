import sys
from pathlib import Path

import numpy as np

try:
    from rknnlite.api import RKNNLite
except Exception as exc:
    raise SystemExit(
        "Cannot import rknnlite.api.RKNNLite. Install RKNN Toolkit Lite2/runtime on the RK3588 device first.\n"
        f"Original error: {exc!r}"
    )


MODEL_PATH = Path(__file__).resolve().parent / "parse_3d_fullres_patch_32x64x64.rknn"


def check_ret(ret: int, step: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{step} failed, ret={ret}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"RKNN model not found: {MODEL_PATH}")

    rknn = RKNNLite()

    ret = rknn.load_rknn(str(MODEL_PATH))
    check_ret(ret, "load_rknn")

    if hasattr(RKNNLite, "NPU_CORE_AUTO"):
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    else:
        ret = rknn.init_runtime()
    check_ret(ret, "init_runtime")

    image_patch = np.random.randn(1, 1, 32, 64, 64).astype(np.float32)
    outputs = rknn.inference(inputs=[image_patch])
    if outputs is None:
        raise RuntimeError("inference returned None")

    print(f"Input shape: {image_patch.shape}, dtype={image_patch.dtype}")
    for index, output in enumerate(outputs):
        array = np.asarray(output)
        print(
            f"Output {index}: shape={array.shape}, dtype={array.dtype}, "
            f"min={array.min():.6f}, max={array.max():.6f}, mean={array.mean():.6f}"
        )

    rknn.release()
    print("RKNN probe finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RKNN probe failed: {exc}", file=sys.stderr)
        raise
