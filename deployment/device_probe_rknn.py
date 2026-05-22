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


def read_device_compatible() -> str:
    try:
        return (
            Path("/proc/device-tree/compatible")
            .read_bytes()
            .replace(b"\x00", b"\n")
            .decode("utf-8", errors="ignore")
            .lower()
        )
    except Exception:
        return ""


def select_npu_core_mask(rknn_lite_cls):
    compatible = read_device_compatible()
    if "rk3588" in compatible:
        candidates = ["NPU_CORE_0_1_2", "NPU_CORE_ALL", "NPU_CORE_AUTO"]
    elif "rk356" in compatible:
        candidates = ["NPU_CORE_0", "NPU_CORE_AUTO"]
    else:
        candidates = ["NPU_CORE_AUTO", "NPU_CORE_ALL", "NPU_CORE_0_1_2", "NPU_CORE_0"]

    for attr in candidates:
        if hasattr(rknn_lite_cls, attr):
            return int(getattr(rknn_lite_cls, attr)), attr
    return None, "default"


def main() -> None:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"RKNN model not found: {model_path}")

    rknn = RKNNLite()

    ret = rknn.load_rknn(str(model_path))
    check_ret(ret, "load_rknn")

    selected_mask, selected_name = select_npu_core_mask(RKNNLite)
    if selected_mask is None:
        print("NPU core mask: default init_runtime()")
        ret = rknn.init_runtime()
    else:
        print(f"NPU core mask: {selected_name}={selected_mask}")
        ret = rknn.init_runtime(core_mask=selected_mask)
        if ret != 0 and selected_name not in ("NPU_CORE_AUTO", "default") and hasattr(RKNNLite, "NPU_CORE_AUTO"):
            print(f"init_runtime with {selected_name} failed, retrying NPU_CORE_AUTO")
            ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
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
