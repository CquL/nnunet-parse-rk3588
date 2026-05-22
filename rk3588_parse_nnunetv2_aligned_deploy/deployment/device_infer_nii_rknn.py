from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    from rknnlite.api import RKNNLite
except Exception as exc:
    raise SystemExit(
        "Cannot import rknnlite.api.RKNNLite. Install RKNN Toolkit Lite2/runtime on the RK3588 device first.\n"
        f"Original error: {exc!r}"
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "parse_3d_fullres_patch_32x64x64.rknn"

# From nnUNet plans.json foreground_intensity_properties_per_channel for CT.
CT_CLIP_LOW = -778.0
CT_CLIP_HIGH = 445.0
CT_MEAN = 42.44314956665039
CT_STD = 301.8079528808594


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.lower().replace(",", "x").split("x")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Shape must look like 32x64x64")
    return tuple(parts)


def normalize_ct(image_zyx: np.ndarray) -> np.ndarray:
    image = image_zyx.astype(np.float32, copy=False)
    image = np.clip(image, CT_CLIP_LOW, CT_CLIP_HIGH)
    image = (image - CT_MEAN) / CT_STD
    return image.astype(np.float32, copy=False)


def start_positions(size: int, patch: int, step: int) -> list[int]:
    if size <= patch:
        return [0]

    starts = list(range(0, size - patch + 1, step))
    last = size - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def pad_to_minimum(image: np.ndarray, patch_size: tuple[int, int, int]) -> tuple[np.ndarray, tuple[int, int, int]]:
    pads = []
    for size, patch in zip(image.shape, patch_size):
        pads.append(max(0, patch - size))
    padded = np.pad(image, [(0, p) for p in pads], mode="constant", constant_values=0)
    return padded, tuple(pads)


def output_to_logits(output: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    expected = (1, 2, *patch_size)
    output = np.asarray(output)

    if output.shape == expected:
        return output[0]

    # Some RKNN layouts are flattened or internally transformed. For this model
    # the total element count is unambiguous: 1 * 2 * D * H * W.
    if output.size == np.prod(expected):
        return output.reshape(expected)[0]

    raise ValueError(f"Unexpected RKNN output shape: {output.shape}, expected {expected}")


def load_rknn(model_path: Path) -> RKNNLite:
    rknn = RKNNLite()

    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed, ret={ret}")

    if hasattr(RKNNLite, "NPU_CORE_AUTO"):
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    else:
        ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")

    return rknn


def infer_patch(rknn: RKNNLite, patch_zyx: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    input_tensor = patch_zyx[None, None].astype(np.float32, copy=False)
    outputs = rknn.inference(inputs=[input_tensor])
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("RKNN inference returned no outputs")
    return output_to_logits(outputs[0], patch_size)


def write_mask(mask_zyx: np.ndarray, reference: sitk.Image, output_path: Path) -> None:
    mask_image = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    mask_image.CopyInformation(reference)
    sitk.WriteImage(mask_image, str(output_path))


def write_patch_mask(
    mask_zyx: np.ndarray,
    reference: sitk.Image,
    start_zyx: tuple[int, int, int],
    output_path: Path,
) -> None:
    mask_image = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    mask_image.SetSpacing(reference.GetSpacing())
    mask_image.SetDirection(reference.GetDirection())

    z, y, x = start_zyx
    origin = reference.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
    mask_image.SetOrigin(origin)
    sitk.WriteImage(mask_image, str(output_path))


def run_center_patch(
    rknn: RKNNLite,
    image: np.ndarray,
    reference: sitk.Image,
    output_path: Path,
    patch_size: tuple[int, int, int],
) -> None:
    image, _ = pad_to_minimum(image, patch_size)
    starts = tuple(max(0, (size - patch) // 2) for size, patch in zip(image.shape, patch_size))
    z, y, x = starts
    dz, dy, dx = patch_size

    patch = image[z : z + dz, y : y + dy, x : x + dx]
    logits = infer_patch(rknn, patch, patch_size)
    mask = np.argmax(logits, axis=0).astype(np.uint8)

    write_patch_mask(mask, reference, starts, output_path)
    print(f"Center patch start zyx={starts}, output={output_path}")


def run_full_volume(
    rknn: RKNNLite,
    image: np.ndarray,
    reference: sitk.Image,
    output_path: Path,
    patch_size: tuple[int, int, int],
    step_size: tuple[int, int, int],
) -> None:
    original_shape = image.shape
    image, _ = pad_to_minimum(image, patch_size)
    dz, dy, dx = patch_size
    sz, sy, sx = step_size

    z_starts = start_positions(image.shape[0], dz, sz)
    y_starts = start_positions(image.shape[1], dy, sy)
    x_starts = start_positions(image.shape[2], dx, sx)
    total = len(z_starts) * len(y_starts) * len(x_starts)

    score_sum = np.zeros((2, *image.shape), dtype=np.float32)
    count = np.zeros(image.shape, dtype=np.uint16)

    done = 0
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = image[z : z + dz, y : y + dy, x : x + dx]
                logits = infer_patch(rknn, patch, patch_size)
                score_sum[:, z : z + dz, y : y + dy, x : x + dx] += logits
                count[z : z + dz, y : y + dy, x : x + dx] += 1

                done += 1
                if done == 1 or done % 25 == 0 or done == total:
                    print(f"Processed patches: {done}/{total}", flush=True)

    count = np.maximum(count, 1)
    score_sum /= count[None]
    mask = np.argmax(score_sum, axis=0).astype(np.uint8)
    mask = mask[: original_shape[0], : original_shape[1], : original_shape[2]]

    write_mask(mask, reference, output_path)
    print(f"Full-volume output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a .nii.gz file through the small RKNN PARSE probe model.")
    parser.add_argument("-i", "--input", required=True, help="Input .nii.gz CT volume")
    parser.add_argument("-o", "--output", required=True, help="Output .nii.gz segmentation mask")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="RKNN model path")
    parser.add_argument("--patch-size", type=parse_shape, default=(32, 64, 64))
    parser.add_argument("--step-size", type=parse_shape, default=(32, 64, 64))
    parser.add_argument(
        "--center-patch-only",
        action="store_true",
        help="Only run one center patch. Use this first to test device inference quickly.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    itk_image = sitk.ReadImage(str(input_path))
    image = sitk.GetArrayFromImage(itk_image)
    image = normalize_ct(image)

    print(f"Input NIfTI shape zyx={image.shape}")
    print(f"RKNN patch input shape nchwd=(1, 1, {args.patch_size[0]}, {args.patch_size[1]}, {args.patch_size[2]})")

    rknn = load_rknn(model_path)
    try:
        if args.center_patch_only:
            run_center_patch(rknn, image, itk_image, output_path, args.patch_size)
        else:
            run_full_volume(rknn, image, itk_image, output_path, args.patch_size, args.step_size)
    finally:
        rknn.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        raise
