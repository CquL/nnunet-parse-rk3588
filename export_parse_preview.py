import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


def normalize_to_uint8(volume: np.ndarray) -> tuple[np.ndarray, float, float]:
    flat = volume.ravel()
    step = max(1, flat.size // 1_000_000)
    sample = flat[::step]
    low, high = np.percentile(sample, [1, 99])
    if high <= low:
        low = float(np.min(sample))
        high = float(np.max(sample))
    if high <= low:
        high = low + 1.0

    scaled = (volume.astype(np.float32) - low) / (high - low)
    scaled = np.clip(scaled, 0, 1)
    return (scaled * 255).astype(np.uint8), float(low), float(high)


def pick_slices(mask: np.ndarray, count: int) -> list[int]:
    positive_slices = np.where(mask.any(axis=(1, 2)))[0]
    if len(positive_slices) == 0:
        z_count = mask.shape[0]
        return [int(x) for x in np.linspace(0, z_count - 1, min(count, z_count))]

    first = int(positive_slices[0])
    last = int(positive_slices[-1])
    picked = [int(round(x)) for x in np.linspace(first, last, min(count, last - first + 1))]
    return sorted(set(picked))


def save_overlay(image_slice: np.ndarray, mask_slice: np.ndarray, output_path: Path) -> None:
    rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float32)
    vessel = mask_slice > 0

    alpha = 0.55
    rgb[vessel, 0] = rgb[vessel, 0] * (1 - alpha) + 255 * alpha
    rgb[vessel, 1] = rgb[vessel, 1] * (1 - alpha)
    rgb[vessel, 2] = rgb[vessel, 2] * (1 - alpha)

    Image.fromarray(rgb.astype(np.uint8)).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PNG previews for nnUNet PARSE predictions.")
    parser.add_argument("--images", default=r"D:\nnunet_parse_run\nnunetv2_PARSE_test_images")
    parser.add_argument("--predictions", default=r"D:\nnunet_parse_run\predictions")
    parser.add_argument("--output", default=r"D:\nnunet_parse_run\preview_png")
    parser.add_argument("--slices", type=int, default=9)
    args = parser.parse_args()

    images_dir = Path(args.images)
    predictions_dir = Path(args.predictions)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = sorted(predictions_dir.glob("*.nii.gz"))
    if not prediction_files:
        raise FileNotFoundError(f"No .nii.gz prediction files found in {predictions_dir}")

    for pred_path in prediction_files:
        case_id = pred_path.name.removesuffix(".nii.gz")
        image_path = images_dir / f"{case_id}_0000.nii.gz"
        if not image_path.exists():
            print(f"Skip {case_id}: input image not found: {image_path}")
            continue

        image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
        if image.shape != mask.shape:
            raise ValueError(f"{case_id}: image shape {image.shape} != mask shape {mask.shape}")

        image_uint8, low, high = normalize_to_uint8(image)
        slices = pick_slices(mask, args.slices)

        print(f"{case_id}: exporting {len(slices)} slices, display window [{low:.1f}, {high:.1f}]")
        for z in slices:
            output_path = output_dir / f"{case_id}_z{z:03d}_overlay.png"
            save_overlay(image_uint8[z], mask[z], output_path)

    print(f"Done. PNG previews are in: {output_dir}")


if __name__ == "__main__":
    main()
