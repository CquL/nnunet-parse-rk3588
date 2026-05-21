import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


def nii_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    flat = volume.ravel()
    step = max(1, flat.size // 1_000_000)
    sample = flat[::step]
    low, high = np.percentile(sample, [1, 99])
    if high <= low:
        low, high = float(np.min(sample)), float(np.max(sample))
    if high <= low:
        high = low + 1.0

    scaled = (volume.astype(np.float32) - low) / (high - low)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def resample_mask_to_image(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    same_grid = (
        mask.GetSize() == image.GetSize()
        and mask.GetSpacing() == image.GetSpacing()
        and mask.GetOrigin() == image.GetOrigin()
        and mask.GetDirection() == image.GetDirection()
    )
    if same_grid:
        return mask

    return sitk.Resample(
        mask,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def pick_slices(mask: np.ndarray, count: int) -> list[int]:
    positive = np.where(mask.any(axis=(1, 2)))[0]
    if len(positive) == 0:
        return [int(x) for x in np.linspace(0, mask.shape[0] - 1, min(count, mask.shape[0]))]

    first = int(positive[0])
    last = int(positive[-1])
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
    parser = argparse.ArgumentParser(description="Export PNG overlays for an RK3588 NIfTI mask.")
    parser.add_argument("--image", required=True, help="Original CT .nii.gz, for example PA000005_0000.nii.gz")
    parser.add_argument("--mask", required=True, help="RK3588 output mask .nii.gz")
    parser.add_argument("--output", required=True, help="Output folder for PNG previews")
    parser.add_argument("--slices", type=int, default=9)
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_itk = sitk.ReadImage(str(image_path))
    mask_itk = sitk.ReadImage(str(mask_path))
    mask_itk = resample_mask_to_image(mask_itk, image_itk)

    image = sitk.GetArrayFromImage(image_itk)
    mask = sitk.GetArrayFromImage(mask_itk)
    image_u8 = normalize_to_uint8(image)

    unique, counts = np.unique(mask, return_counts=True)
    print("Mask labels:", dict(zip(unique.tolist(), counts.tolist())))

    stem = nii_stem(mask_path)
    for z in pick_slices(mask, args.slices):
        save_overlay(image_u8[z], mask[z], output_dir / f"{stem}_z{z:03d}_overlay.png")

    print(f"Done. PNG previews are in: {output_dir}")


if __name__ == "__main__":
    main()
