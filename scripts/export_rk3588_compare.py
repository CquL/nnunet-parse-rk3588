import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw


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
    return sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def pick_slices(mask: np.ndarray, count: int) -> list[int]:
    positive = np.where(mask.any(axis=(1, 2)))[0]
    if len(positive) == 0:
        return [int(x) for x in np.linspace(0, mask.shape[0] - 1, min(count, mask.shape[0]))]
    first = int(positive[0])
    last = int(positive[-1])
    picked = [int(round(x)) for x in np.linspace(first, last, min(count, last - first + 1))]
    return sorted(set(picked))


def make_overlay(image_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float32)
    vessel = mask_slice > 0
    alpha = 0.55
    rgb[vessel, 0] = rgb[vessel, 0] * (1 - alpha) + 255 * alpha
    rgb[vessel, 1] = rgb[vessel, 1] * (1 - alpha)
    rgb[vessel, 2] = rgb[vessel, 2] * (1 - alpha)
    return rgb.astype(np.uint8)


def panel_title(panel: Image.Image, title: str) -> Image.Image:
    titled = Image.new("RGB", (panel.width, panel.height + 28), "white")
    titled.paste(panel, (0, 28))
    draw = ImageDraw.Draw(titled)
    draw.text((8, 7), title, fill=(0, 0, 0))
    return titled


def save_compare(image_slice: np.ndarray, mask_slice: np.ndarray, z: int, output_path: Path) -> None:
    input_panel = Image.fromarray(np.stack([image_slice] * 3, axis=-1))
    mask_panel = Image.fromarray((mask_slice > 0).astype(np.uint8) * 255).convert("RGB")
    overlay_panel = Image.fromarray(make_overlay(image_slice, mask_slice))

    panels = [
        panel_title(input_panel, f"Input CT z={z}"),
        panel_title(mask_panel, "RKNN mask"),
        panel_title(overlay_panel, "Overlay"),
    ]
    canvas = Image.new("RGB", (panels[0].width * 3, panels[0].height), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel.width, 0))
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export side-by-side PNG comparisons for an RK3588 NIfTI mask.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slices", type=int, default=9)
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_itk = sitk.ReadImage(str(image_path))
    mask_itk = resample_mask_to_image(sitk.ReadImage(str(mask_path)), image_itk)
    image = sitk.GetArrayFromImage(image_itk)
    mask = sitk.GetArrayFromImage(mask_itk)
    image_u8 = normalize_to_uint8(image)

    unique, counts = np.unique(mask, return_counts=True)
    print("Mask labels:", dict(zip(unique.tolist(), counts.tolist())))

    stem = nii_stem(mask_path)
    saved = []
    for z in pick_slices(mask, args.slices):
        path = output_dir / f"{stem}_z{z:03d}_compare.png"
        save_compare(image_u8[z], mask[z], z, path)
        saved.append(path)

    print(f"Saved {len(saved)} comparison PNGs in: {output_dir}")


if __name__ == "__main__":
    main()
