from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def dice_score(pred: np.ndarray, label: np.ndarray, foreground_label: int) -> float:
    pred_fg = pred == foreground_label
    label_fg = label == foreground_label
    pred_sum = int(pred_fg.sum())
    label_sum = int(label_fg.sum())
    if pred_sum + label_sum == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_fg, label_fg).sum() / (pred_sum + label_sum))


def hd95_score(pred_image: sitk.Image, label_image: sitk.Image, foreground_label: int) -> float:
    pred_binary = sitk.Cast(sitk.Equal(pred_image, foreground_label), sitk.sitkUInt8)
    label_binary = sitk.Cast(sitk.Equal(label_image, foreground_label), sitk.sitkUInt8)

    pred_arr = sitk.GetArrayViewFromImage(pred_binary)
    label_arr = sitk.GetArrayViewFromImage(label_binary)
    pred_sum = int(pred_arr.sum())
    label_sum = int(label_arr.sum())
    if pred_sum + label_sum == 0:
        return 0.0
    if pred_sum == 0 or label_sum == 0:
        return float("inf")

    pred_surface = sitk.LabelContour(pred_binary)
    label_surface = sitk.LabelContour(label_binary)
    pred_surface_arr = sitk.GetArrayViewFromImage(pred_surface).astype(bool)
    label_surface_arr = sitk.GetArrayViewFromImage(label_surface).astype(bool)

    distance_to_pred = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            pred_binary,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )
    distance_to_label = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            label_binary,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )

    pred_to_label = sitk.GetArrayViewFromImage(distance_to_label)[pred_surface_arr]
    label_to_pred = sitk.GetArrayViewFromImage(distance_to_pred)[label_surface_arr]
    distances = np.concatenate([pred_to_label, label_to_pred]).astype(np.float64, copy=False)
    if distances.size == 0:
        return float("inf")
    return float(np.percentile(distances, 95))


def read_image_and_mask(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute foreground Dice and HD95 for PARSE NIfTI segmentations.")
    parser.add_argument("--pred-dir", required=True, help="Directory with predicted CASE.nii.gz masks")
    parser.add_argument("--label-dir", required=True, help="Directory with label CASE.nii.gz masks")
    parser.add_argument("--label", type=int, default=1, help="Foreground label id")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    label_dir = Path(args.label_dir)
    csv_path = Path(args.csv) if args.csv else pred_dir / "metrics_eval.csv"

    rows: list[dict[str, str]] = []
    dice_values: list[float] = []
    hd95_values: list[float] = []
    for label_path in sorted(label_dir.glob("*.nii.gz")):
        case = label_path.name.removesuffix(".nii.gz")
        pred_path = pred_dir / f"{case}.nii.gz"
        if not pred_path.exists():
            rows.append({"case": case, "dice": "missing_prediction", "hd95_mm": "", "pred_voxels": "", "label_voxels": ""})
            continue

        pred_image, pred = read_image_and_mask(pred_path)
        label_image, label = read_image_and_mask(label_path)
        if pred.shape != label.shape:
            raise ValueError(f"Shape mismatch for {case}: pred={pred.shape}, label={label.shape}")

        pred_voxels = int((pred == args.label).sum())
        label_voxels = int((label == args.label).sum())
        dice = dice_score(pred, label, args.label)
        hd95 = hd95_score(pred_image, label_image, args.label)
        dice_values.append(dice)
        if np.isfinite(hd95):
            hd95_values.append(hd95)
        rows.append(
            {
                "case": case,
                "dice": f"{dice:.6f}",
                "hd95_mm": "inf" if np.isinf(hd95) else f"{hd95:.6f}",
                "pred_voxels": str(pred_voxels),
                "label_voxels": str(label_voxels),
            }
        )
        hd95_text = "inf" if np.isinf(hd95) else f"{hd95:.6f}"
        print(
            f"{case}: Dice={dice:.6f}, HD95={hd95_text} mm, "
            f"pred_voxels={pred_voxels}, label_voxels={label_voxels}"
        )

    if dice_values:
        print(f"Mean Dice={float(np.mean(dice_values)):.6f} over {len(dice_values)} cases")
        if hd95_values:
            print(f"Mean finite HD95={float(np.mean(hd95_values)):.6f} mm over {len(hd95_values)} cases")
    else:
        print("No matched predictions were found.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "dice", "hd95_mm", "pred_voxels", "label_voxels"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
