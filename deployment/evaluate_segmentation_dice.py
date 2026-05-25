from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def nii_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def dice_score(pred: np.ndarray, label: np.ndarray, foreground_label: int) -> float:
    pred_fg = pred == foreground_label
    label_fg = label == foreground_label
    pred_sum = int(pred_fg.sum())
    label_sum = int(label_fg.sum())
    if pred_sum + label_sum == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_fg, label_fg).sum() / (pred_sum + label_sum))


def overlap_counts(pred: np.ndarray, label: np.ndarray, foreground_label: int) -> dict[str, int]:
    pred_fg = pred == foreground_label
    label_fg = label == foreground_label
    intersection = np.logical_and(pred_fg, label_fg)
    false_positive = np.logical_and(pred_fg, ~label_fg)
    false_negative = np.logical_and(~pred_fg, label_fg)
    return {
        "intersection_voxels": int(intersection.sum()),
        "false_positive_voxels": int(false_positive.sum()),
        "false_negative_voxels": int(false_negative.sum()),
    }


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
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON output path")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    label_dir = Path(args.label_dir)
    csv_path = Path(args.csv) if args.csv else pred_dir / "metrics_eval.csv"
    summary_path = Path(args.summary_json) if args.summary_json else pred_dir / "metrics_summary.json"

    rows: list[dict[str, str]] = []
    dice_values: list[float] = []
    hd95_values: list[float] = []
    missing_count = 0
    failed_count = 0
    for label_path in sorted(label_dir.glob("*.nii.gz")):
        case = nii_stem(label_path)
        pred_path = pred_dir / f"{case}.nii.gz"
        if not pred_path.exists():
            missing_count += 1
            rows.append(
                {
                    "case": case,
                    "status": "missing_prediction",
                    "dice": "",
                    "hd95_mm": "",
                    "pred_voxels": "",
                    "label_voxels": "",
                    "intersection_voxels": "",
                    "false_positive_voxels": "",
                    "false_negative_voxels": "",
                    "pred_shape_zyx": "",
                    "label_shape_zyx": "",
                    "spacing_xyz": "",
                    "pred_path": str(pred_path),
                    "label_path": str(label_path),
                }
            )
            continue

        pred_image, pred = read_image_and_mask(pred_path)
        label_image, label = read_image_and_mask(label_path)
        if pred.shape != label.shape:
            failed_count += 1
            rows.append(
                {
                    "case": case,
                    "status": f"shape_mismatch_pred_{pred.shape}_label_{label.shape}",
                    "dice": "",
                    "hd95_mm": "",
                    "pred_voxels": "",
                    "label_voxels": "",
                    "intersection_voxels": "",
                    "false_positive_voxels": "",
                    "false_negative_voxels": "",
                    "pred_shape_zyx": str(tuple(int(i) for i in pred.shape)),
                    "label_shape_zyx": str(tuple(int(i) for i in label.shape)),
                    "spacing_xyz": str(tuple(float(i) for i in label_image.GetSpacing())),
                    "pred_path": str(pred_path),
                    "label_path": str(label_path),
                }
            )
            continue

        pred_voxels = int((pred == args.label).sum())
        label_voxels = int((label == args.label).sum())
        overlaps = overlap_counts(pred, label, args.label)
        dice = dice_score(pred, label, args.label)
        hd95 = hd95_score(pred_image, label_image, args.label)
        dice_values.append(dice)
        if np.isfinite(hd95):
            hd95_values.append(hd95)
        rows.append(
            {
                "case": case,
                "status": "ok",
                "dice": f"{dice:.6f}",
                "hd95_mm": "inf" if np.isinf(hd95) else f"{hd95:.6f}",
                "pred_voxels": str(pred_voxels),
                "label_voxels": str(label_voxels),
                "intersection_voxels": str(overlaps["intersection_voxels"]),
                "false_positive_voxels": str(overlaps["false_positive_voxels"]),
                "false_negative_voxels": str(overlaps["false_negative_voxels"]),
                "pred_shape_zyx": str(tuple(int(i) for i in pred.shape)),
                "label_shape_zyx": str(tuple(int(i) for i in label.shape)),
                "spacing_xyz": str(tuple(float(i) for i in label_image.GetSpacing())),
                "pred_path": str(pred_path),
                "label_path": str(label_path),
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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "status",
                "dice",
                "hd95_mm",
                "pred_voxels",
                "label_voxels",
                "intersection_voxels",
                "false_positive_voxels",
                "false_negative_voxels",
                "pred_shape_zyx",
                "label_shape_zyx",
                "spacing_xyz",
                "pred_path",
                "label_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {csv_path}")

    hd95_arr = np.asarray(hd95_values, dtype=np.float64)
    dice_arr = np.asarray(dice_values, dtype=np.float64)
    summary = {
        "generated_at_iso": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "num_cases": len(rows),
        "num_matched": len(dice_values),
        "num_missing_prediction": missing_count,
        "num_failed": failed_count,
        "dice_mean": float(dice_arr.mean()) if dice_arr.size else None,
        "dice_median": float(np.median(dice_arr)) if dice_arr.size else None,
        "dice_min": float(dice_arr.min()) if dice_arr.size else None,
        "dice_max": float(dice_arr.max()) if dice_arr.size else None,
        "hd95_finite_mean": float(hd95_arr.mean()) if hd95_arr.size else None,
        "hd95_finite_median": float(np.median(hd95_arr)) if hd95_arr.size else None,
        "hd95_finite_min": float(hd95_arr.min()) if hd95_arr.size else None,
        "hd95_finite_max": float(hd95_arr.max()) if hd95_arr.size else None,
        "hd95_inf_count": int(sum(1 for row in rows if row.get("hd95_mm") == "inf")),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Summary JSON written: {summary_path}")


if __name__ == "__main__":
    main()
