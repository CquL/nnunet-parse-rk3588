import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_mask(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two binary NIfTI masks.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    reference = read_mask(Path(args.reference))
    candidate = read_mask(Path(args.candidate))
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, candidate={candidate.shape}")

    intersection = np.logical_and(reference, candidate).sum()
    ref_count = reference.sum()
    cand_count = candidate.sum()
    union = np.logical_or(reference, candidate).sum()
    dice = (2 * intersection) / (ref_count + cand_count) if ref_count + cand_count else 1.0
    iou = intersection / union if union else 1.0

    print(f"shape: {reference.shape}")
    print(f"reference foreground: {int(ref_count)}")
    print(f"candidate foreground: {int(cand_count)}")
    print(f"intersection: {int(intersection)}")
    print(f"dice: {dice:.6f}")
    print(f"iou: {iou:.6f}")


if __name__ == "__main__":
    main()
