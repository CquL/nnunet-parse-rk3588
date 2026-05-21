# nnUNet PARSE RK3588 Deployment Package

This private repository records the local nnUNetv2 PARSE inference and RK3588/RKNN deployment work.

The full binary package is intended to be uploaded as a GitHub Release asset:

- `nnunet_parse_rk3588_full_package.zip`

## What This Contains

- Local nnUNetv2 inference scripts for Windows/Conda.
- RK3588 probe scripts for loading and running the converted RKNN model.
- Conversion scripts for `nnUNetv2 -> ONNX -> RKNN`.
- MegaFlow platform workflow PDF reference.

## Main Documents

- `README_GENERAL_PTH_TO_RKNN.md`  
  General `.pth/.pt -> ONNX -> RKNN` workflow and environment notes.

- `README_PARSE_RK3588_DEPLOYMENT.md`  
  End-to-end notes for this PARSE nnUNetv2 RK3588 deployment.

- `README_RK3588_DEPLOYMENT_ISSUES.md`  
  Problems encountered on RK3588 and the fixes used.

- `README_PTH_TO_RKNN.md`  
  Shorter project-specific conversion notes.

## Main Scripts

- `export_nnunet_patch_onnx.py`  
  PARSE/nnUNetv2-specific exporter from `checkpoint_best.pth` to ONNX.

- `convert_onnx_to_rknn.py`  
  Generic ONNX-to-RKNN converter for RK3588.

- `deployment/device_probe_rknn.py`  
  RK3588 device probe for loading and running the `.rknn` model.

- `deployment/device_infer_nii_rknn.py`  
  RK3588 NIfTI inference wrapper: `.nii.gz -> tensor -> RKNN -> .nii.gz`.

- `export_rk3588_compare.py`  
  Exports side-by-side PNG comparisons for RK3588 `.nii.gz` outputs.

## Important Model Notes

The original nnUNetv2 network uses a 3D patch input:

```text
1 x 1 x 96 x 160 x 160
```

The first RKNN file that successfully compiled is a smaller feasibility probe:

```text
1 x 1 x 32 x 64 x 64
```

This smaller RKNN is for testing whether the RK3588 device can load and execute the model through RKNN Runtime. It is not expected to match the original nnUNetv2 full-resolution segmentation quality yet.

## Main Device Test

On the RK3588 Ubuntu device, after installing RKNN Runtime / RKNN Toolkit Lite2 and Python dependencies:

```bash
cd deployment
bash check_rk3588_env.sh
python3 device_probe_rknn.py
```

Quick NIfTI center-patch test:

```bash
python3 device_infer_nii_rknn.py \
  -i PA000005_0000.nii.gz \
  -o PA000005_center_patch_rknn.nii.gz \
  --center-patch-only
```

## Full Package Layout

Large binary artifacts are intentionally not committed to git. They are uploaded as private GitHub Release assets instead. The release zip includes:

- `nnunetv2_PARSE_model_minimal/`
- `nnunetv2_PARSE_test_images/`
- `deployment/`
- `predictions/`
- `preview_png/`
- Windows/Conda helper scripts
- RKNN/ONNX conversion scripts
- MegaFlow PDF reference

Duplicate original zip files and temporary conversion intermediates are intentionally excluded from the release package to keep the upload size manageable.
