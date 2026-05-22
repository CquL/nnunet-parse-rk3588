# RK3588 Deployment Steps

This folder is for testing the PARSE nnUNetv2 RKNN model on an RK3588 Ubuntu device.

## 1. Install Runtime Dependencies

```bash
cd deployment
bash install_rk3588_deps.sh
```

If the board has no internet, install these packages manually:

```bash
python3 -m pip install numpy SimpleITK rknn-toolkit-lite2
python3 -m pip install scipy scikit-image
```

## 2. Check Device Environment

```bash
bash check_rk3588_env.sh
```

Important lines:

- `uname -m` should be `aarch64`
- `rknnlite import: OK`
- `scipy import: OK` and `scikit-image import: OK` are recommended for the closer nnUNetv2-aligned path. If they are missing, the script can still run with a SimpleITK fallback.

## 3. Probe RKNN Runtime

```bash
python3 device_probe_rknn.py
```

This uses random data and only checks whether RKNN Runtime can load and run:

```text
parse_3d_fullres_patch_32x64x64.rknn
```

## 4. Run One NIfTI Center Patch

Put a test image next to this folder or pass an absolute path:

Simple launcher:

```bash
bash run_nii_inference.sh ../PA000005_0000.nii.gz ../PA000005_center_patch_rknn.nii.gz --center-patch-only
```

Equivalent Python command:

```bash
python3 device_infer_nii_rknn.py \
  -i PA000005_0000.nii.gz \
  -o PA000005_center_patch_rknn.nii.gz \
  --center-patch-only
```

## 5. Run Full Sliding-Window Test

Simple launcher:

```bash
bash run_nii_inference.sh ../PA000005_0000.nii.gz ../PA000005_rknn_mask.nii.gz
```

Equivalent Python command:

```bash
python3 device_infer_nii_rknn.py \
  -i PA000005_0000.nii.gz \
  -o PA000005_rknn_mask.nii.gz
```

## Important

The RKNN file in this folder is a feasibility probe using:

```text
input:  1 x 1 x 32 x 64 x 64
output: 1 x 2 x 32 x 64 x 64
```

The original nnUNetv2 model uses:

```text
input: 1 x 1 x 96 x 160 x 160
```

So this package is for proving the RK3588 NPU deployment path first, not final clinical-quality segmentation.
