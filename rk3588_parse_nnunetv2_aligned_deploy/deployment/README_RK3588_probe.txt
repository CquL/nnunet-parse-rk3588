This folder contains a first RK3588 NPU feasibility probe for the PARSE nnUNetv2 model.

Files:
- parse_3d_fullres_patch_32x64x64.rknn
  RKNN model converted from the nnUNetv2 3D network with a small fixed patch input.

- device_probe_rknn.py
  Loads the RKNN model on an RK3588 device and runs one random input through the NPU.
  This only checks model loading and NPU inference, not medical-image preprocessing.

- device_infer_nii_rknn.py
  Reads a .nii.gz CT volume, normalizes it, cuts 3D patches, runs RKNN inference,
  and writes a .nii.gz segmentation mask.

- check_rk3588_env.sh
  Prints RK3588/Linux/RKNN runtime environment information.

Run on the RK3588 device:

  cd <this folder>
  bash check_rk3588_env.sh
  python3 device_probe_rknn.py

Quick NIfTI test on one center patch:

  python3 device_infer_nii_rknn.py \
    -i PA000005_0000.nii.gz \
    -o PA000005_center_patch_rknn.nii.gz \
    --center-patch-only

Full NIfTI sliding-window test:

  python3 device_infer_nii_rknn.py \
    -i PA000005_0000.nii.gz \
    -o PA000005_rknn_mask.nii.gz

Expected model input:
  float32 shape: 1 x 1 x 32 x 64 x 64

Expected model output:
  logits shape: 1 x 2 x 32 x 64 x 64

Important:
The original nnUNetv2 inference patch is 1 x 1 x 96 x 160 x 160.
That ONNX exports successfully, but RKNN compilation in the current WSL environment was killed by OOM.
The smaller RKNN model is for device/runtime feasibility testing first.
Its segmentation quality is not expected to match the original nnUNetv2 96 x 160 x 160 pipeline yet.
