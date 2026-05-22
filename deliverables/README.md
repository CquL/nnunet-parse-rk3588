# Deliverables

This directory is the clean handoff area for machines outside the development
workstation.

## h100_rknn96_build_package

Use this on a Linux server to export the true nnUNetv2 full-resolution
`96x160x160` patch model:

1. Put `checkpoint_best.pth` into
   `nnunetv2_PARSE_model_minimal/Dataset501_PARSE/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/`.
2. Run `bash run_export_96_onnx.sh`.
3. Run `bash run_convert_96_rknn.sh`.

The checkpoint is intentionally not committed to Git.

## RK3588 Deployment Package

The runtime package is generated from `rk3588_parse_nnunetv2_aligned_deploy/`
and uploaded to the GitHub release as
`rk3588_parse_nnunetv2_aligned_deploy.zip`.

The device only needs the deployment zip, the validation-set zip if you want to
benchmark Dice/HD95, and the RKNN runtime dependencies. It does not need `.pth`
or `.onnx`.
