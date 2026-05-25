# nnUNetv2 PARSE on RK3588

This private repository records the PARSE nnUNetv2 deployment path for RK3588 NPU inference.

The goal is:

```text
checkpoint_best.pth
  -> rebuild nnUNetv2 network with PyTorch
  -> export ONNX
  -> compile RKNN for RK3588
  -> run .nii.gz inference on RK3588 with RKNNLite
  -> evaluate Dice and HD95 on the fold-0 validation set
```

The RK3588 device does not run official `nnUNetv2_predict` directly, because official nnUNetv2 uses PyTorch for network forward. This deployment keeps the non-network nnUNetv2 inference pipeline in Python and replaces only the neural network forward with RKNNLite.

## Current Status

Current deployable model:

```text
parse_3d_fullres_patch_32x64x64.rknn
```

Current official nnUNetv2 plan patch:

```text
96 x 160 x 160
```

Current RKNN patch:

```text
32 x 64 x 64
```

The `32x64x64` RKNN is converted from the same `checkpoint_best.pth` network weights. It is not a retrained smaller model, and it does not remove network layers. The key difference is that every NPU forward sees a smaller 3D patch, so context is smaller and sliding-window stitching happens more often.

The `96x160x160` ONNX can be exported, but RKNN Toolkit2 2.3.2 conversion on the current host failed during `rknn.build()` after high memory/swap usage with:

```text
E RKNN: Unkown op target: 0
```

So the validated RK3588 path currently uses the `32x64x64` RKNN.

## What Is Aligned With Official nnUNetv2

The active RK3588 script is:

```text
deployment/device_infer_nii_rknn_nnunetv2_aligned.py
```

It follows the non-network parts of nnUNetv2 2.7.0 as closely as practical:

- SimpleITK reads and writes `.nii.gz`.
- CT normalization uses the values from `plans.json`.
- Target spacing is `[1.0, 0.65625, 0.65625]`.
- Sliding-window step logic follows nnUNetv2.
- Gaussian window blending is implemented.
- Mirror TTA is implemented and enabled by default in the config.
- Logits are resampled back to the original image grid.
- Fold-0 validation can be evaluated with Dice and HD95.

Important remaining differences:

- RKNN input patch is `32x64x64`, not the official `96x160x160`.
- Network forward runs through RKNN/NPU, not PyTorch FP32.
- If `scipy` and `scikit-image` are missing on the board, preprocessing falls back to SimpleITK resampling.

When the closer preprocessing path is active, the device log prints:

```text
Preprocess mode=official_like
Resampling backend=skimage
```

## Repository Layout

```text
deployment/
  RK3588 runtime scripts, aligned inference, environment checks, monitoring,
  and Dice/HD95 evaluation scripts.

deliverables/
  Clean handoff packages. `h100_rknn96_build_package` is the server-side
  package for exporting/converting the official 96x160x160 RKNN. The checkpoint
  is intentionally not stored in git.

rk3588_parse_nnunetv2_aligned_deploy/
  Unzipped deploy package skeleton. The large .rknn and test image are excluded from git.

scripts/
  Host-side helper scripts for ONNX/RKNN conversion and visualization.

README_*.md
  Detailed conversion, deployment, and troubleshooting notes.

deploy_upload/
  Local upload staging folder. Only small README/sha256 files are committed.
```

Large model/data artifacts are intentionally not committed to normal git:

```text
*.pth
*.onnx
*.rknn
*.nii.gz
*.zip
check_external_data/
```

`.pth` and `.onnx` should not be uploaded to the RK3588 device. They are host-side conversion artifacts and expose more of the model. The device only needs `.rknn`, scripts, configs, and data to test.

## Release Downloads

Large files should be uploaded as private GitHub Release assets, not normal git files.

Expected release page:

```text
https://github.com/CquL/nnunet-parse-rk3588/releases/latest
```

Expected asset links after release upload:

```text
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/rk3588_parse_nnunetv2_aligned_deploy.zip
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/nnunetv2_PARSE_fold0_evalset.zip
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/nnunetv2_PARSE_fold0_evalset.zip.sha256
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/rk3588_parse_nnunetv2_aligned_deploy.zip.sha256
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/h100_rknn96_build_package.zip
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/h100_rknn96_build_package.zip.sha256
```

Local files prepared for upload:

```text
deploy_upload/rk3588_parse_nnunetv2_aligned_deploy.zip
deploy_upload/rk3588_parse_nnunetv2_aligned_deploy.zip.sha256
deploy_upload/nnunetv2_PARSE_fold0_evalset.zip
deploy_upload/nnunetv2_PARSE_fold0_evalset.zip.sha256
deploy_upload/h100_rknn96_build_package.zip
deploy_upload/h100_rknn96_build_package.zip.sha256
```

Current deploy zip SHA256:

```text
b401474e80ed8f0b59a3cc4a4d1ccd137ef746838871fb7ebab780a4ae17193b
```

## RK3588 Quick Start

Upload these two files to the RK3588 board:

```text
rk3588_parse_nnunetv2_aligned_deploy.zip
nnunetv2_PARSE_fold0_evalset.zip
```

Then run on the board:

```bash
cd ~/文档

unzip -o rk3588_parse_nnunetv2_aligned_deploy.zip
unzip -o nnunetv2_PARSE_fold0_evalset.zip

cd rk3588_parse_nnunetv2_aligned_deploy/deployment
bash check_rk3588_env.sh
python3 device_probe_rknn.py ./parse_3d_fullres_patch_32x64x64.rknn
```

Fast single-case test:

```bash
time bash run_aligned32_inference.sh \
  ../test_images/PA000005_0000.nii.gz \
  ../outputs/PA000005_fast_mask.nii.gz \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

Fast fold-0 validation test:

```bash
time bash run_evalset_aligned32.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_aligned32_fast \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta

cat ../outputs/evalset_aligned32_fast/metrics_eval.csv
cat ../outputs/evalset_aligned32_fast/metrics_summary.json
```

Closer nnUNetv2-style validation test:

```bash
time bash run_evalset_aligned32.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_aligned32

cat ../outputs/evalset_aligned32/metrics_eval.csv
cat ../outputs/evalset_aligned32/metrics_summary.json
```

The closer run is much slower because Gaussian blending, overlap, and TTA are enabled.

Each inference run now also writes operational logs:

```text
outputs/.../metrics_infer.jsonl
outputs/.../metrics_eval.csv
outputs/.../metrics_summary.json
outputs/.../logs/*_infer.log
outputs/.../logs/*_monitor.log
outputs/.../logs/inference_runs.tsv
```

The monitor records timestamps, memory, disk, Python process RSS, RKNN core
selection, and readable RKNPU debug load information when the board exposes it.

## H100 / Server Build For 96x160x160 RKNN

Use this package when moving to a larger server:

```text
deliverables/h100_rknn96_build_package/
```

It contains the export/convert scripts, `plans.json`, `dataset.json`, and a
README. It does not include `checkpoint_best.pth`; copy that private file into:

```text
deliverables/h100_rknn96_build_package/nnunetv2_PARSE_model_minimal/Dataset501_PARSE/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

Then run:

```bash
cd deliverables/h100_rknn96_build_package
bash run_export_96_onnx.sh
bash run_convert_96_rknn.sh
```

## Conversion Flow

Host-side conversion uses:

```text
export_nnunet_patch_onnx.py
convert_onnx_to_rknn.py
```

Export ONNX:

```bash
python scripts/export_nnunet_patch_onnx.py \
  --patch-size 32x64x64 \
  --output deployment/parse_3d_fullres_patch_32x64x64.onnx
```

Convert ONNX to RKNN:

```bash
python scripts/convert_onnx_to_rknn.py \
  --onnx deployment/parse_3d_fullres_patch_32x64x64.onnx \
  --rknn deployment/parse_3d_fullres_patch_32x64x64.rknn
```

The conversion environment used RKNN Toolkit2 2.3.2. The RK3588 runtime should use matching RKNNLite/runtime libraries.

## Known Issues

The major issues encountered so far:

- RKNN runtime `1.5.0` could not load model version 6; runtime `2.3.2` was required.
- Direct `96x160x160` RKNN conversion failed during build with high memory pressure.
- ToDesk remote sessions can disconnect during long runs; use `tmux` or SSH.
- Official `nnUNetv2_predict` cannot directly call `.rknn`; the RKNN path must use RKNNLite.
- GitHub normal git cannot store files larger than 100 MB; use Release assets or Git LFS.
- Deploy keys must have `Allow write access` enabled to push.

More details are in:

```text
README_PARSE_RK3588_DEPLOYMENT.md
README_RK3588_DEPLOYMENT_ISSUES.md
deployment/README_NNUNETV2_ALIGNED_DEPLOYMENT.md
```

## Security Notes

Do not upload these unless explicitly needed in a private controlled environment:

```text
checkpoint_best.pth
*.onnx
```

The `.rknn` file still contains the network structure and weights, so treat it as a model artifact and upload only to private release assets or trusted deployment machines.
