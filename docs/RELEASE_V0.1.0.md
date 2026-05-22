# nnUNet PARSE RK3588 package v0.1.0

This private release collects the current RK3588 deployment assets and the
server-side package for continuing the official `96x160x160` RKNN conversion.

## Current Deployable RK3588 Model

- RKNN model in deployment zip: `parse_3d_fullres_patch_32x64x64.rknn`
- Source checkpoint: `checkpoint_best.pth`
- Official nnUNetv2 patch size: `96x160x160`
- Current deployed RKNN patch size: `32x64x64`
- Non-network inference: nnUNetv2-like Python pipeline
- Network forward: RKNNLite on RK3588 NPU
- Validation metrics: Dice and HD95 with `run_evalset_aligned32.sh`

## Download Links

RK3588 deployment package:

```text
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/rk3588_parse_nnunetv2_aligned_deploy.zip
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/rk3588_parse_nnunetv2_aligned_deploy.zip.sha256
```

Fold-0 validation set:

```text
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/nnunetv2_PARSE_fold0_evalset.zip
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/nnunetv2_PARSE_fold0_evalset.zip.sha256
```

H100/Linux server package for generating the official-patch-size RKNN:

```text
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/h100_rknn96_build_package.zip
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/h100_rknn96_build_package.zip.sha256
```

## RK3588 Quick Start

```bash
cd ~/文档
unzip -o rk3588_parse_nnunetv2_aligned_deploy.zip
unzip -o nnunetv2_PARSE_fold0_evalset.zip

cd rk3588_parse_nnunetv2_aligned_deploy/deployment
bash check_rk3588_env.sh
python3 device_probe_rknn.py ./parse_3d_fullres_patch_32x64x64.rknn
```

Fast validation pass:

```bash
time bash run_evalset_aligned32.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_aligned32_fast \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

Closer nnUNetv2-style validation pass:

```bash
time bash run_evalset_aligned32.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_aligned32
```

The outputs include:

```text
metrics_infer.jsonl
metrics_eval.csv
metrics_summary.json
logs/*_infer.log
logs/*_monitor.log
```

## H100 / Linux Server 96x160x160 Build

Unzip `h100_rknn96_build_package.zip`, copy the private
`checkpoint_best.pth` into the expected `fold_0/` directory, then run:

```bash
bash run_export_96_onnx.sh
bash run_convert_96_rknn.sh
```

The RK3588 board only needs the final `.rknn`. Do not upload `.pth` or `.onnx`
to the device unless you intentionally want to expose those model artifacts.
