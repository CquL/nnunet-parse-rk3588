# Current File Organization

## Upload To RK3588

The clean upload folder is:

```text
deploy_upload/
```

It contains:

```text
rk3588_parse_nnunetv2_aligned_deploy.zip
nnunetv2_PARSE_fold0_evalset.zip
rk3588_parse_nnunetv2_aligned_deploy.zip.sha256
nnunetv2_PARSE_fold0_evalset.zip.sha256
h100_rknn96_build_package.zip
h100_rknn96_build_package.zip.sha256
README_UPLOAD_RK3588.md
```

Only the deployment zip and evalset zip are required for running the evalset on
the board. The `.sha256` files are optional integrity checks. The H100 zip is
for server-side 96x160x160 RKNN generation, not for the RK3588 board.

## Upload To H100 / Linux Server

The clean server-side build package is:

```text
deliverables/h100_rknn96_build_package/
```

It contains export/convert scripts, `plans.json`, `dataset.json`, and the 96
patch-size config template. It intentionally does not include:

```text
checkpoint_best.pth
*.onnx
*.rknn
```

Copy `checkpoint_best.pth` into the expected `fold_0/` directory on the server,
then run:

```bash
bash run_export_96_onnx.sh
bash run_convert_96_rknn.sh
```

## Main Deployment Source

```text
deployment/
rk3588_parse_nnunetv2_aligned_deploy/
```

These are the active deployment files. The zip is generated from `rk3588_parse_nnunetv2_aligned_deploy/`.

Runtime logs and metrics are produced by the deployment launchers:

```text
metrics_infer.jsonl
metrics_eval.csv
metrics_summary.json
logs/*_infer.log
logs/*_monitor.log
logs/inference_runs.tsv
```

## Conversion And Debug Scripts

These root-level scripts are for host-side conversion, inspection, and debugging. They are not uploaded to the RK3588 board:

```text
convert_onnx_to_rknn.py
export_nnunet_patch_onnx.py
export_pytorch_to_onnx_general.py
export_parse_preview.py
export_rk3588_compare.py
export_rk3588_preview.py
compare_masks.py
count_model_params.py
```

## Important Data

```text
nnunetv2_PARSE_model_minimal/
nnunetv2_PARSE_fold0_evalset/
nnunetv2_PARSE_fold0_evalset.zip
```

Do not upload `checkpoint_best.pth` or `.onnx` files to the device unless explicitly needed for debugging. The RK3588 deployment only needs `.rknn`.

## Deleted Legacy Packages

These were previous/reference packages. They have been removed locally because
the active deployment is now unified under `rk3588_parse_nnunetv2_aligned_deploy/`.
The names remain in `.gitignore` so they do not accidentally re-enter git if
they are regenerated later:

```text
rk3588_parse_deploy/
rk3588_parse_deploy_package/
rk3588_parse_min_deploy_package/
nnunet_parse_rk3588_full_package/
nnunet-parse-rk3588-0.1.0/
```
