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
README_UPLOAD_RK3588.md
```

Only the two zip files are required for running the evalset on the board. The `.sha256` file is optional.

## Main Deployment Source

```text
deployment/
rk3588_parse_nnunetv2_aligned_deploy/
```

These are the active deployment files. The zip is generated from `rk3588_parse_nnunetv2_aligned_deploy/`.

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

## Old Or Reference Packages

These are previous/reference packages and logs. Keep them for now until RK3588 metrics are finalized:

```text
rk3588_parse_deploy/
rk3588_parse_deploy_package/
rk3588_parse_min_deploy_package/
nnunet_parse_rk3588_full_package/
nnunet-parse-rk3588-0.1.0/
```
