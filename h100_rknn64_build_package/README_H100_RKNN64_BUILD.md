# H100/Linux Server Build Package: PARSE nnUNetv2 64x128x128 RKNN

This package builds a middle-size RKNN patch model:

- PyTorch checkpoint to ONNX: `checkpoint_best.pth -> parse_3d_fullres_patch_64x128x128.onnx`
- ONNX to RKNN: `parse_3d_fullres_patch_64x128x128.onnx -> parse_3d_fullres_patch_64x128x128.rknn`

The goal is to get closer to the original nnUNetv2 `96x160x160` patch while keeping the RKNN file below the observed RKNN Runtime 4GB parsing boundary.

## Required Private Model File

Copy the protected checkpoint into:

```text
nnunetv2_PARSE_model_minimal/
└── Dataset501_PARSE/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        ├── dataset.json
        ├── plans.json
        └── fold_0/
            └── checkpoint_best.pth
```

Only `dataset.json` and `plans.json` are included here. Do not upload `.pth` or `.onnx` to the RK3588 device.

## Environments

ONNX export environment:

```bash
conda create -n nnunet_export python=3.11 -y
conda activate nnunet_export
pip install -r requirements_h100_rknn64.txt
```

RKNN conversion environment:

```bash
conda create -n rknn232 python=3.10 -y
conda activate rknn232
pip install rknn-toolkit2==2.3.2
```

## Export ONNX

```bash
bash run_export_64_onnx.sh
```

Expected:

```text
deployment/parse_3d_fullres_patch_64x128x128.onnx
```

## Convert RKNN

```bash
bash run_convert_64_rknn.sh
```

Expected:

```text
deployment/parse_3d_fullres_patch_64x128x128.rknn
deployment/parse_3d_fullres_patch_64x128x128.rknn.sha256
```

Before copying to RK3588, check size:

```bash
ls -lh deployment/parse_3d_fullres_patch_64x128x128.rknn
```

If the RKNN is still near or above 4GB, try `64x112x112` or `48x128x128` next.
