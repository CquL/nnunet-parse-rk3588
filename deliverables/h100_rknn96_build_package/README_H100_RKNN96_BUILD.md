# H100/Linux Server Build Package: PARSE nnUNetv2 96x160x160 RKNN

This package contains the files needed to generate the official-patch-size RKNN:

- PyTorch checkpoint to ONNX: `checkpoint_best.pth -> parse_3d_fullres_patch_96x160x160.onnx`
- ONNX to RKNN: `parse_3d_fullres_patch_96x160x160.onnx -> parse_3d_fullres_patch_96x160x160.rknn`

The checkpoint is not included in Git. Add it locally before running.

## Required Model File

Place the private weight file here:

```text
nnunetv2_PARSE_model_minimal/
└── Dataset501_PARSE/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        ├── dataset.json
        ├── plans.json
        └── fold_0/
            └── checkpoint_best.pth
```

Only `dataset.json` and `plans.json` are included in this package. Copy
`checkpoint_best.pth` from the protected model store or development machine.

## Suggested Environments

Use separate environments if possible:

```bash
conda create -n nnunet_export python=3.11 -y
conda activate nnunet_export
pip install -r requirements_h100_rknn96.txt
```

For RKNN conversion, use the RKNN Toolkit2 wheel that matches the target runtime.
Our RK3588 device is running RKNNLite/librknnrt `2.3.2`, so prefer
`rknn-toolkit2==2.3.2`.

```bash
conda create -n rknn232 python=3.10 -y
conda activate rknn232
pip install rknn-toolkit2==2.3.2
```

## Export ONNX

From this package root:

```bash
bash run_export_96_onnx.sh
```

Expected output:

```text
deployment/parse_3d_fullres_patch_96x160x160.onnx
```

## Convert RKNN

From this package root, in the RKNN Toolkit2 environment:

```bash
bash run_convert_96_rknn.sh
```

Expected output:

```text
deployment/parse_3d_fullres_patch_96x160x160.rknn
```

If this succeeds, copy only the `.rknn` file into the RK3588 deployment package
and update the deployment config to point to it. Do not upload `.pth` or `.onnx`
to the device unless you intentionally want to expose intermediate model files.

## Known Issue From Local Workstation

The local workstation successfully exported the network and converted the
`32x64x64` RKNN, but `96x160x160` RKNN conversion consumed very high RAM/swap
and failed during `rknn.build()` with:

```text
E RKNN: Unkown op target: 0
```

That is why this package exists: run the official patch-size conversion on a
larger Linux server with a clean RKNN Toolkit2 environment. The GPU helps ONNX
export, but RKNN build itself is mostly a Toolkit2 compiler/RAM problem rather
than an H100 acceleration problem.

## After RKNN Is Built

1. Put `parse_3d_fullres_patch_96x160x160.rknn` in the deployment directory.
2. Copy `deployment/model_config_parse_true_96x160x160.template.json` to a real
   config file and set `"model_file": "parse_3d_fullres_patch_96x160x160.rknn"`.
3. Run the aligned inference script with `--patch-size 96x160x160` or the 96
   config.
4. Validate on `nnunetv2_PARSE_fold0_evalset` and compare Dice/HD95 against the
   official nnUNetv2 baseline.
