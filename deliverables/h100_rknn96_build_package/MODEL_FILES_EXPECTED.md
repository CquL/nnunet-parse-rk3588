# Private Files Expected

These files are required to build the official-patch-size RKNN but are not
stored in Git:

```text
nnunetv2_PARSE_model_minimal/Dataset501_PARSE/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

Recommended local copy command from the development workstation:

```bash
rsync -avP \
  /home/lhj/nnunet-parse-rk3588-main/nnunetv2_PARSE_model_minimal/Dataset501_PARSE/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth \
  <server>:/path/to/h100_rknn96_build_package/nnunetv2_PARSE_model_minimal/Dataset501_PARSE/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/
```

Keep `.pth` and `.onnx` private. The RK3588 device deployment only needs the
final `.rknn`.
