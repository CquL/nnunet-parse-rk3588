# v0.1.0

Initial private package for nnUNetv2 PARSE local inference and RK3588 RKNN feasibility testing.

Included in the release asset:

- nnUNetv2 model folder with `checkpoint_best.pth`, `plans.json`, and `dataset.json`
- two `.nii.gz` test volumes
- local Windows inference scripts
- preview/export scripts
- ONNX export and RKNN conversion scripts
- RK3588 device probe scripts
- small-patch RKNN feasibility model `parse_3d_fullres_patch_32x64x64.rknn`
- exported ONNX files for `96x160x160` and `32x64x64` patch tests

Known limitation:

- The original `96x160x160` RKNN build was killed by WSL out-of-memory during compilation. The generated RKNN model is currently the smaller `32x64x64` feasibility probe.
