# v0.1.0

Initial private package for nnUNetv2 PARSE local inference and RK3588 RKNN feasibility testing.

Included in the release assets:

- `nnunet_parse_rk3588_full_package.zip.part01` ... `part09`
- `SHA256SUMS.txt`
- `REASSEMBLE_RELEASE_ASSET.ps1`
- `REASSEMBLE_RELEASE_ASSET.sh`

Reassemble on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\REASSEMBLE_RELEASE_ASSET.ps1
```

Reassemble on Linux:

```bash
bash REASSEMBLE_RELEASE_ASSET.sh
```

The reassembled zip should match:

```text
4917F7BCB2F358DC8B3EC92C8A0565729A48734A3120AAA0A066B80B2CD2708A  nnunet_parse_rk3588_full_package.zip
```

The full package contains:

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
