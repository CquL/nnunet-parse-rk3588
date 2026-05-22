# Release Assets

Large files for RK3588 testing should be attached to a private GitHub Release instead of committed to normal git.

## Assets To Upload

Current local asset folder:

```text
deploy_upload/
```

Upload these files:

```text
rk3588_parse_nnunetv2_aligned_deploy.zip
rk3588_parse_nnunetv2_aligned_deploy.zip.sha256
nnunetv2_PARSE_fold0_evalset.zip
```

The deployment zip contains the RK3588 scripts, configs, and `parse_3d_fullres_patch_32x64x64.rknn`. It does not contain `.pth` or `.onnx`.

## Suggested Release

Suggested tag:

```text
aligned32-v0.2
```

Suggested title:

```text
RK3588 nnUNetv2-aligned 32x64x64 deployment
```

Suggested release body:

```markdown
This release contains the RK3588 deployment package and fold-0 validation set for the PARSE nnUNetv2 RKNN experiment.

Current status:
- RKNN model: parse_3d_fullres_patch_32x64x64.rknn
- Network weights: converted from checkpoint_best.pth
- Official nnUNetv2 plan patch: 96x160x160
- Current RKNN patch: 32x64x64
- Pre/post processing: nnUNetv2-like Python pipeline, RKNNLite for network forward
- Metrics: run on RK3588 with run_evalset_aligned32.sh, outputs Dice and HD95

Assets:
- rk3588_parse_nnunetv2_aligned_deploy.zip
- rk3588_parse_nnunetv2_aligned_deploy.zip.sha256
- nnunetv2_PARSE_fold0_evalset.zip
```

## Direct Links After Upload

These links work after the release assets are uploaded:

```text
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/rk3588_parse_nnunetv2_aligned_deploy.zip
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/rk3588_parse_nnunetv2_aligned_deploy.zip.sha256
https://github.com/CquL/nnunet-parse-rk3588/releases/latest/download/nnunetv2_PARSE_fold0_evalset.zip
```

## Upload With GitHub CLI

If `gh` is installed and logged in:

```bash
cd /home/lhj/nnunet-parse-rk3588-main

gh release create aligned32-v0.2 \
  deploy_upload/rk3588_parse_nnunetv2_aligned_deploy.zip \
  deploy_upload/rk3588_parse_nnunetv2_aligned_deploy.zip.sha256 \
  deploy_upload/nnunetv2_PARSE_fold0_evalset.zip \
  --repo CquL/nnunet-parse-rk3588 \
  --title "RK3588 nnUNetv2-aligned 32x64x64 deployment" \
  --notes-file docs/RELEASE_ASSETS.md
```

If `gh` is not installed, open the GitHub release page in the browser and upload the three files manually.
