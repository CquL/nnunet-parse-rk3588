# RK3588 Upload Package

Upload these files to the RK3588 device:

```text
rk3588_parse_nnunetv2_aligned_deploy.zip
nnunetv2_PARSE_fold0_evalset.zip
```

`*.sha256` files are optional. They are for checking whether transfer finished
without corruption.

## Run On RK3588

```bash
cd ~/文档

unzip -o rk3588_parse_nnunetv2_aligned_deploy.zip
unzip -o nnunetv2_PARSE_fold0_evalset.zip

cd rk3588_parse_nnunetv2_aligned_deploy/deployment
bash check_rk3588_env.sh

python3 device_probe_rknn.py ./parse_3d_fullres_patch_32x64x64.rknn
```

If you also copy `parse_3d_fullres_patch_96x160x160.rknn` into the deployment
folder, probe it with:

```bash
python3 device_probe_rknn.py ./parse_3d_fullres_patch_96x160x160.rknn --shape 96x160x160
```

If you also copy `parse_3d_fullres_patch_64x128x128.rknn` into the deployment
folder, probe it with:

```bash
python3 device_probe_rknn.py ./parse_3d_fullres_patch_64x128x128.rknn --shape 64x128x128
```

For the closest nnUNetv2-style preprocessing, `check_rk3588_env.sh` should show `scipy import: OK` and `scikit-image import: OK`. If those two are missing, inference still runs, but it falls back to SimpleITK resampling.

Fast single-case test:

```bash
time bash run_aligned32_inference.sh \
  ../test_images/PA000005_0000.nii.gz \
  ../outputs/PA000005_fast_mask.nii.gz \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

Fast fold-0 evalset test:

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

Closer nnUNetv2-style evalset test, slower:

```bash
time bash run_evalset_aligned32.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_aligned32

cat ../outputs/evalset_aligned32/metrics_eval.csv
cat ../outputs/evalset_aligned32/metrics_summary.json
```

Fast 96x160x160 evalset test after the 96 RKNN is copied into `deployment/`:

```bash
time bash run_evalset_aligned96.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_96_fast \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

Fast 64x128x128 evalset test after the 64 RKNN is copied into `deployment/`:

```bash
time bash run_evalset_aligned64.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_64_fast \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

Runtime logs are written under the selected output directory:

```text
metrics_infer.jsonl
logs/*_infer.log
logs/*_monitor.log
logs/inference_runs.tsv
```
