# RK3588 Deployment Steps

This folder is for testing the PARSE nnUNetv2 RKNN model on an RK3588 Ubuntu device.

## 1. Install Runtime Dependencies

```bash
cd deployment
bash install_rk3588_deps.sh
```

If the board has no internet, install these packages manually:

```bash
python3 -m pip install numpy SimpleITK rknn-toolkit-lite2
python3 -m pip install scipy scikit-image
```

## 2. Check Device Environment

```bash
bash check_rk3588_env.sh
```

Important lines:

- `uname -m` should be `aarch64`
- `rknnlite import: OK`
- `scipy import: OK` and `scikit-image import: OK` are recommended for the closer nnUNetv2-aligned path. If they are missing, the script can still run with a SimpleITK fallback.

## 3. Probe RKNN Runtime

```bash
python3 device_probe_rknn.py ./parse_3d_fullres_patch_32x64x64.rknn
```

For the 96x160x160 RKNN after copying it to this directory:

```bash
python3 device_probe_rknn.py ./parse_3d_fullres_patch_96x160x160.rknn --shape 96x160x160
```

This uses random data and only checks whether RKNN Runtime can load and run:

```text
parse_3d_fullres_patch_32x64x64.rknn
```

## 4. Run The nnUNetv2-Aligned Path

Fast single-case smoke test:

```bash
bash run_aligned32_inference.sh \
  ../test_images/PA000005_0000.nii.gz \
  ../outputs/PA000005_fast_mask.nii.gz \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

Closer nnUNetv2-style single case:

```bash
bash run_aligned32_inference.sh \
  ../test_images/PA000005_0000.nii.gz \
  ../outputs/PA000005_aligned_mask.nii.gz
```

The closer run is slower because overlap, Gaussian blending, and TTA are enabled.

## 5. Run Fold-0 Validation

After unzipping `nnunetv2_PARSE_fold0_evalset.zip` next to the deployment folder:

```bash
bash run_evalset_aligned32.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_aligned32_fast \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

Results:

```text
../outputs/evalset_aligned32_fast/metrics_eval.csv
../outputs/evalset_aligned32_fast/metrics_summary.json
../outputs/evalset_aligned32_fast/metrics_infer.jsonl
../outputs/evalset_aligned32_fast/logs/
```

After the 96x160x160 RKNN is copied in, use:

```bash
bash run_evalset_aligned96.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_96_fast \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

## Monitoring

The launchers write timestamped logs and lightweight resource snapshots:

```text
logs/*_infer.log
logs/*_monitor.log
logs/inference_runs.tsv
metrics_infer.jsonl
metrics_eval.csv
metrics_summary.json
```

Set the monitor interval:

```bash
MONITOR_INTERVAL=5 bash run_aligned32_inference.sh INPUT.nii.gz OUTPUT.nii.gz
MONITOR_INTERVAL=0 bash run_aligned32_inference.sh INPUT.nii.gz OUTPUT.nii.gz
```

## Important Current Limit

The RKNN file in this folder is a feasibility probe using:

```text
input:  1 x 1 x 32 x 64 x 64
output: 1 x 2 x 32 x 64 x 64
```

The original nnUNetv2 model uses:

```text
input: 1 x 1 x 96 x 160 x 160
```

So this package is for proving the RK3588 NPU deployment path first, not final clinical-quality segmentation.
