# nnUNetv2-Aligned RK3588 Deployment Notes

This folder now has two device inference paths:

- `device_infer_nii_rknn.py`: the original small feasibility script. It reads `.nii.gz`, applies CT normalization, runs fixed patches, averages overlaps, and writes a mask.
- `device_infer_nii_rknn_nnunetv2_aligned.py`: a closer nnUNetv2-style path. It follows the nnUNetv2 2.7.0 inference order for the non-network part: crop-to-nonzero, CTNormalization from `plans.json`, resampling to the plan spacing, nnUNet-like sliding-window starts, Gaussian blending, mirror TTA by default, then logits resampling/crop restoration back to the original image grid. It uses RKNNLite only for the network forward.

Current available RKNN:

```text
parse_3d_fullres_patch_32x64x64.rknn
input:  1 x 1 x 32 x 64 x 64
output: 1 x 2 x 32 x 64 x 64
```

Original nnUNetv2 plan:

```text
patch_size: 96 x 160 x 160
spacing:    1.0 x 0.65625 x 0.65625  (zyx)
network:    PlainConvUNet, 6 stages, 30,784,450 parameters
```

So the current RKNN can test a more faithful preprocessing/postprocessing path, but it is still not the final architecture-aligned model until `parse_3d_fullres_patch_96x160x160.rknn` exists.

## Run Current 32x64x64 RKNN With Better nnUNetv2-Like Pre/Post

```bash
cd deployment
bash run_aligned32_inference.sh \
  ../test_images/PA000005_0000.nii.gz \
  ../outputs/PA000005_rknn_aligned32_mask.nii.gz
```

For a quicker but less faithful test, reduce overlap:

```bash
bash run_aligned32_inference.sh \
  ../test_images/PA000005_0000.nii.gz \
  ../outputs/PA000005_rknn_aligned32_fast_mask.nii.gz \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

## Run The Fold-0 Evalset On RK3588

Upload and unzip the evalset next to this deployment folder:

```bash
unzip nnunetv2_PARSE_fold0_evalset.zip
cd rk3588_parse_nnunetv2_aligned_deploy/deployment
bash run_evalset_aligned32.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_aligned32
```

The script writes masks as `CASE.nii.gz` and then computes foreground Dice and HD95
against `nnunetv2_PARSE_fold0_evalset/labels`. Results are saved to:

```text
rk3588_parse_nnunetv2_aligned_deploy/outputs/evalset_aligned32/metrics_eval.csv
rk3588_parse_nnunetv2_aligned_deploy/outputs/evalset_aligned32/metrics_summary.json
```

For a first quick pass, use no overlap and no Gaussian:

```bash
bash run_evalset_aligned32.sh \
  ../../nnunetv2_PARSE_fold0_evalset \
  ../outputs/evalset_aligned32_fast \
  --tile-step-size 1.0 \
  --no-gaussian \
  --no-tta
```

## Run True 96x160x160 RKNN After It Is Compiled

Copy `model_config_parse_true_96x160x160.template.json` to a real config, place `parse_3d_fullres_patch_96x160x160.rknn` in this folder, then run:

```bash
cp model_config_parse_true_96x160x160.template.json model_config_parse_true_96x160x160.json
python3 device_infer_nii_rknn_nnunetv2_aligned.py \
  -i ../PA000005_0000.nii.gz \
  -o ../PA000005_rknn_96_mask.nii.gz \
  --config model_config_parse_true_96x160x160.json
```

Mirror TTA is enabled by default to be closer to nnUNetv2 default prediction. Add `--no-tta` only for a faster pass or when comparing against the saved Windows baseline that used `--disable_tta`.

## Device Files Needed

For RK3588 runtime you do not need `.pth` or `.onnx`. You need:

- the final `.rknn`
- `device_infer_nii_rknn_nnunetv2_aligned.py`
- the JSON config matching that RKNN patch size
- Python packages: `numpy`, `SimpleITK`, `rknn-toolkit-lite2`
- Optional but recommended for closer nnUNetv2 preprocessing: `scipy`, `scikit-image`
- matching `librknnrt.so` runtime for the RKNN Toolkit2 version used to compile the model

If `scipy` and `scikit-image` are available, the script prints `Preprocess mode=official_like` and `Resampling backend=skimage` or `scipy_map_coordinates`. If they are missing, it falls back to the older SimpleITK resampling path so the deployment can still run.

Keep `plans.json` and `dataset.json` in the package for traceability and future regeneration, but they are not loaded by the aligned script unless you choose to build a dynamic config loader later.

## Runtime Monitoring

The launcher scripts now write both model metrics and runtime snapshots. For a
single case:

```text
../outputs/metrics_infer.jsonl
../outputs/logs/CASE_infer.log
../outputs/logs/CASE_monitor.log
../outputs/logs/inference_runs.tsv
```

For an evalset run:

```text
../outputs/evalset_aligned32/metrics_eval.csv
../outputs/evalset_aligned32/metrics_summary.json
../outputs/evalset_aligned32/logs/evalset_*.log
```

The JSONL row includes input shape, spacing, patch size, patch count, TTA and
Gaussian status, selected RKNN core mask, preprocess/load/inference/postprocess
timing, patch timing percentiles, output label counts, RSS peak, available
memory, disk space, and readable RKNPU debug load text.

The monitor interval defaults to 10 seconds. Override or disable it with:

```bash
MONITOR_INTERVAL=5 bash run_aligned32_inference.sh INPUT.nii.gz OUTPUT.nii.gz
MONITOR_INTERVAL=0 bash run_aligned32_inference.sh INPUT.nii.gz OUTPUT.nii.gz
```
