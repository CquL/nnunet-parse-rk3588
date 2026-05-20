param(
    [string]$CondaEnv = "d2l_gpu",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$ModelDir = Join-Path $Root "nnunetv2_PARSE_model_minimal"
$InputDir = Join-Path $Root "nnunetv2_PARSE_test_images"
$OutputDir = Join-Path $Root "predictions"

if (-not (Test-Path -LiteralPath $ModelDir)) {
    throw "Model directory not found: $ModelDir"
}
if (-not (Test-Path -LiteralPath $InputDir)) {
    throw "Input image directory not found: $InputDir"
}

$env:nnUNet_results = $ModelDir
$env:nnUNet_raw = Join-Path $Root "nnUNet_raw_placeholder"
$env:nnUNet_preprocessed = Join-Path $Root "nnUNet_preprocessed_placeholder"

New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

conda run -n $CondaEnv nnUNetv2_predict `
    -i $InputDir `
    -o $OutputDir `
    -d 501 `
    -c 3d_fullres `
    -f 0 `
    -tr nnUNetTrainer `
    -p nnUNetPlans `
    -chk checkpoint_best.pth `
    -device $Device `
    --disable_tta `
    --disable_progress_bar `
    -npp 1 `
    -nps 1

Write-Host "Done. Predictions are in: $OutputDir"
