param(
    [string]$SourceRoot = "",
    [string]$WorkRoot = "D:\nnunet_parse_run",
    [string]$CondaEnv = "d2l_gpu",
    [string]$LocalOutputDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = $PSScriptRoot
}

$modelSource = Join-Path $SourceRoot "nnunetv2_PARSE_model_minimal"
$imageSource = Join-Path $SourceRoot "nnunetv2_PARSE_test_images"
$modelWork = Join-Path $WorkRoot "nnunetv2_PARSE_model_minimal"
$imageWork = Join-Path $WorkRoot "nnunetv2_PARSE_test_images"
$outputDir = Join-Path $WorkRoot "predictions"

if ([string]::IsNullOrWhiteSpace($LocalOutputDir)) {
    $LocalOutputDir = Join-Path $SourceRoot "nnunetv2_PARSE_predictions_local"
}

if (-not (Test-Path -LiteralPath $modelSource)) {
    throw "Model directory not found: $modelSource"
}
if (-not (Test-Path -LiteralPath $imageSource)) {
    throw "Image directory not found: $imageSource"
}

New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
Copy-Item -Recurse -Force -LiteralPath $modelSource -Destination $WorkRoot
Copy-Item -Recurse -Force -LiteralPath $imageSource -Destination $WorkRoot
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
New-Item -ItemType Directory -Force -Path $LocalOutputDir | Out-Null

$env:nnUNet_results = $modelWork
$env:nnUNet_raw = Join-Path $WorkRoot "nnUNet_raw_placeholder"
$env:nnUNet_preprocessed = Join-Path $WorkRoot "nnUNet_preprocessed_placeholder"

conda run -n $CondaEnv nnUNetv2_predict `
    -i $imageWork `
    -o $outputDir `
    -d 501 `
    -c 3d_fullres `
    -f 0 `
    -tr nnUNetTrainer `
    -p nnUNetPlans `
    -chk checkpoint_best.pth `
    -device cuda `
    --disable_tta `
    -npp 1 `
    -nps 1

Copy-Item -Force -Path (Join-Path $outputDir "*") -Destination $LocalOutputDir

Write-Host "Done. Predictions are in: $LocalOutputDir"
