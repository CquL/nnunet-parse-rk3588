@echo off
setlocal

set "ROOT=%~dp0"
set "CONDA_ENV=d2l_gpu"
set "DEVICE=cuda"

set "nnUNet_results=%ROOT%nnunetv2_PARSE_model_minimal"
set "nnUNet_raw=%ROOT%nnUNet_raw_placeholder"
set "nnUNet_preprocessed=%ROOT%nnUNet_preprocessed_placeholder"

set "INPUT_DIR=%ROOT%nnunetv2_PARSE_test_images"
set "OUTPUT_DIR=%ROOT%predictions"

if not exist "%nnUNet_results%" (
    echo Model directory not found: %nnUNet_results%
    exit /b 1
)

if not exist "%INPUT_DIR%" (
    echo Input image directory not found: %INPUT_DIR%
    exit /b 1
)

if not exist "%nnUNet_raw%" mkdir "%nnUNet_raw%"
if not exist "%nnUNet_preprocessed%" mkdir "%nnUNet_preprocessed%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

conda run -n %CONDA_ENV% nnUNetv2_predict ^
    -i "%INPUT_DIR%" ^
    -o "%OUTPUT_DIR%" ^
    -d 501 ^
    -c 3d_fullres ^
    -f 0 ^
    -tr nnUNetTrainer ^
    -p nnUNetPlans ^
    -chk checkpoint_best.pth ^
    -device %DEVICE% ^
    --disable_tta ^
    --disable_progress_bar ^
    -npp 1 ^
    -nps 1

if errorlevel 1 (
    echo Inference failed.
    exit /b 1
)

echo Done. Predictions are in: %OUTPUT_DIR%
endlocal
