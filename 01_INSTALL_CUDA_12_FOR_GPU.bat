@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Install CUDA 12.0 for GPU
cd /d "%~dp0"

set "CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_527.41_windows.exe"
set "INSTALLERS_DIR=%CD%\installers"
set "CUDA_INSTALLER=%INSTALLERS_DIR%\cuda_12.0.0_527.41_windows.exe"

if not exist "%INSTALLERS_DIR%" mkdir "%INSTALLERS_DIR%"

echo ==========================================
echo  CUDA 12.0 GPU Setup

echo ==========================================
echo.
echo This is only needed if GPU mode fails with cublas/cudnn/CUDA errors.
echo File: cuda_12.0.0_527.41_windows.exe
echo Download size is large. It may take time.
echo.

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo WARNING: nvidia-smi was not found.
    echo Install or update NVIDIA driver first if CUDA installer fails.
    echo.
) else (
    echo NVIDIA driver detected:
    nvidia-smi
    echo.
)

where cublas64_12.dll >nul 2>nul
if not errorlevel 1 (
    echo cublas64_12.dll already exists in PATH:
    where cublas64_12.dll
    echo.
    echo CUDA may already be installed. You can skip this installer if GPU works.
    echo.
)

if not exist "%CUDA_INSTALLER%" (
    echo Downloading CUDA 12.0 installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%CUDA_URL%' -OutFile '%CUDA_INSTALLER%'"
    if errorlevel 1 (
        echo Failed to download CUDA installer.
        echo You can manually download it from NVIDIA CUDA 12.0 archive.
        pause
        exit /b 1
    )
) else (
    echo CUDA installer already downloaded:
    echo %CUDA_INSTALLER%
)

echo.
echo The NVIDIA installer will open now.
echo Choose Express/Default installation.
echo You may need Administrator permission.
echo.
pause

start /wait "" "%CUDA_INSTALLER%"

echo.
echo Checking cublas64_12.dll...
where cublas64_12.dll
if errorlevel 1 (
    echo.
    echo CUDA installer finished, but cublas64_12.dll was not found in PATH yet.
    echo Restart Windows, then try 02_START_APP.bat again.
) else (
    echo.
    echo CUDA looks visible. Restart Windows if the app still does not see it.
)

echo.
echo Done.
pause
