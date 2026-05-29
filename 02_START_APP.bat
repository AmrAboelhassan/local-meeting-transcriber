@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Local Meeting Transcriber
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo App is not installed yet.
    echo Please run 00_INSTALL_APP_FIRST.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

REM Add NVIDIA DLL folders installed by pip inside this app environment.
set "PATH=%CD%\.venv\Lib\site-packages\nvidia\cublas\bin;%PATH%"
set "PATH=%CD%\.venv\Lib\site-packages\nvidia\cudnn\bin;%PATH%"
set "PATH=%CD%\.venv\Lib\site-packages\nvidia\cuda_runtime\bin;%PATH%"
set "PATH=%CD%\.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin;%PATH%"

REM Add common system-wide CUDA Toolkit folders if installed.
for %%V in (v12.9 v12.8 v12.7 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0) do (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\bin" (
        set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\bin;!PATH!"
    )
)

echo ==========================================
echo Starting Local Meeting Transcriber
echo ==========================================
echo.
echo A browser window should open automatically.
echo If it does not open, copy this link:
echo http://127.0.0.1:7860
echo.
echo Important:
echo  1. First click "Download selected model" inside the app.
echo  2. Then upload the meeting recording and click "Start Transcription".
echo.
echo Keep this black window open while using the app.
echo Closing it will stop the app.
echo.

python app.py

pause
