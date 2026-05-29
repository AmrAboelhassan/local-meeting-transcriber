@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Install Local Meeting Transcriber
cd /d "%~dp0"

set "PYTHON_VERSION=3.11.9"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe"
set "INSTALLERS_DIR=%CD%\installers"
set "PYTHON_INSTALLER=%INSTALLERS_DIR%\python-%PYTHON_VERSION%-amd64.exe"

if not exist "%INSTALLERS_DIR%" mkdir "%INSTALLERS_DIR%"

echo ==========================================
echo  Local Meeting Transcriber - App Setup
echo ==========================================
echo.
echo This script will:
echo  1. Install Python 3.11 if it is missing
echo  2. Create the local app environment
echo  3. Install the app libraries
echo  4. Install NVIDIA cuDNN/cuBLAS Python DLL packages for GPU support
echo.
echo It will NOT download any Whisper model. Models are downloaded later from the app screen.
echo.

call :find_python

if not defined PY_RUN (
    echo Python 3.11 was not found. Downloading Python %PYTHON_VERSION%...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"
    if errorlevel 1 (
        echo Failed to download Python. Check internet connection.
        pause
        exit /b 1
    )

    echo Installing Python for current user...
    start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 Include_test=0

    set "PATH=%LocalAppData%\Programs\Python\Python311;%LocalAppData%\Programs\Python\Python311\Scripts;%PATH%"
    call :find_python
)

if not defined PY_RUN (
    echo Python installation was not detected.
    echo Please restart Windows, then run this file again.
    pause
    exit /b 1
)

echo Using Python command: %PY_RUN%
echo.

if exist ".venv\Scripts\activate.bat" (
    echo Existing .venv found. Reusing it.
) else (
    echo Creating virtual environment...
    %PY_RUN% -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip wheel setuptools
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo Installing app libraries. This may take several minutes...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Library installation failed.
    echo Check internet connection, then run this file again.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Installation finished successfully.
echo Next: run 02_START_APP.bat
echo If GPU still fails, run 01_INSTALL_CUDA_12_FOR_GPU.bat once.
echo ==========================================
pause
exit /b 0

:find_python
set "PY_RUN="
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    set "PY_RUN=py -3.11"
    exit /b 0
)

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,11) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_RUN=python"
    exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PY_RUN="%LocalAppData%\Programs\Python\Python311\python.exe""
    exit /b 0
)
exit /b 0
