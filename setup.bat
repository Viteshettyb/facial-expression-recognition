@echo off
REM ====================================================================
REM  Facial Expression Recognition - one-time setup for Windows
REM
REM  Double-click this file, or run it from the project root.
REM  Safe to run more than once: existing work is reused, not redone.
REM
REM    setup.bat          auto-detect NVIDIA GPU, fall back to CPU
REM    setup.bat cpu      force the smaller CPU-only PyTorch build
REM ====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Facial Expression Recognition - Setup

set "FAILED="

echo.
echo ====================================================================
echo    FACIAL EXPRESSION RECOGNITION - FIRST TIME SETUP
echo ====================================================================
echo.
echo  Project folder: %CD%
echo.
echo  This installs everything needed to run the project.
echo  It can take 5-15 minutes the first time (PyTorch is a big download).
echo.

REM ---------------------------------------------------------------- 1/6
echo [1/6] Looking for Python ...

set "PY="
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3.12"
if not defined PY (
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 set "PY=py -3.11"
)
if not defined PY (
    py -3.10 --version >nul 2>&1
    if not errorlevel 1 set "PY=py -3.10"
)
if not defined PY (
    python --version >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo.
    echo   [FAILED] Python was not found.
    echo.
    echo   Install Python 3.12 from https://www.python.org/downloads/
    echo   During installation you MUST tick "Add python.exe to PATH".
    echo   Then close this window, open a new one, and run setup.bat again.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%V in ('%PY% --version 2^>^&1') do set "PYVER=%%V"
echo       Found !PYVER!
echo.

REM ---------------------------------------------------------------- 2/6
echo [2/6] Python virtual environment (.venv) ...

if exist ".venv\Scripts\python.exe" (
    echo       Already exists - reusing it.
) else (
    echo       Creating .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo.
        echo   [FAILED] Could not create the virtual environment.
        echo   Make sure you have write permission in this folder.
        echo.
        pause
        exit /b 1
    )
    echo       Created.
)
set "VPY=.venv\Scripts\python.exe"
echo.

REM ---------------------------------------------------------------- 3/6
echo [3/6] Updating pip ...
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
if errorlevel 1 (
    echo       [warning] pip could not be upgraded - continuing anyway.
) else (
    echo       Done.
)
echo.

REM ---------------------------------------------------------------- 4/6
echo [4/6] PyTorch ...

set "TVER="
for /f "delims=" %%T in ('.venv\Scripts\python.exe -c "import torch;print(torch.__version__)" 2^>nul') do set "TVER=%%T"

if defined TVER (
    echo       Already installed ^(!TVER!^) - skipping the download.
) else (
    set "MODE=gpu"
    if /i "%~1"=="cpu" set "MODE=cpu"

    if "!MODE!"=="gpu" (
        nvidia-smi >nul 2>&1
        if errorlevel 1 (
            echo       No NVIDIA GPU detected - installing the CPU build.
            set "MODE=cpu"
        ) else (
            echo       NVIDIA GPU detected - installing the CUDA build.
        )
    ) else (
        echo       CPU build requested.
    )

    echo       Downloading PyTorch, please wait ...
    if "!MODE!"=="gpu" (
        "%VPY%" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
        if errorlevel 1 (
            echo.
            echo       CUDA build failed - falling back to the CPU build.
            "%VPY%" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
        )
    ) else (
        "%VPY%" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
    )

    "%VPY%" -c "import torch" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   [FAILED] PyTorch could not be installed.
        echo   Check your internet connection, then try: setup.bat cpu
        echo.
        pause
        exit /b 1
    )
    echo       Installed.
)
echo.

REM ---------------------------------------------------------------- 5/6
echo [5/6] Backend packages ...
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo   [FAILED] Could not install the packages in requirements.txt
    echo.
    pause
    exit /b 1
)
echo       Done.
echo.

REM ---------------------------------------------------------------- 6/6
echo [6/6] Frontend packages ...

where node >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [FAILED] Node.js was not found.
    echo.
    echo   Install the Node.js LTS build from https://nodejs.org/
    echo   Then close this window, open a new one, and run setup.bat again.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%N in ('node --version 2^>^&1') do set "NODEVER=%%N"
echo       Node.js !NODEVER!

REM Reinstall only when package-lock.json has actually changed since last run.
set "LOCKFILE=frontend\package-lock.json"
set "STAMP=frontend\node_modules\.setup-lock-hash"
set "LOCKHASH="
set "OLDHASH="

if exist "%LOCKFILE%" (
    for /f "skip=1 tokens=*" %%H in ('certutil -hashfile "%LOCKFILE%" MD5 2^>nul') do (
        if not defined LOCKHASH set "LOCKHASH=%%H"
    )
    set "LOCKHASH=!LOCKHASH: =!"
)
if exist "%STAMP%" set /p OLDHASH=<"%STAMP%"

set "NEEDINSTALL=1"
if exist "frontend\node_modules" (
    if defined LOCKHASH (
        if "!LOCKHASH!"=="!OLDHASH!" set "NEEDINSTALL="
    )
)

if not defined NEEDINSTALL (
    echo       Dependencies already up to date - skipping npm install.
) else (
    echo       Running npm install, please wait ...
    pushd frontend
    call npm install
    if errorlevel 1 (
        popd
        echo.
        echo   [FAILED] npm install did not complete.
        echo   Check your internet connection and try setup.bat again.
        echo.
        pause
        exit /b 1
    )
    popd
    if defined LOCKHASH (
        if exist "frontend\node_modules" (
            >"%STAMP%" echo !LOCKHASH!
        )
    )
    echo       Done.
)
echo.

REM ------------------------------------------------------- validation
echo ====================================================================
echo    VALIDATING THE INSTALLATION
echo ====================================================================
echo.
"%VPY%" verify_setup.py
if errorlevel 1 (
    echo.
    echo ====================================================================
    echo    SETUP FINISHED WITH PROBLEMS
    echo ====================================================================
    echo.
    echo  Read the messages above, then see SETUP_WINDOWS.md for fixes.
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------ done
echo.
echo ====================================================================
echo    SETUP COMPLETE
echo ====================================================================
echo.
echo  Open this folder in VS Code, then run these in its terminal.
echo.
echo  Backend  (terminal 1, from the project root):
echo.
echo      .venv\Scripts\activate
echo      python -m uvicorn backend.app.main:app --reload --port 8000
echo.
echo  Frontend (terminal 2):
echo.
echo      cd frontend
echo      npm run dev
echo.
echo  --------------------------------------------------------------
echo   Frontend URL:  http://localhost:5173
echo   Backend URL:   http://127.0.0.1:8000/docs
echo  --------------------------------------------------------------
echo.
echo  Start the backend first and wait for "Application startup complete."
echo.
pause
