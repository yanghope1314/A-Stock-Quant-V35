@echo off
setlocal enabledelayedexpansion
title A-Stock Quant System V35

echo.
echo   * * * * * * * * * * * * * * * * * * * * * * * * * * * *
echo   *  A-Share Quant System - V35 Private Equity           *
echo   *  A-share Quantitative Stock Selection System         *
echo   * * * * * * * * * * * * * * * * * * * * * * * * * * * *
echo.

:: ── Resolve Python path ──
set "USE_PYTHON="

:: Step 1: Check current python in PATH
python -c "import django, numpy, pandas, tushare" >nul 2>&1
if !errorlevel! equ 0 (
    set "USE_PYTHON=python"
    echo [OK] Using Python from PATH
    goto :TOKEN_CHECK
)

:: Step 2: Check CONDA_PREFIX (set if running from conda terminal)
if defined CONDA_PREFIX (
    if exist "!CONDA_PREFIX!\python.exe" (
        "!CONDA_PREFIX!\python.exe" -c "import django, numpy, pandas, tushare" >nul 2>&1
        if !errorlevel! equ 0 (
            set "USE_PYTHON=!CONDA_PREFIX!\python.exe"
            echo [OK] Using Python from CONDA_PREFIX
            goto :TOKEN_CHECK
        )
    )
)

:: Step 3: Scan conda env directories
set "CONDA_BASE="
where conda >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%b in ('conda info --base 2^>nul') do set "CONDA_BASE=%%b"
)

if defined CONDA_BASE (
    echo [INFO] Scanning Conda environments...
    set ENV_COUNT=0

    :: Scan envs dir for valid environments
    if exist "!CONDA_BASE!\envs\" (
        for /d %%d in ("!CONDA_BASE!\envs\*") do (
            if exist "%%d\python.exe" (
                "%%d\python.exe" -c "import django, numpy, pandas, tushare" >nul 2>&1
                if !errorlevel! equ 0 (
                    set /a ENV_COUNT+=1
                    set "ENV_!ENV_COUNT!=%%d"
                    for %%n in (%%d) do set "ENV_NAME_!ENV_COUNT!=%%~nxd"
                    echo   [!ENV_COUNT!] %%~nxd
                )
            )
        )
    )

    if !ENV_COUNT! equ 0 (
        echo   No usable environment found (need django+numpy+pandas+tushare)
        goto :NO_ENV
    )

    :: Auto-pick if only one env found
    if !ENV_COUNT! equ 1 (
        set "USE_PYTHON=!ENV_1!\python.exe"
        echo [OK] Auto-selected: !ENV_NAME_1!
        goto :TOKEN_CHECK
    )

    :: Multiple envs - ask user to pick
    echo.
    set /p PICK="Select environment (1-!ENV_COUNT!): "
    if "!PICK!"=="" (
        echo No selection, exiting.
        pause
        exit /b 1
    )
    set "USE_PYTHON="
    for /l %%i in (1,1,!ENV_COUNT!) do (
        if "!PICK!"=="%%i" (
            set "USE_PYTHON=!ENV_%%i!\python.exe"
            echo [OK] Selected: !ENV_NAME_%%i!
        )
    )
    if "!USE_PYTHON!"=="" (
        echo [ERROR] Invalid choice
        pause
        exit /b 1
    )
    goto :TOKEN_CHECK
)

:: ── No working Python found ──
:NO_ENV
echo.
echo   ---------------------------------------------------------
echo   No working Python environment found.
echo.
echo   Setup options:
echo.
echo   1. Conda (recommended):
echo      conda env create -f environment.yml
echo      Then re-run start.bat
echo.
echo   2. Pip (may be slow, torch works best with conda):
echo      pip install -r requirements.txt
echo   ---------------------------------------------------------
echo.
pause
exit /b 1

:: ── Check Tushare Token ──
:TOKEN_CHECK
set HAS_TOKEN=0
if exist .env (
    findstr /c:"TUSHARE_TOKEN" .env >nul 2>&1
    if !errorlevel! equ 0 set HAS_TOKEN=1
)
if !HAS_TOKEN! equ 0 (
    echo.
    echo   ---------------------------------------------------------
    echo   [!] Tushare Token not configured.
    echo       Register at https://tushare.pro to get your Token.
    echo       After startup, paste Token in the Web UI and save.
    echo   ---------------------------------------------------------
    echo.
)

:: ── Launch ──
echo.
echo [Starting] Launching server...
echo.
echo   Open in browser: http://127.0.0.1:8000
echo   Press Ctrl+C to stop
echo.

start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"

"!USE_PYTHON!" manage.py runserver

pause
