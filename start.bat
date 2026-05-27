@echo off
cd /d "%~dp0"
title A-Stock Quant System V35

echo.
echo   * * * * * * * * * * * * * * * * * * * * * * * * * * * *
echo   *  A-Share Quant System - V35 Private Equity           *
echo   *  A-share Quantitative Stock Selection System         *
echo   * * * * * * * * * * * * * * * * * * * * * * * * * * * *
echo.

set USE_PYTHON=

:: Step 1: Check current python in PATH
python -c "import django, numpy, pandas, tushare" >nul 2>&1
if not errorlevel 1 (
    set USE_PYTHON=python
    echo [OK] Using Python from PATH
    goto TOKEN_CHECK
)

:: Step 2: Check CONDA_PREFIX
if not defined CONDA_PREFIX goto SCAN_ENVS
if not exist "%CONDA_PREFIX%\python.exe" goto SCAN_ENVS
"%CONDA_PREFIX%\python.exe" -c "import django, numpy, pandas, tushare" >nul 2>&1
if errorlevel 1 goto SCAN_ENVS
set USE_PYTHON=%CONDA_PREFIX%\python.exe
echo [OK] Using Python from CONDA_PREFIX
goto TOKEN_CHECK

:: Step 3: Scan conda env directories
:SCAN_ENVS
set CONDA_BASE=
where conda >nul 2>&1
if errorlevel 1 goto NO_ENV
for /f "tokens=*" %%b in ('conda info --base 2^>nul') do set CONDA_BASE=%%b
if not defined CONDA_BASE goto NO_ENV

echo [INFO] Scanning Conda environments...
set ENV_COUNT=0
if not exist "%CONDA_BASE%\envs" goto NO_ENV

for /d %%d in ("%CONDA_BASE%\envs\*") do (
    if exist "%%d\python.exe" (
        "%%d\python.exe" -c "import django, numpy, pandas, tushare" >nul 2>&1
        if not errorlevel 1 (
            set /a ENV_COUNT=ENV_COUNT+1
            call set ENV_PATH_%%ENV_COUNT%%=%%d
            call set ENV_NAME_%%ENV_COUNT%%=%%~nxd
            call echo   [%%ENV_COUNT%%] %%~nxd
        )
    )
)

if %ENV_COUNT% equ 0 (
    echo   No usable environment found
    goto NO_ENV
)

:: Auto-pick if only one
if %ENV_COUNT% gtr 1 goto ASK_PICK
call set USE_PYTHON=%%ENV_PATH_1%%\python.exe
call echo [OK] Auto-selected: %%ENV_NAME_1%%
goto TOKEN_CHECK

:ASK_PICK
echo.
set /p PICK=Select environment (1-%ENV_COUNT%):
if "%PICK%"=="" (
    echo No selection, exiting.
    pause
    exit /b 1
)
for /l %%i in (1,1,%ENV_COUNT%) do (
    if "%PICK%"=="%%i" (
        call set USE_PYTHON=%%ENV_PATH_%%i%%\python.exe
        call echo [OK] Selected: %%ENV_NAME_%%i%%
    )
)
if "%USE_PYTHON%"=="" (
    echo [ERROR] Invalid choice
    pause
    exit /b 1
)

:TOKEN_CHECK
set HAS_TOKEN=0
if not exist .env goto LAUNCH
findstr /c:"TUSHARE_TOKEN" .env >nul 2>&1
if errorlevel 1 goto LAUNCH
set HAS_TOKEN=1

:LAUNCH
if %HAS_TOKEN% equ 0 (
    echo.
    echo   ---------------------------------------------------------
    echo   [!] Tushare Token not configured.
    echo       Register at https://tushare.pro to get your Token.
    echo       After startup, paste Token in the Web UI and save.
    echo   ---------------------------------------------------------
    echo.
)

echo.
echo [Starting] Launching server (--noreload, single process)...
echo   Model loading takes 20-40s, browser will open when ready.
echo   Press Ctrl+C to stop.
echo.

:: Step 1: Run Django migrations (show errors for debugging)
echo [1/2] Running database migrations...
"%USE_PYTHON%" manage.py migrate --noinput
if errorlevel 1 (
    echo [ERROR] Migration failed! See the error above.
    echo.
    pause
    exit /b 1
)
echo [OK] Migrations complete.

:: Step 2: Start server directly (NOT in separate window — so errors are visible)
echo [2/2] Starting Django server...
echo   Model loading takes 20-40s, browser will open when ready.
echo   Press Ctrl+C to stop.
echo.

:: Launch server in background via subprocess, wait for it, then open browser
start "A-Stock Quant Server" cmd /k ""%USE_PYTHON%" manage.py runserver --noreload"

:: Wait for server and open browser
echo Waiting for server to be ready...
"%USE_PYTHON%" _open_browser.py

pause
exit /b %errorlevel%

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
echo   2. Pip:
echo      pip install -r requirements.txt
echo   ---------------------------------------------------------
echo.
pause
exit /b 1
