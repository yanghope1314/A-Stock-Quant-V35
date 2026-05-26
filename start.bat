@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title A-Stock Quant System V35

echo.
echo   ╔══════════════════════════════════════════════╗
echo   ║   A-Share Quant System · V35 Private Equity  ║
echo   ║   A股量化选股系统 · V35 私募级                  ║
echo   ╚══════════════════════════════════════════════╝
echo.

:: ── Step 0: Check if current Python already works ──
python -c "import django, numpy, pandas, tushare" >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Python environment ready
    goto :TOKEN_CHECK
)

:: ── Step 1: Try Conda ──
where conda >nul 2>&1
if !errorlevel! neq 0 goto :NO_CONDA

echo [INFO] Conda detected. Searching for usable environment...
echo.

:: Try stock_quant first (project default)
call conda activate stock_quant >nul 2>&1
if !errorlevel! equ 0 (
    python -c "import django, numpy, pandas, tushare" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Conda env 'stock_quant' activated
        goto :TOKEN_CHECK
    )
)

:: stock_quant not found or missing deps — list available envs
echo Available Conda environments:
echo.
set ENV_COUNT=0
for /f "tokens=1" %%i in ('conda env list 2^>nul ^| findstr /v "^#" ^| findstr /v "base" ^| findstr /v "^$"') do (
    set /a ENV_COUNT+=1
    set "ENV_!ENV_COUNT!=%%i"
    echo   [!ENV_COUNT!] %%i
)

if !ENV_COUNT! equ 0 (
    echo   (no environments found besides base)
    goto :NO_ENV
)

echo.
echo   [0] Create new 'stock_quant' from environment.yml (recommended)
echo.
set /p ENV_CHOICE="Select environment number (0-!ENV_COUNT!, default=0): "
if "!ENV_CHOICE!"=="" set ENV_CHOICE=0
if "!ENV_CHOICE!"=="0" goto :CREATE_ENV

:: Validate and activate selected env
set SELECTED_ENV=
for /l %%i in (1,1,!ENV_COUNT!) do (
    if "!ENV_CHOICE!"=="%%i" (
        set SELECTED_ENV=!ENV_%%i!
    )
)
if "!SELECTED_ENV!"=="" (
    echo [ERROR] Invalid choice
    pause
    exit /b 1
)

call conda activate !SELECTED_ENV! >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate '!SELECTED_ENV!'
    pause
    exit /b 1
)
echo [OK] Conda env '!SELECTED_ENV!' activated
goto :TOKEN_CHECK

:CREATE_ENV
echo.
echo [Creating] conda env create -f environment.yml
echo [创建中] 首次创建约需 5-10 分钟，请耐心等待...
call conda env create -f environment.yml
if !errorlevel! neq 0 (
    echo [ERROR] Failed to create environment. Try manually:
    echo   conda env create -f environment.yml
    pause
    exit /b 1
)
call conda activate stock_quant >nul 2>&1
echo [OK] Conda env 'stock_quant' created and activated
goto :TOKEN_CHECK

:NO_ENV
echo.
echo [INFO] No Conda environments found. Create one with:
echo   conda env create -f environment.yml
echo   conda activate stock_quant
echo.
echo Or install with pip:
echo   pip install -r requirements.txt
echo.
set /p PIP_CHOICE="Install with pip now? (y/n): "
if /i "!PIP_CHOICE!"=="y" (
    echo [Installing] pip install -r requirements.txt ...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if !errorlevel! neq 0 (
        echo [ERROR] pip install failed. Try conda instead.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
    goto :TOKEN_CHECK
)
pause
exit /b 1

:NO_CONDA
echo [INFO] Conda not found. Checking Python dependencies...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Python not found. Please install Python 3.12+ and Conda.
    echo [错误] 未找到 Python，请安装 Python 3.12+ 及 Conda。
    pause
    exit /b 1
)
python -c "import django, numpy, pandas, tushare" >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo [INFO] Dependencies missing. Recommended setup:
    echo   1. Install Conda from https://docs.conda.io
    echo   2. Run: conda env create -f environment.yml
    echo   3. Run: conda activate stock_quant
    echo.
    set /p PIP2_CHOICE="Install with pip now? (y/n): "
    if /i "!PIP2_CHOICE!"=="y" (
        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        if !errorlevel! neq 0 (
            echo [ERROR] pip install failed.
            pause
            exit /b 1
        )
    ) else (
        pause
        exit /b 1
    )
)
echo [OK] Python environment ready

:: ── Check .env / Tushare Token ──
:TOKEN_CHECK
set HAS_TOKEN=0
if exist .env (
    findstr /c:"TUSHARE_TOKEN" .env >nul 2>&1
    if !errorlevel! equ 0 (
        set HAS_TOKEN=1
    )
)
if !HAS_TOKEN! equ 0 (
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║  [!] Tushare Token not configured                          ║
    echo ║  [!] Tushare Token 尚未配置                                 ║
    echo ║                                                              ║
    echo ║  1. Register at https://tushare.pro to get your Token       ║
    echo ║  2. After startup, paste Token in the Web UI and click Save ║
    echo ║                                                              ║
    echo ║  1. 去 https://tushare.pro 注册获取 Token                    ║
    echo ║  2. 启动后在网页顶部输入框粘贴 Token 并保存                   ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
)

:: ── Launch ──
echo.
echo [Starting] Launching server...
echo [启动] 正在启动服务器...
echo.
echo   Open in browser: http://127.0.0.1:8000
echo   浏览器打开: http://127.0.0.1:8000
echo.
echo   Press Ctrl+C to stop | 按 Ctrl+C 停止
echo.

:: Try to open browser after 3 seconds
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"

python manage.py runserver

pause
