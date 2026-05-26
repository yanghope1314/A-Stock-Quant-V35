@echo off
chcp 65001 >nul
title A-Stock Quant System V35

echo.
echo   ╔══════════════════════════════════════════════╗
echo   ║   A-Share Quant System · V35 Private Equity  ║
echo   ║   A股量化选股系统 · V35 私募级                  ║
echo   ╚══════════════════════════════════════════════╝
echo.

:: ── Check Python ──
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.12+
    echo [错误] 未找到 Python，请先安装 Python 3.12+
    pause
    exit /b 1
)
echo [OK] Python found

:: ── Check dependencies ──
python -c "import django" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] Dependencies not installed. Installing now...
    echo [提示] 正在安装依赖包，请稍候...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies. Try: pip install -r requirements.txt
        echo [错误] 依赖安装失败，请手动执行: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
)

:: ── Check .env / Tushare Token ──
set HAS_TOKEN=0
if exist .env (
    findstr /c:"TUSHARE_TOKEN" .env >nul 2>&1
    if %errorlevel% equ 0 (
        set HAS_TOKEN=1
    )
)
if %HAS_TOKEN% equ 0 (
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

:: Try to open browser after 2 seconds
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"

python manage.py runserver

pause
