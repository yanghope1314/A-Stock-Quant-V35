@echo off
chcp 65001 >nul
title A股量化选股系统 V35 OpenSource
echo ==========================================
echo    A股量化选股系统 V35 OpenSource
echo ==========================================
echo.

:: 检查是否在 Conda 环境中
if "%CONDA_DEFAULT_ENV%"=="" (
    echo [!] 警告: 未检测到激活的 Conda 环境。
    echo [!] 强烈建议先执行: conda activate stock_312
    echo.
)

:: 尝试安装如果缺失的话
python -c "import dotenv" 2>nul
if %errorlevel% neq 0 (
    echo [提示] 正在安装 python-dotenv...
    pip install python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1
)

python manage.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] 系统启动失败，请检查 Python 环境和依赖项。
)

echo.
pause
