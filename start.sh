#!/usr/bin/env bash
set -e

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   A-Share Quant System · V35 Private Equity  ║"
echo "  ║   A股量化选股系统 · V35 私募级                  ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# ── Check Python ──
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.12+"
    echo "[错误] 未找到 Python3，请先安装 Python 3.12+"
    exit 1
fi
echo "[OK] Python found: $(python3 --version)"

# ── Check dependencies ──
if ! python3 -c "import django" &> /dev/null; then
    echo ""
    echo "[INFO] Dependencies not installed. Installing now..."
    echo "[提示] 正在安装依赖包，请稍候..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        echo "[错误] 依赖安装失败，请手动执行: pip install -r requirements.txt"
        exit 1
    fi
    echo "[OK] Dependencies installed"
fi

# ── Check Tushare Token ──
if ! grep -q "TUSHARE_TOKEN" .env 2>/dev/null; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  [!] Tushare Token not configured                          ║"
    echo "║  [!] Tushare Token 尚未配置                                 ║"
    echo "║                                                              ║"
    echo "║  1. Register at https://tushare.pro to get your Token       ║"
    echo "║  2. After startup, paste Token in the Web UI and click Save ║"
    echo "║                                                              ║"
    echo "║  1. 去 https://tushare.pro 注册获取 Token                    ║"
    echo "║  2. 启动后在网页顶部输入框粘贴 Token 并保存                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
fi

# ── Launch ──
echo ""
echo "[Starting] Launching server..."
echo "[启动] 正在启动服务器..."
echo ""
echo "  Open in browser: http://127.0.0.1:8000"
echo "  浏览器打开: http://127.0.0.1:8000"
echo ""
echo "  Press Ctrl+C to stop | 按 Ctrl+C 停止"
echo ""

# Try to open browser after 3 seconds
(sleep 3 && (open http://127.0.0.1:8000 2>/dev/null || xdg-open http://127.0.0.1:8000 2>/dev/null)) &

python3 manage.py runserver
