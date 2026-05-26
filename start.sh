#!/usr/bin/env bash
# Smart launcher: auto-detects Conda env, falls back to pip, never flash-crashes

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   A-Share Quant System · V35 Private Equity  ║"
echo "  ║   A股量化选股系统 · V35 私募级                  ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# ── Step 0: Check if current Python already works ──
if python3 -c "import django, numpy, pandas, tushare" &> /dev/null 2>&1; then
    echo "[OK] Python environment ready"
else
    # ── Step 1: Try Conda ──
    if command -v conda &> /dev/null; then
        echo "[INFO] Conda detected. Searching for usable environment..."
        echo ""

        # Source conda.sh to enable `conda activate`
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        fi

        # Try stock_quant first (project default)
        if conda activate stock_quant 2>/dev/null && \
           python3 -c "import django, numpy, pandas, tushare" &> /dev/null 2>&1; then
            echo "[OK] Conda env 'stock_quant' activated"
        else
            # List available envs (exclude base)
            echo "Available Conda environments:"
            echo ""
            envs=()
            idx=1
            while IFS= read -r line; do
                name=$(echo "$line" | awk '{print $1}')
                if [ -n "$name" ] && [ "$name" != "base" ] && [ "$name" != "#" ]; then
                    # skip lines with * (current env marker) and empty names
                    clean_name=$(echo "$name" | tr -d '*')
                    if [ -n "$clean_name" ]; then
                        envs+=("$clean_name")
                        echo "  [$idx] $clean_name"
                        ((idx++))
                    fi
                fi
            done < <(conda env list 2>/dev/null)

            if [ ${#envs[@]} -eq 0 ]; then
                echo "  (no environments found besides base)"
                echo ""
                echo "[INFO] Create one with: conda env create -f environment.yml"
                echo "  Or install with pip: pip3 install -r requirements.txt"
                echo ""
                read -p "Install with pip now? (y/n): " pip_choice
                if [ "$pip_choice" = "y" ] || [ "$pip_choice" = "Y" ]; then
                    pip3 install -r requirements.txt || {
                        echo "[ERROR] pip install failed. Try conda instead."
                        exit 1
                    }
                else
                    exit 1
                fi
            else
                echo ""
                echo "  [0] Create new 'stock_quant' from environment.yml (recommended)"
                echo ""
                read -p "Select environment number (0-${#envs[@]}, default=0): " choice
                choice=${choice:-0}

                if [ "$choice" = "0" ]; then
                    echo ""
                    echo "[Creating] conda env create -f environment.yml"
                    echo "[创建中] 首次创建约需 5-10 分钟..."
                    conda env create -f environment.yml || {
                        echo "[ERROR] Failed to create environment."
                        exit 1
                    }
                    conda activate stock_quant 2>/dev/null
                    echo "[OK] Conda env 'stock_quant' created and activated"
                elif [ "$choice" -ge 1 ] 2>/dev/null && [ "$choice" -le "${#envs[@]}" ]; then
                    selected="${envs[$((choice-1))]}"
                    conda activate "$selected" 2>/dev/null || {
                        echo "[ERROR] Failed to activate '$selected'"
                        exit 1
                    }
                    echo "[OK] Conda env '$selected' activated"
                else
                    echo "[ERROR] Invalid choice"
                    exit 1
                fi
            fi
        fi
    else
        # ── No Conda — check Python and deps ──
        if ! command -v python3 &> /dev/null; then
            echo "[ERROR] Python3 not found. Please install Python 3.12+ and Conda."
            echo "[错误] 未找到 Python3，请安装 Python 3.12+ 及 Conda。"
            exit 1
        fi

        if ! python3 -c "import django, numpy, pandas, tushare" &> /dev/null 2>&1; then
            echo ""
            echo "[INFO] Dependencies missing. Recommended setup:"
            echo "  1. Install Conda from https://docs.conda.io"
            echo "  2. Run: conda env create -f environment.yml"
            echo "  3. Run: conda activate stock_quant"
            echo ""
            read -p "Install with pip now? (y/n): " pip2_choice
            if [ "$pip2_choice" = "y" ] || [ "$pip2_choice" = "Y" ]; then
                pip3 install -r requirements.txt || {
                    echo "[ERROR] pip install failed."
                    exit 1
                }
            else
                exit 1
            fi
        fi
    fi
    echo "[OK] Python environment ready"
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
