#!/usr/bin/env bash
# Smart launcher: scans conda env dirs directly, no "conda activate" needed
# Works across all drives (C/D/E) — auto-detects conda base path

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   A-Share Quant System · V35 Private Equity  ║"
echo "  ║   A股量化选股系统 · V35 私募级                  ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

resolve_python() {
    # Returns the path to a working python (with django+numpy+pandas+tushare)
    # or empty string if none found.

    # 1. Try current python3 in PATH
    if python3 -c "import django, numpy, pandas, tushare" &>/dev/null; then
        echo "python3"
        return
    fi

    # 2. Try python in PATH
    if python -c "import django, numpy, pandas, tushare" &>/dev/null; then
        echo "python"
        return
    fi

    # 3. Try CONDA_PREFIX (already in an active conda env)
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
        if "$CONDA_PREFIX/bin/python" -c "import django, numpy, pandas, tushare" &>/dev/null; then
            echo "$CONDA_PREFIX/bin/python"
            return
        fi
    fi

    # 4. Scan conda envs directory
    local conda_base=""
    if command -v conda &>/dev/null; then
        conda_base=$(conda info --base 2>/dev/null)
    fi

    if [ -z "$conda_base" ]; then
        # Try common conda paths
        for try_base in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/usr/local/anaconda3" "/usr/local/miniconda3"; do
            if [ -d "$try_base/envs" ]; then
                conda_base="$try_base"
                break
            fi
        done
    fi

    if [ -n "$conda_base" ] && [ -d "$conda_base/envs" ]; then
        local found=()
        for d in "$conda_base/envs"/*/; do
            local py="$d/bin/python"
            if [ -f "$py" ] || [ -f "$d/python.exe" ]; then
                py="${d}bin/python"
                [ -f "$d/python.exe" ] && py="$d/python.exe"
                if "$py" -c "import django, numpy, pandas, tushare" &>/dev/null 2>&1; then
                    found+=("$py|$(basename "$d")")
                fi
            fi
        done

        if [ ${#found[@]} -eq 1 ]; then
            IFS='|' read -r py name <<< "${found[0]}"
            echo "[OK] Auto-selected: $name" >&2
            echo "$py"
            return
        elif [ ${#found[@]} -gt 1 ]; then
            echo "[INFO] Multiple usable Conda environments found:" >&2
            local idx=1
            for entry in "${found[@]}"; do
                IFS='|' read -r py name <<< "$entry"
                echo "  [$idx] $name" >&2
                ((idx++))
            done
            echo "" >&2
            read -p "Select environment (1-${#found[@]}): " choice
            if [ -n "$choice" ] && [ "$choice" -ge 1 ] 2>/dev/null && [ "$choice" -le "${#found[@]}" ]; then
                IFS='|' read -r py name <<< "${found[$((choice-1))]}"
                echo "[OK] Selected: $name" >&2
                echo "$py"
                return
            fi
        fi
    fi

    # Nothing found
    echo ""
}

USE_PYTHON=$(resolve_python)

if [ -z "$USE_PYTHON" ]; then
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║  No working Python environment found                        ║"
    echo "  ║  未找到可用的 Python 环境                                     ║"
    echo "  ║                                                              ║"
    echo "  ║  Setup options:                                              ║"
    echo "  ║                                                              ║"
    echo "  ║  1. Conda (recommended):                                     ║"
    echo "  ║     conda env create -f environment.yml                      ║"
    echo "  ║     Then re-run: bash start.sh                               ║"
    echo "  ║                                                              ║"
    echo "  ║  2. Pip (may be slow):                                       ║"
    echo "  ║     pip3 install -r requirements.txt                         ║"
    echo "  ║                                                              ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
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

"$USE_PYTHON" manage.py runserver
