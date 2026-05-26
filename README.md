# A-Stock Quant System · V35 Private Equity Edition

## A股量化选股系统 · V35 私募级

[![Status](https://img.shields.io/badge/Status-V35_Private_Equity-teal)](https://github.com/yanghope1314)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Framework-Django_4.2-green)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**150+ Technical Factors · Dual-Model Stacking · AI Graph Neural Networks · Market Timing · WeChat Alerts**

**150+ 技术因子 · 双模型 Stacking · AI 图神经网络 · 大盘择时 · 微信通知**

> Open Source Statement | 开源声明: This project is refactored from private equity production code. For quantitative research reference only. Not financial advice.
> 本项目从私募实战代码重构而来，仅供量化研究参考，不构成任何投资建议。

---

## Core Features | 核心特性

### Factor Engine | 因子引擎
- **150+ Rule-Based Factors**: Momentum (5d/20d/60d), Reversal, Volume-Price, MACD, KDJ, RSI, William %R, Bollinger Bands, Price Position, Amihud Liquidity
- **VIF Orthogonalization**: Iterative Variance Inflation Factor screening removes collinear redundancy, retaining ~30 independent factors
- **Cross-Sectional Neutralization**: Within-day stock ranking (not cross-historical), sector neutralization using latest cross-section only

### Model Architecture | 模型体系
- **Dual-Model Fusion**: Trend model (InterpretableXGBV18) + Bottom-Fishing model, IC-weighted Stacking
- **AI Ensemble**: MLP / Transformer / StockGNN / SpatioTemporalGAT / SmartXGNN (XGB+NN+CatBoost)
- **Dynamic Weights**: Rolling-window Spearman IC optimization, market-regime adaptive

### Market Timing & Risk Control | 择时 & 风控
- **Market Timing**: Cross-sectional mean → market direction signal. Bear market auto-blocks trend signals, bottom-fishing only
- **Risk Neutralization**: Sector + style factor dual stripping
- **Portfolio Optimization**: Risk parity + max drawdown constraints
- **Stop-Loss Monitor**: Built-in position stop-loss dashboard

### WeChat Notifications (Free) | 微信通知（免费）
- **ServerChan** (Free: 5 msgs/day): Priority push for predictions & risk alerts
- **PushPlus**: Auto-failover backup channel
- **Priority Queue**: Risk Alert > Timing Change > Daily Picks > Weekly Report
- Auto-drops low-priority messages when daily limit reached

---

## V35 Prediction Accuracy | V35 预测精度

Four root causes of prediction inaccuracy — all resolved:

| Issue | Status | Fix |
|-------|--------|-----|
| 1. Single 20d label (noisy) | Fixed | Multi-horizon blended: `0.5×cs_rank_5d + 0.3×cs_rank_10d + 0.2×cs_rank_20d` — 5d IC significantly higher in A-shares |
| 2. Fundamental look-ahead bias | Fixed | PE/PB/ROE/ROA/rev/profit/mv/turnover all `shift(1)` per stock — training & inference consistent `_lag1` features |
| 3. Missing market timing | Fixed | Bear market auto-blocks trend signals, bottom-fishing only |
| 4. Factor collinearity | Fixed | VIF iterative screening: 150+ → ~30 independent factors |

---

## Quick Start | 快速开始

### Prerequisites | 前置条件
- Python 3.12+
- [Tushare](https://tushare.pro) account (free registration, points included)
- (Optional) [ServerChan](https://sct.ftqq.com) account (free WeChat push)

### 1. Clone | 克隆

```bash
git clone https://github.com/yanghope1314/A-Stock-Quant-V35.git
cd A-Stock-Quant-V35
```

### 2. Install Dependencies | 安装依赖

```bash
pip install -r requirements.txt
```

### 3. Configure Tushare Token | 配置 Tushare Token

**Method 1 (Recommended)**: After startup, paste your Token in the Web UI top bar and click Save.

**方法一（推荐）**: 启动后在 Web 界面顶部输入框粘贴 Token，点击保存。

**Method 2**: Create `.env` in the project root:

**方法二**: 在项目根目录创建 `.env` 文件：

```
TUSHARE_TOKEN=your_tushare_token
```

> Register at [tushare.pro](https://tushare.pro) → Profile → API Token → Copy
> 去 [tushare.pro](https://tushare.pro) 注册 → 个人主页 → 接口Token → 复制
>
> **Recommended points ≥ 2000** for full functionality (money flow, limit data, margin, shareholder analysis). Points below threshold auto-degrade gracefully.
> **建议积分 ≥ 2000** 以使用全部高级接口。积分不足时系统自动降级。

### 4. Configure WeChat Notifications (Optional) | 配置微信通知（可选）

**Method 1 (Recommended)**: After startup, paste your ServerChan SendKey in the "WeChat Notify" panel and click Save.

**方法一（推荐）**: 启动后在 Web 界面「微信通知」面板粘贴 Server酱 SendKey，点击保存。

**Method 2**: Add to `.env`:

**方法二**: 在 `.env` 文件中添加：

```
SERVERCHAN_SENDKEY=your_sendkey
```

> Register at [sct.ftqq.com](https://sct.ftqq.com) → Get SendKey → Bind WeChat via QR
> 去 [sct.ftqq.com](https://sct.ftqq.com) 注册 → 获取 SendKey → 微信扫码绑定
>
> Free tier: 5 messages/day. System auto-prioritizes: Risk Alerts > Timing > Picks > Weekly.
> 免费版每日5条，系统按优先级自动推送：风险告警 > 择时 > 选股 > 周报。

### 5. Launch (One-Click) | 一键启动

**Windows** — double-click `start.bat`

**Mac / Linux** — run in terminal:
```bash
bash start.sh
```

**Manual start** (if scripts don't work):
```bash
python manage.py runserver
```

The script will:
- Check Python & auto-install missing dependencies
- Warn if Tushare Token is not yet configured
- Launch the server & auto-open browser at `http://127.0.0.1:8000`

> Click "Start Analysis" in the Web UI to begin | 在网页中点击「开始选股分析」即可

### First-Time User Flow | 新用户操作流程

1. Register at [tushare.pro](https://tushare.pro) → get your API Token
2. Launch with `start.bat` / `start.sh`
3. Paste Token in the top bar of the Web UI → click Save
4. (Optional) Paste ServerChan SendKey in the "WeChat" panel → click Save
5. Click "Start Analysis" — the system fetches data, trains models, and outputs stock picks
6. Check WeChat for daily push notifications

### 新用户操作流程

1. 去 [tushare.pro](https://tushare.pro) 注册 → 获取 API Token
2. 双击 `start.bat`（Windows）或运行 `bash start.sh`（Mac/Linux）
3. 在网页顶部输入框粘贴 Token → 点击保存
4. （可选）在「微信通知」面板粘贴 Server酱 SendKey → 点击保存
5. 点击「开始选股分析」→ 系统自动拉数据、训练模型、输出选股结果
6. 微信接收每日推送通知

---

## Architecture | 系统架构

```
A-Stock-Quant-V35/
├── stock_app/                           # Core Quant Engine | 核心量化引擎
│   ├── views.py                         # Django View Dispatcher
│   ├── factor_engine.py                 # 150+ Rule Factor Computation [Standalone]
│   ├── market_timing.py                 # Market Timing Signals [Standalone]
│   ├── factor_selector.py              # VIF Orthogonal Screening [Standalone]
│   ├── wechat_notify.py                # WeChat Push Notifications [Standalone]
│   ├── tree_models.py                  # XGBoost / LightGBM / CatBoost
│   ├── risk_neutralizer.py             # Risk Neutralization
│   ├── portfolio_optimizer.py          # Portfolio Optimization
│   ├── dynamic_weight_optimizer.py     # Dynamic IC Weighting
│   ├── config_v19.py                   # Global Configuration
│   ├── upgrade_v19_nlp_sentiment.py    # NLP Sentiment Engine (RoBERTa)
│   ├── upgrade_v19_small_cap_factors.py # Small-Cap Factor Engine
│   ├── backtest.py                     # Backtest Engine
│   ├── sell_logic.py                   # Exit Strategy / Stop-Loss
│   ├── transaction_cost_model.py       # Transaction Cost Model
│   ├── model_persistence.py            # Model Save/Load
│   ├── models/                         # AI Models (MLP/Transformer/GNN)
│   │   └── ai_models.py
│   └── templates/stock_app/
│       └── index.html                  # Frontend Dashboard
├── models/                              # Trained Model Weights (gitignored)
├── requirements.txt                     # Python Dependencies
├── manage.py                            # Django Entry Point
├── .gitignore                           # Security Exclusions
└── .env                                 # Token Config (gitignored, never committed)
```

---

## Security | 安全说明

- **Token Protection**: All tokens/keys stored via `.env` or browser UI. `.env` excluded in `.gitignore`
- **Never Committed**: `models/`, `*.log`, `db.sqlite3`, `.env`, `__pycache__/`
- **Security Checklist | 安全检查清单**:
  - [x] No hardcoded tokens | 无硬编码 Token
  - [x] No hardcoded passwords/keys | 无硬编码密码/密钥
  - [x] `.env` excluded from version control | `.env` 排除在版本控制之外
  - [x] Model weights excluded | 模型权重文件已排除
  - [x] Log files excluded | 日志文件已排除

---

## Data Sources | 数据源说明

| API | Purpose | Points Required |
|-----|---------|-----------------|
| `daily` | Daily K-line | Free |
| `daily_basic` | Valuation / Market Cap / Turnover | Free |
| `money_flow` | Capital Flow (Institutional/Retail) | 120 |
| `limit_list` | Limit Up/Down Data | 120 |
| `pledge_stat` | Share Pledge Statistics | Free |
| `hsgt_top10` | Northbound Top 10 | Free |
| `margin` | Margin Trading | 120 |
| `forecast` | Earnings Forecast | Free |
| `stk_limit` | Dynamic Limit Price | 2000 |
| `top10_holders` | Top 10 Shareholders | 2000 |
| `top_inst` | Institutional Holdings | 2000 |

> Points ≥ 2000 recommended for all advanced APIs. Systems auto-degrade when points are insufficient.
> 建议积分 ≥ 2000 以使用全部高级接口。积分不足时系统自动降级跳过高级因子。

---

## FAQ | 常见问题

**Q: "Data fetch failed" error after clicking Analyze? | 点击分析后报错「数据获取失败」？**

A: Check Tushare Token is saved correctly. Click "Check Connection" to verify backend status.
检查 Tushare Token 是否正确保存。点击顶部「检测连接」按钮确认后端状态。

**Q: Not receiving WeChat notifications? | 微信收不到通知？**

A: Verify ServerChan SendKey is saved, restart Django. Quiet hours (23:00-07:00) suppress all pushes.
确认 Server酱 SendKey 已保存，重启 Django 服务。免打扰时段(23:00-07:00)不推送。

**Q: Model training is slow? | 模型训练很慢？**

A: First training (XGBoost+LightGBM+CatBoost+Optuna) takes ~5-10 min. Subsequent runs load cached models in seconds.
首次训练约需5-10分钟。后续加载缓存模型只需数秒。

**Q: pip install fails? | pip install 报错？**

A: Use Python 3.12. `torch` and `catboost` may need system-specific installation.
建议使用 Python 3.12。`torch` 和 `catboost` 可能需要根据系统单独安装。

---

## Disclaimer | 免责声明

**This system is for quantitative research and technical study only. It does NOT constitute any investment advice.**

**本系统仅供量化策略研究和技术学习使用，不构成任何投资建议。**

- Markets involve risk. Invest with caution. | 股市有风险，投资需谨慎
- Historical backtests do not guarantee future returns. | 历史回测不代表未来收益
- Users bear full responsibility for their trading decisions. | 使用者应自行承担交易风险
- The author assumes no liability for any profits or losses from use of this system. | 作者不对使用本系统产生的任何盈亏负责

---

## Author | 作者

**Hope Yang** · Dalian Maritime University · Big Data Management & Application

[![GitHub](https://img.shields.io/badge/GitHub-yanghope1314-24292e?logo=github)](https://github.com/yanghope1314)

---

*V35 Private Equity Edition · 2026*
