# A股量化选股系统 · V35 私募级

[![Status](https://img.shields.io/badge/Status-V35_Private_Equity-teal)](https://github.com/yanghope1314)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Framework-Django_4.2-green)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

150+ 技术因子 · 双模型 Stacking · AI 图神经网络 · 大盘择时 · 微信通知

> **开源声明**: 本项目从私募实战代码重构而来，仅供量化研究参考，不构成任何投资建议。

---

## 核心特性

### 因子引擎
- **150+ 规则因子**: 动量(5d/20d/60d)、反转、量价、MACD、KDJ、RSI、William %R、布林带、价格位置、Amihud流动性
- **VIF 正交化**: 迭代方差膨胀因子筛选，剔除共线性冗余，保留 ~30 个独立因子
- **截面中性化**: 同一天内股票间排名比较（非跨历史），行业中性化仅用最新截面

### 模型体系
- **双模型融合**: 涨势股模型(InterpretableXGBV18) + 抄底股模型，IC加权 Stacking
- **AI 集成**: MLP / Transformer / StockGNN / SpatioTemporalGAT / SmartXGNN(XGB+NN+CatBoost)
- **动态权重**: 基于 Spearman IC 的滚动窗口权重优化，市场状态自适应

### 择时 & 风控
- **大盘择时**: 全市场截面均值 → 市场方向判断，熊市自动禁止趋势信号，仅保留抄底
- **风险中性化**: 行业 + 风格双重剥离
- **组合优化**: 风险平价 + 最大回撤约束
- **止损巡检**: 内置持仓止损监控面板

### 微信通知（免费）
- **Server酱** (免费版5条/天): 优先推送预测结果和风险告警
- **PushPlus**: 备用通道自动切换
- **推送优先级**: 风险告警 > 择时变更 > 每日选股 > 周报
- 超过每日限额自动丢弃低优先级消息

---

## 快速开始

### 前置条件
- Python 3.12+
- [Tushare](https://tushare.pro) 账号（免费注册，注册即送积分）
- （可选）[Server酱](https://sct.ftqq.com) 账号（免费微信推送）

### 1. 克隆项目
```bash
git clone https://github.com/yanghope1314/my_stock_project.git
cd my_stock_project
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 Tushare Token
**方法一（推荐）**: 启动后在 Web 界面顶部输入框粘贴 Token，点击保存按钮。

**方法二**: 在项目根目录创建 `.env` 文件：
```
TUSHARE_TOKEN=你的tushare_token
```

> 去 [tushare.pro](https://tushare.pro) 注册 → 个人主页 → 接口Token → 复制

### 4. 配置微信通知（可选）
**方法一（推荐）**: 启动后在 Web 界面「微信通知」面板粘贴 Server酱 SendKey，点击保存。

**方法二**: 在 `.env` 文件中添加：
```
SERVERCHAN_SENDKEY=你的sendkey
```

> 去 [sct.ftqq.com](https://sct.ftqq.com) 注册 → 获取 SendKey → 微信扫码绑定

### 5. 启动系统
```bash
python manage.py runserver
```
浏览器打开 `http://127.0.0.1:8000`，点击「开始选股分析」。

---

## 系统架构

```
my_stock_project/
├── stock_app/                        # 核心量化引擎
│   ├── views.py                      # Django 视图调度中心 (3155行)
│   ├── factor_engine.py              # 150+ 规则因子计算 [独立模块]
│   ├── market_timing.py              # 大盘择时信号 [独立模块]
│   ├── factor_selector.py            # VIF 正交化筛选 [独立模块]
│   ├── wechat_notify.py              # 微信通知推送 [独立模块]
│   ├── tree_models.py                # XGBoost/LightGBM/CatBoost 树模型
│   ├── risk_neutralizer.py           # 风险中性化
│   ├── portfolio_optimizer.py        # 组合优化
│   ├── dynamic_weight_optimizer.py   # 动态 IC 权重
│   ├── config_v19.py                 # 全局配置
│   ├── models/                       # AI模型 (MLP/Transformer/GNN)
│   │   └── ai_models.py
│   └── templates/stock_app/
│       └── index.html                # 前端 Dashboard
├── models/                           # 训练好的模型权重 (gitignore)
├── requirements.txt                  # Python 依赖
├── manage.py                         # Django 入口
├── .gitignore                        # 安全排除规则
└── .env                              # Token 配置 (gitignore, 不会提交)
```

---

## 安全说明

- **Token 保护**: 所有 Token/SendKey 通过 `.env` 文件或浏览器界面保存，`.env` 已在 `.gitignore` 中排除
- **永不提交**: `models/`、`*.log`、`db.sqlite3`、`.env`、`__pycache__/` 均在 gitignore
- **开源检查清单**:
  - [x] 无硬编码 Token
  - [x] 无硬编码密码/密钥
  - [x] `.env` 排除在版本控制之外
  - [x] 模型权重文件排除在版本控制之外
  - [x] 日志文件排除在版本控制之外

---

## 数据源说明

| 接口 | 用途 | 积分要求 |
|------|------|----------|
| `daily` | 日线行情 | 免费 |
| `daily_basic` | 估值/市值/换手率 | 免费 |
| `money_flow` | 资金流向(主力/散户) | 120积分 |
| `limit_list` | 涨跌停数据 | 120积分 |
| `pledge_stat` | 股权质押 | 免费 |
| `hsgt_top10` | 沪深股通十大成交 | 免费 |
| `margin` | 融资融券 | 120积分 |
| `forecast` | 业绩预告 | 免费 |
| `stk_limit` | 涨停价格（动态） | 2000积分 |
| `top10_holders` | 十大股东 | 2000积分 |
| `top_inst` | 机构持股 | 2000积分 |

> **建议积分 ≥ 2000** 以使用全部高级接口（资金流向、涨跌停、融资融券、股东分析等）。
> 注册即送积分，每日签到可累积。积分不足时系统自动降级跳过高级因子。

---

## 常见问题

**Q: 点击分析后报错「数据获取失败」？**
A: 检查 Tushare Token 是否正确保存。点击顶部「检测连接」按钮确认后端状态。

**Q: 微信收不到通知？**
A: 确认 Server酱 SendKey 已保存，重启 Django 服务。免打扰时段(23:00-07:00)不推送。

**Q: 模型训练很慢？**
A: 首次训练(XGBoost+LightGBM+CatBoost+Optuna)需要约5-10分钟。后续加载缓存模型只需数秒。

**Q: pip install 报错？**
A: 建议使用 Python 3.12，`torch` 和 `catboost` 可能需要根据系统单独安装。

---

## 免责声明

本系统仅供量化策略研究和技术学习使用，**不构成任何投资建议**。

- 股市有风险，投资需谨慎
- 历史回测不代表未来收益
- 使用者应自行承担交易风险
- 作者不对使用本系统产生的任何盈亏负责

---

## 作者

**Hope Yang** · 大连海事大学 · 大数据管理与应用

[![GitHub](https://img.shields.io/badge/GitHub-yanghope1314-24292e?logo=github)](https://github.com/yanghope1314)

---

*V35 Private Equity Edition · 2026*
