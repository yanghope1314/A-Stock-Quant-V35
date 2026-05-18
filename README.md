# A股智能选股系统 (A-Stock Quant System) - V35 OpenSource

![Quant UI](https://img.shields.io/badge/Status-V35_OpenSource-teal)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Django](https://img.shields.io/badge/Framework-Django-green)

本系统是一个从私募实战代码重构而来的 A 股智能选股平台。集成了 150+ 因子计算、双模型 Stacking 预测、AI 图神经网络 (GNN) 以及工业级的风险中性化与组合优化模块。

## 🌟 核心特性

- **多模型集成**: 结合了改进的 XGBoost (涨势股) 与底部分析模型，并引入 SmartXGNN 时空预测。
- **工业级因子工程**: 覆盖 150+ 技术规则因子（动量、反转、量价、波动率、财务衍生等）。
- **动态权重优化**: 基于 IC (信息系数) 的秩相关优化，自动感知市场状态并调整模型权重。
- **一键启动**: 集成 Django 后端与现代化 HTML5 前端，实现选股、分析、止损巡检的一站式体验。
- **环境隔离**: 提供 Conda `environment.yml` 与 Docker 容器化支持，解决“代码能跑但环境难装”的问题。

## 🚀 快速开始

### 方式一：Conda 本地安装 (推荐)

1. **创建环境**:

   ```bash
   conda env create -f environment.yml
   conda activate stock_quant
   ```

2. **配置 API**:
   在启动系统后，直接在浏览器界面的 **顶部输入框** 填入您的 [Tushare](https://tushare.pro/) Token，点击保存即可，系统会自动持久化，**无需手动修改源码**，你也可以在config填写tushare的token。

3. **启动系统**:
   在终端执行：
   ```bash
   python manage.py
   ```

### 方式二：Docker 容器启动

```bash
docker-compose up --build
```

## ❓ 常见问题排查

**1. 运行提示 `No module named 'dotenv'` 怎么办？**
这通常是因为您的 CMD 终端当前**未激活** Conda 环境，默认使用了系统自带的 Python。
请确保在运行前执行：

```bash
conda activate stock_312
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**2. 代码编辑器（如 VSCode）大面积红线报错？**
这是编辑器选错了 Python 解释器。请在 VSCode 中按 `Ctrl+Shift+P`，搜索 `Python: Select Interpreter`，然后选择您环境（如 `stock_312`）对应的 Python。

## 🛠️ 系统架构

- `stock_app/`: 核心逻辑包
  - `views.py`: 选股引擎调度中心
  - `tree_models.py`: 树模型实现与训练
  - `risk_neutralizer.py`: 风险中性化模块
  - `portfolio_optimizer.py`: 组合优化 (风险平价)
- `models/`: 模型权重保存路径 (Git 已忽略)
- `templates/`: 现代化 Web 界面

## ⚠️ 开源说明

- **相对路径**: 系统已全面切换为相对路径，支持任意目录部署。
- **数据源**: 默认使用 Tushare。部分高级因子需要 Tushare 积分。
- **免责声明**: 本系统仅供量化研究参考，不构成任何投资建议。股市有风险，入市需谨慎。

---

_Developed by Private Quant Analysts_
