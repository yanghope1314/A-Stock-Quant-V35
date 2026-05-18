# -*- coding: utf-8 -*-
"""
V19增强配置文件
===========
添加动态权重和XGNN配置
"""
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

# ============= Tushare配置 =============
# 优先读取环境变量，若无则为空。不要在此文件硬编码 Token。
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
TUSHARE_POINTS = int(os.environ.get("TUSHARE_POINTS", 2120))  # 默认假设 2120 积分，可由环境变量覆盖

# ============= 模型配置 V35 OpenSource =============
# ─────────────────────────────────────────────────────────────────────
# V35 全开策略：torch_geometric + catboost 依赖已安装，全部启用
# 精度提升路径：rule(25%) → dual_tree(25%) → AI/XGNN(20%) →
#              CatBoost(15%) → NLP(10%) → small_cap(5%)
# ─────────────────────────────────────────────────────────────────────
USE_SMALL_CAP_FACTORS = True
USE_AI_MODELS = True
USE_NLP_SENTIMENT = True
USE_XGBOOST = True
USE_XGNN = True     # V35 重新启用：依赖已修复（torch_scatter/torch_cluster）
USE_CATBOOST = True # V35 新增：CatBoost 有序提升

# 小市值因子配置（中证1000/2000）
SMALL_CAP_CONFIG = {
    'size_threshold_pct': 0.3,
    'min_float_mv': 10,
    'max_float_mv': 200,
}

# AI模型配置 V35（GNN 重新启用）
AI_MODEL_CONFIG = {
    'input_dim': 9,
    'use_mlp': True,
    'use_transformer': True,
    'use_gnn': True,        # V35 重新启用：torch_geometric 已可用
    'device': 'cpu',
    'epochs': 50,           # V35 适当增加（early_stopping 会提前终止，无过拟合风险）
    'batch_size': 512,      # V35 加大 batch（减少梯度噪声）
}

USE_NLP_SENTIMENT = True
NLP_CONFIG = {
    'use_bert': True,                # 若需 BERT 则设为 True
    'bert_model_path': None,         # 如使用本地模型可指定路径
    'sentiment_window': 7,
}
# 🆕 动态权重配置（私募级参数）
DYNAMIC_WEIGHT_CONFIG = {
    'enable': True,                  # 是否启用动态权重
    'min_weight': 0.03,               # 单因子最小权重（更低以允许分散）
    'max_weight': 0.60,               # 单因子最大权重
    'lookback_periods': 20,            # 历史回看期数
    'weight_smooth': 0.5,              # 权重平滑系数（0.5较灵敏，0.8较平滑）
    'risk_parity_alpha': 0.3,          # 风险平价混合系数
    'ic_decay_factor': 10,             # IC衰减因子（越大衰减越慢）
    # 基准权重（根据最新研究调整，总和为1）
    # rule_baseline：规则因子基线（永远存在，保底）
    # dual：双模型融合分（样本足够时启用）
    # ──────────────────────────────────────────────────────────────────
    # V35 权重依据（头部私募IC研究，A股2020-2025回测）：
    #   rule_baseline：规则因子汇总 IC≈0.03-0.05（最稳定，保底）
    #   small_cap：中证1000/2000 alpha IC≈0.05-0.08（2025年最强因子）
    #   dual（tree模型）：XGB+LGB+CatBoost+DART stacking IC≈0.04-0.06
    #   ai（MLP+Transformer+GNN）：非线性交互 IC≈0.03-0.05
    #   nlp（RoBERTa情感）：短期情绪因子 IC≈0.02-0.04
    #   valuation：低估值修复（均衡市场 IC≈0.04-0.06，小盘行情降权）
    # ──────────────────────────────────────────────────────────────────
    'baseline_weights': {
        'rule_baseline':        0.22,  # 规则基线（保底+最稳定，受保护不衰减）
        'small_cap':            0.28,  # 小市值（2025年核心超额来源）
        'dual':                 0.18,  # 双树模型融合（XGB+LGB+Cat+DART stacking）
        'ai':                   0.12,  # AI集成（MLP+Transformer+GNN IC加权）
        'valuation_attractiveness': 0.08,  # 估值吸引力（均衡市场主力）
        'nlp':                  0.06,  # NLP情绪（RoBERTa精度提升后权重上调）
        'xgnn':                 0.04,  # SmartXGNN融合（XGB+NN+CatBoost）
        'original':             0.01,  # 传统因子（已被rule_baseline覆盖，保留兼容）
        'advanced_factors':     0.01,  # 北向/涨停等高频信号
    }
}

# XGNN配置 V35（SmartXGNN: XGBoost + NN + CatBoost 三组件）
XGNN_CONFIG = {
    'enable': True,
    'hidden_dim': 128,          # V35: 增大隐藏维度（从64→128）提升NN表达能力
    'use_rank_loss': True,      # V35: 启用 ICRankLoss 优化排名IC
    # 以下权重由训练后 IC 自动校准，初始值仅作参考
    'xgb_weight': 0.40,         # XGBoost（稳定基础）
    'nn_weight':  0.35,         # 神经网络（非线性）
    'cat_weight': 0.25,         # CatBoost（有序提升，新增）
}

# ============= 回测配置 =============
BACKTEST_CONFIG = {
    'start_date': '20230101',
    'end_date': '20260214',
    'initial_capital': 1000000,
    'commission_rate': 0.0003,
    'slippage': 0.001,
    'rebalance_freq': 20,  # 调仓频率（交易日）
}

# ============= 风控配置 =============
RISK_CONFIG = {
    'max_position_pct': 0.05,
    'max_sector_pct': 0.3,
    'max_drawdown_limit': 0.15,
    'stop_loss_pct': 0.1,
    'min_liquidity': 1000000,  # 最小日均成交额
}

# ============= 日志配置 =============
LOG_CONFIG = {
    'level': 'INFO',
    'file': 'quant_v19_enhanced.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# ============= 增强配置 =============

# 高级因子配置（充分利用Tushare 2120积分）
ADVANCED_FACTORS_CONFIG = {
    'northbound_lookback': 10,  # 北向资金回看天数
    'limit_up_lookback': 60,    # 涨停数据回看天数
    'announcement_lookback': 30,  # 公告数据回看天数
    'margin_lookback': 30,      # 融资融券回看天数
    'enable_cache': True,       # 启用缓存
    'cache_dir': './factor_cache',
}

# A股特色配置
CHINA_STOCK_CONFIG = {
    'price_limit': 0.10,        # 主板涨跌停10%
    'price_limit_st': 0.05,     # ST股涨跌停5%
    'price_limit_cyb': 0.20,    # 创业板/科创板20%
    't_plus_1': True,           # T+1交易制度
    'stamp_duty': 0.001,        # 印花税（卖出）
}

# 真实数据获取配置
DATA_SOURCE_CONFIG = {
    'use_real_data': True,      # 使用真实Tushare数据
    'stock_pool': 'csi1000',    # 股票池：csi1000, csi2000, all
    'start_date': '20230101',
    'end_date': '20260216',
    'batch_size': 50,           # 批量获取股票数
}


# ==================== 工业级优化配置 ====================

# 优化1：特征管理
FEATURE_MANAGEMENT = {
    'enable': True,
    'use_median_fill': True,
    'auto_validate': True,
}

# 优化2：模型持久化
MODEL_PERSISTENCE = {
    'enable': True,
    'model_dir': 'models',
    'max_age_days': 7,
    'auto_save': True,
    'auto_load': True,
}

# 优化3：NLP鲁棒性
NLP_ROBUSTNESS = {
    'enable': True,
    'max_stocks_per_batch': 30,
    'fallback_to_zero_on_error': True,
}

# 优化4：增强日志
ENHANCED_LOGGING = {
    'enable': True,
    'log_weights': True,
    'log_model_status': True,
    'log_prediction_distribution': True,  # 新增：记录预测分布
    'log_ic': True,                        # 新增：记录因子IC
}

# 优化5：自动回测
RUN_BACKTEST = True   # 默认关闭（去掉尾逗号，原来是tuple导致判断失效）
AUTO_BACKTEST = {
    'enable': False,
    'rebalance_freq': 20,
}