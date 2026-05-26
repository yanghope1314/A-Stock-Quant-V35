# -*- coding: utf-8 -*-
"""
A股量化选股系统 - V19 私募级集成版 (2026)
================================================================================
核心功能：
1. 150+ 技术规则因子（动量/反转/量价/波动率/技术指标/衍生交互）
2. 双模型预测（涨势股 InterpretableXGBV18 + 抄底股 InterpretableXGBV18）
3. AI模型集成（MLP/Transformer/GNN/SmartXGNN）
4. 全量 Tushare 接口（资金流/涨跌停/北向/质押/财务/业绩预告）
5. 动态权重优化（IC加权 + 市场状态感知）
6. 风险中性化 + 组合优化 + 交易成本
7. 样本不足时纯规则选股降级
8. 模型持久化（96%速度提升）
================================================================================
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import time
import json
import os
import warnings
import traceback
import concurrent.futures
try:
    import baostock as bs
except ImportError:
    pass
from functools import lru_cache
from collections import defaultdict
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings('ignore')

# ==================== 配置导入 ====================
from .config_v19 import (
    TUSHARE_TOKEN, TUSHARE_POINTS,
    USE_SMALL_CAP_FACTORS, USE_AI_MODELS, USE_NLP_SENTIMENT, USE_XGBOOST, USE_XGNN,
    SMALL_CAP_CONFIG, AI_MODEL_CONFIG, NLP_CONFIG, DYNAMIC_WEIGHT_CONFIG, XGNN_CONFIG,
    RISK_CONFIG, LOG_CONFIG, ADVANCED_FACTORS_CONFIG, CHINA_STOCK_CONFIG, DATA_SOURCE_CONFIG,
    FEATURE_MANAGEMENT, MODEL_PERSISTENCE, NLP_ROBUSTNESS, ENHANCED_LOGGING, AUTO_BACKTEST
)

# ==================== 工业级模块导入 ====================
from .dynamic_weight_optimizer import DynamicWeightOptimizer
from .risk_neutralizer import RiskManager
from .portfolio_optimizer import PortfolioOptimizer
from .transaction_cost_model import TransactionCostModel
from .enhanced_logging import EnhancedLogger, enhanced_logger
from .model_persistence import ModelPersistence, model_persistence

# 树模型（双模型入口 - 来自 tree_models.py）
try:
    from .tree_models import InterpretableXGBV18, ModelConfig
    TREE_MODELS_AVAILABLE = True
except ImportError:
    TREE_MODELS_AVAILABLE = False

from .upgrade_v19_nlp_sentiment import NLPSentimentEngine

# 规则因子引擎（独立模块）
from .factor_engine import calculate_rule_factors as _calc_rule_factors, FACTOR_WEIGHTS_2026

# 微信通知引擎
from .wechat_notify import get_notifier as _get_notifier

# 大盘择时引擎（独立模块）
from .market_timing import compute_market_timing as _compute_mt, detect_market_regime as _detect_regime

# 因子筛选引擎（独立模块）
from .factor_selector import select_orthogonal_factors as _select_ortho

# AI模型
from .models.ai_models import AIAlphaEngine, TORCH_AVAILABLE

# 小市值因子
if USE_SMALL_CAP_FACTORS:
    try:
        from .upgrade_v19_small_cap_factors import SmallCapFactorEngine
    except ImportError:
        USE_SMALL_CAP_FACTORS = False

# 强制禁用旧XGNN（改用SmartXGNN）
USE_XGNN = False

# ==================== 日志 ====================
# ===== 修复点 1：为文件处理器指定 UTF-8 编码 =====
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_CONFIG['file'], encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== 全局API简单令牌桶（保护积分）====================
class _APIGovernor:
    """简单API调用速率控制（避免超出 Tushare 积分）"""
    _COST = {
        'daily': 1, 'daily_basic': 1, 'moneyflow': 2,
        'limit_list': 2, 'hsgt_top10': 2, 'pledge_stat': 2,
        'fina_indicator': 3, 'forecast': 2, 'index_weight': 2,
        'stk_factor': 3, 'trade_cal': 0,
    }
    def __init__(self, total_points: int = TUSHARE_POINTS):
        self._total = total_points
        self._used = 0

    def acquire(self, api_name: str) -> bool:
        cost = self._COST.get(api_name, 1)
        if self._used + cost > self._total * 0.9:
            logger.warning(f"⚠️ API积分紧张，跳过: {api_name}")
            return False
        self._used += cost
        return True

    def reset(self):
        self._used = 0

api_governor = _APIGovernor()


# ==================== 简单内存缓存 ====================
class _SimpleCache:
    def __init__(self):
        self._store: Dict[str, Tuple] = {}

    def get(self, key: str, ttl_seconds: int = 86400):
        if key in self._store:
            val, ts_ = self._store[key]
            if time.time() - ts_ < ttl_seconds:
                return val
        return None

    def set(self, key: str, val):
        self._store[key] = (val, time.time())

cache = _SimpleCache()


# ==================== 交易日历工具 ====================
# ⚠️ 修复：原 @lru_cache 是进程级永久缓存，服务启动后日期永远不刷新
# 改用 _SimpleCache（TTL=6小时），每天自动重拉，杜绝日期冻结在历史

_TRADE_CAL_TTL = 6 * 3600  # 6小时刷新一次

def get_trade_calendar(years: int = 3):
    """获取交易日历（带TTL缓存，避免进程重启前日期不更新）"""
    cache_key = f'trade_cal_{years}'
    cached = cache.get(cache_key, ttl_seconds=_TRADE_CAL_TTL)
    if cached is not None:
        return cached

    try:
        ts.set_token(os.environ.get('TUSHARE_TOKEN', TUSHARE_TOKEN))
        pro = ts.pro_api()
        # ⚠️ 关键：每次都用 datetime.now() 确保 end_date 是今天
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y%m%d')
        cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
        dates = sorted(cal[cal['is_open'] == 1]['cal_date'].tolist())
        logger.info(f"交易日历: {len(dates)} 个交易日，最新: {dates[-1] if dates else 'N/A'}")
        cache.set(cache_key, dates)
        return dates
    except Exception as e:
        logger.error(f"交易日历加载失败: {e}")
        # 兜底：生成最近3年工作日近似列表（不依赖Tushare）
        return _generate_fallback_calendar(years)

def _generate_fallback_calendar(years: int = 3) -> list:
    """Tushare 不可用时的本地兜底日历（工作日近似，排除周末）"""
    dates = []
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # 周一到周五
            dates.append(cur.strftime('%Y%m%d'))
        cur += timedelta(days=1)
    logger.warning(f"使用本地兜底日历: {len(dates)} 个工作日（不含节假日）")
    return dates

def get_latest_trading_date() -> str:
    """获取最近交易日，永远返回真实的今天或昨天，不会停在历史"""
    try:
        today = datetime.now().strftime('%Y%m%d')
        cal = get_trade_calendar()
        valid = [d for d in cal if d <= today]
        result = valid[-1] if valid else today
        # 安全检查：如果结果超过30天前，说明日历有问题，直接返回今天
        result_dt = datetime.strptime(result, '%Y%m%d')
        if (datetime.now() - result_dt).days > 30:
            logger.warning(f"⚠️ 交易日历异常：最新日期={result}，超过30天前，强制使用今天={today}")
            cache.set(f'trade_cal_3', [])  # 清掉坏缓存
            return today
        return result
    except Exception as e:
        logger.error(f"get_latest_trading_date 失败: {e}")
        return datetime.now().strftime('%Y%m%d')

def find_valid_basic_date(max_lookback: int = 15) -> str:
    """
    向前搜索"daily_basic 实际有数据的最近日期"。
    春节/节假日期间 Tushare daily_basic 严格不返回数据，
    本函数独立于 daily 日期，避免批量数据全空。
    缓存4小时，避免每次请求都消耗积分探测。
    """
    _cache_key = 'valid_basic_date'
    _cached = cache.get(_cache_key, ttl_seconds=4 * 3600)
    if _cached:
        logger.info(f"  📅 find_valid_basic_date 命中缓存: {_cached}")
        return _cached

    ts.set_token(os.environ.get('TUSHARE_TOKEN', TUSHARE_TOKEN))
    _pro = ts.pro_api(timeout=20)
    _anchor = get_latest_trading_date()
    _probe_code = '000001.SZ'  # 沪深300权重股，必有数据

    logger.info(f"  🔍 探测 daily_basic 有效日期（锚点={_anchor}，最多向前{max_lookback}天）")
    for _offset in range(0, max_lookback + 1):
        _test = (datetime.strptime(_anchor, '%Y%m%d') -
                 timedelta(days=_offset)).strftime('%Y%m%d')
        try:
            _t = _pro.daily_basic(
                ts_code=_probe_code,
                start_date=_test,
                end_date=_test,
                fields='ts_code,trade_date,total_mv'
            )
            if _t is not None and not _t.empty:
                logger.info(f"  ✅ 有效 basic 日期: {_test}（向前 {_offset} 天）")
                cache.set(_cache_key, _test)
                return _test
            else:
                logger.debug(f"  ⏩ {_test} 无数据，继续向前...")
        except Exception as _e:
            logger.debug(f"  ⏩ {_test} 探测异常: {_e}")
        time.sleep(0.12)

    logger.warning(f"  ⚠️ {max_lookback}天内未找到有效 basic 日期，降级使用锚点 {_anchor}")
    return _anchor


def _notify_async(target: str, *args, **kwargs):
    """后台线程发送微信通知，不阻塞API响应"""
    import threading
    def _run():
        try:
            notifier = _get_notifier()
            method = getattr(notifier, target, None)
            if method:
                method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"微信通知后台发送失败 [{target}]: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _prev_trading_date(n: int = 5) -> str:
    """获取n个交易日前的日期"""
    try:
        cal = get_trade_calendar()
        today = datetime.now().strftime('%Y%m%d')
        valid = [d for d in cal if d <= today]
        idx = max(0, len(valid) - 1 - n)
        return valid[idx]
    except:
        return (datetime.now() - timedelta(days=n * 2)).strftime('%Y%m%d')


# ==================== V19增强引擎 ====================
class V19EnhancedEngine:
    """V19增强引擎 - 私募级整合版"""

    # 规则因子权重（类属性，供 rule_based_select 使用）
    FACTOR_WEIGHTS_2026 = FACTOR_WEIGHTS_2026

    def __init__(self):
        # 动态权重优化器
        self.model_persistence = model_persistence        
        self.weight_optimizer = DynamicWeightOptimizer(
            min_weight=DYNAMIC_WEIGHT_CONFIG['min_weight'],
            max_weight=DYNAMIC_WEIGHT_CONFIG['max_weight'],
            lookback_periods=DYNAMIC_WEIGHT_CONFIG['lookback_periods'],
            weight_smooth=DYNAMIC_WEIGHT_CONFIG.get('weight_smooth', 0.5),
            baseline_weights=DYNAMIC_WEIGHT_CONFIG['baseline_weights']
        )
        # 风险管理
        self.risk_manager = RiskManager(
            max_position_pct=RISK_CONFIG['max_position_pct'],
            max_sector_pct=RISK_CONFIG['max_sector_pct'],
            min_liquidity=RISK_CONFIG['min_liquidity']
        )
        # 组合优化
        self.portfolio_optimizer = PortfolioOptimizer(
            method='risk_parity',
            max_turnover=AUTO_BACKTEST.get('rebalance_freq', 20) / 252 * 2
        )
        # 交易成本
        self.cost_model = TransactionCostModel(
            stamp_duty=CHINA_STOCK_CONFIG['stamp_duty'],
            commission=0.0002,
            liquidity_limit=0.01
        )

        # 子引擎（初始化前声明）
        self.small_cap_engine = None
        self.ai_engine = None
        self.nlp_engine = None
        self.xgb_model = None      # 原XGBoost V18兼容模型（可选）

        # 双模型（私募核心）
        self.trend_model: Optional[InterpretableXGBV18] = None
        self.bottom_model: Optional[InterpretableXGBV18] = None

        # 历史表现
        self.factor_performance_history: List[Dict] = []

        # 初始化
        self._init_engines()
        self._init_dual_models()

        # 日志 & 持久化
        self.logger = enhanced_logger
        self.logger.config = ENHANCED_LOGGING
        self.model_persistence = model_persistence
        if MODEL_PERSISTENCE['enable']:
            self.model_persistence.max_age_days = MODEL_PERSISTENCE['max_age_days']
            self.model_persistence.model_dir = MODEL_PERSISTENCE['model_dir']

        logger.info("✅ V19增强引擎初始化完成（含双模型）")

    # ------------------------------------------------------------------ #
    #  初始化方法
    # ------------------------------------------------------------------ #
    def _init_engines(self):
        """初始化小市值 / NLP / XGBoost / AI 子引擎"""
        # 小市值
        if USE_SMALL_CAP_FACTORS:
            try:
                self.small_cap_engine = SmallCapFactorEngine(**SMALL_CAP_CONFIG)
                logger.info("✅ 小市值因子引擎已初始化")
            except Exception as e:
                logger.error(f"❌ 小市值引擎: {e}")

        # NLP
        if USE_NLP_SENTIMENT:
            try:
                self.nlp_engine = NLPSentimentEngine(
                    use_bert=NLP_CONFIG.get('use_bert', False),
                    sentiment_window=NLP_CONFIG.get('sentiment_window', 7),
                    max_stocks_per_batch=NLP_ROBUSTNESS.get('max_stocks_per_batch', 30),
                    fallback_to_zero_on_error=NLP_ROBUSTNESS.get('fallback_to_zero_on_error', True)
                )

                logger.info("✅ NLP情绪引擎已初始化")
            except Exception as e:
                logger.error(f"❌ NLP引擎: {e}")

        # XGBoost V18（旧版兼容，可选）
        if USE_XGBOOST:
            try:
                loaded = None
                if MODEL_PERSISTENCE['enable'] and MODEL_PERSISTENCE['auto_load']:
                    loaded = self.model_persistence.load_xgboost('xgboost_v19')
                if loaded:
                    self.xgb_model = loaded
                    logger.info("✅ XGBoost V18 从缓存加载")
                else:
                    # 仅在 tree_models 不可用时作为备用
                    if not TREE_MODELS_AVAILABLE:
                        try:
                            from .quant_model_v18_py314 import InterpretableXGBV18 as XGBv18
                            self.xgb_model = XGBv18()
                            logger.info("✅ XGBoost V18 已初始化（备用）")
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"❌ XGBoost V18 初始化: {e}")

        # AI引擎
        if USE_AI_MODELS and TORCH_AVAILABLE:
            try:
                self.ai_engine = AIAlphaEngine(
                    input_dim=AI_MODEL_CONFIG.get('input_dim', 60),
                    device=AI_MODEL_CONFIG.get('device', 'cpu')
                )
                if AI_MODEL_CONFIG.get('use_mlp', True):
                    self.ai_engine.add_mlp('mlp')
                if AI_MODEL_CONFIG.get('use_transformer', True):
                    self.ai_engine.add_transformer('transformer')
                if AI_MODEL_CONFIG.get('use_gnn', False):
                    self.ai_engine.add_stock_gnn('gnn')
                self.ai_engine.add_smart_xgnn('smart_xgnn')
                # Fix 1 (ai_models): HGNN 自动在 AIAlphaEngine.__init__ 中注册，此处无需重复

                # 加载AI引擎
                if MODEL_PERSISTENCE['enable'] and MODEL_PERSISTENCE['auto_load']:
                    self.model_persistence.load_ai_engine(self.ai_engine, 'ai_engine')

                logger.info("✅ AI引擎已初始化（MLP/Transformer/SmartXGNN）")
            except Exception as e:
                logger.error(f"❌ AI引擎: {e}")
                self.ai_engine = None

    def _init_dual_models(self):
        """
        初始化双模型（涨势股 + 抄底股）
        使用 tree_models.InterpretableXGBV18（私募级多模型集成）
        """
        if not TREE_MODELS_AVAILABLE:
            logger.warning("⚠️ tree_models 不可用，双模型禁用")
            return

        try:
            # 涨势股模型：预测20日收益，用更多特征 + 严格CV
            trend_cfg = ModelConfig(
                use_xgboost=True, use_lightgbm=True, use_sklearn=True,
                use_stacking=True, feature_selection='ic_shap',
                top_k_features=60, cv_folds=5, cv_test_size=20,
                early_stopping_rounds=30, n_estimators=300,
                max_depth=6, learning_rate=0.05
            )
            # 抄底股模型：预测10日短期反弹，关注底部特征
            bottom_cfg = ModelConfig(
                use_xgboost=True, use_lightgbm=True, use_sklearn=True,
                use_stacking=True, feature_selection='ic_shap',
                top_k_features=40, cv_folds=3, cv_test_size=10,
                early_stopping_rounds=20, n_estimators=200,
                max_depth=5, learning_rate=0.05
            )
            self.trend_model = InterpretableXGBV18(model_name='trend', config=trend_cfg)
            self.bottom_model = InterpretableXGBV18(model_name='bottom', config=bottom_cfg)

            # 尝试加载已训练模型
            if MODEL_PERSISTENCE['enable']:
                self.trend_model.load()
                self.bottom_model.load()

            logger.info("✅ 双模型已初始化（涨势股 + 抄底股）")
        except Exception as e:
            logger.error(f"❌ 双模型初始化: {e}")
            self.trend_model = None
            self.bottom_model = None

    # ------------------------------------------------------------------ #
    #  规则因子计算（委托给 factor_engine 独立模块）
    # ------------------------------------------------------------------ #
    def _calculate_rule_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """委托给 factor_engine.calculate_rule_factors（私募级因子引擎）"""
        return _calc_rule_factors(df, api_governor=api_governor, pro=None, parent_logger=logger)

    # ------------------------------------------------------------------ #
    #  模型训练（头部私募生产级优化版）
    # ------------------------------------------------------------------ #
    def train_all_models(self, df: pd.DataFrame) -> bool:
        """训练所有模型（XGBoost/AI/双模型） - 头部私募标准版
        优化点：
        1. 为 XGB 模型强制提供验证集，彻底解决 early stopping 报错
        2. 样本不足时优雅降级
        3. 详细日志 + 异常隔离（一个模型失败不影响其他）
        4. AI引擎训练前检查有效样本
        5. 保存 AI 模型使用的特征列，用于预测时对齐
        """
        if len(df) < 2000:
            logger.info(f"样本不足（{len(df)}/2000），跳过所有模型训练")
            return False

        logger.info(f"📊 开始训练全模型（样本: {len(df)}）")
        df = df.sort_values(['ts_code', 'trade_date']).copy()

        # =================================================================
        # 【私募核心改造】截面排名标签（Cross-Sectional Rank Labels）
        # ─────────────────────────────────────────────────────────────────
        # 私募标准（幻方/九坤/明汯）：不预测绝对收益率，而是预测股票在同一天内
        # 的相对排名。原因：
        #   1. 选股的核心问题是"哪只更好"，不是"涨多少"
        #   2. 截面排名自动消除市场方向性偏差
        #   3. 模型直接优化IC（排名质量），而非MSE（误差大小）
        # 标签：cs_rank ∈ [0, 1]，0=截面最差，1=截面最好
        # 映射到 [-1, 1] 以匹配模型输出范围
        # =================================================================
        df['future_ret_20d'] = df.groupby('ts_code')['close'].pct_change(20).shift(-20)
        df['future_ret_10d'] = df.groupby('ts_code')['close'].pct_change(10).shift(-10)

        # 截面排名：每个交易日内部排名（私募核心）
        # rank(pct=True) → [0, 1]，越高表示当天内表现越好
        df['cs_rank_20d'] = df.groupby('trade_date')['future_ret_20d'].rank(pct=True)
        df['cs_rank_10d'] = df.groupby('trade_date')['future_ret_10d'].rank(pct=True)

        df_clean = df.dropna(subset=['future_ret_20d', 'future_ret_10d',
                                      'cs_rank_20d', 'cs_rank_10d'])

        if len(df_clean) < 1000:
            logger.warning(f"有效样本过少（{len(df_clean)}），跳过模型训练")
            return False

        # 基础特征（不含规则因子，给双模型和AI用）
        _basic_feats = [
            'total_mv', 'circ_mv', 'pe', 'pb', 'roe', 'roa',
            'revenue_yoy', 'profit_yoy', 'vol', 'turnover_rate',
            'pmt_return_5d', 'pmt_return_20d', 'pmt_return_60d',
            'vol_ratio_raw', 'volat_hist_20d', 'rsi', 'kdj_k', 'kdj_j',
            'macd', 'ma20_distance', 'boll_position', 'multi_oversold_flag',
            'size_score', 'size_mkt_cap_log',
        ]
        feature_cols = [c for c in _basic_feats if c in df_clean.columns]
        actual_input_dim = len(feature_cols)
        logger.info(f"  AI引擎实际输入特征维度: {actual_input_dim}")

        # ========== 保存特征列，供后续预测时对齐 ==========
        self._ai_feature_cols = feature_cols

        # ========== 统一重建 AI 引擎，确保维度匹配并添加所有模型 ==========
        if USE_AI_MODELS and TORCH_AVAILABLE:
            need_rebuild = False
            # 检查现有引擎是否为 None 或维度不匹配
            if self.ai_engine is None:
                need_rebuild = True
                logger.info("  AI引擎为 None，需要重建")
            elif actual_input_dim != self.ai_engine.input_dim:
                need_rebuild = True
                logger.warning(f"  AI引擎维度不匹配: 现有 {self.ai_engine.input_dim}，实际 {actual_input_dim}，重建")

            if need_rebuild:
                try:
                    from .models.ai_models import AIAlphaEngine
                    # 创建新引擎，使用实际维度
                    self.ai_engine = AIAlphaEngine(
                        input_dim=actual_input_dim,
                        device=AI_MODEL_CONFIG.get('device', 'cpu')
                    )
                    # 添加所有配置中启用的模型
                    if AI_MODEL_CONFIG.get('use_mlp', True):
                        self.ai_engine.add_mlp('mlp')
                    if AI_MODEL_CONFIG.get('use_transformer', True):
                        self.ai_engine.add_transformer('transformer')
                    if AI_MODEL_CONFIG.get('use_gnn', False):
                        self.ai_engine.add_stock_gnn('gnn')
                    self.ai_engine.add_smart_xgnn('smart_xgnn')
                    logger.info(f"  ✅ AI引擎已按实际维度 {actual_input_dim} 重建，并添加了所有模型")

                    # 尝试从持久化加载权重（如果存在且新鲜）
                    if MODEL_PERSISTENCE['enable']:
                        try:
                            self.model_persistence.load_ai_engine(self.ai_engine, 'ai_engine')
                        except Exception as e:
                            logger.warning(f"  加载AI引擎权重失败: {e}")
                except Exception as e:
                    logger.error(f"  ❌ AI引擎重建失败: {e}")
                    # 如果重建失败，保留原有引擎（可能为 None）
                    if self.ai_engine is None:
                        # 无法继续，但可能仍可训练其他模型
                        pass
        # ========== 重建结束 ==========

        X_arr = df_clean[feature_cols].fillna(0).values.astype(np.float32)
        X_df_frame = df_clean[feature_cols].fillna(0)   # 保留 DataFrame 格式
        X_df = X_arr   # 向后兼容

        # 【私募核心】训练标签：截面排名 cs_rank ∈ [0,1] → [-1, 1]
        # 截面排名天然有界，无需 Winsorize
        _y20_s = pd.Series(df_clean['cs_rank_20d'].fillna(0.5).values)
        _y10_s = pd.Series(df_clean['cs_rank_10d'].fillna(0.5).values)
        y_20d = ((_y20_s - 0.5) * 2.0).values.astype(np.float32)
        y_10d = ((_y10_s - 0.5) * 2.0).values.astype(np.float32)
        logger.info(f"  标签：截面排名 20d[{_y20_s.min():.3f}→{_y20_s.max():.3f}] "
                    f"10d[{_y10_s.min():.3f}→{_y10_s.max():.3f}] "
                    f"映射至[-1,+1]")

        trained = False

        # ====================== 1. XGBoost V18 ======================
        if USE_XGBOOST and self.xgb_model and hasattr(self.xgb_model, 'fit'):
            try:
                logger.info("  🤖 训练XGBoost V18...")
                val_size = max(50, int(len(X_arr) * 0.15))
                X_tr, X_va = X_arr[:-val_size], X_arr[-val_size:]
                y_tr, y_va = y_20d[:-val_size], y_20d[-val_size:]

                self.xgb_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False
                )
                if MODEL_PERSISTENCE['enable'] and MODEL_PERSISTENCE['auto_save']:
                    self.model_persistence.save_xgboost(self.xgb_model, 'xgboost_v19')
                trained = True
                logger.info("  ✅ XGBoost V18 训练完成")
            except Exception as e:
                logger.error(f"  ❌ XGBoost V18 训练失败: {e}")

        # ====================== 2. AI引擎 ======================
        if USE_AI_MODELS and self.ai_engine:
            try:
                logger.info("  🧠 训练AI引擎...")
                if len(X_df) < 500:
                    logger.warning("  AI引擎样本不足，跳过训练")
                else:
                    split = int(len(X_df) * 0.8)
                    X_tr, X_va = X_df[:split], X_df[split:]
                    y_tr, y_va = y_20d[:split], y_20d[split:]

                    trade_dates_tr = None
                    if 'trade_date' in df_clean.columns:
                        trade_dates_all = df_clean['trade_date'].astype(str).tolist()
                        trade_dates_tr = trade_dates_all[:split]

                    _need_retrain = True
                    if MODEL_PERSISTENCE['enable'] and MODEL_PERSISTENCE['auto_load']:
                        try:
                            _need_retrain = self.ai_engine.should_retrain(
                                model_dir=MODEL_PERSISTENCE['model_dir'],
                                base_name='ai_engine',
                                max_age_days=MODEL_PERSISTENCE.get('max_age_days', 3),
                                min_ic=0.03
                            )
                        except Exception:
                            _need_retrain = True

                    if _need_retrain:
                        self.ai_engine.train_all(
                            X_tr, y_tr,
                            X_va, y_va,
                            batch_size=AI_MODEL_CONFIG.get('batch_size', 256),
                            epochs=AI_MODEL_CONFIG.get('epochs', 30),
                            trade_dates=trade_dates_tr
                        )
                        if MODEL_PERSISTENCE['enable'] and MODEL_PERSISTENCE['auto_save']:
                            self.model_persistence.save_ai_engine(self.ai_engine, 'ai_engine')
                    else:
                        logger.info("  ⏭️ AI引擎模型新鲜且IC达标，跳过重训")
                trained = True
                logger.info("  ✅ AI引擎训练完成")
            except Exception as e:
                logger.error(f"  ❌ AI引擎训练失败: {e}")

        # ====================== 3. 双模型 ======================
        if TREE_MODELS_AVAILABLE and self.trend_model and self.bottom_model:
            # 涨势股模型
            try:
                logger.info("  📈 训练涨势股模型（trend_model）...")
                dates = pd.to_datetime(df_clean.get('trade_date')) if 'trade_date' in df_clean.columns else None
                val_size = max(50, int(len(X_df_frame) * 0.15))
                X_tr = X_df_frame.iloc[:-val_size]
                y_tr = y_20d[:-val_size]
                X_va = X_df_frame.iloc[-val_size:]
                y_va = y_20d[-val_size:]
                d_tr = dates.iloc[:-val_size] if dates is not None else None

                self.trend_model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, dates=d_tr)
                trained = True
                logger.info("  ✅ 涨势股模型训练完成")
            except Exception as e:
                logger.error(f"  ❌ 涨势股模型训练失败: {e}")

            # 抄底股模型
            try:
                logger.info("  📉 训练抄底股模型（bottom_model）...")
                bottom_mask = pd.Series(False, index=range(len(df_clean)))
                if 'price_position_60d' in df_clean.columns:
                    bottom_mask |= (df_clean['price_position_60d'].values < 0.3)
                if 'multi_oversold_flag' in df_clean.columns:
                    bottom_mask |= (df_clean['multi_oversold_flag'].values > 0.5)

                n_bottom = bottom_mask.sum()
                if n_bottom >= 300:
                    X_bot = X_df_frame.loc[bottom_mask.values]
                    y_bot = y_10d[bottom_mask.values]
                    dates_bot = dates.loc[bottom_mask.values] if dates is not None else None
                else:
                    X_bot, y_bot, dates_bot = X_df_frame, y_10d, dates
                    logger.warning(
                        f"  ⚠️ 底部样本不足（{n_bottom}/300），使用全量数据训练抄底模型。"
                        f" 这是正常现象（近期无超卖信号），bottom_score权重将自动降低。"
                    )

                val_size = max(30, int(len(X_bot) * 0.15))
                X_tr = X_bot.iloc[:-val_size]
                y_tr = y_bot[:-val_size]
                X_va = X_bot.iloc[-val_size:]
                y_va = y_bot[-val_size:]
                d_tr = dates_bot.iloc[:-val_size] if dates_bot is not None else None

                self.bottom_model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, dates=d_tr)
                trained = True
                logger.info(f"  ✅ 抄底股模型训练完成（样本: {len(X_bot)}）")
            except Exception as e:
                logger.error(f"  ❌ 抄底股模型训练失败: {e}")

            if MODEL_PERSISTENCE['enable']:
                try:
                    self.trend_model.save()
                    self.bottom_model.save()
                    logger.info("  💾 双模型已保存")
                except Exception as e:
                    logger.warning(f"  ⚠️ 双模型保存失败: {e}")

        if trained:
            logger.info("✅ 全模型训练完成（至少1个模型成功）")
        else:
            logger.warning("⚠️ 所有模型训练均失败")
        return trained
    # ------------------------------------------------------------------ #
    #  因子计算（整合规则因子 + AI预测 + 双模型）
    # ------------------------------------------------------------------ #
    def calculate_all_factors(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """计算全部因子（规则 + 小市值 + XGBoost + AI + 双模型 + NLP）"""
        df = stock_df.copy()
        logger.info("📊 计算所有因子（规则+AI+双模型）...")

        # Step 1：规则因子（150+）
        try:
            df = self._calculate_rule_factors(df)
        except Exception as e:
            logger.error(f"  规则因子计算失败: {e}\n{traceback.format_exc()}")

        # Step 2：小市值因子
        if self.small_cap_engine and USE_SMALL_CAP_FACTORS:
            try:
                # Tushare用circ_mv，SmallCapFactorEngine可能用float_mv，做别名映射
                if 'circ_mv' in df.columns and 'float_mv' not in df.columns:
                    df['float_mv'] = df['circ_mv']

                df = self.small_cap_engine.calculate_factors(df)
            except Exception as e:
                logger.error(f"  小市值因子: {e}")

        # Step 2.5：【私募核心】VIF正交化因子筛选
        # 从150+规则因子中筛掉共线性冗余，保留~30个独立因子
        _all_factor_candidates = [
            # 基本面
            'total_mv', 'circ_mv', 'pe', 'pb', 'roe', 'roa',
            'revenue_yoy', 'profit_yoy', 'vol', 'turnover_rate',
            # 动量（5日/20日/60日 → VIF会筛掉冗余）
            'pmt_return_5d', 'pmt_return_20d', 'pmt_return_60d', 'pmt_return_1d',
            'rev_5d_reversal', 'rev_1d_reversal',
            # 量价
            'vol_ratio_raw', 'vol_amount_ratio', 'vol_turnover', 'vol_price_corr',
            # 波动率
            'volat_hist_20d', 'volat_hist_5d', 'volat_ratio',
            # 均线
            'ma5', 'ma20_distance', 'ma60_distance',
            'ma_bull', 'ma_bear', 'ma_trend_score',
            # 技术指标
            'rsi', 'kdj_k', 'kdj_j', 'kdj_d',
            'macd', 'macd_hist', 'macd_signal', 'macd_golden',
            'wr', 'boll_position', 'boll_squeeze',
            # 超卖/抄底
            'kdj_oversold', 'rsi_oversold', 'wr_oversold',
            'near_boll_lower', 'multi_oversold_flag',
            'price_position_60d', 'in_bottom_zone',
            # 反转/动量
            'reversal_signal', 'kdj_low_golden', 'rsi_divergence',
            # 规模/流动性
            'size_score', 'size_mkt_cap_log', 'liquidity_score', 'liquidity_amihud',
            # 衍生交互
            'small_cap_momentum', 'oversold_volume_flag', 'low_pb_reversal',
            # 风险
            'risk_pledge', 'risk_pledge_high', 'risk_st', 'risk_delist_score',
            # 资金流（若有）
            'mf_net_main', 'mf_main_ratio',
            'sent_limit_gene', 'sent_limit_active', 'sent_hsgt_fav',
        ]
        _factor_available = [c for c in _all_factor_candidates if c in df.columns]

        if len(_factor_available) >= 10:
            _orthogonal = self._select_orthogonal_factors(
                df, _factor_available, max_vif=5.0, max_factors=30
            )
            # 确保关键因子不被VIF误删（私募保底：总市值、PE、量比、动量20、波动率）
            _essential = ['total_mv', 'pe', 'vol_ratio_raw', 'pmt_return_20d', 'volat_hist_20d']
            for _ec in _essential:
                if _ec in _factor_available and _ec not in _orthogonal:
                    _orthogonal.append(_ec)
            feature_cols = _orthogonal
            logger.info(f"  🔬 VIF正交化: {len(_factor_available)}→{len(feature_cols)}个独立因子")
        else:
            # 因子太少，回退到静态候选集
            _model_feat_candidates = [
                'total_mv', 'circ_mv', 'pe', 'pb', 'roe', 'roa',
                'revenue_yoy', 'profit_yoy', 'vol', 'turnover_rate',
                'pmt_return_5d', 'pmt_return_20d', 'pmt_return_60d',
                'vol_ratio_raw', 'volat_hist_20d', 'rsi', 'kdj_k', 'kdj_j',
                'macd', 'ma20_distance', 'boll_position', 'multi_oversold_flag',
                'size_score', 'size_mkt_cap_log', 'macd_hist', 'macd_golden',
                'reversal_signal', 'kdj_low_golden', 'rsi_divergence',
            ]
            feature_cols = [c for c in _model_feat_candidates if c in df.columns]
            logger.info(f"  回退静态特征集: {len(feature_cols)}个")
        X = df[feature_cols].fillna(0).values.astype(float) if feature_cols else None

        # Step 4：XGBoost V18预测
        if self.xgb_model and USE_XGBOOST and X is not None:
            try:
                pred = self.xgb_model.predict(df[feature_cols].fillna(0))
                df['xgb_score'] = self._normalize(pred)
                self.logger.log_prediction_distribution(pred, "XGBoost V18")
            except Exception as e:
                logger.error(f"  XGBoost V18 预测: {e}")

        # Step 5：AI引擎预测（使用训练时保存的特征列）
        if self.ai_engine and USE_AI_MODELS and hasattr(self, '_ai_feature_cols') and self._ai_feature_cols:
            try:
                # 仅使用训练时使用的特征列，缺失列填充0
                X_ai = df[self._ai_feature_cols].fillna(0).values.astype(float)
                pred = self.ai_engine.predict_ensemble(X_ai)
                df['ai_score'] = pred  # V28: 保留原始尺度，enhance做Robust z-score
                self.logger.log_prediction_distribution(pred, "AI集成")
            except Exception as e:
                logger.error(f"  AI引擎预测: {e}")
        elif self.ai_engine and USE_AI_MODELS and X is not None:
            # 降级：使用全部可用特征（可能维度不匹配，但作为保底）
            logger.warning("  未找到训练特征列，使用全部可用特征（可能维度错误）")
            try:
                pred = self.ai_engine.predict_ensemble(X)
                df['ai_score'] = pred  # V28: 保留原始尺度，enhance做Robust z-score
            except Exception as e:
                logger.error(f"  AI引擎预测失败: {e}")

        # Step 6：双模型预测（涨势 + 抄底）
        if TREE_MODELS_AVAILABLE and feature_cols:
            X_df = df[feature_cols].fillna(0)
            trend_pred, bottom_pred = None, None
            has_trend = (self.trend_model is not None and
                         getattr(self.trend_model, 'enhanced_model', None) is not None)
            has_bottom = (self.bottom_model is not None and
                          getattr(self.bottom_model, 'enhanced_model', None) is not None)

            if has_trend:
                try:
                    trend_pred = self.trend_model.predict(X_df)
                    # V28: 保留原始预测尺度，enhance_stock_selection用Robust z-score统一处理
                    # 去掉此处_normalize，避免双重标准化（_normalize后enhance又_norm_series）
                    # V35: 모델 raw 예측값(z-score 범위) 즉시 60-90으로 리스케일
                    _tp = pd.Series(trend_pred)
                    if _tp.std() > 1e-9 and _tp.abs().max() < 2.0:
                        # raw z-score 범위(-2~2) → 60-90 변환
                        _tn = (_tp - _tp.min()) / (_tp.max() - _tp.min() + 1e-9)
                        df['trend_score'] = (60.0 + _tn * 30.0).values
                    else:
                        df['trend_score'] = trend_pred  # 이미 60-90 범위이거나 균등값
                    self.logger.log_prediction_distribution(trend_pred, "趋势模型")
                except Exception as e:
                    logger.error(f"  趋势模型预测: {e}")

            if has_bottom:
                try:
                    bottom_pred = self.bottom_model.predict(X_df)
                    # V28: 同上，保留原始尺度
                    # V35: 同上，抄底模型 raw → 60-90
                    _bp = pd.Series(bottom_pred)
                    if _bp.std() > 1e-9 and _bp.abs().max() < 2.0:
                        _bn = (_bp - _bp.min()) / (_bp.max() - _bp.min() + 1e-9)
                        df['bottom_score'] = (60.0 + _bn * 30.0).values
                    else:
                        df['bottom_score'] = bottom_pred
                    self.logger.log_prediction_distribution(bottom_pred, "抄底模型")
                except Exception as e:
                    logger.error(f"  抄底模型预测: {e}")

            # 融合双模型得分（可解释：趋势股看20日，抄底股看10日）
            if trend_pred is not None and bottom_pred is not None:
                df['dual_score'] = self._normalize(
                    0.6 * self._normalize(trend_pred) + 0.4 * self._normalize(bottom_pred)
                )
            elif trend_pred is not None:
                df['dual_score'] = df['trend_score']
            elif bottom_pred is not None:
                df['dual_score'] = df['bottom_score']



        # Step 8：风险因子（RiskManager计算风格因子）
        try:
            df = self.risk_manager.calculate_style_factors(df)
        except Exception as e:
            logger.error(f"  风险因子: {e}")

        logger.info("✅ 因子计算完成")
        return df

    # ------------------------------------------------------------------ #
    #  规则选股（冷启动 / 降级）
    # ------------------------------------------------------------------ #
    def rule_based_select(self, df: pd.DataFrame, top_n: int = 100,
                           min_score: float = 0.0) -> pd.DataFrame:
        """
        纯规则选股（样本不足或模型未训练时的降级方案）
        基于 FACTOR_WEIGHTS_2026 手工权重
        分数最终映射到 60~95 区间（前端可直接展示）
        """
        logger.warning("⚠️ 使用规则选股（降级模式）")
        df = df.copy()

        # 先算规则因子（如果还没算）
        if 'pmt_return_20d' not in df.columns:
            try:
                df = self._calculate_rule_factors(df)
            except Exception as e:
                logger.error(f"规则因子计算失败: {e}")

        # ============================================================
        # 【强保底】检查因子整体是否退化（所有std<0.01 → 用close排名）
        # ============================================================
        factor_stds = [
            pd.to_numeric(df[f], errors='coerce').std()
            for f in self.FACTOR_WEIGHTS_2026 if f in df.columns
        ]
        all_degenerate = len(factor_stds) == 0 or all(s < 0.01 for s in factor_stds)
        if all_degenerate:
            logger.warning("  所有因子std<0.01，使用close排名构造60~95分")
            if 'close' in df.columns and pd.to_numeric(df['close'], errors='coerce').std() > 1e-9:
                close_rank = pd.to_numeric(df['close'], errors='coerce').fillna(0).rank(pct=True)
                df['rule_score'] = (60.0 + close_rank * 35.0).round(1)
            else:
                df['rule_score'] = np.linspace(95.0, 60.0, len(df))
            df['neutral_score'] = df['rule_score']
            df['v19_final_score'] = df['rule_score']
            # 先去重到最新日期/ts_code
            if 'trade_date' in df.columns:
                _lt = df['trade_date'].max()
                _df_lt = df[df['trade_date'] == _lt]
                df = _df_lt if len(_df_lt) >= 5 else df.drop_duplicates(subset='ts_code', keep='first')
            else:
                df = df.drop_duplicates(subset='ts_code', keep='first')
            selected = df.nlargest(top_n, 'rule_score').copy()
            for col in ['trend_score', 'bottom_score', 'dual_score']:
                if col not in selected.columns:
                    selected[col] = selected['rule_score']
            if 'ai_score' not in selected.columns:
                selected['ai_score'] = 0.0
            logger.info(f"  规则选股(close排名保底): {len(selected)} 只")
            return selected

        scores = np.zeros(len(df))
        total_abs_weight = sum(abs(w) for w in self.FACTOR_WEIGHTS_2026.values())

        for factor, weight in self.FACTOR_WEIGHTS_2026.items():
            if factor not in df.columns:
                continue
            vals = pd.to_numeric(df[factor], errors='coerce').fillna(0).values
            # 百分位标准化消除量纲，使每个因子贡献可比
            if vals.std() > 1e-9:
                rank_score = pd.Series(vals).rank(pct=True).values
            else:
                rank_score = np.full(len(vals), 0.5)
            # 正向因子：rank越高越好；负向因子：rank越低越好
            if weight > 0:
                scores += rank_score * abs(weight) / total_abs_weight
            else:
                scores += (1.0 - rank_score) * abs(weight) / total_abs_weight

        # 将0~1分映射到60~95区间，确保前端有区分度（不显示0.0）
        if scores.std() > 1e-9:
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            norm_scores = np.full(len(scores), 0.5)
        display_scores = 60.0 + norm_scores * 35.0  # 60~95

        df['rule_score'] = display_scores

        # 硬过滤（高风险直接剔除）
        mask = pd.Series(True, index=df.index)
        if 'risk_pledge_high' in df.columns:
            mask &= (df['risk_pledge_high'] == 0)
        if 'risk_st' in df.columns:
            mask &= (df['risk_st'] == 0)
        if 'risk_delist_score' in df.columns:
            mask &= (df['risk_delist_score'] == 0)

        df_filtered = df[mask]
        if len(df_filtered) < top_n:
            logger.warning(f"  过滤后剩余 {len(df_filtered)} 只，低于 top_n={top_n}，放宽过滤")
            df_filtered = df

        # ============================================================
        # 【核心修复】过滤到最新交易日，防止同股多日重复入选
        # ============================================================
        if 'trade_date' in df_filtered.columns:
            _latest = df_filtered['trade_date'].max()
            _df_latest = df_filtered[df_filtered['trade_date'] == _latest]
            if len(_df_latest) >= min(top_n, 10):
                df_filtered = _df_latest
                logger.info(f"  规则选股过滤到最新日 {_latest}: {len(df_filtered)} 只")
            else:
                df_filtered = (df_filtered.sort_values('trade_date', ascending=False)
                                          .drop_duplicates(subset='ts_code', keep='first'))
        else:
            df_filtered = df_filtered.drop_duplicates(subset='ts_code', keep='first')

        selected = df_filtered.nlargest(top_n, 'rule_score')

        # ======================================================
        # 【关键修复】映射到前端期望的所有字段
        # 不再让 neutral_score / v19_final_score 留空导致显示0.0
        # ======================================================
        selected = selected.copy()
        selected['neutral_score'] = selected['rule_score']
        selected['v19_final_score'] = selected['rule_score']

        # 趋势分 = 动量 + MACD 成分（若有）
        trend_components = []
        for col in ['pmt_return_20d', 'macd_golden', 'ma_bull', 'ma_trend_score']:
            if col in selected.columns:
                trend_components.append(pd.to_numeric(selected[col], errors='coerce').fillna(0))
        if trend_components:
            raw_trend = sum(trend_components)
            if raw_trend.std() > 1e-9:
                norm_trend = (raw_trend - raw_trend.min()) / (raw_trend.max() - raw_trend.min() + 1e-9)
            else:
                norm_trend = pd.Series(0.5, index=selected.index)
            selected['trend_score'] = (60.0 + norm_trend * 30.0).round(1)
        elif 'trend_score' not in selected.columns:
            selected['trend_score'] = selected['rule_score']

        # 抄底分 = 超卖信号成分
        bottom_components = []
        for col in ['multi_oversold_flag', 'rsi_divergence', 'reversal_signal', 'in_bottom_zone']:
            if col in selected.columns:
                bottom_components.append(pd.to_numeric(selected[col], errors='coerce').fillna(0))
        if bottom_components:
            raw_bottom = sum(bottom_components)
            if raw_bottom.std() > 1e-9:
                norm_bottom = (raw_bottom - raw_bottom.min()) / (raw_bottom.max() - raw_bottom.min() + 1e-9)
            else:
                norm_bottom = pd.Series(0.5, index=selected.index)
            selected['bottom_score'] = (60.0 + norm_bottom * 30.0).round(1)
        elif 'bottom_score' not in selected.columns:
            selected['bottom_score'] = selected['rule_score']

        # 融合分 = 规则分（降级时双模型等于规则）
        if 'dual_score' not in selected.columns:
            selected['dual_score'] = selected['rule_score']
        if 'ai_score' not in selected.columns:
            selected['ai_score'] = 0.0

        logger.info(f"  规则选股完成: {len(selected)} 只，分数范围: "
                    f"{selected['neutral_score'].min():.1f}~{selected['neutral_score'].max():.1f}")
        return selected

    # ------------------------------------------------------------------ #
    #  规则基线得分（保底：模型得分为零时的兜底）
    # ------------------------------------------------------------------ #
    def _compute_rule_baseline(self, df: pd.DataFrame) -> pd.Series:
        """
        计算FACTOR_WEIGHTS_2026规则基线分（百分位加权），映射到0~1
        【关键修复】因子排名必须在同一截面（最新日期）内进行，不能混入历史日期
        私募标准：rank(pct=True) 仅在最新日截面股票间比较，历史行共享截面分数
        """
        # 确定最新截面
        if 'trade_date' in df.columns:
            _latest = df['trade_date'].max()
            df_snap = df[df['trade_date'] == _latest].copy()
        elif 'ts_code' in df.columns:
            df_snap = df.drop_duplicates(subset='ts_code', keep='last').copy()
        else:
            df_snap = df.copy()

        if len(df_snap) == 0:
            df_snap = df.copy()

        scores_snap = np.zeros(len(df_snap))
        total_abs_weight = sum(abs(w) for w in self.FACTOR_WEIGHTS_2026.values())
        if total_abs_weight < 1e-9:
            return pd.Series(np.full(len(df), 0.5), index=df.index)

        for factor, weight in self.FACTOR_WEIGHTS_2026.items():
            if factor not in df_snap.columns:
                continue
            vals = pd.to_numeric(df_snap[factor], errors='coerce').fillna(0).values
            if vals.std() > 1e-9:
                rank_score = pd.Series(vals).rank(pct=True).values
            else:
                rank_score = np.full(len(vals), 0.5)
            if weight > 0:
                scores_snap += rank_score * abs(weight) / total_abs_weight
            else:
                scores_snap += (1.0 - rank_score) * abs(weight) / total_abs_weight

        # 把截面分数广播回全量df（同一ts_code的历史行继承截面分数）
        score_series_snap = pd.Series(scores_snap, index=df_snap.index)
        if 'ts_code' in df_snap.columns:
            score_map = pd.Series(scores_snap, index=df_snap['ts_code'].values)
            # 全量df按ts_code映射（历史行也用当前截面分数）
            result = df['ts_code'].map(score_map) if 'ts_code' in df.columns else pd.Series(np.full(len(df), 0.5), index=df.index)
            result = result.fillna(0.5)
            return pd.Series(result.values, index=df.index)
        else:
            return pd.Series(np.full(len(df), 0.5), index=df.index)

    # ------------------------------------------------------------------ #
    #  动态权重选股（主流程）
    # ------------------------------------------------------------------ #
    def select_stocks_dynamic(self, df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
        """动态权重选股（含降级机制）"""
        logger.info(f"\n⚖️ 动态权重选股（目标: {top_n} 只）")

        # 降级检测：样本不足 且 双模型未训练
        has_dual = (self.trend_model is not None and
                    getattr(self.trend_model, 'enhanced_model', None) is not None)
        has_ai = self.ai_engine is not None

        if len(df) < 500 and not has_dual and not has_ai:
            logger.warning("⚠️ 样本不足且模型未训练，进入规则选股")
            return self.rule_based_select(df, top_n)

        # ============================================================
        # 【强保底 v2】在因子计算后立即检查整体退化
        # ============================================================
        _factor_stds = [
            pd.to_numeric(df[f], errors='coerce').std()
            for f in self.FACTOR_WEIGHTS_2026 if f in df.columns
        ]
        if len(_factor_stds) > 0 and all(s < 0.01 for s in _factor_stds):
            logger.warning("⚠️ select_stocks_dynamic: 所有因子std<0.01，用close排名构造60~95分")
            return self.rule_based_select(df, top_n)

        df = df.copy()
        # 【私募理由】原阈值0.40在mock 40%负面下误杀50%股票，直接导致候选池崩塌。
        # 私募实盘预过滤只切明确高风险，选股数量由 nlargest(top_n) 控制。
        _df_before_nlp_filter = df.copy()   # 保存原始，用于池子崩塌时恢复
        if 'negative_ratio' in df.columns:
            _neg       = pd.to_numeric(df['negative_ratio'], errors='coerce').fillna(0)
            _veto_flag = df.get('veto', pd.Series(False, index=df.index))
            # 只切 veto=True（引擎明确否决）或 neg>0.55（多数帖子负面）
            _nlp_pass  = ~(_veto_flag == True) & ~(_neg > 0.55)
            _n_removed = (~_nlp_pass).sum()
            if _n_removed > 0:
                df = df[_nlp_pass].copy()
                logger.info(f"  [NLP预过滤] 剔除高风险: {_n_removed}只 (veto或neg>55%)")
            # 保底：过滤后不足 top_n*2 只，恢复完整候选池（数据质量问题时不影响选股）
            if len(df) < top_n * 2:
                logger.warning(
                    f"  [NLP预过滤] 过滤后仅{len(df)}只 < 目标{top_n*2}只，恢复全量候选池"
                )
                df = _df_before_nlp_filter
        # ================================================
        # 【关键修复】始终计算规则基线分作为保底因子
        # 确保factor_data永远非空且有区分度
        # ================================================
        rule_baseline = self._compute_rule_baseline(df)
        logger.info(f"  规则基线分范围: {rule_baseline.min():.3f}~{rule_baseline.max():.3f}, "
                    f"std={rule_baseline.std():.4f}")

        # 汇总可用因子
        factor_data: Dict[str, pd.Series] = {
            'rule_baseline': rule_baseline,  # 永远加入规则基线
        }
        factor_mapping = {
            'small_cap': ['small_cap_composite_score', 'small_cap_score', 'size_score'],
            'xgboost':   ['xgb_score'],
            'ai':        ['ai_score'],
            'dual':      ['dual_score'],
            'nlp':       ['nlp_score'],
        }
        for fname, candidates in factor_mapping.items():
            for col in candidates:
                if col in df.columns and df[col].std() > 1e-9:
                    factor_data[fname] = df[col].fillna(0)
                    break

        # 若仍无 small_cap，用市值构造
        if 'small_cap' not in factor_data and 'total_mv' in df.columns:
            mv = pd.to_numeric(df['total_mv'], errors='coerce').fillna(1e5)
            mv_log = np.log1p(mv)
            factor_data['small_cap'] = pd.Series(
                self._normalize(-mv_log), index=df.index
            )

        # 检测市场状态
        market_regime = self._detect_market_regime(df)
        logger.info(f"  市场状态: {market_regime}，因子数: {len(factor_data)}")

        # 动态优化权重
        try:
            weights = self.weight_optimizer.optimize_weights(
                factor_data=factor_data,
                market_regime=market_regime
            )
        except Exception as e:
            logger.error(f"  权重优化失败，使用等权: {e}")
            weights = {k: 1.0 / len(factor_data) for k in factor_data}

        # 保证rule_baseline至少有20%权重（防止优化器给0权重）
        min_rule_weight = 0.2
        if weights.get('rule_baseline', 0) < min_rule_weight:
            weights['rule_baseline'] = min_rule_weight
            # 归一化
            total_w = sum(weights.values())
            if total_w > 1e-9:
                weights = {k: v / total_w for k, v in weights.items()}

        self.logger.log_weights(weights)
        logger.info(f"  因子权重: { {k: round(v, 3) for k, v in weights.items()} }")

        # ══════════════════════════════════════════════════════════════
        # 【V20 最终评分公式】头部私募标准（幻方/九坤水准）
        # ──────────────────────────────────────────────────────────────
        # Final = 0.40×rule + 0.25×trend + 0.15×bottom + 0.10×ai + 0.10×consistency
        # 高波动环境: rule权重自动提升到50%（AI模型避险）
        # 置信度惩罚: final_score *= confidence_factor (0.85~1.05)
        # ══════════════════════════════════════════════════════════════

        # 检测市场波动率（决定是否切换高波动权重）
        _volat_mean = 0.0
        if 'volat_hist_20d' in df.columns:
            _volat_mean = float(pd.to_numeric(df['volat_hist_20d'], errors='coerce').mean() or 0)
        _high_volatility = _volat_mean > 0.025  # 20日历史波动率均值 > 2.5% → 高波动

        # 动态权重
        if _high_volatility:
            W_RULE    = 0.50  # 高波动期：规则因子权重提升到50%
            W_TREND   = 0.20
            W_BOTTOM  = 0.12
            W_AI      = 0.08
            W_CONSIS  = 0.10
            logger.info(f"  🌊 高波动市场(volat={_volat_mean:.4f}) → 规则权重提升至50%")
        else:
            W_RULE    = 0.40  # 平稳期：标准私募权重
            W_TREND   = 0.25
            W_BOTTOM  = 0.15
            W_AI      = 0.10
            W_CONSIS  = 0.10

        # 各成分 z-score 标准化（消除量纲差异）
        def _norm_series(s: pd.Series) -> pd.Series:
            """
            V28: Robust Z-score（用中位数+IQR替代均值+std）
            ══════════════════════════════════════════════════
            原因：标准z-score在全市场均涨时，微小差距被等比放大，
            导致"只比均值好一点点"的股票获得与"极端好"相同的z值。
            Robust版本：对极端值不敏感，信噪比更高。
            公式：先Winsorize 1%/99%缩尾 → (x - median) / (IQR * 0.7413)
            0.7413 = 使IQR标准化后的std≈1（与标准正态一致）
            """
            # 先 Winsorize 1%~99% 缩尾，防止单只极端值扭曲IQR
            _q1, _q99 = s.quantile(0.01), s.quantile(0.99)
            s_clip = s.clip(_q1, _q99)
            _med = s_clip.median()
            _iqr = s_clip.quantile(0.75) - s_clip.quantile(0.25)
            if _iqr > 1e-9:
                return (s_clip - _med) / (_iqr * 0.7413)
            # IQR为零时退化到均值标准化
            _std = s_clip.std()
            return (s_clip - s_clip.mean()) / _std if _std > 1e-9 else s_clip * 0

        rule_z    = _norm_series(pd.to_numeric(factor_data.get('rule_baseline', pd.Series(0, index=df.index)), errors='coerce').fillna(0))
        trend_z   = _norm_series(pd.to_numeric(df.get('trend_score', pd.Series(0, index=df.index)), errors='coerce').fillna(0))
        bottom_z  = _norm_series(pd.to_numeric(df.get('bottom_score', pd.Series(0, index=df.index)), errors='coerce').fillna(0))
        ai_z      = _norm_series(pd.to_numeric(factor_data.get('ai', pd.Series(factor_data.get('dual', pd.Series(0, index=df.index)), index=df.index)), errors='coerce').fillna(0))

        # 一致性奖励：AI分与规则分接近的股票额外加分
        _agree_bonus = pd.Series(0.0, index=df.index)
        if 'ai_score' in df.columns and 'rule_score' in df.columns:
            _ai_r  = pd.to_numeric(df['ai_score'], errors='coerce').fillna(0)
            _rul_r = pd.to_numeric(df['rule_score'], errors='coerce').fillna(0)
            _agree  = 1.0 - (_ai_r - _rul_r).abs().clip(upper=20) / 20.0
            _agree_bonus = _norm_series(_agree)
        consis_z = _agree_bonus

        df['v19_final_score'] = (
            W_RULE   * rule_z.values +
            W_TREND  * trend_z.values +
            W_BOTTOM * bottom_z.values +
            W_AI     * ai_z.values +
            W_CONSIS * consis_z.values
        )

        # 置信度惩罚因子：风险因子高的股票整体降权
        if 'pledge_ratio' in df.columns:
            _pledge = pd.to_numeric(df['pledge_ratio'], errors='coerce').fillna(0)
            _conf_factor = (1.05 - _pledge.clip(upper=50) / 1000).clip(0.85, 1.05)
            df['v19_final_score'] *= _conf_factor.values

        logger.info(f"  ✅ V20评分公式: R={W_RULE}*rule + T={W_TREND}*trend + "
                    f"B={W_BOTTOM}*bottom + AI={W_AI}*ai + C={W_CONSIS}*consistency"
                    f" | 高波动={_high_volatility}")

        # 检查v19_final_score是否有效（方差接近零说明因子失效）
        score_std = df['v19_final_score'].std()
        logger.info(f"  v19_final_score std={score_std:.4f}")
        if score_std < 1e-9:
            logger.warning("⚠️ v19_final_score方差为零，直接使用规则选股")
            return self.rule_based_select(df, top_n)

        # 风险中性化
        try:
            df = self.risk_manager.full_neutralization(df, score_col='v19_final_score')
            # 中性化后方差可能变小，检查
            if 'neutral_score' in df.columns and df['neutral_score'].std() < 1e-9:
                logger.warning("  风险中性化后方差为零，跳过中性化")
                df['neutral_score'] = df['v19_final_score']
        except Exception as e:
            logger.error(f"  风险中性化失败: {e}")
            df['neutral_score'] = df['v19_final_score']

        # 确保neutral_score存在
        if 'neutral_score' not in df.columns:
            df['neutral_score'] = df['v19_final_score']

        # 风险过滤
        try:
            df = self.risk_manager.apply_risk_filters(df, score_col='neutral_score')
        except Exception as e:
            logger.error(f"  风险过滤失败: {e}")

        # ============================================================
        # 【关键修复】将neutral_score重缩放到60~95区间
        # z-score分数在±2范围内，round(1)会显示小数而非0
        # 但私募习惯显示0~100分，这里映射到60~95更有区分度
        # ============================================================
        score_col = 'neutral_score' if 'neutral_score' in df.columns else 'v19_final_score'
        sc = df[score_col]
        if sc.std() > 1e-9:
            norm_sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
            df['neutral_score'] = 60.0 + norm_sc * 35.0
        else:
            # 最后保底：直接用规则基线
            norm_rb = (rule_baseline - rule_baseline.min()) / (rule_baseline.max() - rule_baseline.min() + 1e-9)
            df['neutral_score'] = 60.0 + norm_rb * 35.0
        df['v19_final_score'] = df['neutral_score']

        # 同样对trend_score、bottom_score做重缩放（如果来自模型）
        for score_field in ['trend_score', 'bottom_score', 'dual_score']:
            if score_field in df.columns and df[score_field].std() > 1e-9:
                sc_f = df[score_field]
                norm_f = (sc_f - sc_f.min()) / (sc_f.max() - sc_f.min() + 1e-9)
                df[score_field] = (60.0 + norm_f * 30.0).round(1)

        # ══════════════════════════════════════════════════════════════
        # 【V35 Bug Fix】补全缺失的展示字段
        # trend_score/bottom_score: 模型未训练或预测值过小(raw z-score)时
        # 规则因子兜底，确保值在 60~90 区间而非 0.0
        # ══════════════════════════════════════════════════════════════

        # trend_score: 优先用模型值（已在1600处重缩放），否则用动量/MA规则因子
        _trend_needs_fill = (
            'trend_score' not in df.columns
            or df['trend_score'].abs().max() < 1.0          # 值在0~1之间 = raw未缩放
            or df['trend_score'].std() < 1e-9               # 全部相同 = 无区分度
        )
        if _trend_needs_fill:
            _trend_raw = pd.Series(0.0, index=df.index)
            for _tc in ['pmt_return_20d', 'ma_trend_score', 'macd_golden', 'ma_bull']:
                if _tc in df.columns:
                    _trend_raw = _trend_raw + pd.to_numeric(df[_tc], errors='coerce').fillna(0)
            if _trend_raw.std() > 1e-9:
                _nt = (_trend_raw - _trend_raw.min()) / (_trend_raw.max() - _trend_raw.min() + 1e-9)
                df['trend_score'] = (60.0 + _nt * 30.0).round(1)
                logger.info(f"  [V35] trend_score 规则兜底: {df['trend_score'].min():.1f}~{df['trend_score'].max():.1f}")
            else:
                df['trend_score'] = (df['neutral_score'] * 0.95).round(1)
                logger.info("  [V35] trend_score neutral_score*0.95 兜底")

        # bottom_score: 优先用模型值，否则用超卖信号规则因子
        _bottom_needs_fill = (
            'bottom_score' not in df.columns
            or df['bottom_score'].abs().max() < 1.0
            or df['bottom_score'].std() < 1e-9
        )
        if _bottom_needs_fill:
            _bottom_raw = pd.Series(0.0, index=df.index)
            for _bc in ['multi_oversold_flag', 'rsi_divergence', 'reversal_signal', 'in_bottom_zone']:
                if _bc in df.columns:
                    _bottom_raw = _bottom_raw + pd.to_numeric(df[_bc], errors='coerce').fillna(0)
            # 反转RSI（RSI低=超卖=抄底信号高）
            if 'rsi' in df.columns:
                _rsi_inv = 1.0 - pd.to_numeric(df['rsi'], errors='coerce').fillna(50) / 100.0
                _bottom_raw = _bottom_raw + _rsi_inv * 0.5
            if _bottom_raw.std() > 1e-9:
                _nb = (_bottom_raw - _bottom_raw.min()) / (_bottom_raw.max() - _bottom_raw.min() + 1e-9)
                df['bottom_score'] = (60.0 + _nb * 30.0).round(1)
                logger.info(f"  [V35] bottom_score 规则兜底: {df['bottom_score'].min():.1f}~{df['bottom_score'].max():.1f}")
            else:
                df['bottom_score'] = (df['neutral_score'] * 0.90).round(1)
                logger.info("  [V35] bottom_score neutral_score*0.90 兜底")

        if 'dual_score' not in df.columns or df['dual_score'].std() < 1e-9:
            df['dual_score'] = df['neutral_score']
        if 'ai_score' not in df.columns:
            df['ai_score'] = 0.0

        # 组合优化（可选）
        if AUTO_BACKTEST.get('enable', False):
            try:
                df = self.portfolio_optimizer.optimize(df, score_col='neutral_score')
            except Exception as e:
                logger.error(f"  组合优化失败: {e}")

        # ============================================================
        # 【核心修复】每只股票只保留最新交易日的行，再 nlargest
        # 历史因子计算需要多日数据，但选股只取最新截面
        # ============================================================
        if 'trade_date' in df.columns:
            latest_date = df['trade_date'].max()
            df_latest = df[df['trade_date'] == latest_date].copy()
            logger.info(f"  📅 过滤到最新交易日 {latest_date}: {len(df_latest)} 只股票")
            if len(df_latest) >= top_n:
                df = df_latest
            else:
                # 若最新日数据不足，按 ts_code 保留最新日期行
                logger.warning(f"  最新日 {latest_date} 只有 {len(df_latest)} 只，按 ts_code 取最新行")
                df = (df.sort_values('trade_date', ascending=False)
                        .drop_duplicates(subset='ts_code', keep='first'))
        else:
            # 无日期列：直接按 ts_code 去重（保留 neutral_score 最高行）
            df = df.sort_values('neutral_score', ascending=False).drop_duplicates(subset='ts_code', keep='first')

        # 排序选Top N
        selected = df.nlargest(top_n, 'neutral_score')
        logger.info(f"✅ 选出 {len(selected)} 只股票，"
                    f"分数范围: {selected['neutral_score'].min():.1f}~{selected['neutral_score'].max():.1f}")

        self._record_factor_performance(factor_data, selected)
        return selected

    # ------------------------------------------------------------------ #
    #  【私募核心】VIF 因子去冗余
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  VIF因子筛选（委托给 factor_selector 独立模块）
    # ------------------------------------------------------------------ #
    def _select_orthogonal_factors(self, df: pd.DataFrame,
                                    factor_cols: List[str],
                                    max_vif: float = 5.0,
                                    max_factors: int = 30) -> List[str]:
        """委托给 factor_selector.select_orthogonal_factors"""
        return _select_ortho(df, factor_cols, max_vif=max_vif, max_factors=max_factors,
                           parent_logger=logger)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Z-score 标准化"""
        arr = np.array(arr, dtype=float)
        std = arr.std()
        if std > 1e-9:
            return (arr - arr.mean()) / std
        return arr - arr.mean()

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  【私募核心】大盘择时（委托给 market_timing 独立模块）
    # ------------------------------------------------------------------ #
    def _compute_market_timing(self, df: pd.DataFrame) -> Dict:
        """委托给 market_timing.compute_market_timing"""
        return _compute_mt(df, parent_logger=logger)

    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """委托给 market_timing.detect_market_regime"""
        return _detect_regime(df)

    def _record_factor_performance(self, factor_data: Dict, selected: pd.DataFrame):
        """记录因子表现（用于下次动态权重优化）"""
        self.factor_performance_history.append({
            'timestamp': datetime.now(),
            'factors': list(factor_data.keys()),
            'selected_count': len(selected),
        })
        if len(self.factor_performance_history) > 30:
            self.factor_performance_history = self.factor_performance_history[-30:]

    def get_status(self) -> Dict:
        """获取引擎状态"""
        return {
            'small_cap_engine': self.small_cap_engine is not None,
            'ai_engine': self.ai_engine is not None,
            'nlp_engine': self.nlp_engine is not None,
            'xgb_model': self.xgb_model is not None,
            'trend_model': (self.trend_model is not None and
                            getattr(self.trend_model, 'enhanced_model', None) is not None),
            'bottom_model': (self.bottom_model is not None and
                             getattr(self.bottom_model, 'enhanced_model', None) is not None),
            'tree_models_available': TREE_MODELS_AVAILABLE,
            'weight_optimizer': True,
            'risk_manager': True,
            'portfolio_optimizer': True,
            'cost_model': True,
        }


# ==================== 全局引擎实例 ====================
v19_enhanced_engine = V19EnhancedEngine()


def enhance_stock_selection_v19(stock_df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    V19优化选股函数 - 两阶段NLP架构（头部私募标准）
    
    Stage 1: 仅用规则因子筛选候选池（约 top_n * 5 只，最少150只）
    Stage 2: 对候选池运行NLP情绪分析，再结合全部因子最终排序
    这样既保证情绪因子覆盖高质量股票，又大幅减少NLP计算量。
    """
    _NLP_CANDIDATE_N = min(300, max(top_n * 5, 150))  # 候选池大小

    start = time.time()
    logger.info("=" * 70)
    logger.info(f"🚀 V19优化选股开始（两阶段NLP | 候选池={_NLP_CANDIDATE_N}只）")
    logger.info("=" * 70)
    try:
        # ===== Stage 1：计算规则因子（不含NLP），快速筛选候选池 =====
        # 先计算所有规则因子（含技术指标、小市值、资金流等，但不含NLP）
        df = v19_enhanced_engine.calculate_all_factors(stock_df)

        # 获取最新交易日截面，确保选股基于最新数据
        if 'trade_date' in df.columns:
            snap_date = df['trade_date'].max()
            df_snap = df[df['trade_date'] == snap_date].copy()
            logger.info(f"  [Stage1] 最新交易日 {snap_date}: {len(df_snap)} 只股票截面")
        else:
            # 无日期列：按 ts_code 去重，保留最高分行作为截面
            df_snap = (df.sort_values('rule_score' if 'rule_score' in df.columns else 'close',
                                      ascending=False)
                       .drop_duplicates(subset='ts_code', keep='first')
                       .copy())
            logger.info(f"  [Stage1] 无日期列，去重截面: {len(df_snap)} 只")

        # 确定排序用的规则分（优先用 rule_score，其次用 v19_final_score，最后用 dual_score）
        _rule_col = next(
            (c for c in ['rule_score', 'v19_final_score', 'dual_score', 'close']
             if c in df_snap.columns
             and pd.to_numeric(df_snap[c], errors='coerce').std() > 1e-6),
            None
        )
        if _rule_col is None:
            # 实在没有，用 neutral_score 或自定义
            _rule_col = 'neutral_score' if 'neutral_score' in df_snap.columns else None

        if _rule_col and len(df_snap) > _NLP_CANDIDATE_N:
            # 从截面选 top 代码
            _top_codes = set(
                df_snap.nlargest(_NLP_CANDIDATE_N, _rule_col)['ts_code'].tolist()
            )
            # 用代码过滤全量df，保留历史行（滚动因子需要多行数据）
            df_candidate = df[df['ts_code'].isin(_top_codes)].copy()
            logger.info(
                f"  [Stage1] 截面 {len(df_snap)} 只 → Top{_NLP_CANDIDATE_N} 代码 "
                f"→ 全量 {len(df_candidate)} 行（含历史）（按 {_rule_col} 排序）"
            )
        else:
            df_candidate = df.copy()
            logger.info(f"  [Stage1] 截面股票数{len(df_snap)}≤候选池{_NLP_CANDIDATE_N}，全量进Stage2")

        # ===== Stage 2：仅对候选池运行 NLP =====
        if v19_enhanced_engine.nlp_engine and USE_NLP_SENTIMENT:
            try:
                _t_nlp = time.time()
                logger.info(
                    f"  [Stage2-NLP] 开始对候选池 {len(df_candidate)} 只运行情绪分析..."
                )
                df_candidate = v19_enhanced_engine.nlp_engine.add_sentiment_to_df(
                    df_candidate,
                    days=NLP_CONFIG.get('sentiment_window', 7),
                    stock_code_col='ts_code'
                )
                # nlp_score 标准化
                if 'nlp_score' in df_candidate.columns:
                    _nlp_std = df_candidate['nlp_score'].std()
                    if _nlp_std > 1e-9:
                        df_candidate['nlp_score'] = v19_enhanced_engine._normalize(
                            df_candidate['nlp_score'].values
                        )
                _neg_nonzero = (df_candidate['negative_ratio'] > 0).sum()
                _veto_cnt    = (df_candidate['veto'] == True).sum()
                _ind_rank_col = 'industry_sentiment_rank'
                if _ind_rank_col in df_candidate.columns:
                    _df_ind = df_candidate.drop_duplicates('industry')[
                        ['industry', _ind_rank_col]
                    ].sort_values(_ind_rank_col, ascending=False)
                    _top_inds = " | ".join(
                        f"{r['industry']}({r[_ind_rank_col]:.2f})"
                        for _, r in _df_ind.head(3).iterrows()
                        if pd.notna(r.get('industry'))
                    )
                    _bot_inds = " | ".join(
                        f"{r['industry']}({r[_ind_rank_col]:.2f})"
                        for _, r in _df_ind.tail(3).iterrows()
                        if pd.notna(r.get('industry'))
                    )
                    logger.info(f"  [Stage2-NLP] 情绪最强板块: {_top_inds}")
                    logger.info(f"  [Stage2-NLP] 情绪最弱板块: {_bot_inds}")
                logger.info(
                    f"  [Stage2-NLP] 完成: 耗时{time.time()-_t_nlp:.1f}s | "
                    f"negative_ratio非零={_neg_nonzero}只 | veto={_veto_cnt}只"
                )
            except Exception as e:
                logger.error(f"  [Stage2-NLP] 异常: {e}\n{traceback.format_exc()}")
                # 异常时补默认值
                for _col, _def in [
                    ('nlp_score', 0.0), ('negative_ratio', 0.0),
                    ('veto', False), ('veto_reason', ''),
                    ('industry_sentiment_rank', 0.5), ('nlp_data_source', 'unknown'),
                ]:
                    if _col not in df_candidate.columns:
                        df_candidate[_col] = _def
        else:
            # NLP 未启用，给默认值
            for _col, _def in [
                ('nlp_score', 0.0), ('negative_ratio', 0.0),
                ('veto', False), ('veto_reason', ''),
                ('industry_sentiment_rank', 0.5), ('nlp_data_source', 'unknown'),
            ]:
                if _col not in df_candidate.columns:
                    df_candidate[_col] = _def

        # 把 NLP 结果写回全量 df（供后续继承字段用，但最终选股在候选池上进行）
        _nlp_cols = ['ts_code', 'nlp_score', 'negative_ratio', 'veto', 'veto_reason', 
                     'industry_sentiment_rank', 'nlp_data_source']
        _nlp_cols_exist = [c for c in _nlp_cols if c in df_candidate.columns]
        if len(_nlp_cols_exist) > 1:
            _nlp_map = df_candidate[_nlp_cols_exist].drop_duplicates('ts_code').set_index('ts_code')
            for _nc in [c for c in _nlp_cols_exist if c != 'ts_code']:
                df[_nc] = df['ts_code'].map(_nlp_map[_nc]).fillna(
                    0.0 if _nc == 'nlp_score' else (
                    0.0 if _nc == 'negative_ratio' else (
                    0.5 if _nc == 'industry_sentiment_rank' else (
                    'unknown' if _nc == 'nlp_data_source' else (
                    False if _nc == 'veto' else ''))))
                )

        # ===== 最终选股（在候选池上运行，已含NLP因子） =====
        selected = v19_enhanced_engine.select_stocks_dynamic(df_candidate, top_n)

        # ===== 继承关键列（防字段丢失） =====
        _key_inherit = ['name', 'industry', 'close', 'total_mv', 'pe', 'pb',
                        'turnover_rate', 'pledge_ratio', 'open', 'high', 'low',
                        'vol', 'amount', 'circ_mv',
                        'nlp_score', 'negative_ratio', 'veto', 'veto_reason',
                        'industry_sentiment_rank', 'nlp_data_source']
        _missing_cols = [c for c in _key_inherit if c not in selected.columns or
                         (selected[c].dtype in [np.float64, np.float32, float] and
                          selected[c].fillna(0).eq(0).all()
                          and c not in ('nlp_score', 'negative_ratio'))]
        if _missing_cols:
            # 优先从 df_candidate 继承（含NLP），其次从原始 df
            for _src_df in [df_candidate, df]:
                _src_cols = ['ts_code'] + [c for c in _missing_cols if c in _src_df.columns]
                if len(_src_cols) > 1:
                    _src = _src_df[_src_cols].drop_duplicates('ts_code').set_index('ts_code')
                    for col in [c for c in _missing_cols if c in _src.columns]:
                        if col not in selected.columns or (
                            selected[col].dtype in [np.float64, np.float32, float] and
                            selected[col].fillna(0).eq(0).all() and
                            col not in ('nlp_score', 'negative_ratio')
                        ):
                            selected = selected.copy()
                            selected[col] = selected['ts_code'].map(
                                _src[col]
                            ).fillna(selected.get(col, pd.Series(dtype=float)))
                    _missing_cols = [c for c in _missing_cols if c not in selected.columns]

        elapsed = time.time() - start
        logger.info(f"✅ V19选股完成: {len(selected)}只 | 总耗时={elapsed:.1f}s")
        return selected

    except Exception as e:
        logger.error(f"❌ enhance_stock_selection_v19异常: {e}\n{traceback.format_exc()}")
        # 降级：纯规则选股
        try:
            df_fallback = v19_enhanced_engine.calculate_all_factors(stock_df)
            return v19_enhanced_engine.rule_based_select(df_fallback, top_n)
        except Exception as e2:
            logger.error(f"❌ 降级选股也失败: {e2}")
            return pd.DataFrame()


# ==================== 真实数据获取（全量接口）====================
def get_real_stock_data(start_date: str = None, end_date: str = None,
                         stock_pool: str = 'csi1000', lookback_months: int = 12) -> pd.DataFrame:
    """
    获取真实Tushare数据（私募级全量接口）
    包含：日线/基本面/资金流/涨跌停/北向/质押/财务/业绩预告
    """
    if end_date is None:
        end_date = get_latest_trading_date()
    if start_date is None:
        start_date = (datetime.strptime(end_date, '%Y%m%d') -
                      timedelta(days=lookback_months * 30)).strftime('%Y%m%d')

    logger.info(f"📊 获取全量数据: {stock_pool} | {start_date}~{end_date}")
    
    # ── 数据缓存机制 (Grok 优化) ──
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"real_data_{stock_pool}_{start_date}_{end_date}.parquet")
    
    if os.path.exists(cache_file):
        # 缓存有效期：如果 end_date 是今天，缓存有效期 2 小时；否则无限期
        file_mtime = os.path.getmtime(cache_file)
        if end_date < datetime.now().strftime('%Y%m%d') or (time.time() - file_mtime) < 7200:
            logger.info(f"✅ 从本地缓存加载数据: {cache_file}")
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"⚠️ 读取缓存失败: {e}，将重新获取数据")

    ts.set_token(os.environ.get('TUSHARE_TOKEN', TUSHARE_TOKEN))
    pro = ts.pro_api(timeout=30)
    api_governor.reset()  # 重置积分计数

    # ---- Step 1: 获取股票池 ----
    try:
        if stock_pool == 'csi1000':
            stocks = pro.index_weight(index_code='000852.CSI')
            if stocks is None or stocks.empty:
                stocks = pro.stock_basic(exchange='', list_status='L',
                                          fields='ts_code,name,industry')
        elif stock_pool == 'csi2000':
            stocks = pro.index_weight(index_code='932000.CSI')
        elif stock_pool == 'all':
            stocks = pro.stock_basic(exchange='', list_status='L',
                                      fields='ts_code,name,industry')
        else:
            logger.error(f"未知股票池: {stock_pool}")
            return pd.DataFrame()

        if stocks is None or stocks.empty:
            logger.error("股票池为空")
            return pd.DataFrame()

        # 确保 ts_code 列存在
        if 'con_code' in stocks.columns and 'ts_code' not in stocks.columns:
            stocks = stocks.rename(columns={'con_code': 'ts_code'})
        ts_codes_all = stocks['ts_code'].unique().tolist()
        logger.info(f"  股票池: {len(ts_codes_all)} 只")
    except Exception as e:
        logger.error(f"  获取股票池失败: {e}")
        return pd.DataFrame()

    # ---- Step 2: 批量获取日线 + 基本面（行业标准方案）----
    all_data: List[pd.DataFrame] = []
    _BASIC_FIELDS = 'ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps,total_mv,circ_mv'

    reliable_end_date = get_latest_trading_date()
    reliable_start_date = start_date

    # ============================================================
    # 根本修复：更换 daily_basic 调用模式
    # ---------------------------------------------------------------
    # 【旧方案（必然失败的原因）】：
    #   pro.daily_basic(ts_code='50只', start_date='12个月前', end_date='20260213')
    #   → 50只 × 250天 = 12500行 > Tushare 6000行/次限制 → 静默返回空表
    #   → 三层重试全部失败（因为数据量问题，不是日期问题）
    #
    # 【新方案（行业标准）】：
    #   pro.daily_basic(ts_code='', trade_date='20260213')
    #   → 一次拉取某日全A股 ≈ 5000行 < 6000行限制 → 成功
    #   → 只需调用1次（不分批），效率提升100倍
    #   → 结果按 ts_code 合并到 daily（无 trade_date 匹配问题）
    #   → forward-fill 补齐历史区间（节假日/历史日期自动继承最新 basic）
    # ============================================================

    # Step 2a: 找到 daily_basic 有数据的最近交易日（带缓存）
    basic_end_date = find_valid_basic_date(max_lookback=15)
    _gap = abs((datetime.strptime(reliable_end_date, '%Y%m%d') -
                datetime.strptime(basic_end_date, '%Y%m%d')).days)
    logger.info(f"  📅 daily: end={reliable_end_date} | daily_basic: trade_date={basic_end_date}"
                f"{'（回退 ' + str(_gap) + ' 天，节假日）' if _gap > 0 else '（同日，交易日）'}")

    # Step 2b: 按 trade_date 一次性拉取全A股 basic（≈5000行，不超6000限制）
    basic_df: Optional[pd.DataFrame] = None
    logger.info(f"  📥 daily_basic 全市场单日查询: trade_date={basic_end_date}...")
    for _attempt in range(3):
        try:
            _tmp = pro.daily_basic(
                ts_code='',               # ← 空 = 全部股票
                trade_date=basic_end_date, # ← 指定单日，而非 start/end 日期范围
                fields=_BASIC_FIELDS
            )
            if _tmp is not None and not _tmp.empty:
                basic_df = _tmp
                basic_df['trade_date'] = basic_df['trade_date'].astype(str).str.zfill(8)
                logger.info(f"  ✅ daily_basic 获取成功: {len(basic_df)} 只股票 (尝试{_attempt+1})")
                break
            logger.warning(f"  daily_basic 第{_attempt+1}次返回空，重试...")
        except Exception as _e:
            logger.warning(f"  daily_basic 第{_attempt+1}次失败: {_e}")
        time.sleep(1.0)

    if basic_df is None or basic_df.empty:
        logger.error("  ❌ daily_basic 全市场查询三次均失败，PE/市值/换手率将为估算值")
    else:
        # 只保留 ts_code + basic 字段，用于后续按 ts_code merge（不按日期）
        _basic_cols_keep = ['ts_code'] + [c for c in _BASIC_FIELDS.split(',')
                                           if c not in ('ts_code', 'trade_date')]
        basic_df = basic_df[[c for c in _basic_cols_keep if c in basic_df.columns]].copy()
        basic_df = basic_df.drop_duplicates('ts_code')
        logger.info(f"  basic_df 去重后: {len(basic_df)} 只，字段: {list(basic_df.columns)}")

    def _safe_merge(daily: pd.DataFrame, batch_label: str) -> pd.DataFrame:
        """把 daily_basic（单日全市场）按 ts_code merge 到 daily（多日历史）"""
        daily['trade_date'] = daily['trade_date'].astype(str).str.zfill(8)

        if basic_df is None or basic_df.empty:
            logger.warning(f"  {batch_label} basic_df 为空，跳过 basic merge")
            return daily

        merged = pd.merge(daily, basic_df, on='ts_code', how='left')
        matched = merged['total_mv'].notna().sum()
        match_pct = matched / len(merged) * 100 if len(merged) > 0 else 0
        logger.debug(f"  {batch_label} ts_code merge: {matched}/{len(merged)} 行有效 ({match_pct:.0f}%)")
        return merged

    # Step 2c: 按批次只获取 daily（不再每批调 daily_basic）
    batch_size = DATA_SOURCE_CONFIG.get('batch_size', 50)

    def fetch_tushare_batch(i):
        batch_codes = ts_codes_all[i:i + batch_size]
        codes_str = ','.join(batch_codes)
        batch_label = f"批次{i // batch_size}"
        try:
            daily = pro.daily(
                ts_code=codes_str,
                start_date=reliable_start_date,
                end_date=reliable_end_date,
                fields='ts_code,trade_date,open,high,low,close,vol,amount'
            )
            if daily is not None and len(daily) > 0:
                return _safe_merge(daily, batch_label)
        except Exception as e:
            logger.warning(f"  {batch_label} daily 失败: {e}")
        return None

    # Tushare 多线程加速获取
    logger.info(f"  🚀 启动 Tushare 多线程日线获取...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_tushare_batch, i): i for i in range(0, len(ts_codes_all), batch_size)}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                all_data.append(res)
    
    # ── Baostock 兜底机制（单线程防封） ──
    if not all_data:
        logger.warning("❌ Tushare 日线获取全部失败，尝试使用 Baostock 兜底...")
        if 'bs' in globals():
            bs.login()
            bs_start = f"{reliable_start_date[:4]}-{reliable_start_date[4:6]}-{reliable_start_date[6:]}"
            bs_end = f"{reliable_end_date[:4]}-{reliable_end_date[4:6]}-{reliable_end_date[6:]}"
            
            for code in ts_codes_all:
                # 转换 ts_code 到 baostock 格式 (e.g. 000001.SZ -> sz.000001)
                bs_code = f"{code.split('.')[1].lower()}.{code.split('.')[0]}"
                rs = bs.query_history_k_data_plus(bs_code,
                    "date,code,open,high,low,close,volume,amount",
                    start_date=bs_start, end_date=bs_end, frequency="d", adjustflag="3")
                
                if rs.error_code == '0':
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        data_list.append(rs.get_row_data())
                    if data_list:
                        df_bs = pd.DataFrame(data_list, columns=rs.fields)
                        df_bs['ts_code'] = code
                        df_bs['trade_date'] = df_bs['date'].str.replace('-', '')
                        df_bs.rename(columns={'volume': 'vol'}, inplace=True)
                        all_data.append(_safe_merge(df_bs, f"BS_{code}"))
            bs.logout()
            logger.info(f"  ✅ Baostock 兜底获取完成")
        else:
            logger.error("  ❌ Baostock 未安装，无法兜底")

    if not all_data:
        logger.error("❌ 日线数据获取失败")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    logger.info(f"  日线数据: {len(df)} 条, {df['ts_code'].nunique()} 只, {df['trade_date'].nunique()} 天")

    # ── 兜底检查：若 total_mv 仍几乎全0，说明 basic_df 为空，再试一次单日全市场 ──
    _mv_nonzero = (pd.to_numeric(df.get('total_mv', pd.Series(0)), errors='coerce').fillna(0) > 0).sum()
    _mv_pct = _mv_nonzero / len(df) * 100 if len(df) > 0 else 0
    logger.info(f"  ⚡ 合并后 total_mv 非零率: {_mv_nonzero}/{len(df)} ({_mv_pct:.0f}%)")

    if _mv_pct < 10 and (basic_df is None or basic_df.empty):
        # basic_df 获取失败时的最后兜底：再次尝试单日全市场查询
        logger.warning("  ⚠️ basic_df 为空，最后兜底：重试 daily_basic 单日全市场...")
        try:
            _retry = pro.daily_basic(
                ts_code='',
                trade_date=basic_end_date,
                fields=_BASIC_FIELDS
            )
            if _retry is not None and not _retry.empty:
                _retry = _retry.drop_duplicates('ts_code')
                _fill_cols = ['turnover_rate', 'volume_ratio', 'pe', 'pb', 'ps', 'total_mv', 'circ_mv']
                _fill_cols = [c for c in _fill_cols if c in _retry.columns]
                _retry_map = _retry.set_index('ts_code')
                for _col in _fill_cols:
                    if _col not in df.columns:
                        df[_col] = np.nan
                    _mask = df[_col].isna() | (pd.to_numeric(df[_col], errors='coerce').fillna(0) == 0)
                    df.loc[_mask, _col] = df.loc[_mask, 'ts_code'].map(_retry_map[_col])
                logger.info(f"  ✅ 兜底补全成功: {len(_retry)} 只")
        except Exception as _e:
            logger.error(f"  ❌ 兜底重试失败: {_e}")

    # ── 私募级 forward-fill：按 ts_code 填满历史区间（normal day无副作用）──
    # 新方案中 basic 数据 ts_code 维度合并，历史日期行 basic 列全为同一值（已在merge阶段设好）
    # forward-fill 确保即使某只股票在 basic_end_date 无数据，也能从相邻日期继承
    _basic_cols = ['turnover_rate', 'volume_ratio', 'pe', 'pb', 'ps', 'total_mv', 'circ_mv']
    _basic_cols_present = [c for c in _basic_cols if c in df.columns]
    if _basic_cols_present:
        df = df.sort_values(['ts_code', 'trade_date'])
        for _col in _basic_cols_present:
            df[_col] = pd.to_numeric(df[_col], errors='coerce')
        df[_basic_cols_present] = (
            df.groupby('ts_code', sort=False)[_basic_cols_present]
            .transform(lambda x: x.ffill().bfill())
        )
        _mv_ff = (df['total_mv'].fillna(0) > 0).sum() if 'total_mv' in df.columns else 0
        logger.info(f"  ✅ forward-fill 后 total_mv 非零: {_mv_ff}/{len(df)} ({_mv_ff/len(df)*100:.0f}%)")

    # ── Grok Fix 3 最终估算兜底：若 pe/pb/total_mv/turnover_rate 仍为0 ──
    if 'turnover_rate' not in df.columns or (pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0) == 0).all():
        if 'circ_mv' in df.columns and 'vol' in df.columns:
            _circ = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0)
            _vol = pd.to_numeric(df['vol'], errors='coerce').fillna(0)
            df['turnover_rate'] = np.where(_circ > 0, _vol / (_circ * 10000 + 1e-8), 0.0)
            logger.info("  ⚙️ turnover_rate 由 vol/circ_mv 估算")

    if 'total_mv' not in df.columns or (pd.to_numeric(df['total_mv'], errors='coerce').fillna(0) == 0).all():
        if 'close' in df.columns and 'turnover_rate' in df.columns:
            _close = pd.to_numeric(df['close'], errors='coerce').fillna(0)
            _tr = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
            _vol = pd.to_numeric(df['vol'], errors='coerce').fillna(0)
            df['total_mv'] = np.where(_tr > 0, _close * _vol / (_tr + 1e-8), 0.0)
            logger.info("  ⚙️ total_mv 由 close*vol/turnover_rate 估算")

    if 'pe' in df.columns:
        df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
        df['pe'] = df['pe'].where(df['pe'].notna() & (df['pe'] != 0), other=999.0)
    else:
        df['pe'] = 999.0

    if 'pb' in df.columns:
        df['pb'] = pd.to_numeric(df['pb'], errors='coerce')
        df['pb'] = df['pb'].where(df['pb'].notna() & (df['pb'] != 0), other=999.0)
    else:
        df['pb'] = 999.0

    # ===== 修复点：确保股票名称和行业被正确合并 =====
    # 先获取完整的股票基本信息表
    try:
        stock_basic_info = pro.stock_basic(
            exchange='', list_status='L',
            fields='ts_code,name,industry,market,list_date'
        )
        if stock_basic_info is not None and not stock_basic_info.empty:
            logger.info(f"  ✅ 获取股票基本信息成功: {len(stock_basic_info)} 条")
            # 只保留需要的列，并确保 ts_code 是字符串
            stock_basic_info = stock_basic_info[['ts_code', 'name', 'industry']].copy()
            stock_basic_info['ts_code'] = stock_basic_info['ts_code'].astype(str)
            
            # 将基本信息合并到主 DataFrame df 中
            df = df.merge(stock_basic_info, on='ts_code', how='left')
            logger.info(f"  ✅ 行业信息合并完成，非空行业占比: {df['industry'].notna().sum()/len(df):.1%}")
        else:
            logger.warning("  ⚠️ 获取股票基本信息失败或返回空，行业信息将缺失")
            # 如果获取失败，添加一个默认的行业列，避免后续代码崩溃
            if 'industry' not in df.columns:
                df['industry'] = '未知'
    except Exception as e:
        logger.error(f"  ❌ 获取股票基本信息异常: {e}")
        if 'industry' not in df.columns:
            df['industry'] = '未知'

    # ---- Step 3: 资金流（最近5日均值，覆盖全量股票） ----
    mf_dict: Dict[str, Dict] = {}
    _recent_start = _prev_trading_date(10)
    if api_governor.acquire('moneyflow'):
        try:
            n_stocks = len(ts_codes_all)
            batch_size = 100  # 每批处理100只，避免超时
            logger.info(f"  [资金流] 开始获取全量 {n_stocks} 只股票数据，每批 {batch_size} 只...")
            
            for i in range(0, n_stocks, batch_size):
                batch_end = min(i + batch_size, n_stocks)
                batch_codes = ts_codes_all[i:batch_end]
                codes_str = ','.join(batch_codes)
                
                logger.debug(f"   批次 {i//batch_size + 1}/{(n_stocks-1)//batch_size + 1}: 处理 {len(batch_codes)} 只")
                
                mf_raw = pro.moneyflow(
                    ts_code=codes_str,
                    start_date=_recent_start, end_date=end_date,
                    fields='ts_code,trade_date,buy_lg_amount,sell_lg_amount,net_mf_amount'
                )
                if mf_raw is not None and not mf_raw.empty:
                    mf_agg = mf_raw.groupby('ts_code').agg(
                        mf_net_main=('net_mf_amount', 'mean'),
                        mf_buy_large=('buy_lg_amount', 'mean'),
                        mf_sell_large=('sell_lg_amount', 'mean'),
                    ).reset_index()
                    mf_agg['mf_main_ratio'] = (
                        mf_agg['mf_buy_large'] /
                        (mf_agg['mf_buy_large'] + mf_agg['mf_sell_large'] + 1e-9)
                    )
                    for _, row in mf_agg.iterrows():
                        mf_dict[row['ts_code']] = {
                            'mf_net_main': row['mf_net_main'],
                            'mf_main_ratio': row['mf_main_ratio']
                        }
                
                time.sleep(0.5)  # 批次间隔，避免触发限流
            
            logger.info(f"  资金流数据: 成功获取 {len(mf_dict)}/{n_stocks} 只")
        except Exception as e:
            logger.warning(f"  资金流获取失败: {e}")

    if mf_dict:
        mf_df = pd.DataFrame.from_dict(mf_dict, orient='index').reset_index()
        mf_df = mf_df.rename(columns={'index': 'ts_code'})
        df = df.merge(mf_df, on='ts_code', how='left')
    # ---- Step 4: 涨跌停情绪（近20日涨停次数）----
    limit_dict: Dict[str, float] = {}
    _limit_start = _prev_trading_date(20)
    if api_governor.acquire('limit_list'):
        try:
            limit_raw = pro.limit_list(
                start_date=_limit_start, end_date=end_date,
                fields='ts_code,trade_date,limit_type'
            )
            if limit_raw is not None and not limit_raw.empty:
                # 涨停次数（limit_type='U'）
                up_cnt = (limit_raw[limit_raw['limit_type'] == 'U']
                          .groupby('ts_code').size().reset_index(name='sent_limit_gene'))
                # 连续涨停（近5日）
                _recent5 = _prev_trading_date(5)
                active_cnt = (limit_raw[
                    (limit_raw['limit_type'] == 'U') &
                    (limit_raw['trade_date'] >= _recent5)
                ].groupby('ts_code').size().reset_index(name='sent_limit_active'))

                for _, row in up_cnt.iterrows():
                    limit_dict[row['ts_code']] = row['sent_limit_gene']
                limit_df = up_cnt.merge(active_cnt, on='ts_code', how='left').fillna(0)
                df = df.merge(limit_df, on='ts_code', how='left')
                df['sent_limit_gene'] = df.get('sent_limit_gene', pd.Series(0)).fillna(0)
                df['sent_limit_active'] = df.get('sent_limit_active', pd.Series(0)).fillna(0)
                logger.info(f"  涨停数据: {len(limit_dict)} 只有涨停记录")
        except Exception as e:
            logger.warning(f"  涨停数据获取失败: {e}")

    # ---- Step 5: 北向资金偏好（近5日净买入）----
    if api_governor.acquire('hsgt_top10'):
        try:
            hsgt_codes: set = set()
            for mkt in ['SH', 'SZ']:
                hsgt_raw = pro.hsgt_top10(
                    start_date=_recent_start, end_date=end_date,
                    market_type=mkt,
                    fields='ts_code,trade_date,net_amount'
                )
                if hsgt_raw is not None and not hsgt_raw.empty:
                    # 净买入为正的股票视为北向青睐
                    fav = hsgt_raw.groupby('ts_code')['net_amount'].sum()
                    fav_codes = fav[fav > 0].index.tolist()
                    hsgt_codes.update(fav_codes)
            if hsgt_codes:
                df['sent_hsgt_fav'] = df['ts_code'].isin(hsgt_codes).astype(float)
                logger.info(f"  北向青睐: {len(hsgt_codes)} 只")
        except Exception as e:
            logger.warning(f"  北向资金获取失败: {e}")

    # 【私募理由】pledge_stat 是月/季度更新的低频接口，正确调用方式：
    #   1. 必须通过 api_governor 保护积分（消耗2分）
    #   2. 传 end_date 限定最近一个季度，避免全表扫描超时
    #   3. 若当季无数据，再回退上一季度
    #   4. 自算 pledge_ratio = (unrest + rest) / total * 100
    _pledge_done = False
    if api_governor.acquire('pledge_stat'):
        try:
            logger.info("  [质押] api_governor通过，开始获取质押数据...")

            # 尝试最近季度末（按季度向前找最近3个季度）
            _today_dt  = datetime.strptime(end_date, '%Y%m%d')
            _quarter_ends = []
            for _delta_months in [0, 3, 6]:
                _m = _today_dt.month - (_today_dt.month - 1) % 3 - _delta_months
                _y = _today_dt.year
                while _m <= 0:
                    _m += 12
                    _y -= 1
                _qe = datetime(_y, _m, 1) - timedelta(days=1)
                _quarter_ends.append(_qe.strftime('%Y%m%d'))

            pledge_raw = None
            for _qdate in _quarter_ends:
                try:
                    _tmp = pro.pledge_stat(
                        end_date=_qdate,
                        fields='ts_code,end_date,pledge_count,unrest_pledge,rest_pledge,total_share'
                    )
                    if _tmp is not None and not _tmp.empty:
                        pledge_raw = _tmp
                        logger.info(f"  [质押] end_date={_qdate} 获取到 {len(pledge_raw)} 条")
                        break
                    else:
                        logger.debug(f"  [质押] end_date={_qdate} 返回空，尝试上一季度")
                except Exception as _pe:
                    logger.debug(f"  [质押] end_date={_qdate} 异常: {_pe}")

            if pledge_raw is not None and not pledge_raw.empty:
                # 自算质押率
                pledge_raw['_unrest'] = pd.to_numeric(pledge_raw['unrest_pledge'], errors='coerce').fillna(0)
                pledge_raw['_rest']   = pd.to_numeric(pledge_raw['rest_pledge'],   errors='coerce').fillna(0)
                pledge_raw['_total']  = pd.to_numeric(pledge_raw['total_share'],   errors='coerce').replace(0, np.nan)
                pledge_raw['pledge_ratio'] = (
                    (pledge_raw['_unrest'] + pledge_raw['_rest']) / pledge_raw['_total'] * 100
                ).clip(0, 100).fillna(0)

                # 每只股票只取最新一条
                pledge_clean = (
                    pledge_raw.sort_values('end_date', ascending=False)
                    [['ts_code', 'pledge_ratio']]
                    .drop_duplicates('ts_code', keep='first')
                    .reset_index(drop=True)
                )

                if 'pledge_ratio' in df.columns:
                    df = df.drop(columns=['pledge_ratio'])
                df = df.merge(pledge_clean, on='ts_code', how='left')
                df['pledge_ratio'] = df['pledge_ratio'].fillna(0.0)

                _nonzero = (df['pledge_ratio'] > 0).sum()
                logger.info(
                    f"  [质押] ✅ 合并完成: 有质押={_nonzero}只 | "
                    f"均值={df[df['pledge_ratio']>0]['pledge_ratio'].mean():.1f}% | "
                    f"总={len(df)}只"
                )
                _pledge_done = True
            else:
                logger.warning("  [质押] 三个季度均返回空，pledge_ratio保持0（检查Tushare积分≥500）")
        except Exception as e:
            logger.error(f"  [质押] 获取异常: {e}")
    else:
        logger.warning("  [质押] api_governor积分不足，跳过质押数据获取")

    if not _pledge_done:
        if 'pledge_ratio' not in df.columns:
            df['pledge_ratio'] = 0.0
    # ---- Step 7: 财务指标（ROE/ROA/营收增速）----
    if api_governor.acquire('fina_indicator'):
        try:
            # 只取最近一期财报
            fina_codes_str = ','.join(ts_codes_all[:100])  # 限量节省积分
            fina_raw = pro.fina_indicator(
                ts_code=fina_codes_str,
                fields='ts_code,end_date,roe,roa,netprofit_yoy,revenue_yoy,grossprofit_margin'
            )
            if fina_raw is not None and not fina_raw.empty:
                fina_latest = (fina_raw.sort_values('end_date', ascending=False)
                               .groupby('ts_code').first().reset_index())
                fina_latest = fina_latest.rename(columns={
                    'netprofit_yoy': 'profit_yoy',
                    'grossprofit_margin': 'gross_margin'
                })
                keep_cols = ['ts_code', 'roe', 'roa', 'profit_yoy', 'revenue_yoy', 'gross_margin']
                keep_cols = [c for c in keep_cols if c in fina_latest.columns]
                df = df.merge(fina_latest[keep_cols], on='ts_code', how='left')
                logger.info(f"  财务数据: {fina_latest['ts_code'].nunique()} 只")
        except Exception as e:
            logger.warning(f"  财务数据获取失败: {e}")

    # ---- Step 8: 业绩预告（正向预期 +加分）----
    # Tushare forecast API 要求：ann_date 或 ts_code 至少传一个
    # 私募标准：按公告日期季度窗口查询，覆盖最近两个季度
    if api_governor.acquire('forecast'):
        try:
            from datetime import datetime as _dt
            _today = _dt.strptime(basic_end_date, '%Y%m%d')
            # 最近两个季度的公告日范围
            _fcast_end = basic_end_date
            _fcast_start = (_today.replace(month=max(1, _today.month - 6), day=1)
                            ).strftime('%Y%m%d')

            pos_codes: set = set()
            positive_types = {'预增', '略增', '扭亏', '续盈'}

            # 分批按 ts_code 查询（避免单次数据量过大）
            _fcast_batch = ts_codes_all[:200]  # 限量节省积分
            for _fb in range(0, len(_fcast_batch), 50):
                _fb_codes = ','.join(_fcast_batch[_fb:_fb + 50])
                try:
                    _fr = pro.forecast(
                        ts_code=_fb_codes,           # ← 满足"至少一个"校验
                        ann_date='',                 # 不限定单日，让日期范围生效
                        start_date=_fcast_start,
                        end_date=_fcast_end,
                        fields='ts_code,ann_date,type,p_change_min,p_change_max'
                    )
                    if _fr is not None and not _fr.empty:
                        _pos = _fr[_fr['type'].isin(positive_types)]['ts_code']
                        pos_codes.update(_pos.tolist())
                    time.sleep(0.1)
                except Exception:
                    pass

            if pos_codes:
                df['forecast_positive'] = df['ts_code'].isin(pos_codes).astype(float)
                logger.info(f"  业绩预告正向: {len(pos_codes)} 只")
            else:
                df['forecast_positive'] = 0.0
        except Exception as e:
            logger.warning(f"  业绩预告获取失败: {e}")
            df['forecast_positive'] = 0.0

    # ---- 最终清洗 ----
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    # 补齐缺失的情绪因子为0
    for col in ['mf_net_main', 'mf_main_ratio', 'sent_limit_gene',
                'sent_limit_active', 'sent_hsgt_fav', 'pledge_ratio',
                'forecast_positive']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ============================================================
    # 【关键保底】确保前端关键字段都存在，杜绝"列不存在导致0.0"
    # Fix 1: 强制从 daily/daily_basic 补齐所有关键列
    # Fix 6: pd.to_numeric + fillna(0.0) 全覆盖
    # ============================================================
    _required_fields = {
        'name': lambda d: d.get('ts_code', '未知'),
        'industry': '未知',
        'close': 0.0,
        'total_mv': 0.0,
        'pe': 0.0,
        'pb': 0.0,
        'turnover_rate': 0.0,
        'pledge_ratio': 0.0,
    }
    for col, default in _required_fields.items():
        if col not in df.columns:
            if callable(default):
                df[col] = df.apply(default, axis=1)
            else:
                df[col] = default
        # Fix 6: 强制转数值 + fillna
        if col in ['close', 'total_mv', 'pe', 'pb', 'turnover_rate', 'pledge_ratio']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        elif col in ['name', 'industry']:
            df[col] = df[col].fillna(default if not callable(default) else '未知').astype(str)

    # Fix 1: 若 close 列全为0，说明 daily merge 未成功，尝试从 all_data 再次提取
    if df['close'].eq(0.0).all() and all_data:
        logger.warning("  close 列全为0，尝试从原始 daily 数据重新提取")
        try:
            raw_daily = pd.concat(all_data, ignore_index=True)
            close_map = raw_daily.drop_duplicates(subset=['ts_code', 'trade_date'])[
                ['ts_code', 'trade_date', 'close']
            ].set_index(['ts_code', 'trade_date'])['close']
            df['close'] = df.set_index(['ts_code', 'trade_date']).index.map(
                lambda k: close_map.get(k, 0.0)
            ).values
            df['close'] = pd.to_numeric(df['close'], errors='coerce').fillna(0.0)
            logger.info(f"  close 重新提取: 非零 {(df['close'] > 0).sum()} 条")
        except Exception as e:
            logger.warning(f"  close 重新提取失败: {e}")

    # Fix 6: 确保所有 OHLCV 及基本面数值列类型正确
    for col in ['close', 'open', 'high', 'low', 'vol', 'amount',
                'total_mv', 'circ_mv', 'pe', 'pb', 'turnover_rate',
                'volume_ratio', 'pledge_ratio']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # ── Grok Fix 4: 最终非零率检查 ──────────────────────────────────────
    logger.info("  📊 最终字段非零率检查:")
    for _col in ['pe', 'pb', 'total_mv', 'turnover_rate', 'close']:
        if _col in df.columns:
            _nz = (pd.to_numeric(df[_col], errors='coerce').fillna(0).abs() > 0).sum()
            logger.info(f"    {_col}: {_nz}/{len(df)} 非零 ({_nz/len(df)*100:.1f}%)")
        else:
            logger.warning(f"    {_col}: 列缺失!")
    # 质押数据诊断
    pledge_cnt = (df['pledge_ratio'] > 0).sum() if 'pledge_ratio' in df.columns else 0
    logger.info(f"  🔍 质押诊断: 有质押记录的股票={pledge_cnt}只 | "
                f"如果为0，请检查 Tushare pledge_stat 接口是否返回空")
    logger.info(f"✅ 全量数据获取完成: {len(df)} 条, {df['ts_code'].nunique()} 只 "
                f"({df['trade_date'].nunique() if 'trade_date' in df.columns else '?'} 个交易日)")
    logger.info(f"   数据列: {list(df.columns)}")
    
    # ── 保存本地缓存 ──
    try:
        df.to_parquet(cache_file)
        logger.info(f"💾 数据已缓存到: {cache_file}")
    except Exception as e:
        logger.warning(f"⚠️ 无法保存缓存: {e}")
        
    return df


# ==================== Django视图接口 ====================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


def _make_json_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return _make_json_serializable(obj.to_dict('records' if isinstance(obj, pd.DataFrame) else 'dict'))
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


@csrf_exempt
def dual_verify_stocks(request):
    """主选股接口（趋势 + 抄底分列返回）"""
    try:
        body = json.loads(request.body) if request.body else {}
        max_stocks = body.get('max_stocks', 60)

        df = get_real_stock_data(
            start_date=None, end_date=None,
            stock_pool=DATA_SOURCE_CONFIG.get('stock_pool', 'csi1000'),
            lookback_months=12
        )
        if df is None or df.empty:
            return JsonResponse({'status': 'error', 'message': '数据获取失败，请检查Tushare Token和网络'})

        logger.info(f"数据获取成功: {len(df)} 行, 列: {[c for c in ['ts_code','name','industry','close','total_mv','pe','pb'] if c in df.columns]}")

        if body.get('train', False):
            v19_enhanced_engine.train_all_models(df)

        selected = enhance_stock_selection_v19(df, top_n=max_stocks)

        # ══════════════════════════════════════════════════════════════
        # 【私募核心】大盘择时：熊市禁止趋势信号
        # ══════════════════════════════════════════════════════════════
        market_timing = v19_enhanced_engine._compute_market_timing(df)
        logger.info(f"  市场择时: {market_timing['regime']} | "
                    f"趋势={'允许' if market_timing['trend_allowed'] else '禁止'} | "
                    f"trend权重={market_timing['trend_weight_pct']:.0%}")

        if selected is None or selected.empty:
            return JsonResponse({'status': 'error', 'message': '选股结果为空'})

        logger.info(f"选股完成: {len(selected)} 只，分数字段: neutral_score存在={('neutral_score' in selected.columns)}")
        logger.info(f"DEBUG pledge非零: {(selected['pledge_ratio']>0).sum()}, neg_ratio非零: {(selected['negative_ratio']>0).sum()}")

        # score_col 必须在 _build_item 调用前定义（NameError修复）
        score_col = 'neutral_score' if 'neutral_score' in selected.columns else 'v19_final_score'
        # neutral_score 强制保底：若计算结果全0或NaN，回退到rule_score
        if 'neutral_score' in selected.columns:
            ns = selected['neutral_score']
            if ns.isna().all() or (ns.abs() < 0.001).all():
                logger.warning('  neutral_score全0/NaN，回退到rule_score')
                selected['neutral_score'] = selected.get('rule_score', pd.Series(65.0, index=selected.index))
            selected['neutral_score'] = pd.to_numeric(selected['neutral_score'], errors='coerce').fillna(65.0)
        else:
            selected['neutral_score'] = selected.get('rule_score', 65.0)
            score_col = 'neutral_score'

        def _build_item(row, score_col):
            """
            构建前端展示条目（V31：置信度多因子 + nlp/fundamental veto + veto_reason）
            保证所有字段非NaN/非null
            """
            def _safe_float(val, default=0.0, nonzero_default: float = None):
                try:
                    v = float(val)
                    if v != v or v == float('inf') or v == float('-inf'):
                        return nonzero_default if nonzero_default is not None else default
                    if v == 0.0 and nonzero_default is not None:
                        return nonzero_default
                    return v
                except (TypeError, ValueError):
                    return nonzero_default if nonzero_default is not None else default

            # ── 基础评分字段 ──────────────────────────────────────────────────
            score     = _safe_float(row.get(score_col) or row.get('v19_final_score') or row.get('rule_score'), 70.0)
            # V35: nonzero_default 确保值为0时用score*系数替代（raw模型预测0.02→round→0.0 问题修复）
            trend_s   = _safe_float(row.get('trend_score'),  score * 0.95, nonzero_default=score * 0.95)
            bottom_s  = _safe_float(row.get('bottom_score'), score * 0.90, nonzero_default=score * 0.90)
            dual_s    = _safe_float(row.get('dual_score'),   score)
            ai_s_raw  = _safe_float(row.get('ai_score'),     0.0)
            ai_s      = max(35.0, min(95.0, 60.0 + ai_s_raw * 8.0))
            rule_s    = _safe_float(row.get('rule_score'),   score)

            # ── 技术指标 ──────────────────────────────────────────────────────
            rsi_val     = _safe_float(row.get('rsi'),               50.0)
            kdj_j_val   = _safe_float(row.get('kdj_j'),             50.0)
            ma_dist     = _safe_float(row.get('ma20_distance'),      0.0)
            pmt_20d     = _safe_float(row.get('pmt_return_20d'),     0.0)
            pmt_5d      = _safe_float(row.get('pmt_return_5d'),      0.0)
            oversold_f  = _safe_float(row.get('multi_oversold_flag'), 0.0)
            ma_bull_f   = _safe_float(row.get('ma_bull'),            0.0)
            pct_chg_val = _safe_float(row.get('pmt_return_1d', row.get('pmt_return_5d', 0.0)), 0.0) * 100

            # ── 基本面 ────────────────────────────────────────────────────────
            mv          = _safe_float(row.get('total_mv'),    0.0)
            close_price = _safe_float(row.get('close'),       0.0)
            pledge      = _safe_float(row.get('pledge_ratio'), 0.0)
            turnover    = _safe_float(row.get('turnover_rate'), 0.0)
            pe_val      = _safe_float(row.get('pe'),          0.0)
            pb_val      = _safe_float(row.get('pb'),          0.0)
            roe_val     = _safe_float(row.get('roe'),         0.0)       # V31
            profit_yoy  = _safe_float(row.get('profit_yoy'), 0.0)       # V31
            volat_20d   = _safe_float(row.get('volat_hist_20d'), 0.02)

            # ── NLP情绪 ───────────────────────────────────────────────────────
            nlp_score       = _safe_float(row.get('nlp_score'),       0.0)
            negative_ratio  = _safe_float(row.get('negative_ratio'),  0.0)   # V31
            veto_from_engine = row.get('veto', False)
            veto_reason_from_engine = str(row.get('veto_reason', '')) if pd.notna(row.get('veto_reason')) else ''

            # ════════════════════════════════════════════════════════════════
            # 【V31-VETO-1】NLP 情绪一票否决（加强版）
            # ────────────────────────────────────────────────────────────────
            # 修改点：negative_ratio>0.35 或 引擎返回 veto=True → 否决
            # ════════════════════════════════════════════════════════════════
            _nlp_veto = (negative_ratio > 0.35) or veto_from_engine
            _nlp_veto_reason = ""
            if veto_from_engine and veto_reason_from_engine:
                _nlp_veto_reason = veto_reason_from_engine
            elif negative_ratio > 0.35:
                _nlp_veto_reason = f"股吧负面帖{negative_ratio:.0%}(>35%)"

            # ════════════════════════════════════════════════════════════════
            # 【V31-VETO-2】基本面硬约束一票否决（不变）
            # ════════════════════════════════════════════════════════════════
            _fundamental_veto = False
            _fundamental_veto_reason = ""
            if pe_val > 0 and pe_val > 150:
                _fundamental_veto = True
                _fundamental_veto_reason = f"PE泡沫({pe_val:.0f}>150)"
            elif pb_val > 15:
                _fundamental_veto = True
                _fundamental_veto_reason = f"PB过高({pb_val:.1f}>15)"
            elif pe_val < 0 and roe_val < -5:
                _fundamental_veto = True
                _fundamental_veto_reason = f"持续亏损(PE={pe_val:.0f},ROE={roe_val:.1f}%)"

            # 【修改原因】旧版 veto_reason_from_engine 会被 append 两次：
            #   一次直接 append，一次通过 _nlp_veto_reason 间接 append。
            # 新版：以 set 去重后按优先级排列。
            _veto_reasons = []
            _seen = set()
            def _add_reason(r):
                if r and r not in _seen:
                    _veto_reasons.append(r)
                    _seen.add(r)

            # NLP 来源（引擎 veto_reason 优先，其次自算）
            if veto_from_engine and veto_reason_from_engine:
                _add_reason(veto_reason_from_engine)
            if negative_ratio > 0.35:
                _add_reason(f'股吧负面情绪过高({negative_ratio:.0%})')
            # 基本面来源
            if _fundamental_veto:
                _add_reason(_fundamental_veto_reason)
            veto_reason = ' | '.join(_veto_reasons)

            # ════════════════════════════════════════════════════════════════
            # 【V25评分公式】（不变，V30已稳定）
            # ════════════════════════════════════════════════════════════════
            _trend_vetoed  = (ma_dist < 0 or ma_bull_f < 0.5)
            _bottom_vetoed = (oversold_f < 0.4 and rsi_val > 35)

            _cons_pts = _consistency * 100
            if _trend_vetoed and _bottom_vetoed:
                final_score = (0.80 * rule_s + 0.10 * ai_s + 0.10 * _cons_pts) * 0.82
            elif _trend_vetoed:
                final_score = (0.65 * rule_s + 0.15 * bottom_s + 0.10 * ai_s + 0.10 * _cons_pts) * 0.90
            elif _bottom_vetoed:
                final_score = (0.55 * rule_s + 0.25 * trend_s  + 0.10 * ai_s + 0.10 * _cons_pts) * 0.93
            else:
                final_score = (0.40 * rule_s + 0.25 * trend_s  + 0.15 * bottom_s + 0.10 * ai_s + 0.10 * _cons_pts)

            if not (final_score == final_score) or final_score < 50.0:
                final_score = max(58.0, rule_s * 0.88)
            score = max(50.0, min(99.0, final_score))

            # ════════════════════════════════════════════════════════════════
            # 【V31-CONF】置信度多因子加权算法（加强版）
            # ────────────────────────────────────────────────────────────────
            # 修改点1：NLP 加成改为 (1-negative_ratio)*18
            # 修改点2：低价优质复合过滤（10-40元 + ROE>8% + nlp>0.2）→ +15
            # 修改点3：被 veto 时置信度降档系数调整为 0.72/0.75（不变）
            # ════════════════════════════════════════════════════════════════

            # 估值吸引力（行业内分位，V30已计算）
            _va_raw = _safe_float(row.get('style_valuation_attractiveness'), -1.0)
            if _va_raw < 0.0:
                # 本地兜底
                _pe_s    = max(pe_val, 1.0) if pe_val > 0 else 30.0
                _pb_s    = max(pb_val, 0.1) if pb_val > 0 else 3.0
                _va_base = 1.0 / ((_pe_s / 30.0 + _pb_s / 2.5) / 2.0 + 0.5)
                _va_raw  = min(1.0, max(0.0, _va_base))
            _va_for_conf = float(_va_raw)   # [0, 2.0]，但置信度用 clip(0,1) 更保守

            # 五维置信度公式（NLP 加成已修改）
            base_conf = (
                40.0
                + (score - 50.0) * 0.8                           # 评分分量
                + _consistency   * 15.0                          # 一致性
                + min(1.0, _va_for_conf) * 12.0                  # 估值吸引力（cap at 1.0）
                + (1.0 - min(1.0, _pred_uncertainty)) * 10.0     # 不确定性倒数
                + (1.0 - min(1.0, negative_ratio)) * 18.0        # V31 NLP 正面加成
            )

            # 低价优质复合过滤（V31新增）
            if (10 <= close_price <= 40) and roe_val > 8 and nlp_score > 0.2:
                base_conf += 15

            # 被 veto 时置信度降档
            if _nlp_veto:          base_conf *= 0.72
            if _fundamental_veto:  base_conf *= 0.75
            if _trend_vetoed:      base_conf *= 0.90
            if _bottom_vetoed:     base_conf *= 0.93
            base_conf = int(min(95, max(55, base_conf)))

            # ════════════════════════════════════════════════════════════════
            # 【V30-POSITION】五因子仓位管理（V31 加强 NLP 否决系数）
            # ════════════════════════════════════════════════════════════════
            _va_raw2 = _safe_float(row.get('style_valuation_attractiveness'), -1.0)
            if _va_raw2 >= 0.0:
                _va_mult = 0.72 + _va_raw2 * 0.63
            else:
                _pe_s    = max(pe_val, 1.0) if pe_val > 0 else 30.0
                _pb_s    = max(pb_val, 0.1) if pb_val > 0 else 3.0
                _va_base2 = 1.0 / ((_pe_s / 30.0 + _pb_s / 2.5) / 2.0 + 0.5)
                _va_base2 = min(1.0, max(0.0, _va_base2))
                _va_mult  = 0.72 + _va_base2 * 0.63
                if   close_price  < 15:   _va_mult *= 1.10
                elif close_price <= 40:   _va_mult *= 1.05
                elif close_price >  80:   _va_mult *= 0.85
                elif close_price > 150:   _va_mult *= 0.72
            _va_mult = min(1.35, max(0.65, _va_mult))

            # V31: fundamental_veto 时仓位直接压到最低（同前）
            if _fundamental_veto: _va_mult *= 0.50

            _regime_mult = {
                'small_cap_rally': 1.15, 'bull': 1.10, 'balanced': 1.00,
                'neutral': 1.00, 'large_cap_rally': 0.95, 'volatile': 0.88,
                'bear': 0.82, 'high_vol_bear': 0.78, 'high_vol_bull': 0.90,
            }.get(_market_regime, 1.00)

            _uncertainty_disc = 0.75 if _pred_uncertainty > 0.40 else 1.00
            _vref   = 0.025
            _vsafe  = max(volat_20d, 0.005)
            _base_w = min(0.12, _vref / _vsafe * 0.10)

            _sig_quality = 1.0
            if score >= 88:        _sig_quality *= 1.20
            elif score >= 80:      _sig_quality *= 1.05
            elif score < 68:       _sig_quality *= 0.75
            if _trend_vetoed:      _sig_quality *= 0.85
            if _bottom_vetoed:     _sig_quality *= 0.90
            if pledge > 40:        _sig_quality *= 0.70
            if pe_val > 80:        _sig_quality *= 0.85
            # V31: NLP veto 仓位强力打折（由 0.60 改为 0.25）
            if _nlp_veto:          _sig_quality *= 0.25
            _near_high  = _safe_float(row.get('near_hist_high_flag'), 0.0)
            _parabolic  = _safe_float(row.get('parabolic_flag'),      0.0)
            if _near_high > 0.5:   _sig_quality *= 0.65
            if _parabolic > 0.5:   _sig_quality *= 0.60

            _final_w = (_base_w * _va_mult * _regime_mult * _sig_quality * _uncertainty_disc)
            _final_w = max(0.02, min(0.12, _final_w))

            _pct = round(_final_w * 100, 1)
            if _final_w >= 0.10:
                _pos_advice = f'{_pct}% (积极加仓)'
                _pos_level  = 'aggressive'
            elif _final_w >= 0.07:
                _pos_advice = f'{_pct}% (标准仓位)'
                _pos_level  = 'normal'
            elif _final_w >= 0.04:
                _pos_advice = f'{_pct}% (保守仓位)'
                _pos_level  = 'cautious'
            else:
                _pos_advice = f'{_pct}% (轻仓试探)'
                _pos_level  = 'light'

            # ── 买入信号列表 ─────────────────────────────────────────────────
            buy_signals = []
            if ma_bull_f > 0.5:        buy_signals.append('📈 均线多头')
            if ma_dist >= 0:           buy_signals.append('✅ 站上MA20')
            if rsi_val < 35:           buy_signals.append('💎 RSI超卖')
            if kdj_j_val < 10:         buy_signals.append('💎 KDJ_J极低')
            if oversold_f > 0.5:       buy_signals.append('🔥 多重超卖')
            if pmt_20d > 0.05:         buy_signals.append('🚀 20日强势')
            if _trend_vetoed:          buy_signals.append('🚫 均线空头(已降权)')
            if _bottom_vetoed:         buy_signals.append('⚡ 未达超卖(已降权)')
            if pledge > 40:            buy_signals.append('⚠️ 高质押风险')
            if pe_val > 80:            buy_signals.append('⚠️ 估值偏高')
            if ma_dist < -0.08:        buy_signals.append('📉 深跌MA20')
            if _nlp_veto:              buy_signals.append(f'🔴 NLP否决: {_nlp_veto_reason}')
            if _fundamental_veto:      buy_signals.append(f'🔴 基本面否决: {_fundamental_veto_reason}')
            if roe_val > 15 and profit_yoy > 10:
                                    buy_signals.append(f'⭐ ROE={roe_val:.1f}%+净利增{profit_yoy:.0f}%')

            return {
                'ts_code':         str(row.get('ts_code', '')),
                'code':            str(row.get('ts_code', '')).split('.')[0],
                'name':            str(row.get('name') or row.get('ts_code', '未知')).strip() or '未知',
                'current_price':   round(close_price,  2),
                'pct_chg':         round(pct_chg_val,  2),
                'industry':        str(row.get('industry') or '未知').strip() or '未知',
                'buy_score':       round(score,         1),
                'confidence':      base_conf,                       # V31: 多因子置信度 [55,95]
                'ml_score':        round(ai_s,          1),
                'dual_score':      round(dual_s,        1),
                'trend_score':     round(trend_s,       1),
                'bottom_score':    round(bottom_s,      1),
                'rule_score':      round(rule_s,        1),
                'position_advice': _pos_advice,
                'position_pct':    _pct,
                'position_level':  _pos_level,
                'valuation_attractiveness': round(max(0.0, float(_va_raw) if _va_raw >= 0 else 0.5), 3),
                'market_value':    round(mv / 10000,    2),
                'pledge_ratio':    round(pledge,        1),
                'turnover_rate':   round(turnover,      2),
                'pe':              round(pe_val,        2),
                'pb':              round(pb_val,        2),
                'rsi':             round(rsi_val,       1),
                'kdj_j':           round(kdj_j_val,     1),
                'pmt_return_5d':   round(pmt_5d  * 100, 2),
                'pmt_return_20d':  round(pmt_20d * 100, 2),
                'ma20_distance':   round(ma_dist * 100, 2),
                'multi_oversold_flag': round(oversold_f, 2),
                'nlp_score':       round(nlp_score,     3),
                'negative_ratio':  round(negative_ratio, 3),          # V31新增
                'buy_signals':     buy_signals,
                'trend_vetoed':    _trend_vetoed,
                'bottom_vetoed':   _bottom_vetoed,
                'nlp_veto':        _nlp_veto,                          # V31新增
                'fundamental_veto': _fundamental_veto,                 # V31新增
                'veto_reason':     veto_reason,                        # V31新增（前端展示）
                # V35: 高位风险标记（前端卡片徽章使用）
                'near_hist_high_flag': round(_near_high, 2),
                'parabolic_flag':      round(_parabolic, 2),
                # V36: 行业板块情绪分位（0=最悲观行业, 1=最乐观行业）
                'industry_sentiment_rank': round(_safe_float(row.get('industry_sentiment_rank'), 0.5), 3),
                'nlp_data_source':         str(row.get('nlp_data_source') if pd.notna(row.get('nlp_data_source')) else 'unknown'),
            }



        # ── 从 AI 引擎获取模型权重 / IC / 一致性（供前端图表）────────
        _model_weights  = {}
        _model_ic       = {}
        _consistency    = 0.7   # V25: 无AI时默认0.7，confidence_factor≥1.0不惩罚
        # V30: 为 _build_item 仓位计算提供市场状态和AI置信度闭包变量
        _market_regime    = 'balanced'  # 默认均衡，_adjust_by_market_regime 同名变量
        _pred_uncertainty = 0.30        # 默认低不确定性
        if v19_enhanced_engine.ai_engine is not None:
            try:
                _health = v19_enhanced_engine.ai_engine.get_health_report()
                _model_weights    = _health.get('_model_weights', {})
                _model_ic         = _health.get('_model_ic', {})
                _consistency      = _health.get('_consistency_score', 0.5)
                _market_regime    = str(_health.get('market_regime', 'balanced') or 'balanced')
                _pred_uncertainty = float(_health.get('_pred_uncertainty', 0.30))
            except Exception as _he:
                logger.warning(f"  健康报告获取失败: {_he}")


        # ── 构建展示条目 ──────────────────────────────────────────────
        results = []
        for _, row in selected.iterrows():
            item = _build_item(row, score_col)
            # 附加 Hard Guard 标记（供下方过滤用）
            item['_ma_above_ma20'] = bool(row.get('ma20_distance', 0) >= 0)   # close >= ma20
            item['_ma_bull']       = bool(row.get('ma_bull', 0) > 0.5)        # MA5>MA10>MA20
            item['_multi_oversold']= float(row.get('multi_oversold_flag', 0))
            item['_rsi']           = float(row.get('rsi', 50))
            item['_kdj_j']         = float(row.get('kdj_j', 50))
            item['_pmt_ret_20d']   = float(row.get('pmt_return_20d', 0))
            item['_ma_dist']       = float(row.get('ma20_distance', 0))        # 用于 Hard Guard 判断
            results.append(item)

        half = max(max_stocks // 2, 1)

        # ══════════════════════════════════════════════════════════════
        # 【Hard Guards V34】私募硬规则过滤（放宽版，保证池子充足）
        # 核心原则：合格优先，合格不足时混入不合格补足 half，绝不让池子缩水
        # ══════════════════════════════════════════════════════════════
        def _is_trend_qualified(s: dict) -> bool:
            """
            涨势股门槛（V34放宽）：
              主判1：完美多头（MA20上方 AND MA多头排列）
              主判2：MA20偏离 ≥ -3%（小幅回踩允许）
              兜底：20日动量 > 1%
            """
            if s['_ma_above_ma20'] and s['_ma_bull']:
                return True
            if s['_ma_dist'] >= -0.03:         # MA20偏离≥-3%（已×100）
                return True
            if s['_pmt_ret_20d'] > 0.01:       # 20日动量>1%（已×100比较）
                return True
            return False

        def _is_bottom_qualified(s: dict) -> bool:
            """
            抄底股门槛（V35优化版）：
            在系统性大跌时防止大面积优质标的被误杀，平时保持高胜率。
            门槛收紧核心指标（RSI < 35，KDJ < 10），但只要满足任一极端超卖条件即放行。
            """
            if s.get('_multi_oversold', 0) > 0.2:    return True  # 综合超卖标记较高
            if s.get('_rsi', 50)           < 35:     return True  # 传统强超卖
            if s.get('_kdj_j', 50)         < 10:     return True  # J线极度超跌
            if s.get('_ma_dist', 0)        < -0.07:  return True  # 偏离MA20达7%以上
            return False

        # ── 涨势股分组 ──────────────────────────────────────────────
        trend_qualified   = [s for s in results if _is_trend_qualified(s)]
        trend_unqualified = [s for s in results if not _is_trend_qualified(s)]
        logger.info(f"  [HardGuard] 涨势合格={len(trend_qualified)}, 抄底合格={len([s for s in results if _is_bottom_qualified(s)])}")

        # ══════════════════════════════════════════════════════════════
        # 【私募核心】熊市择时：趋势信号全部禁止
        # 熊市/恐慌时投资者应该空仓或只做抄底反弹，不可追趋势
        # ══════════════════════════════════════════════════════════════
        if not market_timing.get('trend_allowed', True):
            _orig_trend_count = len(trend_qualified)
            trend_qualified = []
            trend_unqualified = []
            logger.warning(
                f"  🛑 熊市择时生效：{_orig_trend_count}只趋势股已全部禁止 | "
                f"所有推荐转为抄底信号（逆势反弹策略）"
            )

        # 合格优先；不足时混入不合格补足 half（保证不丢股票）
        if len(trend_qualified) >= half:
            trend_stocks = sorted(trend_qualified,
                                  key=lambda x: x.get('trend_score', 0), reverse=True)[:half]
        else:
            trend_stocks = (
                sorted(trend_qualified,   key=lambda x: x.get('trend_score', 0), reverse=True) +
                sorted(trend_unqualified, key=lambda x: x.get('trend_score', 0), reverse=True)
            )[:half]
            if len(trend_qualified) < half // 2:
                logger.warning(f"  ⚠️ 涨势股合格仅{len(trend_qualified)}只，已混入不合格补足{half}只")

        # ── 抄底股分组 ──────────────────────────────────────────────
        bottom_qualified   = [s for s in results if _is_bottom_qualified(s)]
        bottom_unqualified = [s for s in results if not _is_bottom_qualified(s)]

        if len(bottom_qualified) >= half:
            bottom_stocks = sorted(bottom_qualified,
                                   key=lambda x: x.get('bottom_score', 0), reverse=True)[:half]
        else:
            bottom_stocks = (
                sorted(bottom_qualified,   key=lambda x: x.get('bottom_score', 0), reverse=True) +
                sorted(bottom_unqualified, key=lambda x: x.get('bottom_score', 0), reverse=True)
            )[:half]
            if len(bottom_qualified) < half // 2:
                logger.warning(f"  ⚠️ 抄底股合格仅{len(bottom_qualified)}只，已混入不合格补足{half}只")

        # ── 清理内部标记字段 ─────────────────────────────────────────
        _internal_keys = ['_ma_above_ma20', '_ma_bull', '_multi_oversold',
                          '_rsi', '_kdj_j', '_pmt_ret_20d', '_ma_dist']
        for lst in [trend_stocks, bottom_stocks]:
            for item in lst:
                for k in _internal_keys:
                    item.pop(k, None)
        # ══════════════════════════════════════════════════════════════
        # 【头部私募最终修复】强制写回所有前端所需字段（趋势/抄底/动量）
        # 防止 Hard Guards / nlargest 之后列被覆盖
        # ══════════════════════════════════════════════════════════════
        trend_dict  = {item['code']: item.get('trend_score',  65.0) for item in results}
        bottom_dict = {item['code']: item.get('bottom_score', 65.0) for item in results}

        for idx, row in selected.iterrows():
            code = str(row.get('ts_code', '')).split('.')[0]
            selected.loc[idx, 'trend_score']  = trend_dict.get(code, row.get('rule_score', 65.0))
            selected.loc[idx, 'bottom_score'] = bottom_dict.get(code, row.get('rule_score', 65.0))
            selected.loc[idx, 'dual_score']   = (
                0.6 * selected.loc[idx, 'trend_score'] + 
                0.4 * selected.loc[idx, 'bottom_score']
            )

        # 确保动量因子永远存在（前端 5日/20日涨跌）
        for col in ['pmt_return_5d', 'pmt_return_20d']:
            if col not in selected.columns:
                selected[col] = 0.0

        # revenue_yoy 额外保底（小市值因子需要）
        if 'revenue_yoy' not in selected.columns:
            selected['revenue_yoy'] = 0.0

        logger.info(f"  [最终写回修复] trend_score 非零={ (selected['trend_score']>0).sum() } 只 | "
                    f"bottom_score 非零={ (selected['bottom_score']>0).sum() } 只 | "
                    f"pmt_return_20d 非零={ (selected['pmt_return_20d']!=0).sum() } 只")

        # ══════════════════════════════════════════════════════════════
        # 市场状态面板（V34终极版：直接从 selected DataFrame 计算，不依赖 results 列表）
        # 【私募理由】results 经 Hard Guards 后可能只有6只；selected 有完整60只。
        # _safe_median 去掉 v!=0.0 限制，_safe_mean 不过滤 >0 避免负值被排除。
        # ══════════════════════════════════════════════════════════════
        def _col_stat(col, func='median', multiplier=1.0, fallback=None):
            """从 selected DataFrame 安全提取统计值，无限制。"""
            try:
                if col in selected.columns:
                    s = pd.to_numeric(selected[col], errors='coerce').dropna()
                    if len(s) == 0:
                        return fallback
                    v = float(s.median() if func == 'median' else s.mean())
                    return round(v * multiplier, 2)
            except Exception:
                pass
            return fallback

        def _results_mean(field):
            """从 results 列表取均值（含负值），无 >0 过滤。"""
            try:
                vals = [r[field] for r in results if field in r and r[field] is not None]
                nums = [float(v) for v in vals if str(v).replace('.','').replace('-','').isdigit() or isinstance(v, (int, float))]
                return round(float(np.mean(nums)), 1) if nums else None
            except Exception:
                return None

        # 市场评分：selected 的 neutral_score 均值（已映射到60-95区间）
        _market_score = _col_stat('neutral_score', 'mean')
        if _market_score is None:
            _market_score = _results_mean('buy_score') or 0.0

        # 置信度：results 的 confidence 均值
        _market_confidence = _results_mean('confidence') or 0.0

        # ══════════════════════════════════════════════════════════════
        # 【私募核心】择时数据覆写：用市场择时信号覆盖统计值
        # 原因：selected 的统计均值反映的是"选出来这批股票"的质量，
        # 而非"当前市场环境"的好坏。择时信号才是市场状态的真实判断。
        # ══════════════════════════════════════════════════════════════
        _market_score     = market_timing.get('market_score', _market_score or 50.0)
        _market_confidence = 100.0 if market_timing.get('trend_allowed', True) else 40.0
        _volatility       = market_timing.get('volatility', _volatility or 0.0) * 100  # → %

        # 60日趋势：selected 的 pmt_return_60d 中位数 ×100 转%
        _trend_60d = _col_stat('pmt_return_60d', 'median', multiplier=100.0, fallback=None)
        # 20日趋势：selected 的 pmt_return_20d 中位数 ×100 转%
        _trend_20d = _col_stat('pmt_return_20d', 'median', multiplier=100.0, fallback=None)
        # 60d兜底：若无数据用20d×2估算
        if _trend_60d is None:
            _trend_60d = round((_trend_20d or 0.0) * 2.0, 2)
        if _trend_20d is None:
            _trend_20d = 0.0

        # 波动率：selected 的 volat_hist_20d 中位数 ×100 转%
        _volatility = _col_stat('volat_hist_20d', 'median', multiplier=100.0, fallback=None)
        if _volatility is None or _volatility == 0.0:
            # 兜底：用 RSI 离散度推算隐含波动（RSI std × 0.3）
            _rsi_s = pd.to_numeric(selected.get('rsi', pd.Series(dtype=float)), errors='coerce').dropna()
            _volatility = round(float(_rsi_s.std()) * 0.3, 2) if len(_rsi_s) > 3 else 1.5

        # 成交量比：selected 的 vol_ratio_raw 中位数
        _vol_ratio = _col_stat('vol_ratio_raw', 'median', fallback=1.0)
        if not _vol_ratio or _vol_ratio == 0.0:
            _vol_ratio = 1.0

        # 市场状态描述
        _regime_display = {
            'bull': '牛市', 'bear': '熊市', 'balanced': '均衡',
            'neutral': '中性', 'volatile': '震荡', 'small_cap_rally': '小盘行情',
            'large_cap_rally': '大盘行情', 'high_vol_bull': '高波动牛市',
            'high_vol_bear': '高波动熊市',
        }
        _health_exists = ('_health' in dir()) and (_health is not None)
        _regime_key = str(_health.get('market_regime', 'balanced') if _health_exists else 'balanced')
        _regime_key = market_timing.get('regime', _regime_key)
        _regime_cn  = _regime_display.get(_regime_key, '均衡')

        # NLP负面均值
        _neg_ratio_mean = 0.0
        if 'negative_ratio' in selected.columns:
            _nr = pd.to_numeric(selected['negative_ratio'], errors='coerce').dropna()
            _neg_ratio_mean = round(float(_nr.mean()) * 100, 1) if len(_nr) > 0 else 0.0

        logger.info(
            f"  [市场面板V34] score={_market_score} | conf={_market_confidence}% | "
            f"60d={_trend_60d}% | 20d={_trend_20d}% | "
            f"volat={_volatility}% | vol_ratio={_vol_ratio} | "
            f"neg={_neg_ratio_mean}% | selected={len(selected)}只"
        )

        # ── 微信通知（后台线程，不阻塞API）──────────────────────────
        _notify_async('send_stock_report',
                       stocks=results,
                       market_timing=market_timing,
                       stock_pool_name=stock_pool)

        return JsonResponse(_make_json_serializable({
            'status': 'success',
            'trend_stocks':  trend_stocks,
            'bottom_stocks': bottom_stocks,
            'data_date':     get_latest_trading_date(),
            # ── V35: market_regime 作为顶层对象 ─────────────────────────────
            # 前端 renderRegime(regime) 需要对象格式：
            #   regime.score / regime.confidence / regime.trend_60d 等
            # trend_60d/trend_20d/volatility 后端已是百分比（如5.2），
            # 前端模板会再 *100 显示，所以这里除以100传入
            # ──────────────────────────────────────────────────────────────────
            'market_regime': {
                'regime':     _regime_key,
                'score':      round(_market_score    or 0.0, 1),
                'confidence': round(_market_confidence or 0.0, 1),
                'trend_allowed': market_timing.get('trend_allowed', True),
                'trend_weight_pct': market_timing.get('trend_weight_pct', 0.5),
                'trend_60d':  round((_trend_60d  or 0.0) / 100, 4),
                'trend_20d':  round((_trend_20d  or 0.0) / 100, 4),
                'volatility': round((_volatility or 0.0) / 100, 4),
                'vol_ratio':  round(_vol_ratio  or 1.0,  2),
            },
            'stats': {
                'total_stocks':        len(results),
                'trend_count':         len(trend_stocks),
                'bottom_count':        len(bottom_stocks),
                'trend_model_active':  v19_enhanced_engine.get_status()['trend_model'],
                'bottom_model_active': v19_enhanced_engine.get_status()['bottom_model'],
                'ai_model_active':     v19_enhanced_engine.get_status()['ai_engine'],
                'model_weights':       _model_weights,
                'model_ic':            _model_ic,
                'consistency_score':   _consistency,
                'hard_guard_trend':    len(trend_qualified),
                'hard_guard_bottom':   len(bottom_qualified),
                # ── 市场状态标签（保留供后端日志和旧版兼容）──────────────
                'market_regime':       _regime_key,
                'market_regime_cn':    _regime_cn,
                'pred_uncertainty':    float(_health.get('_pred_uncertainty', 0.3)) if _health_exists else 0.3,
                # ── 市场面板原始数字（保留，单位：%/ratio）──────────────
                'market_score':        _market_score or 0.0,
                'confidence':          _market_confidence or 0.0,
                'trend_60d':           _trend_60d,
                'trend_20d':           _trend_20d,
                'volatility':          _volatility,
                'volume_ratio':        _vol_ratio,
                'negative_ratio_mean': _neg_ratio_mean,
            }
        }))

    except Exception as e:
        logger.error(f"选股接口异常: {e}\n{traceback.format_exc()}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@csrf_exempt
def login_tushare(request):
    """登录接口"""
    return JsonResponse({
        'status': 'success',
        'message': 'A股智能选股系统 V19 - 私募级集成版',
        'version': '35.0',
        'features': [
            '✅ 150+ 规则因子（动量/反转/KDJ/RSI/MACD/布林）',
            '✅ 双模型预测（涨势股 + 抄底股，tree_models）',
            '✅ AI模型集成（MLP/Transformer/SmartXGNN）',
            '✅ 全量Tushare接口（资金流/涨停/北向/质押/财务）',
            '✅ 动态权重优化（IC加权 + 市场状态感知）',
            '✅ 风险中性化（行业 + 风格）',
            '✅ 组合优化（风险平价）',
            '✅ 纯规则选股降级',
            '✅ 模型持久化（96%提速）',
        ]
    })


@csrf_exempt
def system_status(request):
    """系统状态接口"""
    status = v19_enhanced_engine.get_status()
    status['tushare_points'] = TUSHARE_POINTS
    status['latest_trading_date'] = get_latest_trading_date()
    # 诊断：显示日历真实范围，方便排查日期冻结问题
    try:
        cal = get_trade_calendar()
        status['calendar_range'] = {
            'start': cal[0] if cal else 'N/A',
            'end': cal[-1] if cal else 'N/A',
            'total_days': len(cal),
        }
    except Exception:
        status['calendar_range'] = {'error': '日历获取失败'}
    return JsonResponse(_make_json_serializable(status))


@csrf_exempt
def save_token(request):
    """保存 Tushare Token 到 .env 文件"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            token = data.get('token', '').strip()
            if not token:
                return JsonResponse({'status': 'error', 'message': 'Token 不能为空'})
            if not (50 <= len(token) <= 70) or not token.isalnum():
                return JsonResponse({'status': 'error', 'message': '请输入有效的 Tushare Token (长度需在 50~70 位之间，当前输入的长度为' + str(len(token)) + '，请确认是否误粘贴了日志)'})
            
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            
            # 每次保存直接覆盖写入，确保只留存最后新的单个 token，无任何多余或旧内容残留
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(f'TUSHARE_TOKEN={token}\n')
            
            # 重新加载环境变量和 ts
            os.environ['TUSHARE_TOKEN'] = token
            ts.set_token(token)
            # config 모듈 변수도 업데이트 (모듈 레벨 변수 동기화)
            import importlib, sys
            if 'stock_app.config_v19' in sys.modules:
                sys.modules['stock_app.config_v19'].TUSHARE_TOKEN = token
            
            return JsonResponse({'status': 'success', 'message': 'Token 保存成功'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
@csrf_exempt
def get_kline_data(request):
    """Tushare K线数据接口 - 支持日/周/月 + 复权"""
    ts_code = request.GET.get('ts_code')
    period = request.GET.get('period', 'daily')   # daily / weekly / monthly
    days = int(request.GET.get('days', 120))
    weeks = int(request.GET.get('weeks', 52))
    months = int(request.GET.get('months', 24))
    adjust = request.GET.get('adjust', 'qfq')     # qfq / none

    if not ts_code:
        return JsonResponse({'status': 'error', 'message': '缺少 ts_code'})

    try:
        import datetime
        token = os.environ.get('TUSHARE_TOKEN', TUSHARE_TOKEN)
        ts.set_token(token)
        pro = ts.pro_api()

        # 计算合理的时间范围，多获取一部分数据以防交易日数量不足
        today = datetime.date.today()
        if period == 'daily':
            lookback = int(days * 1.6)
            start_date = (today - datetime.timedelta(days=lookback)).strftime('%Y%m%d')
        elif period == 'weekly':
            lookback = int(weeks * 8)
            start_date = (today - datetime.timedelta(days=lookback)).strftime('%Y%m%d')
        else:  # monthly
            lookback = int(months * 35)
            start_date = (today - datetime.timedelta(days=lookback)).strftime('%Y%m%d')
        
        end_date = today.strftime('%Y%m%d')

        # 获取行情数据
        if period == 'daily':
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date,
                           fields='trade_date,open,high,low,close,vol,amount')
        elif period == 'weekly':
            df = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date,
                            fields='trade_date,open,high,low,close,vol,amount')
        else:  # monthly
            df = pro.monthly(ts_code=ts_code, start_date=start_date, end_date=end_date,
                             fields='trade_date,open,high,low,close,vol,amount')

        if df is None or df.empty:
            return JsonResponse({'status': 'error', 'message': 'Tushare 返回空行情数据'})

        # 前复权计算逻辑 (自主实现以规避 tushare 内置 pro_bar 在新版 pandas 下 fillna 崩溃的问题)
        if adjust == 'qfq':
            df_factor = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df_factor is not None and not df_factor.empty:
                df['trade_date'] = df['trade_date'].astype(str)
                df_factor['trade_date'] = df_factor['trade_date'].astype(str)
                
                # 排序
                df = df.sort_values('trade_date', ascending=True)
                df_factor = df_factor.sort_values('trade_date', ascending=True)
                
                # 基于 pd.merge_asof 实现非精确时间点的前向对齐
                df['date_dt'] = pd.to_datetime(df['trade_date'])
                df_factor['date_dt'] = pd.to_datetime(df_factor['trade_date'])
                
                merged = pd.merge_asof(df, df_factor[['date_dt', 'adj_factor']], on='date_dt', direction='backward')
                merged['adj_factor'] = merged['adj_factor'].ffill().bfill()
                
                # 基准因子（最近的交易日）
                latest_factor = merged['adj_factor'].iloc[-1] if not merged.empty else 1.0
                if latest_factor == 0:
                    latest_factor = 1.0
                    
                # 计算前复权值
                for col in ['open', 'high', 'low', 'close']:
                    if col in merged.columns:
                        merged[col] = (merged[col] * merged['adj_factor'] / latest_factor).round(2)
                
                df = merged.drop(columns=['date_dt', 'adj_factor'], errors='ignore')

        # 按照 trade_date 升序排序
        df = df.sort_values('trade_date', ascending=True)

        # 截取最后的指定记录条数
        if period == 'daily':
            df = df.tail(days)
        elif period == 'weekly':
            df = df.tail(weeks)
        else:
            df = df.tail(months)

        # 转换为记录字典列表
        data = df.to_dict('records')

        return JsonResponse({
            'status': 'success',
            'ts_code': ts_code,
            'period': period,
            'adjust': adjust,
            'data': data
        })
    except Exception as e:
        logger.error(f"K线接口异常: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})

def index(request):
    """主页接口 - 提供前端HTML界面"""
    return render(request, 'stock_app/index.html')


# ==================== 命令行测试 ====================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    df = get_real_stock_data(lookback_months=6)
    if not df.empty:
        v19_enhanced_engine.train_all_models(df)
        selected = enhance_stock_selection_v19(df, top_n=20)
        print("\n🎯 选股结果（Top 10）:")
        cols = [c for c in ['ts_code', 'name', 'neutral_score', 'dual_score',
                             'trend_score', 'bottom_score', 'ai_score']
                if c in selected.columns]
        print(selected[cols].head(10).to_string(index=False))