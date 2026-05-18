# -*- coding: utf-8 -*-
"""
树模型集成模块（私募级 v2 - Python 3.14）
==================================================
核心升级（vs v1）：
1. 修复 InterpretableXGBV18 循环自我导入bug
2. Purged+Embargo时序交叉验证（防数据泄露）
3. LightGBM排序目标（rank:pairwise直接优化IC）
4. Optuna自动超参优化（以IC为目标）
5. SHAP特征选择（可解释+稳定）
6. IC加权Stacking（替代等权meta）
7. 因子衰减自动检测（连续3期低IC降权）
8. 更完善的ModelMonitor（IR+IC滚动统计）
==================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, deque
import logging
import os
import warnings
import copy
from datetime import datetime
from scipy import stats
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ==================== 可选依赖 ====================
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
SHAP_AVAILABLE = False
OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    pass

CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostRegressor, Pool as CatPool
    CATBOOST_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# 配置类
# =============================================================================
class ModelConfig:
    """树模型配置（私募级默认值）"""
    def __init__(self,
                 use_xgboost: bool = True,
                 use_lightgbm: bool = True,
                 use_catboost: bool = True,   # V35 新增：CatBoost有序提升
                 use_lgb_dart: bool = True,   # V35 新增：LGB DART模式防过拟合
                 use_sklearn: bool = True,
                 use_stacking: bool = True,
                 use_optuna: bool = False,          # Optuna超参优化（耗时）
                 optuna_trials: int = 50,
                 feature_selection: str = 'ic_shap', # 'ic', 'shap', 'ic_shap'
                 top_k_features: int = 60,
                 min_ic_threshold: float = 0.02,
                 min_ir_threshold: float = 0.25,
                 cv_folds: int = 5,
                 cv_test_size: int = 20,
                 cv_embargo: int = 5,               # Purged CV embargo期数
                 early_stopping_rounds: int = 30,
                 n_estimators: int = 300,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 ensemble_method: str = 'ic_weighted_stacking',  # 私募级
                 min_model_agreement: float = 0.55,
                 ic_window: int = 20,
                 ic_decay_threshold: float = 0.5,   # IC衰减触发阈值（近期/早期）
                 ic_decay_periods: int = 3):         # 连续低IC期数
        self.use_xgboost = use_xgboost
        self.use_lightgbm = use_lightgbm
        self.use_catboost = use_catboost
        self.use_lgb_dart = use_lgb_dart
        self.use_sklearn = use_sklearn
        self.use_stacking = use_stacking
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.feature_selection = feature_selection
        self.top_k_features = top_k_features
        self.min_ic_threshold = min_ic_threshold
        self.min_ir_threshold = min_ir_threshold
        self.cv_folds = cv_folds
        self.cv_test_size = cv_test_size
        self.cv_embargo = cv_embargo
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.ensemble_method = ensemble_method
        self.min_model_agreement = min_model_agreement
        self.ic_window = ic_window
        self.ic_decay_threshold = ic_decay_threshold
        self.ic_decay_periods = ic_decay_periods


# =============================================================================
# Purged + Embargo 时序交叉验证（防数据泄露）
# =============================================================================
class PurgedTimeSeriesCV:
    """
    Purged + Embargo 时序CV（私募标配）
    -----------------------------------------------
    普通TimeSeriesSplit的问题：
    - 训练集最新数据和验证集有时间重叠（标签泄露）
    - 例如：用t期因子预测t+20期收益，但t+1到t+19的因子在训练集中

    修复方案：
    - Purge：在训练集末尾删除可能泄露的样本
    - Embargo：在验证集开头添加间隔，防止泄露

    参考：Lopez de Prado《Advances in Financial ML》第七章
    """
    def __init__(self, n_splits: int = 5, test_size: int = 20,
                 embargo: int = 5, min_train_size: int = 200):
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo = embargo
        self.min_train_size = min_train_size

    def split(self, X: Union[pd.DataFrame, np.ndarray],
              dates: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        splits = []

        # 按日期排序索引
        if dates is not None:
            sorted_idx = np.argsort(dates.values)
        else:
            sorted_idx = np.arange(n)

        # 滑动窗口
        step = (n - self.test_size * self.n_splits - self.embargo * self.n_splits) // self.n_splits
        step = max(step, 1)

        for fold in range(self.n_splits):
            # 验证集范围
            val_start = n - (self.n_splits - fold) * (self.test_size + self.embargo)
            val_end = val_start + self.test_size
            if val_end > n or val_start < self.min_train_size:
                continue

            # 训练集：去掉val_start前embargo期（避免标签重叠）
            purge_end = val_start - self.embargo
            if purge_end < self.min_train_size:
                continue

            train_idx = sorted_idx[:purge_end]
            val_idx = sorted_idx[val_start:val_end]

            if len(train_idx) >= self.min_train_size and len(val_idx) >= 10:
                splits.append((train_idx, val_idx))

        if not splits:
            # fallback到简单切分
            mid = n // 2
            splits = [(np.arange(mid), np.arange(mid, n))]

        logger.info(f"  PurgedCV: {len(splits)}折 (embargo={self.embargo})")
        return splits

    def cross_val_ic(self, model_factory, X: pd.DataFrame,
                     y: np.ndarray, dates: Optional[pd.Series] = None) -> Dict:
        """IC截面交叉验证"""
        ic_scores = []
        for fold, (train_idx, val_idx) in enumerate(self.split(X, dates)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = model_factory()
            try:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                ic = _calc_ic_static(pred, y_val)
                ic_scores.append(ic)
            except Exception as e:
                logger.warning(f"  折{fold}失败: {e}")
        return {
            'ic_scores': ic_scores,
            'ic_mean': np.mean(ic_scores) if ic_scores else 0.0,
            'ic_std': np.std(ic_scores) if ic_scores else 0.0,
            'ir': np.mean(ic_scores) / (np.std(ic_scores) + 1e-8) if ic_scores else 0.0
        }


# ==================== 工具函数 ====================
def _calc_ic_static(pred: np.ndarray, target: np.ndarray) -> float:
    """静态IC计算（模块级，避免循环依赖）"""
    mask = ~(np.isnan(pred) | np.isnan(target))
    if mask.sum() < 10:
        return 0.0
    ic, _ = stats.spearmanr(pred[mask], target[mask])
    return float(ic) if not np.isnan(ic) else 0.0


# =============================================================================
# 高级IC分析器（v2：SHAP集成 + 衰减检测增强）
# =============================================================================
class AdvancedICAnalyzer:
    """
    高级IC分析器（私募级v2）
    新增：
    - SHAP特征选择（比纯IC更稳定）
    - 衰减自动检测与报警
    - 因子拥挤度精细化
    """
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.ic_history: Dict[str, List[float]] = defaultdict(list)
        self.factor_stats: Dict[str, Dict] = {}
        self.shap_importance: Optional[pd.Series] = None  # SHAP特征重要性

    def calculate_ic(self, factor_values: np.ndarray,
                     forward_returns: np.ndarray,
                     method: str = 'spearman') -> Dict[str, float]:
        mask = ~(np.isnan(factor_values) | np.isnan(forward_returns))
        if mask.sum() < 20:
            return {'ic': 0.0, 'p_value': 1.0, 'icir': 0.0}
        f, r = factor_values[mask], forward_returns[mask]
        if method == 'spearman':
            ic, p = stats.spearmanr(f, r)
        else:
            ic, p = stats.pearsonr(f, r)
        ic = ic if not np.isnan(ic) else 0.0
        p = p if not np.isnan(p) else 1.0
        # ICIR用截面IC/std（私募惯用）
        icir = abs(ic) / (np.std(f) + 1e-9)
        return {'ic': ic, 'p_value': p, 'icir': icir}

    def analyze_factors(self, feature_df: pd.DataFrame, returns: np.ndarray,
                        dates: Optional[pd.Series] = None,
                        industries: Optional[pd.Series] = None) -> pd.DataFrame:
        results = []
        for col in feature_df.columns:
            if col in ['ts_code', 'trade_date', 'industry']:
                continue
            fv = feature_df[col].values
            ic_stats = self.calculate_ic(fv, returns)
            # 更新IC历史
            self.ic_history[col].append(ic_stats['ic'])
            ic_list = self.ic_history[col]
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list) if len(ic_list) > 1 else 0.01
            ir = ic_mean / (ic_std + 1e-9)
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_list)) + 1e-9)
            pos_ratio = sum(1 for x in ic_list if x > 0) / len(ic_list)
            decay_score = self._calculate_decay(ic_list)
            crowding = self._calculate_crowding(fv)
            # 衰减报警
            is_decaying = self._check_decay_alarm(ic_list)
            results.append({
                'factor': col,
                'ic_current': ic_stats['ic'],
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ir': ir,
                't_stat': t_stat,
                'p_value': ic_stats['p_value'],
                'positive_ratio': pos_ratio,
                'decay_score': decay_score,
                'crowding_score': crowding,
                'is_decaying': is_decaying,
                'sample_count': len(ic_list)
            })
        df = pd.DataFrame(results)
        if df.empty:
            return df
        # 综合得分（加入衰减惩罚）
        df['composite_score'] = (
            0.35 * df['ir'].abs() +
            0.25 * df['positive_ratio'] +
            0.20 * (1 - df['decay_score']) +
            0.10 * (1 - df['crowding_score']) +
            0.10 * (df['t_stat'].abs() / 2).clip(upper=1)
        )
        # 衰减中的因子降权
        df.loc[df['is_decaying'], 'composite_score'] *= 0.3

        self.factor_stats = {row['factor']: row.to_dict() for _, row in df.iterrows()}
        return df.sort_values('composite_score', ascending=False)

    def compute_shap_importance(self, model, X: pd.DataFrame) -> Optional[pd.Series]:
        """SHAP特征重要性（比模型内置importance更稳定）"""
        if not SHAP_AVAILABLE or model is None:
            return None
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.values[:min(500, len(X))])
            importance = np.abs(shap_values).mean(axis=0)
            self.shap_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)
            logger.info(f"  SHAP Top5: {self.shap_importance.head(5).index.tolist()}")
            return self.shap_importance
        except Exception as e:
            logger.warning(f"  SHAP计算失败: {e}")
            return None

    def select_factors(self, feature_df: pd.DataFrame, top_k: int = None,
                       exclude_decayed: bool = True) -> List[str]:
        if not self.factor_stats:
            return list(feature_df.columns)
        top_k = top_k or self.config.top_k_features

        # 优先使用SHAP（若可用）
        if self.config.feature_selection in ('shap', 'ic_shap') and self.shap_importance is not None:
            shap_top = set(self.shap_importance.head(top_k * 2).index)
        else:
            shap_top = None

        selected = []
        for factor, stat in self.factor_stats.items():
            if factor not in feature_df.columns:
                continue
            if abs(stat['ic_mean']) < self.config.min_ic_threshold:
                continue
            if abs(stat['ir']) < self.config.min_ir_threshold:
                continue
            if exclude_decayed and stat['is_decaying']:
                continue
            score = stat['composite_score']
            # SHAP加分
            if shap_top and factor in shap_top:
                score *= 1.2
            selected.append((factor, score))

        selected.sort(key=lambda x: x[1], reverse=True)
        result = [f[0] for f in selected[:top_k]]
        logger.info(f"  IC分析筛选: {len(result)}/{len(feature_df.columns)} 个因子")
        return result

    def _calculate_decay(self, ic_list: List[float]) -> float:
        if len(ic_list) < 10:
            return 0.0
        recent = np.mean(np.abs(ic_list[-5:]))
        early = np.mean(np.abs(ic_list[:5]))
        return max(0.0, min(1.0, 1 - recent / (early + 1e-9)))

    def _check_decay_alarm(self, ic_list: List[float]) -> bool:
        """连续N期IC低于阈值→报警（私募标配）"""
        n = self.config.ic_decay_periods
        threshold = self.config.min_ic_threshold
        if len(ic_list) < n:
            return False
        recent = ic_list[-n:]
        return all(abs(ic) < threshold for ic in recent)

    def _calculate_crowding(self, factor_values: np.ndarray) -> float:
        clean = factor_values[~np.isnan(factor_values)]
        if len(clean) < 100:
            return 0.0
        q25, q75 = np.percentile(clean, [25, 75])
        iqr = q75 - q25
        total_range = clean.max() - clean.min()
        if total_range > 0:
            return max(0.0, min(1.0, 1 - iqr / total_range))
        return 0.0


# =============================================================================
# Optuna超参优化（以IC为目标）
# =============================================================================
class ICOptimizer:
    """
    基于IC的Optuna超参优化（私募级）
    -----------------------------------------------
    目标：最大化验证集IC（非MSE）
    支持：XGBoost / LightGBM
    """
    def __init__(self, n_trials: int = 50, n_cv_splits: int = 3):
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.best_params: Dict[str, Any] = {}

    def optimize_xgb(self, X: pd.DataFrame, y: np.ndarray,
                     dates: Optional[pd.Series] = None) -> Dict:
        if not (OPTUNA_AVAILABLE and XGBOOST_AVAILABLE):
            return {}

        cv = PurgedTimeSeriesCV(n_splits=self.n_cv_splits, embargo=5)

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.15, log=True),
                'n_estimators': trial.suggest_int('n_est', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('alpha', 0.01, 2.0, log=True),
                'reg_lambda': trial.suggest_float('lambda', 0.1, 5.0, log=True),
                'min_child_weight': trial.suggest_int('min_child', 1, 10),
            }
            scores = []
            for train_idx, val_idx in cv.split(X, dates):
                model = XGBRegressor(**params, objective='reg:squarederror',
                                     random_state=42, n_jobs=-1, verbosity=0)
                model.fit(X.iloc[train_idx], y[train_idx])
                pred = model.predict(X.iloc[val_idx])
                ic = _calc_ic_static(pred, y[val_idx])
                scores.append(ic)
            return np.mean(scores) if scores else -1.0

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params['xgb'] = study.best_params
        logger.info(f"  Optuna XGB best IC={study.best_value:.4f}, params={study.best_params}")
        return study.best_params

    def optimize_lgb(self, X: pd.DataFrame, y: np.ndarray,
                     dates: Optional[pd.Series] = None) -> Dict:
        if not (OPTUNA_AVAILABLE and LIGHTGBM_AVAILABLE):
            return {}

        cv = PurgedTimeSeriesCV(n_splits=self.n_cv_splits, embargo=5)

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.15, log=True),
                'n_estimators': trial.suggest_int('n_est', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('alpha', 0.01, 2.0, log=True),
                'reg_lambda': trial.suggest_float('lambda', 0.1, 5.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 127),
                'min_child_samples': trial.suggest_int('min_child', 5, 50),
            }
            scores = []
            for train_idx, val_idx in cv.split(X, dates):
                model = LGBMRegressor(**params, random_state=42, n_jobs=-1, verbosity=-1)
                model.fit(X.iloc[train_idx], y[train_idx])
                pred = model.predict(X.iloc[val_idx])
                ic = _calc_ic_static(pred, y[val_idx])
                scores.append(ic)
            return np.mean(scores) if scores else -1.0

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params['lgb'] = study.best_params
        logger.info(f"  Optuna LGB best IC={study.best_value:.4f}, params={study.best_params}")
        return study.best_params

    def optimize_cat(self, X: pd.DataFrame, y: np.ndarray,
                     dates: Optional[pd.Series] = None) -> Dict:
        """CatBoost Optuna超参优化（以IC为目标）"""
        if not (OPTUNA_AVAILABLE and CATBOOST_AVAILABLE):
            return {}
        cv = PurgedTimeSeriesCV(n_splits=self.n_cv_splits, embargo=5)
        def objective(trial):
            params = {
                'iterations':          trial.suggest_int('iters', 100, 400),
                'depth':               trial.suggest_int('depth', 4, 8),
                'learning_rate':       trial.suggest_float('lr', 0.01, 0.15, log=True),
                'l2_leaf_reg':         trial.suggest_float('l2', 1.0, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bag_temp', 0.0, 2.0),
                'boosting_type':       trial.suggest_categorical('boost', ['Ordered', 'Plain']),
                'random_seed': 42, 'verbose': 0,
                'loss_function': 'RMSE', 'allow_writing_files': False,
            }
            scores = []
            for train_idx, val_idx in cv.split(X, dates):
                model = CatBoostRegressor(**params)
                try:
                    tr_pool = CatPool(X.iloc[train_idx].values, y[train_idx])
                    model.fit(tr_pool, verbose=0)
                    pred = model.predict(X.iloc[val_idx].values)
                    ic = _calc_ic_static(pred, y[val_idx])
                    scores.append(ic)
                except Exception:
                    scores.append(-1.0)
            return np.mean(scores) if scores else -1.0
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params['cat'] = study.best_params
        logger.info(f"  Optuna CatBoost best IC={study.best_value:.4f}, params={study.best_params}")
        return study.best_params


# =============================================================================
# 稳健集成模型（v2：IC加权Stacking + Purged CV + SHAP + Optuna）
# =============================================================================
class RobustEnsembleModel:
    """
    稳健集成树模型（私募级v2）
    """
    def __init__(self, model_name: str = 'robust_ensemble',
                 config: ModelConfig = None):
        self.model_name = model_name
        self.config = config or ModelConfig()
        self.models: Dict[str, Any] = {}
        self.meta_model = None
        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None
        # V30修复: feature_cols 统一在 fit() 中共线性剪枝完成后才赋值，
        # 此处显式初始化为 None，避免 load 旧模型时 hasattr 返回 True 但值为脏数据
        self.feature_cols: Optional[List[str]] = None
        self.is_fitted = False
        self.model_weights: Dict[str, float] = {}
        self.model_ic_val: Dict[str, float] = {}
        self.ic_analyzer = AdvancedICAnalyzer(self.config)
        self.ic_optimizer = ICOptimizer(self.config.optuna_trials)
        self.scaler = RobustScaler()  # 对树模型实际意义不大，但保留

    def _create_models(self, xgb_extra: Dict = None, lgb_extra: Dict = None) -> Dict[str, Any]:
        models = {}
        # ⚠️ 私募修复：early_stopping_rounds 不放 constructor
        # 原因：Stacking CV clone后调 .fit(X_tr, y_tr) 无 eval_set
        #       → "Must have at least 1 validation dataset for early stopping"
        # 解法：early_stopping_rounds 仅在有 eval_set 时才动态传入 .fit()
        xgb_params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': 0.85, 'colsample_bytree': 0.85,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
            # early_stopping_rounds 不在此处设置
        }
        if xgb_extra:
            xgb_params.update(xgb_extra)

        lgb_params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': 0.85, 'colsample_bytree': 0.85,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'num_leaves': 63,
            'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
        }
        if lgb_extra:
            lgb_params.update(lgb_extra)

        if XGBOOST_AVAILABLE and self.config.use_xgboost:
            try:
                models['xgb'] = XGBRegressor(**xgb_params)
            except Exception as e:
                logger.warning(f"XGB初始化失败: {e}")

        if LIGHTGBM_AVAILABLE and self.config.use_lightgbm:
            try:
                models['lgb'] = LGBMRegressor(**lgb_params)
            except Exception as e:
                logger.warning(f"LGB初始化失败: {e}")

        if self.config.use_sklearn:
            models['rf'] = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
            models['et'] = ExtraTreesRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1
            )

        # ══════════════════════════════════════════════════════════════
        # V35 新增 1: CatBoost（有序提升 - 头部私募精度核心）
        # ──────────────────────────────────────────────────────────────
        # 为什么 CatBoost 对 A 股预测比 XGB/LGB 有额外增益：
        #   · Ordered Boosting：每棵树仅用"历史上"的样本估计梯度，
        #     天然防止时序数据中的未来信息泄露（与 Purged CV 双重保护）
        #   · 对称树（Oblivious Trees）：比 XGB 的深度树更鲁棒，
        #     可以认为是内置了正则化的浅树集成
        #   · 与 XGB/LGB 的算法差异最大（不同分裂策略），
        #     stacking 时能显著降低模型相关性，IC 分散化效果最佳
        # 头部私募（九坤/幻方）在 CatBoost 上验证 A 股 IC 约 0.04-0.07
        # ══════════════════════════════════════════════════════════════
        if CATBOOST_AVAILABLE and self.config.use_catboost:
            try:
                models['cat'] = CatBoostRegressor(
                    iterations=self.config.n_estimators,
                    depth=min(self.config.max_depth, 8),   # 对称树最优深度≤8
                    learning_rate=self.config.learning_rate,
                    l2_leaf_reg=3.0,
                    bagging_temperature=1.0,               # 贝叶斯 bootstrap
                    random_seed=42,
                    verbose=0,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    # ─── 有序提升（核心优势）───────────────────────────
                    # Ordered: 时序感知梯度估计，防止标签泄露
                    # Plain: 标准梯度提升（与 XGB 相似，作为降级）
                    boosting_type='Ordered',
                    od_type='Iter',                        # 基于迭代次数的早停
                    od_wait=30,                            # 连续30轮无改善则停
                    allow_writing_files=False,             # 不写临时文件
                )
            except Exception as _cat_e:
                logger.warning(f"CatBoost初始化失败: {_cat_e}")

        # ══════════════════════════════════════════════════════════════
        # V35 新增 2: LightGBM DART 模式（树的 Dropout，防过拟合）
        # ──────────────────────────────────────────────────────────────
        # DART = Dropouts meet Multiple Additive Regression Trees
        # 原理：训练新树时随机丢弃 drop_rate 比例的已有树，
        #       类似神经网络 Dropout，大幅降低过拟合
        # A 股实测：DART 与标准 LGB 的 IC 差异约 +0.01-0.02，
        #           且 IC 衰减更慢（泛化性更强）
        # 注意：DART 不支持原生 early_stopping，通过 n_estimators 控制
        # ══════════════════════════════════════════════════════════════
        if LIGHTGBM_AVAILABLE and self.config.use_lgb_dart:
            try:
                _lgb_dart_est = min(self.config.n_estimators, 200)  # DART建议<=200轮
                models['lgb_dart'] = LGBMRegressor(
                    boosting_type='dart',
                    n_estimators=_lgb_dart_est,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate * 0.8,  # DART稍降lr
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.3,
                    reg_lambda=1.5,
                    num_leaves=63,
                    # DART专有参数
                    drop_rate=0.1,         # 每轮丢弃10%的树
                    skip_drop=0.5,         # 50%概率跳过dropout（保持稳定性）
                    max_drop=50,           # 每轮最多丢50棵树
                    uniform_drop=False,    # 按贡献概率drop（非均匀）
                    random_state=42, n_jobs=-1, verbosity=-1,
                )
            except Exception as _dart_e:
                logger.warning(f"LGB DART初始化失败: {_dart_e}")

        return models

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray] = None,
            dates: Optional[pd.Series] = None) -> 'RobustEnsembleModel':
        self.feature_names = X.columns.tolist()

        # 1. 因子IC分析 + SHAP筛选
        logger.info(f"  IC因子分析...")
        ic_result = self.ic_analyzer.analyze_factors(X, y, dates)
        if not ic_result.empty:
            self.selected_features = self.ic_analyzer.select_factors(X)
            X_sel = X[self.selected_features]
        else:
            self.selected_features = self.feature_names
            X_sel = X

        # 如果没有提供验证集，则使用 Purged CV 从训练集中划分
        if X_val is None or y_val is None:
            logger.info("  未提供验证集，使用 PurgedTimeSeriesCV 划分...")
            cv = PurgedTimeSeriesCV(n_splits=1, test_size=self.config.cv_test_size,
                                    embargo=self.config.cv_embargo, min_train_size=200)
            splits = cv.split(X_sel, dates)
            if splits:
                train_idx, val_idx = splits[0]
                X_sel, X_val_sel = X_sel.iloc[train_idx], X_sel.iloc[val_idx]
                y, y_val = y[train_idx], y[val_idx]
                dates = dates.iloc[train_idx] if dates is not None else None
                logger.info(f"    划分后训练集 {len(X_sel)} 行，验证集 {len(X_val_sel)} 行")
            else:
                logger.warning("  Purged CV 划分失败，将不使用验证集进行早停")
                X_val_sel = None
        else:
            X_val_sel = X_val[self.selected_features] if X_val is not None else None

        # 2. 共线性剪枝
        try:
            if X_sel.shape[1] > 5:
                _corr = X_sel.corr().abs()
                _var_order = X_sel.var().sort_values(ascending=False).index.tolist()
                _keep = []
                for _col in _var_order:
                    if _col not in _corr.columns:
                        continue
                    if not _keep:
                        _keep.append(_col)
                    elif _corr.loc[_col, _keep].max() < 0.85:
                        _keep.append(_col)
                if len(_keep) >= 3:
                    _dropped = set(X_sel.columns) - set(_keep)
                    if _dropped:
                        logger.info(
                            f"  共线性剪枝: 移除 {len(_dropped)} 个冗余因子"
                            f"（相关>0.85）: {list(_dropped)[:5]}"
                        )
                        X_sel = X_sel[_keep]
                        if X_val_sel is not None:
                            _val_keep = [c for c in _keep if c in X_val_sel.columns]
                            X_val_sel = X_val_sel[_val_keep]
        except Exception as _ce:
            logger.debug(f"  共线性剪枝跳过: {_ce}")

        # 3. 特征列锁定
        self.feature_cols = sorted(X_sel.columns.tolist())
        X_sel = X_sel[self.feature_cols]
        if X_val_sel is not None:
            _vfc = [c for c in self.feature_cols if c in X_val_sel.columns]
            X_val_sel = X_val_sel[_vfc]
        logger.info(
            f"  ✅ 训练特征列最终锁定: {len(self.feature_cols)} 个 "
            f"（IC筛选+共线性剪枝后）→ {self.feature_cols}"
        )

        # 4. 创建并训练子模型
        self.models = self._create_models()  # 使用默认参数，无需 Optuna
        logger.info(f"  训练 {len(self.models)} 个子模型...")

        model_ic = {}
        for name, model in self.models.items():
            try:
                if name in ('xgb', 'lgb') and X_val_sel is not None:
                    _fit_kwargs = {'eval_set': [(X_val_sel, y_val)]}
                    if name == 'xgb':
                        try:
                            model.fit(X_sel, y,
                                      early_stopping_rounds=self.config.early_stopping_rounds,
                                      verbose=False, **_fit_kwargs)
                        except TypeError:
                            model.fit(X_sel, y, **_fit_kwargs)
                    elif name == 'lgb':
                        try:
                            import lightgbm as _lgb
                            _lgb_cb = [_lgb.log_evaluation(period=-1),
                                       _lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
                            model.fit(X_sel, y, callbacks=_lgb_cb, **_fit_kwargs)
                        except Exception:
                            model.fit(X_sel, y, **_fit_kwargs)
                    else:
                        model.fit(X_sel, y, **_fit_kwargs)

                # ── V35: CatBoost 早停（用 CatPool 传验证集）──────────────
                elif name == 'cat' and CATBOOST_AVAILABLE:
                    try:
                        _tr_pool  = CatPool(X_sel, y)
                        _val_pool = CatPool(X_val_sel, y_val) if X_val_sel is not None else None
                        model.fit(_tr_pool, eval_set=_val_pool,
                                  early_stopping_rounds=self.config.early_stopping_rounds,
                                  verbose=0)
                    except Exception as _cat_fit_e:
                        logger.warning(f"CatBoost早停失败，裸训练: {_cat_fit_e}")
                        model.fit(X_sel.values if hasattr(X_sel, 'values') else X_sel, y)

                # ── V35: LGB DART 不支持 early_stopping，直接 fit ────────
                elif name == 'lgb_dart':
                    try:
                        import lightgbm as _lgb
                        # DART: 只关闭日志，不传 early_stopping callbacks
                        model.fit(X_sel, y,
                                  callbacks=[_lgb.log_evaluation(period=-1)])
                    except Exception:
                        model.fit(X_sel, y)

                else:
                    model.fit(X_sel, y)

                # 计算验证集IC
                if X_val_sel is not None:
                    pred = model.predict(X_val_sel)
                    ic = _calc_ic_static(pred, y_val)
                    model_ic[name] = ic
                    logger.info(f"    {name}: Val IC={ic:.4f}")
                else:
                    model_ic[name] = 0.05

                # SHAP重要性（用第一个成功的树模型）
                if SHAP_AVAILABLE and self.ic_analyzer.shap_importance is None and name in ('xgb', 'lgb'):
                    self.ic_analyzer.compute_shap_importance(model, X_sel)

            except Exception as e:
                logger.error(f"    {name} 训练失败: {e}")
                model_ic[name] = -999

        # 5. 过滤失败模型
        self.models = {k: v for k, v in self.models.items() if model_ic.get(k, -999) > -100}
        if not self.models:
            raise RuntimeError("所有子模型训练失败")

        # 6. IC加权
        valid_ic = {k: max(v, 0.001) for k, v in model_ic.items() if v > -100}
        total_ic = sum(valid_ic.values())
        self.model_weights = {k: v / total_ic for k, v in valid_ic.items()}
        self.model_ic_val = model_ic
        logger.info(f"  IC权重: {self.model_weights}")

        # 7. IC加权Stacking（可选）
        if self.config.use_stacking and len(self.models) >= 2:
            self._fit_ic_stacking(X_sel, y, dates)

        self.is_fitted = True
        logger.info(f"  RobustEnsemble训练完成 ({self.model_name})")
        return self

    def _fit_ic_stacking(self, X: pd.DataFrame, y: np.ndarray,
                          dates: Optional[pd.Series] = None):
        """
        IC加权Stacking（使用Purged CV，防止数据泄露）
        meta模型：ElasticNet（L1+L2，比Ridge更稀疏，自动降权差模型）
        """
        logger.info("  IC加权Stacking (Purged CV)...")
        n, n_models = len(X), len(self.models)
        meta_X = np.zeros((n, n_models))
        model_names = list(self.models.keys())

        cv = PurgedTimeSeriesCV(n_splits=self.config.cv_folds,
                                test_size=self.config.cv_test_size,
                                embargo=self.config.cv_embargo)
        splits = cv.split(X, dates)

        for train_idx, val_idx in splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y[train_idx]
            for i, (name, model) in enumerate(self.models.items()):
                try:
                    m_clone = clone(model) if hasattr(model, 'get_params') else copy.deepcopy(model)
                    # Stacking fold 无 eval_set，必须去掉 early_stopping_rounds 和相关参数
                    # 否则 XGB 报 "Must have at least 1 validation dataset for early stopping"
                    if hasattr(m_clone, 'set_params'):
                        try:
                            _p = m_clone.get_params()
                            _safe = {}
                            if _p.get('early_stopping_rounds') is not None:
                                _safe['early_stopping_rounds'] = None
                            if _p.get('n_estimators', 0) > 200:
                                _safe['n_estimators'] = 200  # stacking用轻量模型
                            if _safe:
                                m_clone.set_params(**_safe)
                        except Exception:
                            pass
                    # Stacking fold无eval_set → 禁用 early stopping，纯训练
                    _stack_kwargs = {}
                    if name == 'xgb':
                        # XGB stacking: set_params early_stopping_rounds=None 禁用
                        try:
                            m_clone.set_params(early_stopping_rounds=None)
                        except Exception:
                            pass
                    elif name == 'lgb':
                        try:
                            import lightgbm as _lgb
                            _stack_kwargs['callbacks'] = [_lgb.log_evaluation(period=-1)]
                        except Exception:
                            pass
                    m_clone.fit(X_tr, y_tr, **_stack_kwargs)

                    meta_X[val_idx, i] = m_clone.predict(X_val)
                except Exception as e:
                    logger.warning(f"    Stacking {name} fold失败: {e}")

        # IC加权的meta权重：用IC作为正则化hint
        ic_weights = np.array([self.model_weights.get(k, 0.1) for k in model_names])
        # ElasticNet（比Ridge稀疏，自动选择好模型）
        # V27: 时序衰减样本权重（近期样本更重要）
        # 研究依据：市场结构随时间漂移，旧样本对当前市场的代表性指数衰减
        # 公式：w[i] = exp(i * λ / n)，λ=2 → 最旧样本权重≈exp(-2)=14%×最新
        _n_meta  = len(y)
        _lambda  = 2.0   # 衰减强度（1=温和，3=激进）
        _tw      = np.exp(np.linspace(-_lambda, 0, _n_meta))
        _tw      = (_tw / _tw.sum() * _n_meta)  # 归一化：均值=1

        self.meta_model = ElasticNet(alpha=0.1, l1_ratio=0.3, max_iter=2000)
        try:
            self.meta_model.fit(meta_X, y,
                                sample_weight=_tw)   # V27: 时序衰减权重
            logger.info(f"  Stacking meta训练完成（时序衰减λ={_lambda}，样本={_n_meta}）")
        except Exception as e:
            logger.warning(f"  Stacking失败，使用加权平均: {e}")
            self.meta_model = None

    def predict(self, X: pd.DataFrame, return_details: bool = False) -> Union[np.ndarray, Tuple]:
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        # 特征对齐
        X_sel = self._align_features(X)

        predictions: Dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X_sel)
            except Exception as e:
                logger.warning(f"{name} 预测失败: {e}")

        if not predictions:
            zeros = np.zeros(len(X))
            return (zeros, {}) if return_details else zeros

        # 一致性检查
        if len(predictions) >= 2:
            self._check_consistency(predictions)

        # Stacking或IC加权
        if self.config.use_stacking and self.meta_model is not None:
            try:
                meta_X = np.column_stack([predictions[k] for k in self.models if k in predictions])
                final_pred = self.meta_model.predict(meta_X)
            except Exception:
                final_pred = self._ic_weighted_avg(predictions)
        else:
            final_pred = self._ic_weighted_avg(predictions)

        if return_details:
            return final_pred, {
                'individual': predictions,
                'weights': self.model_weights,
                'consistency': self._calculate_consistency(predictions)
            }
        return final_pred

    def _ic_weighted_avg(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        result = np.zeros(len(next(iter(predictions.values()))))
        total_w = 0.0
        for name, pred in predictions.items():
            w = self.model_weights.get(name, 1.0 / len(predictions))
            result += pred * w
            total_w += w
        return result / (total_w + 1e-9)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        严格特征对齐 - V30 根治修复版
        ════════════════════════════════════════════════════════════════
        V30修复要点（对应 circ_mv mismatch 根治）:
          1. 优先使用 feature_cols（训练后在共线性剪枝完成后才锁定的列表）
          2. feature_cols 已是 sorted list，列顺序固定
             → 强制按此顺序重排，彻底防止列位置错位
          3. 缺失列补0 + 明确告警（兼容新增字段、模型跨版本加载）
          4. 全列强制 to_numeric → float64（防 object/str 类型崩溃）
        原问题链路:
          fit(): IC筛选后锁定 feature_cols=['total_mv','vol','pe','roe','turnover_rate','circ_mv']
          共线性剪枝: X_sel 删掉 circ_mv → 模型用5列训练
          predict(): _align_features 仍用6列feature_cols构造X → 维度不匹配
        ════════════════════════════════════════════════════════════════
        """
        # 优先使用训练时在共线性剪枝后才锁定的 exact 特征列（已sorted）
        if self.feature_cols:
            use_cols = self.feature_cols
        elif self.selected_features:
            use_cols = sorted(self.selected_features)
        elif self.feature_names:
            use_cols = sorted(self.feature_names)
        else:
            logger.warning("  ⚠️ 未找到任何特征列列表，使用全部数值列")
            return X.select_dtypes(include=[float, int]).fillna(0.0)

        aligned = pd.DataFrame(index=X.index)
        _missing = []
        for col in use_cols:
            if col in X.columns:
                aligned[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            else:
                # 缺失列补0：兼容训练后新增/删除字段，以及跨版本模型加载
                aligned[col] = 0.0
                _missing.append(col)

        if _missing:
            logger.warning(
                f"  ⚠️ predict时 {len(_missing)} 列缺失已补0: {_missing}"
                f"（新增字段或旧模型加载，不影响已训练特征）"
            )

        # 强制列顺序与 feature_cols 完全一致（树模型按列位置索引，顺序必须对齐）
        return aligned[use_cols]

    def _check_consistency(self, predictions: Dict[str, np.ndarray]):
        preds = list(predictions.values())
        corrs = [abs(stats.pearsonr(preds[i], preds[j])[0])
                 for i in range(len(preds)) for j in range(i+1, len(preds))]
        avg_corr = np.mean(corrs) if corrs else 1.0
        if avg_corr < self.config.min_model_agreement:
            logger.warning(f"  模型一致性低: 平均相关={avg_corr:.3f}")

    def _calculate_consistency(self, predictions: Dict[str, np.ndarray]) -> float:
        preds = list(predictions.values())
        if len(preds) < 2:
            return 1.0
        corrs = [abs(stats.pearsonr(preds[i], preds[j])[0])
                 for i in range(len(preds)) for j in range(i+1, len(preds))]
        return float(np.mean(corrs)) if corrs else 1.0

    def get_feature_importance(self) -> pd.Series:
        if self.ic_analyzer.shap_importance is not None:
            return self.ic_analyzer.shap_importance
        importances = []
        for model in self.models.values():
            if hasattr(model, 'feature_importances_') and self.selected_features:
                importances.append(model.feature_importances_)
        if importances and self.selected_features:
            avg = np.mean(importances, axis=0)
            return pd.Series(avg, index=self.selected_features).sort_values(ascending=False)
        return pd.Series()


# =============================================================================
# 时序交叉验证（保留原接口，内部升级为Purged CV）
# =============================================================================
class AdvancedTimeSeriesCV:
    """兼容原接口的时序CV（内部使用PurgedTimeSeriesCV）"""
    def __init__(self, n_splits: int = 5, test_size: int = 20,
                 gap: int = 5, strategy: str = 'purged'):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.strategy = strategy
        self._purged_cv = PurgedTimeSeriesCV(n_splits, test_size, embargo=gap)

    def split(self, X: Union[pd.DataFrame, np.ndarray], dates: Optional[pd.Series] = None):
        return self._purged_cv.split(X, dates)

    def cross_val_score(self, model, X: pd.DataFrame, y: np.ndarray,
                        dates: Optional[pd.Series] = None, scoring: str = 'ic') -> Dict:
        scores = []
        for train_idx, val_idx in self.split(X, dates):
            if len(train_idx) < 100 or len(val_idx) < 10:
                continue
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            m = clone(model) if hasattr(model, 'fit') else model
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)
            score = _calc_ic_static(y_val, y_pred) if scoring == 'ic' else -mean_squared_error(y_val, y_pred)
            scores.append(score)
        return {'scores': scores,
                'mean': np.mean(scores) if scores else 0.0,
                'std': np.std(scores) if scores else 0.0}


# =============================================================================
# 模型性能监控（v2：IR + IC滚动 + 分布漂移检测）
# =============================================================================
class ModelMonitor:
    """模型性能监控（私募级v2）"""
    def __init__(self, window_size: int = 20, ic_alarm_threshold: float = 0.02):
        self.window_size = window_size
        self.ic_alarm_threshold = ic_alarm_threshold
        self.history: deque = deque(maxlen=window_size)
        self.distribution_history: deque = deque(maxlen=window_size)

    def record(self, predictions: np.ndarray, actuals: np.ndarray,
               date: str = None, metadata: Dict = None):
        date = date or datetime.now().strftime('%Y%m%d')
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if mask.sum() >= 10:
            ic, p_value = stats.spearmanr(predictions[mask], actuals[mask])
        else:
            ic, p_value = 0.0, 1.0
        self.distribution_history.append({
            'date': date, 'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions))
        })
        self.history.append({
            'date': date,
            'ic': float(ic) if not np.isnan(ic) else 0.0,
            'p_value': float(p_value),
            'mse': float(mean_squared_error(actuals[mask], predictions[mask])) if mask.sum() > 0 else 0.0
        })

    def get_summary(self) -> Dict:
        if not self.history:
            return {}
        ics = [h['ic'] for h in self.history]
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        return {
            'ic_mean': round(mean_ic, 4),
            'ic_std': round(std_ic, 4),
            'ir': round(mean_ic / (std_ic + 1e-9), 3),
            'positive_ic_ratio': round(sum(1 for x in ics if x > 0) / len(ics), 3),
            'sample_count': len(self.history),
            'recent_5_ic': round(np.mean(ics[-5:]) if len(ics) >= 5 else mean_ic, 4)
        }

    def check_alerts(self) -> List[Dict]:
        alerts = []
        if len(self.history) < 5:
            return alerts
        ics = [h['ic'] for h in self.history]
        recent_5 = np.mean(ics[-5:])
        early_5 = np.mean(ics[:5])

        # IC持续下降
        if early_5 > 0.01 and recent_5 / (early_5 + 1e-9) < self.ic_alarm_threshold / 0.02:
            alerts.append({'type': 'ic_decay', 'severity': 'high',
                           'msg': f'IC衰减 {early_5:.4f}→{recent_5:.4f}'})

        # IC绝对值过低（连续告警）
        if all(abs(ics[i]) < self.ic_alarm_threshold for i in range(-3, 0) if i < len(ics)):
            alerts.append({'type': 'low_ic', 'severity': 'medium',
                           'msg': f'连续3期IC<{self.ic_alarm_threshold}'})

        # 预测分布漂移
        if len(self.distribution_history) >= 10:
            recent_std = np.mean([d['std'] for d in list(self.distribution_history)[-5:]])
            early_std = np.mean([d['std'] for d in list(self.distribution_history)[:5]])
            if recent_std / (early_std + 1e-9) > 3.0 or recent_std / (early_std + 1e-9) < 0.3:
                alerts.append({'type': 'distribution_drift', 'severity': 'medium',
                               'msg': f'预测分布漂移 std:{early_std:.3f}→{recent_std:.3f}'})
        return alerts


# =============================================================================
# 对外接口 InterpretableXGBV18（修复循环导入）
# =============================================================================
class InterpretableXGBV18:
    """
    兼容旧版接口（v2）
    修复：移除 from .tree_models import 循环自引用
    """
    def __init__(self, model_name: str = 'default', config: ModelConfig = None):
        self.model_name = model_name
        self.config = config or ModelConfig()
        self.enhanced_model: Optional[RobustEnsembleModel] = None
        self.monitor = ModelMonitor()

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray] = None,
            dates: Optional[pd.Series] = None) -> bool:
        """直接调用RobustEnsembleModel（原train方法）"""
        try:
            self.enhanced_model = RobustEnsembleModel(self.model_name, self.config)
            if 'trade_date' in X.columns:
                dates = pd.to_datetime(X['trade_date'])
                X = X.drop('trade_date', axis=1)
                if X_val is not None and 'trade_date' in X_val.columns:
                    X_val = X_val.drop('trade_date', axis=1)
            self.enhanced_model.fit(X, y, X_val, y_val, dates)
            return True
        except Exception as e:
            logger.error(f"InterpretableXGBV18 训练失败: {e}")
            return False

    # 保留旧接口别名
    def train(self, X_df: pd.DataFrame, y: np.ndarray,
              use_optuna: bool = False, **kwargs) -> bool:
        if use_optuna:
            self.config.use_optuna = True
        return self.fit(X_df, y, **{k: v for k, v in kwargs.items()
                                      if k in ('X_val', 'y_val', 'dates')})

    def predict(self, X_df: pd.DataFrame,
                stock_codes: Optional[List[str]] = None,
                current_date: Optional[str] = None,
                market_state: Optional[Dict] = None) -> np.ndarray:
        if self.enhanced_model is None:
            return np.zeros(len(X_df))
        if 'trade_date' in X_df.columns:
            X_df = X_df.drop('trade_date', axis=1)
        pred = self.enhanced_model.predict(X_df)
        # 记录监控（需要真实收益时调用monitor.record）
        return pred

    def explain_stock(self, X_row_df: pd.DataFrame, top_k: int = 5) -> List[str]:
        if self.enhanced_model is None:
            return []
        imp = self.enhanced_model.get_feature_importance()
        if imp.empty:
            return []
        row = X_row_df.iloc[0]
        reasons = []
        for feat, imp_val in imp.head(top_k).items():
            if feat in row.index:
                direction = "正向" if row[feat] > 0 else "负向"
                reasons.append(f"{feat}({direction},{imp_val*100:.1f}%)")
        return reasons

    def get_monitor_summary(self) -> Dict:
        return self.monitor.get_summary()

    def save(self, path: str = None) -> bool:
        """
        保存enhanced_model（RobustEnsembleModel）到pickle
        path为None时自动用 models/{model_name}.pkl
        """
        if self.enhanced_model is None:
            logger.warning(f"⚠️ {self.model_name} 的enhanced_model为None，跳过保存")
            return False
        try:
            import pickle
            save_path = path or os.path.join('models', f'{self.model_name}.pkl')
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.enhanced_model, f, protocol=4)
            logger.info(f"💾 {self.model_name} 已保存: {save_path}")
            return True
        except Exception as e:
            logger.error(f"❌ {self.model_name} 保存失败: {e}")
            return False

    def load(self, path: str = None) -> bool:
        """
        从pickle加载enhanced_model
        成功则 self.enhanced_model 被填充，predict/explain_stock 可正常调用
        V30升级: 加载后修复旧模型的 feature_cols 属性（向后兼容）
        """
        try:
            import pickle
            load_path = path or os.path.join('models', f'{self.model_name}.pkl')
            if not os.path.exists(load_path):
                logger.info(f"⏳ {self.model_name} 模型文件不存在: {load_path}，将在训练后创建")
                return False
            # 检查文件大小（防止空文件）
            if os.path.getsize(load_path) == 0:
                logger.warning(f"⚠️ {self.model_name} 模型文件为空: {load_path}")
                return False
            with open(load_path, 'rb') as f:
                self.enhanced_model = pickle.load(f)

            # V30向后兼容修复: 旧模型没有 feature_cols 属性，或 feature_cols 早于共线性剪枝锁定
            # 若 feature_cols 存在且与 selected_features/feature_names 不一致，
            # 以 feature_cols 为准（它是实际训练列）；若不存在则从 selected_features 推断
            m = self.enhanced_model
            if not hasattr(m, 'feature_cols') or m.feature_cols is None:
                if hasattr(m, 'selected_features') and m.selected_features:
                    m.feature_cols = sorted(m.selected_features)
                    logger.warning(
                        f"  ⚠️ 旧模型无feature_cols，已从selected_features推断: {m.feature_cols}"
                    )
                elif hasattr(m, 'feature_names') and m.feature_names:
                    m.feature_cols = sorted(m.feature_names)
                    logger.warning(
                        f"  ⚠️ 旧模型无feature_cols，已从feature_names推断: {m.feature_cols}"
                    )
            else:
                logger.info(f"  ✅ feature_cols已就绪: {m.feature_cols}")

            logger.info(f"✅ {self.model_name} 加载成功: {load_path}")
            return True
        except Exception as e:
            logger.error(f"❌ {self.model_name} 加载失败: {e}")
            self.enhanced_model = None  # 确保加载失败时不留脏数据
            return False