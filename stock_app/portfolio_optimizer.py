# -*- coding: utf-8 -*-
"""
组合优化模块（私募级）
功能：
- 等权重
- 均值-方差优化
- 风险平价
- 带换手率约束的优化
===============================================
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("⚠️ CVXPY未安装，组合优化功能受限")


class PortfolioOptimizer:
    """组合优化器"""

    def __init__(self, method='risk_parity', max_turnover=2.0, max_weight=0.1, risk_aversion=1.0):
        """
        初始化
        参数:
            method: 优化方法 ('equal_weight', 'mean_variance', 'risk_parity')
            max_turnover: 最大年化换手率（倍）
            max_weight: 单股最大权重
            risk_aversion: 风险厌恶系数（均值-方差用）
        """
        self.method = method
        self.max_turnover = max_turnover
        self.max_weight = max_weight
        self.risk_aversion = risk_aversion
        logger.info(f"📊 组合优化器初始化 (方法={method}, 最大换手率={max_turnover}倍)")

    def equal_weight(self, n):
        """等权重"""
        return np.ones(n) / n

    def mean_variance_optimization(self, expected_returns, cov_matrix):
        """均值-方差优化"""
        if not CVXPY_AVAILABLE:
            return self.equal_weight(len(expected_returns))

        n = len(expected_returns)
        w = cp.Variable(n)
        ret = expected_returns @ w
        risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(ret - self.risk_aversion / 2 * risk)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight,
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status == 'optimal':
                logger.info(f"   ✅ 均值-方差优化成功，预期收益={ret.value:.2%}, 风险={np.sqrt(risk.value):.2%}")
                return w.value
            else:
                logger.warning(f"   ⚠️ 优化失败 (status={prob.status})，使用等权")
                return self.equal_weight(n)
        except Exception as e:
            logger.error(f"   ❌ 优化异常: {e}")
            return self.equal_weight(n)

    def risk_parity(self, cov_matrix, max_iter=100, tol=1e-6):
        """风险平价（迭代算法）"""
        n = cov_matrix.shape[0]
        w = np.ones(n) / n
        for i in range(max_iter):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_risk = cov_matrix @ w / port_vol
            risk_contrib = w * marginal_risk
            target_risk = port_vol / n
            w_new = w * (target_risk / risk_contrib)
            w_new = w_new / w_new.sum()
            if np.max(np.abs(w_new - w)) < tol:
                logger.info(f"   ✅ 风险平价收敛 (迭代{i+1})")
                return w_new
            w = w_new
        logger.warning(f"   ⚠️ 风险平价未收敛，返回最后迭代结果")
        return w

    def optimize_with_turnover(self, expected_returns, cov_matrix, prev_weights):
        """带换手率约束的均值-方差优化"""
        if not CVXPY_AVAILABLE or prev_weights is None:
            return self.mean_variance_optimization(expected_returns, cov_matrix)

        n = len(expected_returns)
        w = cp.Variable(n)
        ret = expected_returns @ w
        risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(ret - self.risk_aversion / 2 * risk)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight,
            cp.norm(w - prev_weights, 1) <= self.max_turnover / 252  # 日换手率
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status == 'optimal':
                turnover = np.sum(np.abs(w.value - prev_weights))
                logger.info(f"   ✅ 带约束优化成功，换手率={turnover:.2%}")
                return w.value
            else:
                logger.warning(f"   ⚠️ 带约束优化失败，使用无约束版本")
                return self.mean_variance_optimization(expected_returns, cov_matrix)
        except Exception as e:
            logger.error(f"   ❌ 优化异常: {e}")
            return self.mean_variance_optimization(expected_returns, cov_matrix)

    def optimize(self, df, score_col='neutral_score', prev_weights=None):
        """
        主优化函数 - Tushare列名兼容版（ts_code / trade_date）
        参数:
            df: DataFrame，必须包含 score_col
            prev_weights: 前期权重数组，与 df 顺序一致
        返回:
            DataFrame: 添加 'optimal_weight' 列
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"📊 组合优化 (方法={self.method})")
        logger.info("=" * 60)

        df = df.copy()
        n = len(df)

        # 保底：score_col不存在时用等权
        if score_col not in df.columns or n == 0:
            logger.warning(f"⚠️ {score_col}不存在或数据为空，使用等权重")
            df['optimal_weight'] = 1.0 / n if n > 0 else 0
            return df

        # 预期收益（基于得分标准化）
        scores = pd.to_numeric(df[score_col], errors='coerce').fillna(0).values
        std_s = scores.std()
        if std_s > 1e-9:
            expected_returns = (scores - scores.mean()) / std_s
        else:
            expected_returns = np.zeros(n)

        # 协方差矩阵（优先用 trade_date+ts_code，fallback到单位阵）
        cov_matrix = np.eye(n) * 0.01  # 默认对角阵
        # 检测列名（兼容 trade_date/date、ts_code/symbol）
        date_col = 'trade_date' if 'trade_date' in df.columns else ('date' if 'date' in df.columns else None)
        code_col = 'ts_code'  if 'ts_code'  in df.columns else ('symbol' if 'symbol' in df.columns else None)

        if 'close' in df.columns and date_col and code_col:
            try:
                # 用收益率面板构建协方差
                df['_ret'] = df.groupby(code_col)['close'].pct_change(1).fillna(0)
                pivot = df.pivot_table(index=date_col, columns=code_col, values='_ret', aggfunc='mean')
                if pivot.shape[0] >= 5:
                    cov = pivot.cov().fillna(0)
                    # 确保列顺序与df一致
                    codes_in_df = df[code_col].values if code_col else []
                    valid_codes = [c for c in codes_in_df if c in cov.columns]
                    if len(valid_codes) == n:
                        cov_matrix = cov.loc[valid_codes, valid_codes].values + np.eye(n) * 1e-6
                df.drop(columns=['_ret'], inplace=True, errors='ignore')
            except Exception as e:
                logger.warning(f"  协方差矩阵构建失败，用单位阵: {e}")
                cov_matrix = np.eye(n) * 0.01

        # 选择优化方法
        if self.method == 'equal_weight':
            weights = self.equal_weight(n)
        elif self.method == 'mean_variance':
            if prev_weights is not None:
                weights = self.optimize_with_turnover(expected_returns, cov_matrix, prev_weights)
            else:
                weights = self.mean_variance_optimization(expected_returns, cov_matrix)
        elif self.method == 'risk_parity':
            weights = self.risk_parity(cov_matrix)
        else:
            logger.warning(f"未知方法 {self.method}，使用等权")
            weights = self.equal_weight(n)

        # 保证权重合法
        weights = np.clip(weights, 0, self.max_weight)
        total_w = weights.sum()
        if total_w > 1e-9:
            weights = weights / total_w
        else:
            weights = self.equal_weight(n)

        df['optimal_weight'] = weights

        # ════════════════════════════════════════════════════════
        # V28: 行业集中度约束（私募合规标准：单行业≤25%）
        # 逻辑：若任一行业持仓占比>25%，对该行业所有股票权重×0.7
        # 然后重归一化，保证权重之和=1
        # 经济意义：防止策略因行业轮动偏向而过度集中单一赛道
        # ════════════════════════════════════════════════════════
        if 'industry' in df.columns and df['optimal_weight'].sum() > 1e-9:
            try:
                _ind_w = df.groupby('industry')['optimal_weight'].sum()
                _top_ind = _ind_w.idxmax()
                _top_pct = _ind_w.max()
                if _top_pct > 0.25:
                    _mask = df['industry'] == _top_ind
                    df.loc[_mask, 'optimal_weight'] *= 0.70
                    _total = df['optimal_weight'].sum()
                    if _total > 1e-9:
                        df['optimal_weight'] /= _total
                    logger.warning(
                        f"  ⚠️ 行业集中度超标: {_top_ind} 原占比={_top_pct:.1%} → 降权后重归一化"
                    )
                else:
                    logger.info(f"  ✅ 行业集中度合格: 最大行业={_top_ind} {_top_pct:.1%} ≤25%")
            except Exception as _ie:
                logger.warning(f"  行业集中度约束失败（跳过）: {_ie}")

        weights_final = df['optimal_weight'].values
        logger.info(f"\n权重统计: 最大={weights_final.max():.2%}, 最小={weights_final.min():.2%}, 平均={weights_final.mean():.2%}")
        logger.info("=" * 60)
        return df


# 全局实例
portfolio_optimizer = PortfolioOptimizer()


if __name__ == '__main__':
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'symbol': [f'00000{i}.SZ' for i in range(n)],
        'neutral_score': np.random.uniform(0, 1, n),
    })
    df = portfolio_optimizer.optimize(df)
    print(df.nlargest(10, 'optimal_weight')[['symbol', 'optimal_weight']])