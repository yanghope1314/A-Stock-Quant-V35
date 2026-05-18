# -*- coding: utf-8 -*-
"""
V19升级模块1: 小市值因子增强
=================================
针对中证1000/2000的专门优化
基于2025年小盘股表现优异的市场特征
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class SmallCapFactorEngine:
    """
    小市值因子引擎
    ==============
    专门针对小微盘股票优化
    
    背景：
    - 2025年中证1000指增平均收益49.78%，超额17.49%
    - 2025年中证2000累计收益61.84%，超额24.45%
    - 小市值因子成为最强Alpha来源
    """
    
    def __init__(self, 
                 size_threshold_pct: float = 0.3,  # 市值分位数阈值
                 min_float_mv: float = 10,  # 最小流通市值（亿）
                 max_float_mv: float = 200):  # 最大流通市值（亿）
        """
        参数：
            size_threshold_pct: 小盘股市值分位数阈值（0.3 = 后30%）
            min_float_mv: 最小流通市值，过滤掉极小市值
            max_float_mv: 最大流通市值，确保是小盘股
        """
        self.size_threshold_pct = size_threshold_pct
        self.min_float_mv = min_float_mv
        self.max_float_mv = max_float_mv
        
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有小市值相关因子
        
        输入DataFrame需包含：
        - total_mv: 总市值
        - float_mv: 流通市值
        - close: 收盘价
        - volume: 成交量
        - revenue_yoy: 营收同比增长率
        - profit_yoy: 利润同比增长率
        - roe: 净资产收益率
        - pe: 市盈率
        - pb: 市净率
        """
        df = df.copy()
        
        # 1. 基础市值因子
        df = self._calculate_size_factors(df)
        
        # 2. 小盘成长因子
        df = self._calculate_small_cap_growth(df)
        
        # 3. 小盘动量因子
        df = self._calculate_small_cap_momentum(df)
        
        # 4. 小盘质量因子
        df = self._calculate_small_cap_quality(df)
        
        # 5. 小盘流动性因子
        df = self._calculate_small_cap_liquidity(df)
        
        # 6. 小盘波动率因子
        df = self._calculate_small_cap_volatility(df)
        
        # 7. 小盘估值因子
        df = self._calculate_small_cap_valuation(df)
        
        # 8. 小盘综合得分
        df = self._calculate_composite_score(df)
        
        return df
    
    def _calculate_size_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """市值相关基础因子"""
        # 市值分位数（越小越好）
        df['size_quantile'] = df['total_mv'].rank(pct=True)
        df['float_size_quantile'] = df['float_mv'].rank(pct=True)
        
        # 流通比例
        df['float_ratio'] = df['float_mv'] / (df['total_mv'] + 1e-9)
        
        # 市值对数
        df['log_size'] = np.log(df['total_mv'] + 1)
        df['log_float_size'] = np.log(df['float_mv'] + 1)
        
        # 小盘股标识（1=小盘，0=非小盘）
        df['is_small_cap'] = (
            (df['float_size_quantile'] <= self.size_threshold_pct) &
            (df['float_mv'] >= self.min_float_mv) &
            (df['float_mv'] <= self.max_float_mv)
        ).astype(int)
        
        # 微盘股标识（更小的盘）
        df['is_micro_cap'] = (
            (df['float_size_quantile'] <= 0.1) &
            (df['float_mv'] >= self.min_float_mv)
        ).astype(int)
        
        return df
    
    def _calculate_small_cap_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """小盘成长因子"""
        # 小盘 × 营收增长
        df['small_cap_revenue_growth'] = (
            (1 - df['size_quantile']) * df['revenue_yoy'].fillna(0)
        )
        
        # 小盘 × 利润增长
        df['small_cap_profit_growth'] = (
            (1 - df['size_quantile']) * df['profit_yoy'].fillna(0)
        )
        
        # 小盘高成长（营收和利润都高增长）
        df['small_cap_high_growth'] = (
            df['is_small_cap'] * 
            ((df['revenue_yoy'] > 20) & (df['profit_yoy'] > 20)).astype(int)
        )
        
        # 成长加速度
        if 'revenue_yoy_qoq' in df.columns:  # 环比增长
            df['small_cap_growth_accel'] = (
                (1 - df['size_quantile']) * 
                (df['revenue_yoy_qoq'] - df['revenue_yoy']).fillna(0)
            )
        
        return df
    
    def _calculate_small_cap_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """小盘动量因子"""
        # 计算历史收益率（假设已有或需要计算）
        if 'ret_5' not in df.columns and 'close' in df.columns:
            # 这里假设df已按股票和日期排序
            for period in [5, 10, 20, 60]:
                df[f'ret_{period}'] = df.groupby('ts_code')['close'].pct_change(period)
        
        # 小盘 × 短期动量
        df['small_cap_momentum_5d'] = (
            (1 - df['size_quantile']) * df.get('ret_5', 0).fillna(0)
        )
        
        df['small_cap_momentum_20d'] = (
            (1 - df['size_quantile']) * df.get('ret_20', 0).fillna(0)
        )
        
        # 小盘反转因子（小盘股更容易反转）
        df['small_cap_reversal'] = (
            df['is_small_cap'] * (-df.get('ret_5', 0).fillna(0))
        )
        
        # 小盘动量强度（动量 × 成交量）
        if 'volume' in df.columns:
            volume_ma = df.groupby('ts_code')['volume'].transform(
                lambda x: x.rolling(20, min_periods=1).mean()
            )
            df['small_cap_momentum_strength'] = (
                (1 - df['size_quantile']) * 
                df.get('ret_20', 0).fillna(0) *
                (df['volume'] / (volume_ma + 1e-9))
            )
        
        return df
    
    def _calculate_small_cap_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """小盘质量因子"""
        # 小盘 × ROE
        df['small_cap_roe'] = (
            (1 - df['size_quantile']) * df.get('roe', 0).fillna(0)
        )
        
        # 小盘 × 盈利能力
        if 'net_profit_margin' in df.columns:
            df['small_cap_profitability'] = (
                (1 - df['size_quantile']) * 
                df['net_profit_margin'].fillna(0)
            )
        
        # 小盘优质股（高ROE + 低负债）
        if 'debt_to_assets' in df.columns:
            df['small_cap_quality_score'] = (
                df['is_small_cap'] *
                (df['roe'].fillna(0) > 10) *
                (df['debt_to_assets'].fillna(100) < 60)
            ).astype(int)
        
        return df
    
    def _calculate_small_cap_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """小盘流动性因子"""
        if 'volume' not in df.columns:
            return df
        
        # 换手率
        if 'float_share' in df.columns:
            df['turnover_rate'] = (
                df['volume'] / (df['float_share'] * 100 + 1e-9)
            )
        else:
            # 用流通市值估算
            df['turnover_rate'] = (
                df['volume'] * df['close'] / (df['float_mv'] * 1e8 + 1e-9)
            )
        
        # 小盘 × 流动性
        df['small_cap_liquidity'] = (
            (1 - df['size_quantile']) * df['turnover_rate']
        )
        
        # 流动性异常（小盘股突然放量）
        volume_ma = df.groupby('ts_code')['volume'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df['small_cap_liquidity_spike'] = (
            df['is_small_cap'] *
            (df['volume'] / (volume_ma + 1e-9))
        )
        
        return df
    
    def _calculate_small_cap_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """小盘波动率因子"""
        if 'close' not in df.columns:
            return df
        
        # 计算收益率波动率
        returns = df.groupby('ts_code')['close'].pct_change()
        volatility_20d = returns.rolling(20, min_periods=5).std()
        
        df['volatility_20d'] = volatility_20d
        
        # 小盘 × 波动率（高波动 = 高风险高收益）
        df['small_cap_volatility'] = (
            (1 - df['size_quantile']) * df['volatility_20d'].fillna(0)
        )
        
        # 波动率分位数
        df['volatility_quantile'] = df['volatility_20d'].rank(pct=True)
        
        # 小盘低波（防御性小盘股）
        df['small_cap_low_vol'] = (
            df['is_small_cap'] * 
            (df['volatility_quantile'] < 0.3).astype(int)
        )
        
        return df
    
    def _calculate_small_cap_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """小盘估值因子"""
        # 小盘 × PB
        df['small_cap_pb'] = (
            (1 - df['size_quantile']) * (1 / (df.get('pb', 1) + 1e-9))
        )
        
        # 小盘 × PE
        df['small_cap_pe'] = (
            (1 - df['size_quantile']) * (1 / (df.get('pe', 1) + 1e-9))
        )
        
        # 小盘低估值（PB < 2 且 PE < 30）
        df['small_cap_undervalued'] = (
            df['is_small_cap'] *
            (df.get('pb', 100) < 2) *
            (df.get('pe', 100) < 30) *
            (df.get('pe', 0) > 0)  # 排除亏损
        ).astype(int)
        
        return df
    
    def _calculate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """小盘股综合得分"""
        # 标准化各个因子
        factors = [
            'small_cap_revenue_growth',
            'small_cap_momentum_20d',
            'small_cap_roe',
            'small_cap_liquidity',
            'small_cap_pb'
        ]
        
        # 确保因子存在
        existing_factors = [f for f in factors if f in df.columns]
        
        if not existing_factors:
            df['small_cap_composite_score'] = 0
            return df
        
        # Z-score标准化
        for factor in existing_factors:
            mean = df[factor].mean()
            std = df[factor].std()
            if std > 1e-9:
                df[f'{factor}_zscore'] = (df[factor] - mean) / std
            else:
                df[f'{factor}_zscore'] = 0
        
        # 综合得分（仅针对小盘股）
        zscore_cols = [f'{f}_zscore' for f in existing_factors]
        df['small_cap_composite_score'] = (
            df['is_small_cap'] * df[zscore_cols].mean(axis=1)
        )
        
        return df
    
    def select_small_cap_stocks(self, 
                                df: pd.DataFrame,
                                top_n: int = 100,
                                min_quality_score: float = 0.0) -> pd.DataFrame:
        """
        选择优质小盘股
        
        参数：
            df: 包含因子的DataFrame
            top_n: 选择股票数量
            min_quality_score: 最小质量分数（过滤垃圾股）
        
        返回：
            选出的股票DataFrame
        """
        # 过滤条件
        filtered = df[
            (df['is_small_cap'] == 1) &
            (df['small_cap_composite_score'] >= min_quality_score)
        ].copy()
        
        if len(filtered) == 0:
            return pd.DataFrame()
        
        # 按综合得分排序
        selected = filtered.nlargest(top_n, 'small_cap_composite_score')
        
        return selected[['ts_code', 'small_cap_composite_score', 
                        'float_mv', 'is_small_cap'] + 
                       [col for col in df.columns if 'small_cap_' in col]]


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    # 1. 创建引擎
    engine = SmallCapFactorEngine(
        size_threshold_pct=0.3,  # 后30%市值
        min_float_mv=10,  # 最小流通市值10亿
        max_float_mv=200  # 最大流通市值200亿
    )
    
    # 2. 准备数据（示例数据结构）
    # df = tushare_get_daily_data()  # 从Tushare获取
    
    # 3. 计算因子
    # df_with_factors = engine.calculate_factors(df)
    
    # 4. 选股
    # selected_stocks = engine.select_small_cap_stocks(
    #     df_with_factors,
    #     top_n=100,
    #     min_quality_score=0.5
    # )
    
    # 5. 输出结果
    # print(f"选出 {len(selected_stocks)} 只小盘股")
    # print(selected_stocks[['ts_code', 'small_cap_composite_score']].head(10))
    
    pass


if __name__ == '__main__':
    example_usage()