# -*- coding: utf-8 -*-
"""特征列统一管理器 - 优化1"""
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class FeatureManager:
    CORE_FEATURES = [
        'total_mv', 'float_mv', 'circ_mv',
        'revenue_yoy', 'profit_yoy', 'roe', 'roa',
        'pe', 'pb', 'ps', 'pcf',
        'close', 'volume', 'amount',
    ]
    
    REQUIRED_FEATURES = ['ts_code', 'trade_date', 'close', 'volume', 'total_mv', 'float_mv']
    
    def validate_and_prepare(self, df):
        """验证并准备特征"""
        missing = [c for c in self.REQUIRED_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必须字段: {missing}")
        
        available = [c for c in self.CORE_FEATURES if c in df.columns]
        logger.info(f"✅ 可用特征: {len(available)}/{len(self.CORE_FEATURES)}")
        
        df = df.copy()
        for col in available:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        
        return available, df

feature_manager = FeatureManager()
