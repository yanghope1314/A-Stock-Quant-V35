# -*- coding: utf-8 -*-
"""
私募级因子筛选引擎 (Factor Selector)
======================================
从 views.py 拆分出的独立 VIF 正交化模块。
150+ 因子 → VIF 迭代筛选 → ~30 个独立因子

用途：
  from .factor_selector import select_orthogonal_factors

私募标准（幻方/九坤）：
  - VIF > 5 → 存在显著共线性，剔除
  - VIF > 10 → 严重共线性，必须剔除
  - 仅用截面数据（最新日），不混入历史重复行
  - 保留必需因子（essential）不被剔除
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def select_orthogonal_factors(
    df: pd.DataFrame,
    factor_cols: List[str],
    max_vif: float = 5.0,
    max_factors: int = 30,
    essential: List[str] = None,
    parent_logger=None,
) -> List[str]:
    """
    VIF（方差膨胀因子）迭代筛选独立因子

    原理：
      - VIF_j = 1 / (1 - R²_j)，其中 R²_j 是用其他因子回归第j个因子的R²
      - VIF > 5 → 存在显著共线性
      - VIF > 10 → 严重共线性，必须剔除

    算法：
      1. 计算每个因子的VIF
      2. 如果有VIF > max_vif，剔除VIF最大的那个（保护essential）
      3. 重复直到所有VIF < max_vif 或因子数 ≤ max_factors

    参数:
      df: 包含因子列的DataFrame（仅截面数据，不含多日历史）
      factor_cols: 待筛选的因子列名列表
      max_vif: VIF阈值（默认5，私募标准）
      max_factors: 最多保留因子数
      essential: 受保护的必需因子列表（不被VIF剔除）
      parent_logger: 父模块logger

    返回:
      筛选后的因子列名列表
    """
    log = parent_logger or logger
    essential = essential or []

    if len(factor_cols) < 5:
        return factor_cols

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        log.warning("  VIF筛选需要statsmodels，跳过（pip install statsmodels）")
        return factor_cols[:max_factors] if len(factor_cols) > max_factors else factor_cols

    import warnings as _vw

    # 仅取截面数据（最新日），避免历史重复行干扰
    if 'trade_date' in df.columns:
        _latest = df['trade_date'].max()
        df_slice = df[df['trade_date'] == _latest][factor_cols].copy()
    else:
        df_slice = df[factor_cols].copy()

    # 清洗：dropna + 去常数列
    df_slice = df_slice.dropna(axis=0).dropna(axis=1)
    numeric_cols = []
    for c in df_slice.columns:
        s = pd.to_numeric(df_slice[c], errors='coerce')
        if s.std() > 1e-9 and not s.isna().all():
            numeric_cols.append(c)

    if len(numeric_cols) < 5:
        log.warning(f"  VIF筛选: 有效因子仅{len(numeric_cols)}个，跳过")
        return numeric_cols

    df_num = df_slice[numeric_cols].fillna(0).astype(float)

    working = list(numeric_cols)
    n_removed = 0

    log.info(f"  🔬 VIF因子筛选: 初始{len(working)}个 → 目标≤{max_factors}个 | VIF阈值={max_vif}")

    while len(working) > max(5, max_factors) or True:
        if len(working) < 5:
            break

        try:
            with _vw.catch_warnings():
                _vw.simplefilter('ignore')
                vif_list = []
                X_mat = df_num[working].values
                for j in range(len(working)):
                    vif = variance_inflation_factor(X_mat, j)
                    vif_list.append((working[j], vif))

            # 找到VIF最大的可移除因子（保护essential）
            removable = [(name, val) for name, val in vif_list if name not in essential]
            if not removable:
                log.info(f"  ✅ VIF收敛: {len(working)}因子（剩余均为必需因子）")
                break

            max_name, max_val = max(removable, key=lambda x: x[1])

            if max_val <= max_vif and len(working) <= max_factors:
                log.info(f"  ✅ VIF收敛: {len(working)}因子 | 最大VIF={max_val:.1f}({max_name})")
                break

            if max_val > max_vif:
                working.remove(max_name)
                df_num = df_num[working]
                n_removed += 1
                if n_removed <= 5:
                    log.debug(f"  VIF剔除: {max_name} (VIF={max_val:.1f})")
            elif len(working) > max_factors:
                working.remove(max_name)
                df_num = df_num[working]
                n_removed += 1
            else:
                break

        except Exception as e:
            log.warning(f"  VIF计算异常(已剔除{n_removed}个): {e}")
            break

    log.info(f"  🎯 VIF筛选完成: {len(working)}个独立因子 (剔除了{n_removed}个冗余因子)")
    return working
