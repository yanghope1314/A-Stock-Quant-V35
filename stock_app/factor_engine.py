# -*- coding: utf-8 -*-
"""
私募级规则因子引擎 (Factor Engine)
====================================
从 views.py 拆分出的独立因子计算模块。
150+ 技术规则因子：动量/反转/量价/波动率/技术指标/衍生交互。

用途：
  from .factor_engine import calculate_rule_factors, FACTOR_WEIGHTS_2026

私募标准：
  - 截面中性化（同一天内股票间比较，非跨历史）
  - 因子排名用 rank(pct=True)，消除量纲
  - 行业中性化仅用最新截面（不混入历史数据）
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# 规则因子权重字典（2026版，私募校准）
# ══════════════════════════════════════════════════════════════════════
FACTOR_WEIGHTS_2026: Dict[str, float] = {
    # 动量因子（正向）
    'pmt_return_5d':     15.0,
    'pmt_return_20d':    20.0,
    'pmt_return_60d':    12.0,
    # 反转因子（正向：超卖反转）
    'rev_5d_reversal':    8.0,
    # 量价因子
    'vol_ratio_raw':     10.0,
    'vol_turnover':       8.0,
    # 估值因子（负向：PE/PB越高越差）
    'fund_pb':          -18.0,
    'fund_pe':          -15.0,
    # 规模因子
    'size_score':        25.0,
    'size_mkt_cap_log':  -5.0,
    # 资金流向
    'mf_net_main':       18.0,
    'mf_main_ratio':     12.0,
    # 情绪因子
    'sent_limit_gene':   22.0,
    'sent_limit_active': 15.0,
    'sent_hsgt_fav':    20.0,
    # 波动率（负向）
    'volat_hist_20d':    -8.0,
    # 风险因子（负向）
    'risk_pledge':      -15.0,
    'risk_pledge_high': -20.0,
    'risk_st':          -25.0,
    'risk_delist_score': -30.0,
    # 技术指标（抄底信号，正向）
    'kdj_oversold':      10.0,
    'kdj_low_golden':    15.0,
    'rsi_oversold':      12.0,
    'rsi_divergence':    18.0,
    'wr_oversold':        8.0,
    'boll_position':    -10.0,
    'near_boll_lower':   12.0,
    'multi_oversold_flag': 20.0,
    'reversal_signal':   25.0,
    'in_bottom_zone':    15.0,
    'price_position_60d': -30.0,
    'pmt_return_1d':      5.0,
    'liquidity_amihud':   8.0,
}


def calculate_rule_factors(
    df: pd.DataFrame,
    api_governor=None,
    pro=None,
    parent_logger=None,
) -> pd.DataFrame:
    """
    V23 生产级规则因子计算（彻底修复所有 grp KeyError）
    ─────────────────────────────────────────────────────────
    私募修复历史：
      ① macd_signal → 重新groupby（macd列创建后）
      ② kdj_low_golden → 重新groupby（kdj_k/d列创建后）
      ③ rsi_divergence → 重新groupby（rsi列创建后）
      ④ float_mv 列名映射（circ_mv → float_mv）
      ⑤ revenue_yoy 缺失默认0
      ⑥ size_score 截面排名（仅最新日，不混历史）
      ⑦ 行业中性化截面版（仅最新截面计算行业中值）

    参数:
      df: 含股票日线数据的DataFrame（需含 ts_code, trade_date, open, high, low, close, vol, amount）
      api_governor: APIGovernor 实例（可选，用于质押数据补充）
      pro: tushare pro_api 实例（可选）
      parent_logger: 父模块logger（可选）

    返回:
      添加了所有规则因子的DataFrame
    """
    log = parent_logger or logger
    df = df.copy()

    # ── 质押保底补充 ──────────────────────────────────────────────────
    _pledge_missing = (
        'pledge_ratio' not in df.columns or
        (pd.to_numeric(df['pledge_ratio'], errors='coerce').fillna(0) == 0).all()
    )
    if _pledge_missing and api_governor is not None and pro is not None:
        if api_governor.acquire('pledge_stat'):
            log.warning("  [质押-保底] pledge_ratio缺失/全零，从Tushare补充...")
            try:
                from datetime import datetime as _dt2, timedelta as _td2
                _today2 = _dt2.now()
                _qm = _today2.month - (_today2.month - 1) % 3
                _qend_dt = _dt2(_today2.year, _qm, 1) - _td2(days=1)
                _qdate2 = _qend_dt.strftime('%Y%m%d')

                _pr2 = pro.pledge_stat(
                    end_date=_qdate2,
                    fields='ts_code,end_date,unrest_pledge,rest_pledge,total_share'
                )
                if _pr2 is None or _pr2.empty:
                    _qend_dt2 = _dt2(_qend_dt.year, _qend_dt.month, 1) - _td2(days=1)
                    _pr2 = pro.pledge_stat(
                        end_date=_qend_dt2.strftime('%Y%m%d'),
                        fields='ts_code,end_date,unrest_pledge,rest_pledge,total_share'
                    )

                if _pr2 is not None and not _pr2.empty:
                    _pr2['_u'] = pd.to_numeric(_pr2['unrest_pledge'], errors='coerce').fillna(0)
                    _pr2['_r'] = pd.to_numeric(_pr2['rest_pledge'], errors='coerce').fillna(0)
                    _pr2['_t'] = pd.to_numeric(_pr2['total_share'], errors='coerce').replace(0, np.nan)
                    _pr2['pledge_ratio'] = ((_pr2['_u'] + _pr2['_r']) / _pr2['_t'] * 100).clip(0, 100).fillna(0)
                    _pc2 = (
                        _pr2.sort_values('end_date', ascending=False)
                        [['ts_code', 'pledge_ratio']]
                        .drop_duplicates('ts_code', keep='first')
                    )
                    if 'pledge_ratio' in df.columns:
                        df = df.drop(columns=['pledge_ratio'])
                    df = df.merge(_pc2, on='ts_code', how='left')
                    df['pledge_ratio'] = df['pledge_ratio'].fillna(0.0)
                    _nz2 = (df['pledge_ratio'] > 0).sum()
                    log.info(f"  [质押-保底] ✅ 成功: {_nz2}只非零")
                else:
                    log.warning("  [质押-保底] 仍返回空，pledge_ratio=0")
                    if 'pledge_ratio' not in df.columns:
                        df['pledge_ratio'] = 0.0
            except Exception as _pe2:
                log.error(f"  [质押-保底] 异常: {_pe2}")
                if 'pledge_ratio' not in df.columns:
                    df['pledge_ratio'] = 0.0
    elif _pledge_missing:
        if 'pledge_ratio' not in df.columns:
            df['pledge_ratio'] = 0.0

    # ── 0. 数值类型强制 + 列名兼容映射 ────────────────────────────────
    for c in ['open', 'high', 'low', 'close', 'vol', 'amount']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['total_mv', 'circ_mv', 'pe', 'pb', 'turnover_rate']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'circ_mv' in df.columns and 'float_mv' not in df.columns:
        df['float_mv'] = df['circ_mv']
    if 'revenue_yoy' not in df.columns:
        df['revenue_yoy'] = 0.0
    if 'profit_yoy' not in df.columns:
        df['profit_yoy'] = 0.0

    # ── 1. 初始 groupby ──────────────────────────────────────────────
    df = df.sort_values(['ts_code', 'trade_date'])
    grp = df.groupby('ts_code', sort=False)

    # ── 2. 动量因子 ──────────────────────────────────────────────────
    for n, col in [(5, 'pmt_return_5d'), (20, 'pmt_return_20d'), (60, 'pmt_return_60d')]:
        df[col] = grp['close'].pct_change(n)
    df['rev_5d_reversal'] = -df['pmt_return_5d']
    df['rev_1d_reversal'] = -grp['close'].pct_change(1)

    # ── 3. 量价因子 ──────────────────────────────────────────────────
    df['vol_ratio_raw'] = grp['vol'].transform(
        lambda x: x / (x.rolling(20, min_periods=5).mean().shift(1) + 1e-9)
    )
    df['vol_amount_ratio'] = grp['amount'].transform(
        lambda x: x / (x.rolling(20, min_periods=5).mean().shift(1) + 1e-9)
    )
    if 'turnover_rate' in df.columns:
        df['vol_turnover'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
    else:
        df['vol_turnover'] = df['vol_ratio_raw'].fillna(1.0)
    price_chg = grp['close'].pct_change(1)
    vol_chg = grp['vol'].pct_change(1)
    df['vol_price_corr'] = np.sign(price_chg) * np.sign(vol_chg)

    # ── 4. 估值因子 ──────────────────────────────────────────────────
    if 'pb' in df.columns:
        df['fund_pb'] = pd.to_numeric(df['pb'], errors='coerce')
    if 'pe' in df.columns:
        df['fund_pe'] = pd.to_numeric(df['pe'], errors='coerce').clip(-100, 200)

    # ── 5. 规模因子（截面排名，仅最新日）──────────────────────────────
    if 'total_mv' in df.columns:
        _mv = pd.to_numeric(df['total_mv'], errors='coerce')
        _mv = _mv.fillna(_mv.median() if _mv.notna().any() else 1e5)
        df['size_mkt_cap_log'] = np.log1p(_mv)
        if 'trade_date' in df.columns:
            _latest_date = df['trade_date'].max()
            _snap_mv = df.loc[df['trade_date'] == _latest_date,
                              ['ts_code', 'size_mkt_cap_log']].drop_duplicates('ts_code')
            _snap_mv['size_score_snap'] = 1.0 - _snap_mv['size_mkt_cap_log'].rank(pct=True)
            _score_map = _snap_mv.set_index('ts_code')['size_score_snap']
            df['size_score'] = df['ts_code'].map(_score_map).fillna(0.5)
        else:
            df['size_score'] = 1.0 - df['size_mkt_cap_log'].rank(pct=True)
    else:
        df['size_score'] = 0.5
        df['size_mkt_cap_log'] = 10.0

    # ── 6. 波动率因子 ────────────────────────────────────────────────
    df['volat_hist_20d'] = grp['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=5).std()
    )
    df['volat_hist_5d'] = grp['close'].transform(
        lambda x: x.pct_change().rolling(5, min_periods=3).std()
    )
    df['volat_ratio'] = df['volat_hist_5d'] / (df['volat_hist_20d'] + 1e-9)
    df['pmt_return_1d'] = grp['close'].pct_change(1)

    # Amihud 流动性比率
    daily_ret_abs = grp['close'].pct_change(1).abs()
    amount_100m = df['amount'] / 1e8
    amount_100m = amount_100m.replace(0, np.nan).ffill()
    df['amihud_illiq'] = daily_ret_abs / (amount_100m + 1e-9)
    df['amihud_illiq_20d'] = grp['amihud_illiq'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    df['liquidity_amihud'] = -np.log1p(df['amihud_illiq_20d'].clip(0, 10))

    # ── 7. 均线趋势因子 ──────────────────────────────────────────────
    for n in [5, 10, 20, 60]:
        df[f'ma{n}'] = grp['close'].transform(lambda x: x.rolling(n, min_periods=1).mean())
    df['ma_bull'] = ((df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])).astype(float)
    df['ma_bear'] = ((df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20'])).astype(float)
    df['ma_trend_score'] = df['ma_bull'] - df['ma_bear']
    df['ma20_distance'] = (df['close'] - df['ma20']) / (df['ma20'] + 1e-9)
    if 'ma60' in df.columns:
        df['ma60_distance'] = (df['close'] - df['ma60']) / (df['ma60'] + 1e-9)

    # ── 8. MACD ─────────────────────────────────────────────────────
    ema12 = grp['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = grp['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = ema12 - ema26
    _grp_macd = df.groupby('ts_code', sort=False)
    df['macd_signal'] = _grp_macd['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_golden'] = (df['macd'] > df['macd_signal']).astype(float)

    # ── 9. KDJ ──────────────────────────────────────────────────────
    low_min = grp['low'].transform(lambda x: x.rolling(9, min_periods=1).min())
    high_max = grp['high'].transform(lambda x: x.rolling(9, min_periods=1).max())
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-9) * 100
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    df['kdj_oversold'] = (df['kdj_j'] < 20).astype(float)
    df['kdj_overbought'] = (df['kdj_j'] > 80).astype(float)
    _grp_kdj = df.groupby('ts_code', sort=False)
    prev_k = _grp_kdj['kdj_k'].shift(1)
    prev_d = _grp_kdj['kdj_d'].shift(1)
    df['kdj_low_golden'] = (
        (df['kdj_k'] > df['kdj_d']) & (prev_k <= prev_d) & (df['kdj_k'] < 50)
    ).astype(float)

    # ── 10. RSI ─────────────────────────────────────────────────────
    delta = grp['close'].diff()
    avg_gain = delta.clip(lower=0).groupby(df['ts_code']).transform(
        lambda x: x.ewm(com=13, adjust=False).mean()
    )
    avg_loss = (-delta).clip(lower=0).groupby(df['ts_code']).transform(
        lambda x: x.ewm(com=13, adjust=False).mean()
    )
    df['rsi'] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-9))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    _grp_rsi = df.groupby('ts_code', sort=False)
    price_low = _grp_rsi['close'].transform(lambda x: x.rolling(20, min_periods=5).min())
    rsi_low_5 = _grp_rsi['rsi'].transform(lambda x: x.rolling(5, min_periods=3).min())
    df['rsi_divergence'] = (
        (df['close'] == price_low) & (df['rsi'] > rsi_low_5 + 5)
    ).astype(float)

    # ── 11. William %R ──────────────────────────────────────────────
    df['wr'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-9)
    df['wr_oversold'] = (df['wr'] < -80).astype(float)

    # ── 12. Bollinger Bands ─────────────────────────────────────────
    ma20_std = grp['close'].transform(lambda x: x.rolling(20, min_periods=5).std())
    df['boll_upper'] = df['ma20'] + 2 * ma20_std
    df['boll_lower'] = df['ma20'] - 2 * ma20_std
    boll_range = df['boll_upper'] - df['boll_lower'] + 1e-9
    df['boll_position'] = (df['close'] - df['boll_lower']) / boll_range
    df['near_boll_lower'] = (df['boll_position'] < 0.2).astype(float)
    df['boll_squeeze'] = (ma20_std / (df['ma20'] + 1e-9) < 0.02).astype(float)

    # ── 13. 价格位置 ────────────────────────────────────────────────
    df['price_position_60d'] = grp['close'].transform(
        lambda x: x.rolling(60, min_periods=10).apply(
            lambda w: (w[-1] - w.min()) / (w.max() - w.min() + 1e-9), raw=True
        )
    )
    df['in_bottom_zone'] = (df['price_position_60d'] < 0.2).astype(float)

    # ── 14. 多重超卖合成 ────────────────────────────────────────────
    oversold_flags = ['kdj_oversold', 'rsi_oversold', 'wr_oversold', 'near_boll_lower']
    avail = [f for f in oversold_flags if f in df.columns]
    df['multi_oversold_flag'] = df[avail].sum(axis=1) / len(avail) if avail else 0.0

    # ── 15. 反转信号 ────────────────────────────────────────────────
    df['reversal_signal'] = (
        (df['multi_oversold_flag'] > 0.5) &
        (df['vol_ratio_raw'].fillna(1) > 1.5) &
        (df['pmt_return_5d'].fillna(0) < -0.05)
    ).astype(float)

    # ── 16. 风险因子 ────────────────────────────────────────────────
    if 'pledge_ratio' not in df.columns:
        df['pledge_ratio'] = 0.0
    df['risk_pledge'] = (df['pledge_ratio'] > 30).astype(float)
    df['risk_pledge_high'] = (df['pledge_ratio'] > 60).astype(float)
    if 'name' in df.columns:
        df['risk_st'] = df['name'].str.contains('ST|退', na=False).astype(float)
    else:
        df['risk_st'] = 0.0
    if 'de_listed_date' in df.columns:
        df['risk_delist_score'] = df['de_listed_date'].notna().astype(float)

    # ── 17. 资金流默认填充 ──────────────────────────────────────────
    for col in ['mf_net_main', 'mf_main_ratio', 'sent_limit_gene',
                'sent_limit_active', 'sent_hsgt_fav']:
        if col not in df.columns:
            df[col] = 0.0

    # ── 18. 流动性因子 ──────────────────────────────────────────────
    df['liquidity_score'] = grp['amount'].transform(
        lambda x: np.log1p(x.rolling(20, min_periods=5).mean())
    )

    # ── 19. 衍生交互因子 ────────────────────────────────────────────
    df['small_cap_momentum'] = df['size_score'] * df['pmt_return_20d'].fillna(0)
    df['oversold_volume_flag'] = df['multi_oversold_flag'] * df['vol_ratio_raw'].fillna(1)
    df['low_pb_reversal'] = (
        (df.get('fund_pb', pd.Series(99.0, index=df.index)) < 2.0) &
        (df['near_boll_lower'] > 0)
    ).astype(float)

    # ── 20. 行业中性化（截面 alpha 提取）─────────────────────────────
    if 'industry' in df.columns:
        _neutralize_cols = [
            'pmt_return_5d', 'pmt_return_20d', 'pmt_return_60d',
            'ma20_distance', 'ma_trend_score', 'macd', 'rsi',
        ]
        _avail = [c for c in _neutralize_cols if c in df.columns]
        if _avail:
            if 'trade_date' in df.columns:
                _latest = df['trade_date'].max()
                _snap = df[df['trade_date'] == _latest].copy()
            else:
                _snap = df.copy()
            for col in _avail:
                try:
                    _v = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    _industry_map = df[['ts_code', 'industry']].drop_duplicates('ts_code').set_index('ts_code')['industry']
                    _snap_col = pd.to_numeric(_snap[col], errors='coerce').fillna(0)
                    _snap_ind = _snap['industry']
                    _ind_median = _snap_col.groupby(_snap_ind).median()
                    _row_ind = df['ts_code'].map(_industry_map).fillna('未知')
                    _row_median = _row_ind.map(_ind_median).fillna(0.0)
                    df[f'{col}_neutral'] = _v - _row_median
                except Exception:
                    pass
            log.info(f"  行业中性化完成（截面版）: {len(_avail)} 因子")

    # ── 21. NaN 填充 ────────────────────────────────────────────────
    _skip = {'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
             'vol', 'amount', 'name', 'industry'}
    for c in df.columns:
        if c not in _skip:
            try:
                if df[c].dtype.kind in ('f', 'i', 'u'):
                    df[c] = df[c].fillna(0.0)
            except Exception:
                pass

    log.info(f"  ✅ 规则因子计算完成: {df.shape[1]} 列")
    return df
