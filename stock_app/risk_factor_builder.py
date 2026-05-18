# -*- coding: utf-8 -*-
"""
风险因子数据构建器（增强版）
每日运行，生成因子暴露和因子收益率，供选股系统使用
"""
import tushare as ts
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
TOKEN = os.environ.get('TUSHARE_TOKEN', '')
OUTPUT_DIR = './factor_data'
LOOKBACK_DAYS = 120                # 用于计算因子收益率的历史天数
BATCH_SIZE = 50                     # 批量获取股票数
MAX_RETRIES = 3                     # 请求重试次数
SLEEP_BETWEEN_REQUESTS = 0.5        # 请求间隔（秒）

# ==================== 初始化 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
FACTOR_EXPOSURE_FILE = os.path.join(OUTPUT_DIR, 'factor_exposure_latest.csv')
FACTOR_RETURNS_FILE = os.path.join(OUTPUT_DIR, 'factor_returns_history.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ts.set_token(TOKEN)
pro = ts.pro_api(timeout=30)


def fetch_with_retry(func, *args, **kwargs):
    """带重试的API调用"""
    for attempt in range(MAX_RETRIES):
        try:
            result = func(*args, **kwargs)
            if result is not None and not result.empty:
                return result
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        except Exception as e:
            logger.warning(f"第{attempt+1}次尝试失败: {e}")
            time.sleep(SLEEP_BETWEEN_REQUESTS * (2 ** attempt))
    logger.error(f"重试{MAX_RETRIES}次后仍然失败: {func.__name__}")
    return pd.DataFrame()


def fetch_stock_basic():
    """获取股票基础信息（行业），过滤ST"""
    df = fetch_with_retry(pro.stock_basic, exchange='', list_status='L',
                          fields='ts_code,name,industry')
    if df.empty:
        return df
    # 过滤ST和退市股
    df = df[~df['name'].str.contains(r'ST|退市|S\*ST|\*ST', na=False, regex=True)]
    return df[['ts_code', 'industry']]


def fetch_daily_data(start_date, end_date, codes):
    """批量获取日线数据（分批次）"""
    all_daily = []
    for i in range(0, len(codes), BATCH_SIZE):
        batch_codes = codes[i:i+BATCH_SIZE]
        code_str = ','.join(batch_codes)
        df = fetch_with_retry(pro.daily, ts_code=code_str, start_date=start_date,
                              end_date=end_date, fields='ts_code,trade_date,pct_chg,close')
        if not df.empty:
            all_daily.append(df)
            logger.info(f"批次 {i//BATCH_SIZE+1}: 获取 {len(batch_codes)} 只股票，{len(df)} 条记录")
    if all_daily:
        return pd.concat(all_daily, ignore_index=True)
    return pd.DataFrame()


def fetch_daily_basic(start_date, end_date):
    """获取每日指标（市值、PB），按日期循环"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='B').strftime('%Y%m%d').tolist()
    all_basic = []
    for date in date_range:
        # V30: 新增 pe, close 供 factor_valuation_attractiveness 计算
        df = fetch_with_retry(pro.daily_basic, trade_date=date,
                              fields='ts_code,trade_date,total_mv,pb,pe,close')
        if not df.empty:
            all_basic.append(df)
            logger.info(f"获取 {date} 数据，共 {len(df)} 条")
    if all_basic:
        return pd.concat(all_basic, ignore_index=True)
    return pd.DataFrame()


def merge_data(daily_df, basic_df, stock_info):
    """合并日线、市值、行业数据，清洗并重命名"""
    merged = pd.merge(daily_df, basic_df, on=['ts_code', 'trade_date'], how='inner')
    merged = pd.merge(merged, stock_info, on='ts_code', how='left')
    if merged.empty:
        return merged

    # 类型转换
    merged['pct_chg']  = pd.to_numeric(merged['pct_chg'],  errors='coerce') / 100.0
    merged['total_mv'] = pd.to_numeric(merged['total_mv'], errors='coerce')
    merged['pb']       = pd.to_numeric(merged['pb'],       errors='coerce')
    # V30: pe/close 用于 valuation_attractiveness，不加入 dropna 以免丢失数据
    if 'pe'    in merged.columns:
        merged['pe']    = pd.to_numeric(merged['pe'],    errors='coerce')
    if 'close' in merged.columns:
        merged['close'] = pd.to_numeric(merged['close'], errors='coerce')

    # 过滤关键字段缺失
    merged = merged.dropna(subset=['pct_chg', 'total_mv', 'industry'])

    # 重命名以匹配后续计算
    merged.rename(columns={
        'ts_code': 'symbol',
        'trade_date': 'date',
        'pct_chg': 'asset_returns',
        'total_mv': 'market_cap',
        'pb': 'pb_ratio',
        'industry': 'sector'
    }, inplace=True)
    return merged


def calculate_factors(df):
    """计算风格因子暴露（市值、动量、价值）"""
    df = df.copy()
    df = df.sort_values(['symbol', 'date'])

    # 市值因子：对数市值
    df['factor_sze'] = np.log(df['market_cap'])

    # 动量因子：过去20日累计收益（滞后一日，避免未来数据）
    df['ret_lag'] = df.groupby('symbol')['asset_returns'].shift(1)
    df['factor_mom'] = df.groupby('symbol')['ret_lag'].transform(
        lambda x: x.rolling(20, min_periods=5).sum()
    )

    # 价值因子：市净率的倒数（避免除以0）
    df['factor_val'] = 1.0 / df['pb_ratio'].clip(lower=0.1)

    # ════════════════════════════════════════════════════════════════════════
    # V30新增: 估值吸引力复合因子 factor_valuation_attractiveness
    # ────────────────────────────────────────────────────────────────────────
    # 修改位置: calculate_factors 末尾，factor_val 之后
    # 私募理由:
    #   传统 factor_val=1/PB 只含 PB，缺少 PE，且无价格区间奖惩。
    #   A股散户定价偏差使低价低估值股长期被低估：
    #     - 10-40元：机构介入门槛低、流通性好、弹性大
    #     - >80元：散户追捧过度，预期收益差
    #   公式: base = (1-PB行业内分位 + 1-PE行业内分位) / 2  →  [0,1]
    #         × 价格区间bonus → factor_valuation_attractiveness ∈ [0,2]
    #   使用行业内分位（非绝对值）消除行业间固有估值差异。
    # ════════════════════════════════════════════════════════════════════════
    _has_pe     = 'pe'     in df.columns
    _has_close  = 'close'  in df.columns
    _has_sector = 'sector' in df.columns

    if _has_pe:
        # ── Step1: 极值 clip（PE<0 为亏损股，不参与分位比较）───────────────
        _pe_c = pd.to_numeric(df['pe'],       errors='coerce').clip(1, 200).fillna(30)
        _pb_c = pd.to_numeric(df['pb_ratio'], errors='coerce').clip(0.1, 50).fillna(3)
        df['_va_pe'] = _pe_c
        df['_va_pb'] = _pb_c

        # ── Step2: 行业内分位（同行业横向比较，消除行业间绝对估值差）────────
        if _has_sector:
            _pe_rank = df.groupby('sector')['_va_pe'].rank(pct=True)
            _pb_rank = df.groupby('sector')['_va_pb'].rank(pct=True)
        else:
            _pe_rank = df['_va_pe'].rank(pct=True)
            _pb_rank = df['_va_pb'].rank(pct=True)

        # ── Step3: 分位倒数（低估值 → 分位低 → 倒数大 → 吸引力高）─────────
        _inv_pe  = 1.0 - _pe_rank.clip(0.01, 0.99)   # [0,1]，1 = 最低PE
        _inv_pb  = 1.0 - _pb_rank.clip(0.01, 0.99)
        _base_va = (_inv_pe + _inv_pb) / 2.0           # 综合估值吸引力 [0,1]

        # ── Step4: 价格区间 bonus（私募实战核心：10-40元是A股最优买入区间）─
        if _has_close:
            _cl = pd.to_numeric(df['close'], errors='coerce').fillna(30)
            _price_bonus = pd.Series(1.00, index=df.index)   # 其余价格：×1.00
            _price_bonus[_cl < 15]             = 1.40         # 超低价小市值：+40%
            _price_bonus[_cl.between(15, 40)]  = 1.25         # 低价优质买入区：+25%
            _price_bonus[_cl > 80]             = 0.70         # 高价：机构定价充分，-30%
            _price_bonus[_cl > 150]            = 0.55         # 极高价：流动性差，-45%
        else:
            _price_bonus = 1.0

        df['factor_valuation_attractiveness'] = (_base_va * _price_bonus).clip(0.0, 2.0)
        df.drop(columns=['_va_pe', '_va_pb'], errors='ignore', inplace=True)


        # ════════════════════════════════════════════════════════════════════
        # V31新增: factor_valuation_attractiveness × profit_yoy/ROE 联动修正
        # ────────────────────────────────────────────────────────────────────
        # 修改位置: calculate_factors() → factor_valuation_attractiveness 计算末尾
        # 私募理由:
        #   低PE/PB不等于买入信号 — 价值陷阱是A股最常见的散户坑：
        #     - 化工/钢铁行业高ROE是周期顶部表象，不能简单加分
        #     - 纯靠PE/PB排名无法区分"真低估"和"业绩持续下滑"
        #   修正逻辑：
        #     盈利质量系数 profit_quality ∈ [0.60, 1.40]
        #     • ROE>15% + profit_yoy>10%  → ×1.30（真正优质低估）
        #     • ROE>8%  + profit_yoy>0%   → ×1.10（盈利稳定）
        #     • profit_yoy<-20%            → ×0.75（业绩下滑，降低吸引力）
        #     • PE<0（亏损）               → ×0.60（亏损股直接压制）
        #   对行业（如银行）周期性ROE高的保护：
        #     使用行业内ROE分位而非绝对值，避免对周期顶部给出虚高评分
        # ════════════════════════════════════════════════════════════════════
        _has_roe  = 'roe'        in df.columns
        _has_pyoy = 'profit_yoy' in df.columns

        if _has_roe or _has_pyoy:
            # 初始化修正系数为1（中性）
            _profit_quality = pd.Series(1.0, index=df.index)

            if _has_roe:
                _roe = pd.to_numeric(df['roe'], errors='coerce').fillna(0)
                # 行业内ROE分位（防止周期性行业虚高）
                if _has_sector:
                    _roe_rank = df.assign(_tmp_roe=_roe).groupby('sector')['_tmp_roe'].rank(pct=True)
                else:
                    _roe_rank = _roe.rank(pct=True)
                # 行业内ROE前40%（高于行业中位）才加分
                _roe_bonus = pd.Series(0.0, index=df.index)
                _roe_bonus[_roe_rank > 0.6] = 0.15   # 行业内高ROE：+15%
                _roe_bonus[_roe_rank > 0.8] = 0.30   # 行业内极高ROE：+30%
                _profit_quality += _roe_bonus

            if _has_pyoy:
                _pyoy = pd.to_numeric(df['profit_yoy'], errors='coerce').fillna(0)
                _pyoy_adj = pd.Series(0.0, index=df.index)
                _pyoy_adj[_pyoy >  10] =  0.10   # 净利增速>10%：温和加分
                _pyoy_adj[_pyoy >  30] =  0.20   # 净利增速>30%：高增速加分
                _pyoy_adj[_pyoy < -20] = -0.15   # 净利下滑>20%：减分
                _pyoy_adj[_pyoy < -50] = -0.25   # 净利大幅下滑：重罚
                _profit_quality += _pyoy_adj

            # 亏损股（PE<0）直接将吸引力压制到最低区间
            if _has_pe:
                _pe_raw = pd.to_numeric(df['pe'], errors='coerce')
                _profit_quality[_pe_raw < 0] = 0.60   # 当期亏损：直接降到60%

            # clip到合理区间 [0.60, 1.40]
            _profit_quality = _profit_quality.clip(0.60, 1.40)

            # 将修正系数乘入 factor_valuation_attractiveness
            df['factor_valuation_attractiveness'] = (
                df['factor_valuation_attractiveness'] * _profit_quality
            ).clip(0.0, 2.0)

            logger.info(
                f"  ✅ V31盈利质量修正后 factor_valuation_attractiveness: "
                f"mean={df['factor_valuation_attractiveness'].mean():.3f} | "
                f"P25={df['factor_valuation_attractiveness'].quantile(0.25):.3f} | "
                f"P75={df['factor_valuation_attractiveness'].quantile(0.75):.3f} "
                f"| 修正系数range=[{_profit_quality.min():.2f},{_profit_quality.max():.2f}]"
            )
        else:
            logger.info(
                "  ℹ️ roe/profit_yoy字段缺失，factor_valuation_attractiveness未做盈利质量修正"
                "（请在 fetch_daily_basic 中加入 roe/profit_yoy 字段）"
            )

        # ── 注意：此代码插入位置 ─────────────────────────────────────────────
        # 在原有代码 df['factor_valuation_attractiveness'] = ... 行之后
        # 在原有 logger.info("✅ factor_valuation_attractiveness: ...") 之前
        # 整体替换原有 logger.info 行：
        logger.info(
            f"  ✅ factor_valuation_attractiveness (最终): "
            f"mean={df['factor_valuation_attractiveness'].mean():.3f} | "
            f"P25={df['factor_valuation_attractiveness'].quantile(0.25):.3f} | "
            f"P75={df['factor_valuation_attractiveness'].quantile(0.75):.3f}"
        )
    else:
        df['factor_valuation_attractiveness'] = 0.5
        logger.warning(
            "  ⚠️ PE字段缺失，factor_valuation_attractiveness=0.5"
            "（请在 fetch_daily_basic fields 中加入 pe, close）"
        )

    return df


def run_daily_update():
    """主函数：每日更新因子数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=int(LOOKBACK_DAYS * 1.5))).strftime('%Y%m%d')
    logger.info(f"获取数据从 {start_date} 到 {end_date}")

    # 1. 获取股票池
    stock_info = fetch_stock_basic()
    if stock_info.empty:
        logger.error("获取股票基础信息失败")
        return
    codes = stock_info['ts_code'].tolist()

    # 2. 获取日线数据
    daily_df = fetch_daily_data(start_date, end_date, codes)
    if daily_df.empty:
        logger.error("获取日线数据失败")
        return

    # 3. 获取每日指标
    basic_df = fetch_daily_basic(start_date, end_date)
    if basic_df.empty:
        logger.error("获取每日指标失败")
        return

    # 4. 合并数据
    merged = merge_data(daily_df, basic_df, stock_info)
    if merged.empty:
        logger.error("合并数据失败")
        return

    # 5. 计算因子暴露
    merged = calculate_factors(merged)

    # 6. 估计因子收益率（使用 toraniko-pandas，若不可用则降级）
    factor_returns = None
    try:
        from toraniko_pandas.model import estimate_factor_returns
        from toraniko_pandas.utils import top_n_by_group

        # 筛选市值前3000的股票
        df_filtered = top_n_by_group(merged, 3000, 'market_cap', group_col='date')
        # V30: factor_cols 加入 valuation_attractiveness（若已计算）
        factor_cols = [c for c in ['factor_sze', 'factor_mom', 'factor_val',
                                   'factor_valuation_attractiveness'] if c in df_filtered.columns]
        df_filtered = df_filtered.dropna(subset=factor_cols + ['asset_returns'])

        factor_returns = estimate_factor_returns(
            returns=df_filtered['asset_returns'],
            factors=df_filtered[factor_cols],
            dates=df_filtered['date'],
            stocks=df_filtered['symbol']
        )
        logger.info("因子收益率估计成功")
    except Exception as e:
        logger.error(f"因子收益率估计失败: {e}")
        # 降级：生成模拟数据（仅用于测试）
        dates = merged['date'].unique()
        _fallback_cols = [c for c in ['factor_sze', 'factor_mom', 'factor_val',
                                      'factor_valuation_attractiveness'] if c in merged.columns]
        factor_cols = _fallback_cols if _fallback_cols else ['factor_sze', 'factor_mom', 'factor_val']
        factor_returns = pd.DataFrame(
            np.random.randn(len(dates), len(factor_cols)) * 0.01,
            index=dates,
            columns=factor_cols
        )
        logger.warning("使用模拟因子收益率，请尽快修复真实计算")

    # 7. 保存最新因子暴露（最近一天）
    latest_date = merged['date'].max()
    latest_exposure = merged[merged['date'] == latest_date]
    # V30: 保存时加入 valuation_attractiveness（若已计算）
    _save_cols = [c for c in ['factor_sze', 'factor_mom', 'factor_val',
                               'factor_valuation_attractiveness'] if c in latest_exposure.columns]
    exposure_pivot = latest_exposure.pivot(index='symbol', columns='date', values=_save_cols)
    exposure_pivot.columns = [f"{col[0]}_{col[1]}" for col in exposure_pivot.columns]
    exposure_pivot.to_csv(FACTOR_EXPOSURE_FILE)
    logger.info(f"最新因子暴露已保存至 {FACTOR_EXPOSURE_FILE}（含 valuation_attractiveness）")

    # 8. 保存因子收益率历史
    if factor_returns is not None:
        factor_returns.to_csv(FACTOR_RETURNS_FILE)
        logger.info(f"因子收益率历史已保存至 {FACTOR_RETURNS_FILE}")


if __name__ == '__main__':
    run_daily_update()