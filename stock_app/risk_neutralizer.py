# -*- coding: utf-8 -*-
"""
风险中性化与风险管理模块（私募级）
===============================================
功能：
- 行业中性化
- 风格中性化（类Barra）
- 风险因子计算（波动率、流动性、估值）
- 风险过滤（ST、流动性、综合风险）
- 行业集中度检查
===============================================
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class RiskManager:
    """风险中性化与风险管理器"""

    def __init__(self,
                 max_position_pct=0.05,
                 max_sector_pct=0.30,
                 min_liquidity=1000000,
                 max_risk_score=0.8):
        """
        初始化
        参数:
            max_position_pct: 单股最大仓位
            max_sector_pct: 行业最大占比
            min_liquidity: 最小日均成交额（流动性过滤）
            max_risk_score: 综合风险评分上限
        """
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.min_liquidity = min_liquidity
        self.max_risk_score = max_risk_score
        self.industry_mapping = self._build_industry_mapping()
        self.style_factors = ['size', 'value', 'growth', 'momentum']

    def _build_industry_mapping(self):
        """构建行业映射（申万一级）"""
        return {
            '银行': 'finance', '非银金融': 'finance', '房地产': 'realestate',
            '建筑材料': 'materials', '建筑装饰': 'construction', '钢铁': 'materials',
            '有色金属': 'materials', '化工': 'materials', '机械设备': 'industrials',
            '电气设备': 'industrials', '国防军工': 'industrials', '汽车': 'consumer',
            '家用电器': 'consumer', '食品饮料': 'consumer', '纺织服装': 'consumer',
            '轻工制造': 'consumer', '医药生物': 'healthcare', '公用事业': 'utilities',
            '交通运输': 'transportation', '电子': 'technology', '计算机': 'technology',
            '通信': 'technology', '传媒': 'media', '商业贸易': 'trade',
            '休闲服务': 'services', '农林牧渔': 'agriculture', '采掘': 'energy',
            '综合': 'others'
        }

    # ---------- 行业中性化 ----------
    def neutralize_industry(self, df, score_col='v19_final_score', industry_col='industry'):
        """行业中性化：在每个行业内标准化得分"""
        if industry_col not in df.columns or score_col not in df.columns:
            logger.warning(f"⚠️ 缺少{industry_col}或{score_col}列，跳过行业中性化")
            df['industry_neutral_score'] = df.get(score_col, 0)
            return df

        logger.info("🎯 行业中性化...")
        df = df.copy()
        # 统计行业分布
        industry_counts = df[industry_col].value_counts()
        logger.info(f"   行业数量: {len(industry_counts)}")
        logger.info(f"   最大行业: {industry_counts.index[0]} ({industry_counts.iloc[0]}只)")

        # 行业内部Z-score
        df['industry_neutral_score'] = df.groupby(industry_col)[score_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9) if len(x) > 1 else 0
        )
        # 全局标准化
        mean_score = df['industry_neutral_score'].mean()
        std_score = df['industry_neutral_score'].std()
        df['industry_neutral_score'] = (df['industry_neutral_score'] - mean_score) / (std_score + 1e-9)
        logger.info("   ✅ 行业中性化完成")
        return df

    # ---------- 风格因子计算 ----------
    def calculate_style_factors(self, df):
        """计算风格因子（Size, Value, Growth, Momentum, Volatility, Liquidity, Valuation）"""
        logger.info("📊 计算风格因子...")
        df = df.copy()

        # ---- 统一列名适配（A股数据用ts_code/vol，Tushare标准）----
        code_col = 'ts_code' if 'ts_code' in df.columns else ('symbol' if 'symbol' in df.columns else None)
        vol_col  = 'vol'    if 'vol'    in df.columns else ('volume' if 'volume' in df.columns else None)
        amt_col  = 'amount' if 'amount' in df.columns else None

        # Size
        if 'total_mv' in df.columns:
            df['style_size'] = np.log(pd.to_numeric(df['total_mv'], errors='coerce').fillna(1e4) + 1)
            df['style_size'] = (df['style_size'] - df['style_size'].mean()) / (df['style_size'].std() + 1e-9)
        else:
            df['style_size'] = 0

        # Value (PB/PE越低越价值)
        if 'pb' in df.columns and 'pe' in df.columns:
            pb = pd.to_numeric(df['pb'], errors='coerce').fillna(df.get('pb', pd.Series(3)).median())
            pe = pd.to_numeric(df['pe'], errors='coerce').clip(-100, 200).fillna(30)
            df['style_value'] = -(pb + pe)
            df['style_value'] = (df['style_value'] - df['style_value'].mean()) / (df['style_value'].std() + 1e-9)
        else:
            df['style_value'] = 0

        # Growth
        if 'revenue_yoy' in df.columns and 'profit_yoy' in df.columns:
            df['style_growth'] = (pd.to_numeric(df['revenue_yoy'], errors='coerce').fillna(0) +
                                  pd.to_numeric(df['profit_yoy'],  errors='coerce').fillna(0))
            df['style_growth'] = (df['style_growth'] - df['style_growth'].mean()) / (df['style_growth'].std() + 1e-9)
        else:
            df['style_growth'] = 0

        # Momentum（用预计算的 pmt_return_20d，避免groupby symbol/ts_code 不一致）
        if 'pmt_return_20d' in df.columns:
            mom = pd.to_numeric(df['pmt_return_20d'], errors='coerce').fillna(0)
            df['style_momentum'] = (mom - mom.mean()) / (mom.std() + 1e-9)
        elif 'close' in df.columns and code_col:
            df['style_momentum'] = df.groupby(code_col)['close'].pct_change(20).fillna(0)
            df['style_momentum'] = (df['style_momentum'] - df['style_momentum'].mean()) / (df['style_momentum'].std() + 1e-9)
        else:
            df['style_momentum'] = 0

        # ---- 额外风险因子 ----
        # 波动率风险（用预计算 volat_hist_20d 优先）
        if 'volat_hist_20d' in df.columns:
            v = pd.to_numeric(df['volat_hist_20d'], errors='coerce').fillna(0)
            df['volatility_risk'] = (v - v.min()) / (v.max() - v.min() + 1e-9)
        elif 'close' in df.columns and code_col:
            ret = df.groupby(code_col)['close'].pct_change()
            volatility = ret.rolling(20, min_periods=5).std()
            df['volatility_risk'] = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-9)
            df['volatility_risk'] = df['volatility_risk'].fillna(0.5)
        else:
            df['volatility_risk'] = 0.5

        # 流动性风险（用 amount 或 vol*close）
        if amt_col:
            amt = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
            df['liquidity_risk'] = 1 - (amt - amt.min()) / (amt.max() - amt.min() + 1e-9)
        elif vol_col and 'close' in df.columns:
            turnover = (pd.to_numeric(df[vol_col], errors='coerce').fillna(0) *
                        pd.to_numeric(df['close'],  errors='coerce').fillna(0))
            df['liquidity_risk'] = 1 - (turnover - turnover.min()) / (turnover.max() - turnover.min() + 1e-9)
        else:
            df['liquidity_risk'] = 0.5

        # 估值风险
        if 'pe' in df.columns and 'pb' in df.columns:
            pe = pd.to_numeric(df['pe'], errors='coerce').clip(0, 300).fillna(30)
            pb = pd.to_numeric(df['pb'], errors='coerce').clip(0, 30).fillna(3)
            pe_q90 = pe.quantile(0.9) or 1.0
            pb_q90 = pb.quantile(0.9) or 1.0
            df['valuation_risk'] = np.clip((pe / pe_q90 + pb / pb_q90) / 2, 0, 2).fillna(1.0)
        else:
            df['valuation_risk'] = 1.0

        # 综合风险评分
        risk_cols = [c for c in ['volatility_risk', 'liquidity_risk', 'valuation_risk'] if c in df.columns]
        df['综合风险评分'] = df[risk_cols].mean(axis=1) if risk_cols else 0.5

        # ════════════════════════════════════════════════════════════════════
        # V30新增: 估值吸引力风格因子 style_valuation_attractiveness
        # ────────────────────────────────────────────────────────────────────
        # 修改位置: calculate_style_factors 末尾，综合风险评分之后
        # 私募理由:
        #   neutral_score 已做行业+风格中性化，再乘估值因子会破坏中性化效果。
        #   本因子作为独立字段传给 _build_item，仅影响仓位计算，
        #   让低价低估值股自然获得更高仓位，而不改变选股排序。
        #   使用行业内分位（非全局绝对值），消除行业间固有估值差异。
        # ════════════════════════════════════════════════════════════════════
        if 'pe' in df.columns and 'pb' in df.columns:
            _pe = pd.to_numeric(df['pe'], errors='coerce').clip(1, 200).fillna(30)
            _pb = pd.to_numeric(df['pb'], errors='coerce').clip(0.1, 50).fillna(3)
            df['_sva_pe'] = _pe
            df['_sva_pb'] = _pb

            # 行业内分位：同行业横向比较，低PE/PB股获得高吸引力
            if 'industry' in df.columns:
                _pe_rank = df.groupby('industry')['_sva_pe'].rank(pct=True)
                _pb_rank = df.groupby('industry')['_sva_pb'].rank(pct=True)
            else:
                _pe_rank = df['_sva_pe'].rank(pct=True)
                _pb_rank = df['_sva_pb'].rank(pct=True)

            # 分位倒数：低估值分位低 → 倒数大 → 吸引力高
            _inv_pe = 1.0 - _pe_rank.clip(0.01, 0.99)
            _inv_pb = 1.0 - _pb_rank.clip(0.01, 0.99)
            _base   = (_inv_pe + _inv_pb) / 2.0   # 综合估值吸引力 [0,1]

            # 价格区间乘数（仅供仓位计算，不修改 neutral_score）
            if 'close' in df.columns:
                _cl = pd.to_numeric(df['close'], errors='coerce').fillna(30)
                _pm = pd.Series(1.00, index=df.index)
                _pm[_cl < 15]             = 1.40   # 超低价小市值
                _pm[_cl.between(15, 40)]  = 1.25   # 低价优质买入区
                _pm[_cl > 80]             = 0.70   # 高价压制
                _pm[_cl > 150]            = 0.55   # 极高价强压
            else:
                _pm = 1.0

            df['style_valuation_attractiveness'] = (_base * _pm).clip(0.0, 2.0)
            df.drop(columns=['_sva_pe', '_sva_pb'], errors='ignore', inplace=True)

            logger.info(
                f"   style_valuation_attractiveness: "
                f"mean={df['style_valuation_attractiveness'].mean():.3f} | "
                f"P25={df['style_valuation_attractiveness'].quantile(0.25):.3f} | "
                f"P75={df['style_valuation_attractiveness'].quantile(0.75):.3f} "
                f"（行业内分位，不写入neutral_score）"
            )
        else:
            df['style_valuation_attractiveness'] = 0.5
            logger.warning("   ⚠️ PE/PB字段缺失，style_valuation_attractiveness=0.5")

        logger.info("   ✅ 风格因子及风险因子计算完成")
        return df

    # V31新增: fundamental_attractiveness 综合基本面吸引力因子
    # ────────────────────────────────────────────────────────────────────
    # 修改位置: RiskManager 类内，calculate_style_factors 方法之后新增
    # 私募理由:
    #   风险中性化后的 neutral_score 不含基本面质量，
    #   check_neutrality 无法识别"低分高质"和"高分高风险"的结构性偏差。
    #   本方法计算后写入 df['fundamental_attractiveness']，
    #   供 check_neutrality Top100诊断使用，不写入 neutral_score。
    # ════════════════════════════════════════════════════════════════════
    def calculate_fundamental_attractiveness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合基本面吸引力因子 fundamental_attractiveness ∈ [0, 2]。

        公式:
            valuation_base = (1-PE行业分位 + 1-PB行业分位) / 2   →  [0,1]
            growth_quality = clip((ROE/15 + profit_yoy增速加成) / 2, 0, 1)  →  [0,1]
            fundamental_attractiveness = valuation_base × (1 + growth_quality)  →  [0,2]

        字段依赖: pe, pb, roe, profit_yoy（均可缺失，缺失时用保守默认值）
        """
        df = df.copy()
        has_pe   = 'pe'         in df.columns
        has_pb   = 'pb'         in df.columns
        has_roe  = 'roe'        in df.columns
        has_pyoy = 'profit_yoy' in df.columns
        has_ind  = 'industry'   in df.columns

        # ── Step1: 估值基础分 ──────────────────────────────────────────────
        if has_pe and has_pb:
            _pe = pd.to_numeric(df['pe'], errors='coerce').clip(1, 200).fillna(30)
            _pb = pd.to_numeric(df['pb'], errors='coerce').clip(0.1, 50).fillna(3)
            if has_ind:
                _pe_rank = df.groupby('industry')['_fa_pe_tmp'].rank(pct=True) \
                    if '_fa_pe_tmp' in df.columns else \
                    df.assign(_fa_pe_tmp=_pe).groupby('industry')['_fa_pe_tmp'].rank(pct=True)
                # 更简洁写法：
                df['_fa_pe'] = _pe
                df['_fa_pb'] = _pb
                _pe_rank = df.groupby('industry')['_fa_pe'].rank(pct=True)
                _pb_rank = df.groupby('industry')['_fa_pb'].rank(pct=True)
            else:
                df['_fa_pe'] = _pe
                df['_fa_pb'] = _pb
                _pe_rank = df['_fa_pe'].rank(pct=True)
                _pb_rank = df['_fa_pb'].rank(pct=True)
            valuation_base = ((1 - _pe_rank.clip(0.01, 0.99)) +
                              (1 - _pb_rank.clip(0.01, 0.99))) / 2.0
            df.drop(columns=['_fa_pe', '_fa_pb'], errors='ignore', inplace=True)
        else:
            valuation_base = pd.Series(0.5, index=df.index)   # 无估值数据：中性

        # ── Step2: 成长质量分 ──────────────────────────────────────────────
        # ROE评分: ROE=15% 对应 1.0，ROE=0% 对应 0.0，ROE<0 对应负分（clip到0）
        if has_roe:
            roe = pd.to_numeric(df['roe'], errors='coerce').fillna(0)
            roe_score = (roe / 15.0).clip(0.0, 2.0)  # ROE=15%→1.0，30%→2.0
        else:
            roe_score = pd.Series(0.5, index=df.index)

        # 净利润增速加成: yoy>30% → +0.3加分；yoy<-20% → -0.2惩罚
        if has_pyoy:
            pyoy = pd.to_numeric(df['profit_yoy'], errors='coerce').fillna(0)
            growth_bonus = pd.Series(0.0, index=df.index)
            growth_bonus[pyoy >  30] =  0.30
            growth_bonus[pyoy >  60] =  0.50   # 高增速进一步奖励
            growth_bonus[pyoy < -20] = -0.20
            growth_bonus[pyoy < -50] = -0.40   # 大幅下滑重罚
        else:
            growth_bonus = pd.Series(0.0, index=df.index)

        growth_quality = ((roe_score + growth_bonus) / 2.0).clip(0.0, 1.0)

        # ── Step3: 综合吸引力 ─────────────────────────────────────────────
        df['fundamental_attractiveness'] = (valuation_base * (1.0 + growth_quality)).clip(0.0, 2.0)
        # ── V31新增：negative_ratio 惩罚 ──────────────────────────────────
        if 'negative_ratio' in df.columns:
            mask = pd.to_numeric(df['negative_ratio'], errors='coerce') > 0.3
            df.loc[mask, 'fundamental_attractiveness'] *= 0.70
            logger.info(f"   negative_ratio惩罚: 共 {mask.sum()} 只股票负面比例>30%，吸引力×0.70")
        # ───────────────────────────────────────────────────────────────────
        logger.info(
            f"   fundamental_attractiveness: "
            f"mean={df['fundamental_attractiveness'].mean():.3f} | "
            f"P25={df['fundamental_attractiveness'].quantile(0.25):.3f} | "
            f"P75={df['fundamental_attractiveness'].quantile(0.75):.3f}"
        )
        return df
    # ---------- 风格中性化 ----------
    def neutralize_style(self, df, score_col='industry_neutral_score'):
        """风格中性化：对得分进行风格因子回归，取残差"""
        if score_col not in df.columns:
            logger.warning(f"⚠️ 缺少{score_col}列，跳过风格中性化")
            df['style_neutral_score'] = df.get('v19_final_score', 0)
            return df

        logger.info("🎯 风格中性化...")
        df = df.copy()
        df = self.calculate_style_factors(df)

        style_cols = ['style_size', 'style_value', 'style_growth', 'style_momentum']
        available_style = [c for c in style_cols if c in df.columns]
        if not available_style:
            df['style_neutral_score'] = df[score_col]
            return df

        X = df[available_style].fillna(0).values
        y = pd.to_numeric(df[score_col], errors='coerce').fillna(0).values

        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            df['style_neutral_score'] = y - y_pred

            for i, name in enumerate(available_style):
                logger.info(f"     {name:15s}: {model.coef_[i]:7.3f}")
            logger.info("   ✅ 风格中性化完成")
        except Exception as e:
            logger.error(f"   ❌ 风格中性化失败: {e}")
            df['style_neutral_score'] = df[score_col]

        return df

    # ---------- 风险过滤 ----------
    def apply_risk_filters(self, df, score_col='style_neutral_score'):
        """应用风险过滤：ST、流动性、综合风险评分"""
        logger.info("🔍 应用风险过滤...")
        df = df.copy()
        initial_count = len(df)

        # ST股过滤（name列存在时才过滤）
        if 'name' in df.columns:
            df = df[~df['name'].str.contains(r'ST|退市|S\*ST|\*ST', na=False, regex=True)]

        # 流动性过滤（用 amount 或 vol*close，兼容Tushare列名）
        liquidity_checked = False
        if 'amount' in df.columns:
            amt = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            df = df[amt >= self.min_liquidity]
            liquidity_checked = True
        elif 'vol' in df.columns and 'close' in df.columns:
            turnover = (pd.to_numeric(df['vol'],   errors='coerce').fillna(0) *
                        pd.to_numeric(df['close'], errors='coerce').fillna(0))
            df = df[turnover >= self.min_liquidity]
            liquidity_checked = True

        # 综合风险评分过滤（分数 < max_risk_score 才保留）
        if '综合风险评分' in df.columns:
            df = df[df['综合风险评分'] < self.max_risk_score]

        logger.info(f"   风险过滤: {initial_count}→{len(df)} 只股票"
                    + (f" (流动性已过滤)" if liquidity_checked else " (无流动性数据)"))
        return df

    # ---------- 行业集中度调整 ----------
    def adjust_sector_concentration(self, df, score_col='final_score'):
        """如果某行业占比过高，降低该行业得分"""
        if 'industry' not in df.columns or score_col not in df.columns:
            return df

        df = df.copy()
        industry_counts = df['industry'].value_counts(normalize=True)
        top_industry = industry_counts.index[0]
        top_pct = industry_counts.iloc[0]

        if top_pct > self.max_sector_pct:
            logger.warning(f"行业 {top_industry} 占比 {top_pct:.1%} 超过限制 {self.max_sector_pct:.0%}，降低其得分")
            df.loc[df['industry'] == top_industry, score_col] *= 0.7

        return df

    # ---------- 完整流程 ----------
    def full_neutralization(self, df, score_col='v19_final_score'):
        """
        完整中性化流程（行业 + 风格）
        【修复】中性化必须在截面（最新日期）内进行，不能跨历史日期混合，否则中性化失真。
        策略：对最新日截面做中性化，结果映射回全量df。
        """
        if score_col not in df.columns:
            logger.warning(f"缺少得分列 {score_col}，跳过中性化")
            df['neutral_score'] = df.get(score_col, 0)
            return df

        df = df.copy()
        # 只对最新截面做中性化（避免跨日混合）
        if 'trade_date' in df.columns:
            _latest = df['trade_date'].max()
            df_snap = df[df['trade_date'] == _latest].copy()
            _use_snap = True
        else:
            df_snap = df.copy()
            _use_snap = False

        logger.info(f"🎯 全量中性化（截面: {len(df_snap)} 只）...")

        # Step 1: 行业中性化（截面内）
        if 'industry' in df_snap.columns and score_col in df_snap.columns:
            df_snap = self.neutralize_industry(df_snap, score_col)
            score_col_after = 'industry_neutral_score'
        else:
            df_snap['industry_neutral_score'] = df_snap.get(score_col, 0)
            score_col_after = 'industry_neutral_score'

        # Step 2: 风格中性化（截面内）
        style_cols = [c for c in ['style_size', 'style_value', 'style_momentum',
                                   'style_growth', 'style_volatility', 'style_liquidity']
                      if c in df_snap.columns]
        if style_cols and score_col_after in df_snap.columns:
            try:
                X_style = df_snap[style_cols].fillna(0).values
                y_score = df_snap[score_col_after].values
                from scipy import stats
                residuals = y_score.copy()
                for j in range(X_style.shape[1]):
                    sl = X_style[:, j]
                    if sl.std() > 1e-9:
                        slope, intercept, _, _, _ = stats.linregress(sl, y_score)
                        residuals -= slope * sl
                df_snap['neutral_score'] = pd.Series(residuals, index=df_snap.index)
            except Exception as e:
                logger.error(f"  风格中性化失败: {e}")
                df_snap['neutral_score'] = df_snap[score_col_after]
        else:
            df_snap['neutral_score'] = df_snap.get(score_col_after, df_snap.get(score_col, 0))

        # 把截面中性化结果映射回全量df
        if _use_snap and 'ts_code' in df_snap.columns:
            _neutral_map = df_snap.set_index('ts_code')['neutral_score']
            df['neutral_score'] = df['ts_code'].map(_neutral_map)
            # 历史行和未匹配行用 score_col 填充（非最新日无中性化结果）
            _mask_na = df['neutral_score'].isna()
            if _mask_na.any():
                df.loc[_mask_na, 'neutral_score'] = df.loc[_mask_na, score_col]
        else:
            df['neutral_score'] = df_snap.get('neutral_score', df.get(score_col, 0))

        logger.info("   ✅ 全量中性化（截面版）完成")
        return df

# 【修改原因】原版第454行 def check_neutrality 缩进为0（模块级函数），
# self.check_neutrality() 调用时抛 AttributeError。
# 修复：整体缩进4格，作为 RiskManager 实例方法。
# 替换范围：原文件第454行 ~ 第590行（def check_neutrality 整个函数体）

    def check_neutrality(self, df, score_col='neutral_score'):
        """
        检查中性化效果（V31增强版：行业/风格/估值/基本面/情绪五维诊断）
        """
        logger.info("\n📊 中性化效果检查 (V31):")
        report = {}

        # ── 检查1：行业集中度 ──────────────────────────────────────────────
        if 'industry' in df.columns and score_col in df.columns:
            top_stocks = df.nlargest(100, score_col)
            industry_dist = top_stocks['industry'].value_counts()
            max_industry_pct = industry_dist.iloc[0] / len(top_stocks)
            report['max_industry_concentration'] = max_industry_pct
            logger.info(f"  行业集中度: 最大行业占比 {max_industry_pct:.1%}")
            if max_industry_pct > 0.3:
                logger.warning("  ⚠️ 行业集中度较高（>30%）")
            else:
                logger.info("  ✅ 行业集中度良好（<30%）")

        # ── 检查2：风格暴露 ────────────────────────────────────────────────
        style_cols = ['style_size', 'style_value', 'style_growth']
        if all(col in df.columns for col in style_cols) and score_col in df.columns:
            top_stocks = df.nlargest(100, score_col)
            exposures = {col: top_stocks[col].mean() for col in style_cols}
            for name, val in exposures.items():
                report[name] = val
                flag = '✅' if abs(val) < 0.3 else '⚠️'
                logger.info(f"   {name:12s}: {val:7.3f} ({flag})")

        # ── 检查3：估值吸引力分布 ──────────────────────────────────────────
        if 'style_valuation_attractiveness' in df.columns and score_col in df.columns:
            _top = df.nlargest(100, score_col)
            _va_m   = _top['style_valuation_attractiveness'].mean()
            _va_p25 = _top['style_valuation_attractiveness'].quantile(0.25)
            _va_p75 = _top['style_valuation_attractiveness'].quantile(0.75)
            report['va_top100_mean'] = _va_m
            _flag = ('✅ 低估值主导' if _va_m > 0.55 else ('⚠️ 高估值偏多' if _va_m < 0.38 else '📊 均衡'))
            logger.info(
                f"  📊 Top100估值吸引力: "
                f"mean={_va_m:.3f} P25={_va_p25:.3f} P75={_va_p75:.3f}  {_flag}"
            )
            if 'close' in _top.columns:
                _cl = pd.to_numeric(_top['close'], errors='coerce').fillna(0)
                _lp = _cl.between(10, 40).mean()
                _hp = (_cl > 80).mean()
                logger.info(
                    f"  📊 Top100价格区间: "
                    f"10-40元={_lp:.0%} {'✅' if _lp > 0.40 else '⚠️'} | "
                    f">80元={_hp:.0%}   {'✅' if _hp < 0.20 else '⚠️'}"
                )

        # ── 检查4：Top100 基本面质量诊断 ───────────────────────────────────
        if score_col in df.columns:
            _top100 = df.nlargest(100, score_col)

            if 'pe' in _top100.columns:
                _pe_ser     = pd.to_numeric(_top100['pe'], errors='coerce')
                _loss_ratio = (_pe_ser < 0).sum() / max(1, len(_top100))
                _pe_pos_mean = _pe_ser[_pe_ser > 0].mean() if (_pe_ser > 0).any() else float('nan')
                report['loss_stock_ratio_top100'] = _loss_ratio
                _lf = ('✅' if _loss_ratio < 0.10 else
                        ('⚠️ 亏损股偏多' if _loss_ratio < 0.25 else '🔴 亏损股过多'))
                _pe_info = f"| 有效PE均值={_pe_pos_mean:.1f}" if not pd.isna(_pe_pos_mean) else ''
                logger.info(f"  📊 Top100亏损股占比: {_loss_ratio:.0%}  {_lf}  {_pe_info}")

            if 'roe' in _top100.columns:
                _roe_mean = pd.to_numeric(_top100['roe'], errors='coerce').mean()
                report['roe_mean_top100'] = float(_roe_mean) if not pd.isna(_roe_mean) else 0.0
                _rf = '✅' if _roe_mean > 10 else ('📊 中性' if _roe_mean > 5 else '⚠️ 质地偏弱')
                logger.info(f"  📊 Top100 ROE均值: {_roe_mean:.1f}%  {_rf}")

            if 'profit_yoy' in _top100.columns:
                _pyoy_mean = pd.to_numeric(_top100['profit_yoy'], errors='coerce').mean()
                report['profit_yoy_mean_top100'] = float(_pyoy_mean) if not pd.isna(_pyoy_mean) else 0.0
                _pf = ('✅ 成长良好' if _pyoy_mean > 10 else
                        ('📊 平稳' if _pyoy_mean > -5 else '⚠️ 业绩下滑'))
                logger.info(f"  📊 Top100净利增速均值: {_pyoy_mean:.1f}%  {_pf}")

        # ── 检查5：Top100 NLP情绪诊断（V31新增）─────────────────────────────
        if score_col in df.columns:
            _top100 = df.nlargest(100, score_col)

            if 'nlp_score' in _top100.columns:
                _nlp_mean = pd.to_numeric(_top100['nlp_score'], errors='coerce').mean()
                report['nlp_mean_top100'] = float(_nlp_mean) if not pd.isna(_nlp_mean) else 0.0
                _nf = ('✅ 情绪正面' if _nlp_mean > 0.1 else
                        ('📊 情绪中性' if _nlp_mean > -0.1 else '⚠️ 情绪偏负面'))
                logger.info(f"  📊 Top100 NLP情绪均值: {_nlp_mean:.3f}  {_nf}")

            if 'negative_ratio' in _top100.columns:
                _nr_mean = pd.to_numeric(_top100['negative_ratio'], errors='coerce').mean()
                report['negative_ratio_mean_top100'] = float(_nr_mean) if not pd.isna(_nr_mean) else 0.0
                _nrf = ('✅' if _nr_mean < 0.30 else
                         ('⚠️ 负面帖偏多' if _nr_mean < 0.50 else '🔴 负面帖大量'))
                logger.info(f"  📊 Top100股吧负面帖比: {_nr_mean:.0%}  {_nrf}")

            if 'fundamental_attractiveness' in _top100.columns:
                _fa_mean = _top100['fundamental_attractiveness'].mean()
                report['fa_mean_top100'] = float(_fa_mean)
                _faf = ('✅ 基本面优质' if _fa_mean > 0.65 else
                         ('📊 基本面中性' if _fa_mean > 0.40 else '⚠️ 基本面偏弱'))
                logger.info(f"  📊 Top100综合基本面吸引力: {_fa_mean:.3f}  {_faf}")

        # ── 检查6：Top100 veto_reason 统计 ────────────────────────────────
        if score_col in df.columns:
            _top100 = df.nlargest(100, score_col)
            if 'veto_reason' in _top100.columns:
                reasons = _top100['veto_reason'].dropna()
                reasons = reasons[reasons != '']
                if len(reasons) > 0:
                    reason_counts = reasons.value_counts().head(5)
                    logger.info("  📊 Top100 veto_reason 前5:")
                    for reason, count in reason_counts.items():
                        logger.info(f"      {reason}: {count} 次")
                else:
                    logger.info("  📊 Top100 veto_reason: 全部通过，无否决")

        return report

# 全局实例（供主流程调用）
risk_manager = RiskManager()

# 别名（兼容 views.py 中 from .risk_neutralizer import RiskManager）
RiskNeutralizer = RiskManager


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    df = pd.DataFrame({
        'symbol': [f'00000{i}.SZ' for i in range(500)],
        'industry': np.random.choice(['银行', '医药生物', '电子', '食品饮料', '计算机'], 500),
        'total_mv': np.random.uniform(50, 500, 500),
        'pb': np.random.uniform(1, 10, 500),
        'pe': np.random.uniform(10, 50, 500),
        'revenue_yoy': np.random.uniform(-10, 30, 500),
        'profit_yoy': np.random.uniform(-20, 40, 500),
        'close': np.random.uniform(10, 100, 500),
        'v19_final_score': np.random.uniform(0, 1, 500)
    })
    df = risk_manager.full_neutralization(df)
    risk_manager.check_neutrality(df)