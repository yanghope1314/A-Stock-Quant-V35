# -*- coding: utf-8 -*-
"""
动态权重优化器 + Alpha衰减监控（私募级合并版）
================================================
功能：
- 动态权重优化（基于IC、市场状态）
- Alpha衰减监控（IC/IR跟踪、自动降权）
- 因子拥挤度检测
- 权重平滑与约束
================================================
"""

import numpy as np
import pandas as pd
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AlphaDecayMonitor:
    """
    Alpha衰减监控器（私募级）
    跟踪因子IC历史，检测衰减，提供调整建议
    """

    def __init__(self,
                 ic_threshold: float = 0.02,      # IC阈值
                 ir_threshold: float = 0.3,       # IR阈值
                 lookback_periods: int = 20,       # 回看期数
                 decay_periods: int = 3,           # 连续衰减期数
                 min_weight: float = 0.01):        # 最小权重
        self.ic_threshold = ic_threshold
        self.ir_threshold = ir_threshold
        self.lookback_periods = lookback_periods
        self.decay_periods = decay_periods
        self.min_weight = min_weight

        # 历史IC记录 {factor_name: deque([ic1, ic2, ...])}
        self.ic_history = defaultdict(lambda: deque(maxlen=lookback_periods))

    def update_ic(self, factor_name: str, ic_value: float):
        """更新因子IC历史"""
        self.ic_history[factor_name].append(ic_value)

    def calculate_ir(self, factor_name: str) -> float:
        """计算信息比率 IR = mean(IC) / std(IC)"""
        ic_list = list(self.ic_history.get(factor_name, []))
        if len(ic_list) < 2:
            return 0.0
        mean_ic = np.mean(ic_list)
        std_ic = np.std(ic_list)
        return mean_ic / (std_ic + 1e-9)

    def check_decay(self, factor_name: str) -> Dict:
        """
        检查因子是否衰减
        返回：{'is_decaying': bool, 'reason': str, 'action': str}
        """
        ic_list = list(self.ic_history.get(factor_name, []))
        if len(ic_list) < self.decay_periods:
            return {'is_decaying': False, 'reason': '数据不足', 'action': 'keep'}

        recent_ics = ic_list[-self.decay_periods:]
        continuous_decay = all(ic < self.ic_threshold for ic in recent_ics)

        ir = self.calculate_ir(factor_name)
        low_ir = ir < self.ir_threshold

        if continuous_decay:
            return {
                'is_decaying': True,
                'reason': f'连续{self.decay_periods}期IC<{self.ic_threshold:.3f}',
                'action': 'reduce',
                'recent_ic': np.mean(recent_ics),
                'ir': ir
            }
        elif low_ir:
            return {
                'is_decaying': True,
                'reason': f'IR={ir:.2f}<{self.ir_threshold:.2f}',
                'action': 'reduce',
                'recent_ic': np.mean(recent_ics),
                'ir': ir
            }
        else:
            return {
                'is_decaying': False,
                'reason': '健康',
                'action': 'keep',
                'recent_ic': np.mean(recent_ics),
                'ir': ir
            }

    def adjust_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """
        根据衰减检测调整权重（将衰减因子权重降至 min_weight，然后重新归一化）
        """
        adjusted = raw_weights.copy()
        actions = []

        for factor, w in raw_weights.items():
            decay_info = self.check_decay(factor)
            if decay_info['is_decaying']:
                old = w
                adjusted[factor] = self.min_weight
                actions.append({
                    'factor': factor,
                    'old': old,
                    'new': self.min_weight,
                    'reason': decay_info['reason']
                })
                logger.warning(f"  衰减因子 {factor}: {old:.1%} → {self.min_weight:.1%}，原因：{decay_info['reason']}")

        # 重新归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        else:
            n = len(adjusted)
            adjusted = {k: 1.0 / n for k in adjusted}

        if actions:
            logger.info(f"  权重调整完成，共处理 {len(actions)} 个衰减因子")
        return adjusted

    def detect_crowding(self, factor_matrix: np.ndarray, threshold: float = 0.7) -> List[tuple]:
        """
        检测因子拥挤度（因子间高相关性）
        返回高相关因子对列表
        """
        if factor_matrix.shape[1] < 2:
            return []
        corr = np.corrcoef(factor_matrix.T)
        n = corr.shape[0]
        crowded = []
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr[i, j]) > threshold:
                    crowded.append((i, j, corr[i, j]))
        if crowded:
            logger.warning(f"发现 {len(crowded)} 对高相关因子（相关性>{threshold}）")
        return crowded

    def get_health_report(self) -> Dict:
        """生成因子健康报告"""
        report = {}
        for factor, ic_deque in self.ic_history.items():
            ic_list = list(ic_deque)
            if not ic_list:
                continue
            decay_info = self.check_decay(factor)
            report[factor] = {
                'ic_mean': np.mean(ic_list),
                'ic_std': np.std(ic_list),
                'ir': self.calculate_ir(factor),
                'is_decaying': decay_info['is_decaying'],
                'reason': decay_info['reason'],
                'periods': len(ic_list)
            }
        return report


class DynamicWeightOptimizer:
    """
    动态权重优化器（含衰减监控）
    1. IC加权 + 波动率调整
    2. 市场状态划分
    3. Alpha衰减监控与调整
    4. 权重平滑与约束
    """

    def __init__(
        self,
        min_weight: float = 0.03,
        max_weight: float = 0.60,
        lookback_periods: int = 20,
        weight_smooth: float = 0.5,
        baseline_weights: Optional[Dict[str, float]] = None,
        # 衰减监控参数
        ic_threshold: float = 0.02,
        ir_threshold: float = 0.3,
        decay_periods: int = 3,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.lookback_periods = lookback_periods
        self.weight_smooth = weight_smooth

        # ════════════════════════════════════════════════════════════════
        # 基准权重（V28修复：覆盖views.py实际传入的所有factor_data Key）
        # 之前的硬编码缺少 'dual'/'rule_baseline' → 这些Key在DWO中fallback=0.02
        # 导致 dual（含动量成分）权重随机化，在市场均涨时暗中放大追高
        # 修复：① 补全所有已知Key；② optimize_weights中对未知Key给保守默认值
        # ════════════════════════════════════════════════════════════════
        if baseline_weights is None:
            self.baseline_weights = {
                # ── views.py factor_mapping实际输出的Key ───────────────
                'rule_baseline':            0.25,   # 规则因子汇总（最稳定，保持高权重）
                'small_cap':                0.30,   # 小市值因子
                'xgboost':                  0.15,   # XGBoost预测
                'ai':                       0.12,   # AI模型（GNN/MLP等集成）
                'dual':                     0.08,   # dual=0.6*trend+0.4*bottom（含动量，保守权重）
                'nlp':                      0.05,   # NLP情绪
                # ════════════════════════════════════════════════════════
                # V31新增: valuation_attractiveness
                # 私募理由: A股估值因子长期IC约0.04，高于动量(0.02-0.03)
                #   基础权重0.10，均衡市场可动态提升到0.15
                # ════════════════════════════════════════════════════════
                'valuation_attractiveness': 0.10,   # V31: 综合估值吸引力因子
                # ── 扩展Key（若不存在则忽略）─────────────────────────────
                'xgnn':                     0.03,
                'original':                 0.01,
                'advanced_factors':         0.01,
            }
        else:
            self.baseline_weights = baseline_weights.copy()

        # V28：动态兜底机制 — optimize_weights遇到未知Key时用保守权重而非0.02
        # 0.02 对 dual 这类动量因子太高（会在无IC历史时给予不合理权重）
        self._default_weight_conservative = 0.01   # 未知Key的保守兜底

        # IC历史（用于计算IC权重）
        self.ic_history = defaultdict(list)
        self.last_weights = None

        # 衰减监控器
        self.decay_monitor = AlphaDecayMonitor(
            ic_threshold=ic_threshold,
            ir_threshold=ir_threshold,
            lookback_periods=lookback_periods,
            decay_periods=decay_periods,
            min_weight=min_weight
        )

        logger.info(f"✅ 动态权重优化器（含衰减监控）初始化 | 平滑系数={weight_smooth}")

    def calculate_factor_ic(self, factor_scores: pd.DataFrame, future_returns: pd.Series) -> Dict[str, float]:
        """
        计算截面IC（Spearman），并更新衰减监控器的IC历史
        """
        ic_dict = {}
        for factor in factor_scores.columns:
            try:
                f = factor_scores[factor].clip(
                    factor_scores[factor].quantile(0.01),
                    factor_scores[factor].quantile(0.99)
                )
                ic = f.corr(future_returns, method='spearman')
                if np.isnan(ic):
                    ic = 0.0

                # 更新IC历史（用于动态权重）
                self.ic_history[factor].append(ic)
                if len(self.ic_history[factor]) > self.lookback_periods:
                    self.ic_history[factor] = self.ic_history[factor][-self.lookback_periods:]

                # 更新衰减监控器的IC历史
                self.decay_monitor.update_ic(factor, ic)

                ic_dict[factor] = ic
            except Exception as e:
                logger.warning(f"{factor} IC计算失败: {e}")
                ic_dict[factor] = 0.0

        return ic_dict

    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """市场状态检测（简化版，依赖size_quantile和ret_20）"""
        try:
            if all(c in market_data.columns for c in ['size_quantile', 'ret_20']):
                small_ret = market_data[market_data['size_quantile'] <= 0.3]['ret_20'].mean()
                large_ret = market_data[market_data['size_quantile'] >= 0.7]['ret_20'].mean()
                vol = market_data['ret_20'].std()
                ret_spread = small_ret - large_ret

                if ret_spread > 0.025:
                    return 'small_cap_rally'
                elif ret_spread < -0.025:
                    return 'large_cap_rally'
                elif vol > 0.045:
                    return 'volatile'
                else:
                    return 'balanced'
            return 'balanced'
        except:
            return 'balanced'

    def optimize_weights(self, factor_data: Dict[str, 'pd.Series'],
                         market_regime: str = 'balanced') -> Dict[str, float]:
        """
        主优化流程：
        1. 从IC历史计算IC权重（EWMA衰减）
        2. 结合基准权重（70% IC + 30% 基准）
        3. 市场状态调整
        4. 平滑
        5. 约束 + 归一化
        6. 衰减监控调整（将衰减因子降权，但rule_baseline受保护）
        """
        logger.info("⚖️  开始机构级权重优化...")

        available_factors = list(factor_data.keys())
        if not available_factors:
            return {}

        # ---------- 1. IC 加权（EWMA衰减） ----------
        # V28：dual因子含动量成分，在高位市场(bear/high_vol)需要额外降权
        _dual_regime_cap = 0.12  # 正常市场dual最高12%
        if market_regime in ('high_vol_bear', 'bear', 'high_vol_bull'):
            _dual_regime_cap = 0.06  # 高位/熊市/高波：动量因子上限收紧到6%

        ic_weights = {}
        for f in available_factors:
            if f == 'rule_baseline':
                # rule_baseline永远给基准权重，不参与IC计算（因为它是规则因子汇总）
                ic_weights[f] = self.baseline_weights.get(f, 0.20)
                continue
            ic_list = self.ic_history.get(f, [])
            if len(ic_list) >= 3:
                weights = np.exp(np.arange(len(ic_list)) / 10)
                weights = weights / weights.sum()
                ic_weights[f] = max(0.01, np.average(ic_list, weights=weights))
            else:
                # 无IC历史时用基准权重（V28：未知Key用保守兜底而非0.02）
                _fb = self.baseline_weights.get(f, self._default_weight_conservative)
                ic_weights[f] = max(self._default_weight_conservative, _fb)

        total_ic = sum(ic_weights.values())
        if total_ic > 1e-6:
            ic_weights = {k: v / total_ic for k, v in ic_weights.items()}

        # ---------- 2. 基准权重（仅保留可用因子）----------
        base = {f: self.baseline_weights.get(f, 0.02) for f in available_factors}
        base_total = sum(base.values())
        if base_total > 0:
            base = {k: v / base_total for k, v in base.items()}

        # ---------- 3. 综合：70% IC + 30% 基准 ----------
        raw_weights = {f: 0.7 * ic_weights[f] + 0.3 * base[f] for f in available_factors}

        # V28：dual因子上限（含动量，高位市场需收紧）
        if 'dual' in raw_weights:
            raw_weights['dual'] = min(raw_weights['dual'], _dual_regime_cap)

        # ---------- 3.5 因子拥挤度调整（接入已有detect_crowding方法）----------
        # 私募标准：若两个因子相关性>0.75，说明持仓高度重叠（拥挤），
        # 同时给高权重等于持仓集中 → 各降权20%，减少有效暴露重复
        if len(factor_data) >= 3:
            try:
                _factor_keys = list(factor_data.keys())
                _fmat = np.column_stack([
                    factor_data[k].fillna(0).values
                    if hasattr(factor_data[k], 'fillna')
                    else np.array(factor_data[k], dtype=float)
                    for k in _factor_keys
                ])
                # 修改此处：self.detect_crowding → self.decay_monitor.detect_crowding
                _crowded_pairs = self.decay_monitor.detect_crowding(_fmat, threshold=0.75)
                for _ci, _cj, _corr in _crowded_pairs:
                    _fi, _fj = _factor_keys[_ci], _factor_keys[_cj]
                    # 只降权非保护因子（rule_baseline受保护）
                    if _fi != 'rule_baseline':
                        raw_weights[_fi] = raw_weights.get(_fi, 0) * 0.80
                    if _fj != 'rule_baseline':
                        raw_weights[_fj] = raw_weights.get(_fj, 0) * 0.80
                    logger.info(
                        f"  ⚠️ 因子拥挤: {_fi}↔{_fj} 相关={_corr:.2f} → 各降权20%"
                    )
            except Exception as _ce:
                logger.warning(f"  因子拥挤度检测失败（跳过）: {_ce}")

        # ---------- 4. 市场状态调整 ----------
        raw_weights = self._adjust_by_market_regime(raw_weights, market_regime)

        # ---------- 5. 平滑 ----------
        if self.last_weights is not None:
            for f in available_factors:
                old = self.last_weights.get(f, raw_weights[f])
                raw_weights[f] = self.weight_smooth * old + (1 - self.weight_smooth) * raw_weights[f]

        # ---------- 6. 约束 + 归一化 ----------
        normalized = self._normalize_weights(raw_weights)

        # ---------- 7. 衰减监控调整（rule_baseline受保护）----------
        protected = {'rule_baseline'}  # 受保护因子不参与衰减降权
        unprotected = {k: v for k, v in normalized.items() if k not in protected}
        protect_part = {k: v for k, v in normalized.items() if k in protected}

        if unprotected:
            adjusted_unprotected = self.decay_monitor.adjust_weights(unprotected)
            # 合并：保护因子保持，非保护因子衰减调整后归一化
            total_protect_w = sum(protect_part.values())
            remaining_w = 1.0 - total_protect_w
            if remaining_w > 0:
                final_weights = {k: v * remaining_w for k, v in adjusted_unprotected.items()}
                final_weights.update(protect_part)
            else:
                final_weights = protect_part
        else:
            final_weights = normalized

        self.last_weights = final_weights.copy()

        logger.info("✅ 权重优化完成（含衰减调整）")
        for k, v in sorted(final_weights.items()):
            logger.info(f"   {k:<18}: {v:.4f}")

        return final_weights

    def _adjust_by_market_regime(self, w: Dict[str, float], regime: str) -> Dict[str, float]:
            """
            根据市场状态调整因子权重（V31：新增 valuation_attractiveness 动态倾斜）

            修改位置: _adjust_by_market_regime 全函数替换
            """
            adj = w.copy()

            if regime == 'small_cap_rally':
                # 小盘行情：重点提升 small_cap，AI模型对小盘预测精度更高
                if 'small_cap' in adj:
                    _old = adj['small_cap']
                    adj['small_cap'] = min(0.55, adj['small_cap'] * 1.40)
                    logger.info(
                        f"  📈 small_cap权重提升: {_old:.4f}→{adj['small_cap']:.4f} (×1.40) "
                        f"[市场状态: small_cap_rally]"
                    )
                # ════════════════════════════════════════
                # V31: small_cap_rally 时估值因子适度降权
                # 私募理由: 小盘行情中低估值修复往往已完成，追估值效果差
                #   同时避免 small_cap+valuation 双重拥挤
                # ════════════════════════════════════════
                if 'valuation_attractiveness' in adj:
                    _old_va = adj['valuation_attractiveness']
                    adj['valuation_attractiveness'] = adj['valuation_attractiveness'] * 0.85
                    logger.info(
                        f"  📉 valuation_attractiveness权重降低: "
                        f"{_old_va:.4f}→{adj['valuation_attractiveness']:.4f} (×0.85) "
                        f"[small_cap_rally时避免因子拥挤]"
                    )
                if 'ai'       in adj: adj['ai']       *= 1.10
                if 'xgnn'     in adj: adj['xgnn']     *= 1.10
                if 'original' in adj: adj['original'] *= 0.80
                logger.info("  📊 市场状态[small_cap_rally]: small_cap↑×1.40 | va↓×0.85 | ai↑×1.10 | original↓×0.80")

            elif regime == 'balanced':
                # 均衡市场：small_cap温和强化 + valuation_attractiveness主动强化
                if 'small_cap' in adj:
                    _old = adj['small_cap']
                    adj['small_cap'] = min(0.40, adj['small_cap'] * 1.10)
                    logger.info(
                        f"  📊 small_cap权重微提: {_old:.4f}→{adj['small_cap']:.4f} (×1.10) "
                        f"[市场状态: balanced]"
                    )
                # ════════════════════════════════════════════════════════════════
                # V31新增: 均衡市场估值因子主动强化
                # 私募理由: 均衡市场是估值修复行情的主战场，IC最稳定（约0.04-0.06）
                #   baseline 0.10 × 1.25 = 0.125，保守但高于动量因子权重
                # ════════════════════════════════════════════════════════════════
                if 'valuation_attractiveness' in adj:
                    _old_va = adj['valuation_attractiveness']
                    adj['valuation_attractiveness'] = min(0.18, adj['valuation_attractiveness'] * 1.25)
                    logger.info(
                        f"  📈 valuation_attractiveness权重提升: "
                        f"{_old_va:.4f}→{adj['valuation_attractiveness']:.4f} (×1.25) "
                        f"[balanced市场估值修复行情]"
                    )
                if 'rule_baseline' in adj:
                    adj['rule_baseline'] = min(0.35, adj['rule_baseline'] * 1.05)
                logger.info(
                    "  📊 市场状态[balanced]: small_cap↑×1.10 | "
                    "valuation_attractiveness↑×1.25 | rule_baseline↑×1.05"
                )

            elif regime == 'large_cap_rally':
                # 大盘行情：small_cap降权；估值因子此时对大盘股有效，保持不变
                if 'small_cap' in adj:
                    _old = adj['small_cap']
                    adj['small_cap'] *= 0.75
                    logger.info(
                        f"  📉 small_cap权重降低: {_old:.4f}→{adj['small_cap']:.4f} (×0.75) "
                        f"[市场状态: large_cap_rally]"
                    )
                # V31: 大盘行情估值因子对大盘蓝筹有效，温和提升
                if 'valuation_attractiveness' in adj:
                    adj['valuation_attractiveness'] = min(0.15, adj['valuation_attractiveness'] * 1.10)
                    logger.info(
                        f"  📈 valuation_attractiveness: ×1.10 [large_cap_rally，蓝筹估值修复]"
                    )
                if 'xgboost' in adj: adj['xgboost'] *= 1.10
                if 'nlp'     in adj: adj['nlp']     *= 1.10
                logger.info("  📊 市场状态[large_cap_rally]: small_cap↓×0.75 | va↑×1.10 | xgboost↑ | nlp↑")

            elif regime == 'volatile':
                # 高波市：small_cap降权，估值因子也降权（价值陷阱增多，低PE可能是下滑前兆）
                if 'small_cap' in adj:
                    _old = adj['small_cap']
                    adj['small_cap'] *= 0.85
                    logger.info(
                        f"  ⚠️  small_cap权重降低: {_old:.4f}→{adj['small_cap']:.4f} (×0.85) "
                        f"[市场状态: volatile，高波市控小票风险]"
                    )
                # ════════════════════════════════════════
                # V31: volatile 时估值因子适度降权
                # 私募理由: 高波动期间价值陷阱增多，低PE可能伴随业绩下滑
                # ════════════════════════════════════════
                if 'valuation_attractiveness' in adj:
                    _old_va = adj['valuation_attractiveness']
                    adj['valuation_attractiveness'] *= 0.80
                    logger.info(
                        f"  ⚠️  valuation_attractiveness: "
                        f"{_old_va:.4f}→{adj['valuation_attractiveness']:.4f} (×0.80) "
                        f"[volatile，价值陷阱风险↑]"
                    )
                if 'xgboost'          in adj: adj['xgboost']          *= 1.15
                if 'nlp'              in adj: adj['nlp']              *= 0.85
                if 'advanced_factors' in adj: adj['advanced_factors'] *= 1.10
                logger.info(
                    "  📊 市场状态[volatile]: small_cap↓×0.85 | va↓×0.80 | "
                    "xgboost↑×1.15 | nlp↓×0.85 | advanced_factors↑×1.10"
                )

            elif regime in ('bear', 'high_vol_bear'):
                # ════════════════════════════════════════
                # V31: 熊市/高波熊 估值因子大幅降权
                # 私募理由: 熊市中"便宜货"往往继续跌，价值陷阱最多
                #   此时 rule_baseline 和防御性因子更可靠
                # ════════════════════════════════════════
                if 'valuation_attractiveness' in adj:
                    _old_va = adj['valuation_attractiveness']
                    adj['valuation_attractiveness'] *= 0.65
                    logger.info(
                        f"  🔴 valuation_attractiveness: "
                        f"{_old_va:.4f}→{adj['valuation_attractiveness']:.4f} (×0.65) "
                        f"[{regime}，熊市价值陷阱风险高]"
                    )
                logger.info(
                    f"  📊 市场状态[{regime}]: va↓×0.65 | "
                    f"其他因子风控由veto和position_size层处理"
                )

            else:
                # high_vol_bull / neutral 等
                logger.info(f"  📊 市场状态[{regime}]: 权重不调整（风控由veto和position_size层处理）")

            return adj

    def _normalize_weights(self, w: Dict[str, float]) -> Dict[str, float]:
        """权重裁剪 + 归一化"""
        for f in w:
            w[f] = np.clip(w[f], self.min_weight, self.max_weight)
        total = sum(w.values())
        if total < 1e-6:
            n = len(w)
            return {f: 1 / n for f in w}
        return {f: v / total for f, v in w.items()}