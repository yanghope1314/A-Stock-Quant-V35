# -*- coding: utf-8 -*-
"""
止损/止盈/巡检模块 (Exit Strategy) V29
===============================================
私募级退出策略：三层过滤 + 每日巡检
  Layer1 HARD_STOP  → 立即卖出，无条件执行
  Layer2 SOFT_STOP  → 建议减仓50%，次日执行
  Layer3 WATCH      → 预警，加密监控

调用方式（每日收盘后运行）：
  from sell_logic import ExitManager
  em = ExitManager()
  report = em.daily_scan(positions, current_data)
  for signal in report['hard_stops']:
      print(f"❌ {signal['code']} 立即止损: {signal['reason']}")
===============================================
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


# ── 可配置参数（可覆盖）──────────────────────────────────────────
EXIT_CONFIG = {
    # ── Layer1: 硬止损（无条件立即卖出）───────────────────────────
    'hard_stop_loss':        0.08,   # 买入价跌8%触发（机构标准7-10%）
    'hard_time_stop_days':   10,     # 持有>10天且累计收益<-2%，时间止损
    'hard_time_stop_ret':   -0.02,   # 时间止损收益率门槛

    # ── Layer2: 软止损（建议减仓）──────────────────────────────────
    'soft_ma20_break':       True,   # 收盘价跌破MA20（趋势转空）
    'soft_ai_drop':          25,     # AI评分从买入时下降>25分
    'soft_score_drop':       20,     # 综合评分下降>20分
    'soft_high_vol_sell':    0.055,  # 20日波动率超过5.5%（妖股化）

    # ── Layer3: 止盈（动态移动止盈）────────────────────────────────
    'trailing_stop_pct':     0.12,   # 从最高点回撤>12%触发
    'take_profit_fixed':     0.20,   # 固定止盈20%（强制兑现）

    # ── 时间止盈（持有期限）───────────────────────────────────────
    'time_profit_days':      5,      # 持有>5天且收益<1%，强制换股
    'time_profit_ret':       0.01,   # 时间止盈收益率门槛
}


class Position:
    """持仓记录（从前端/数据库传入）"""
    def __init__(self, code: str, buy_price: float, buy_date: str,
                 shares: int = 100, buy_score: float = 70.0,
                 buy_ai_score: float = 60.0):
        self.code       = code
        self.buy_price  = buy_price
        self.buy_date   = datetime.strptime(buy_date, '%Y%m%d') if isinstance(buy_date, str) else buy_date
        self.shares     = shares
        self.buy_score  = buy_score      # 买入时的综合评分
        self.buy_ai_score = buy_ai_score # 买入时的AI评分
        self.peak_price = buy_price      # 持仓期最高价（移动止盈用）


class ExitSignal:
    """退出信号"""
    HARD_STOP = 'HARD_STOP'   # 立即卖出
    SOFT_STOP = 'SOFT_STOP'   # 减仓50%
    TAKE_PROFIT = 'TAKE_PROFIT' # 止盈卖出
    WATCH = 'WATCH'            # 预警观察

    def __init__(self, code: str, signal_type: str, reason: str,
                 current_price: float, buy_price: float,
                 action: str, pnl_pct: float):
        self.code          = code
        self.signal_type   = signal_type
        self.reason        = reason
        self.current_price = current_price
        self.buy_price     = buy_price
        self.action        = action       # '立即清仓' / '减仓50%' / '加密监控'
        self.pnl_pct       = pnl_pct      # 当前盈亏%
        self.ts            = datetime.now().strftime('%Y-%m-%d %H:%M')

    def to_dict(self) -> Dict:
        return {
            'code':          self.code,
            'signal_type':   self.signal_type,
            'reason':        self.reason,
            'current_price': round(self.current_price, 2),
            'buy_price':     round(self.buy_price, 2),
            'pnl_pct':       round(self.pnl_pct * 100, 2),
            'action':        self.action,
            'timestamp':     self.ts,
        }


class ExitManager:
    """
    每日巡检管理器
    ════════════════════════════════════════════
    使用方式：
      em = ExitManager()
      report = em.daily_scan(
          positions=[pos1, pos2, ...],
          current_data={
              '000001': {
                  'close': 12.5,
                  'ma20': 12.0,
                  'ai_score': 55.0,
                  'buy_score': 72.0,
                  'volat_hist_20d': 0.03,
              }
          }
      )
    ════════════════════════════════════════════
    """

    def __init__(self, config: Dict = None):
        self.config = config or EXIT_CONFIG.copy()
        logger.info("📋 ExitManager 初始化完成")

    def _calc_hold_days(self, pos: Position) -> int:
        """计算持仓天数（自然日，实际交易日≈×0.71）"""
        return (datetime.now() - pos.buy_date).days

    def _calc_pnl(self, pos: Position, current_price: float) -> float:
        return (current_price - pos.buy_price) / pos.buy_price

    def check_hard_stop(self, pos: Position, cur: Dict) -> Optional[ExitSignal]:
        """Layer1: 硬止损检查（无条件，优先级最高）"""
        cp    = float(cur.get('close', pos.buy_price))
        pnl   = self._calc_pnl(pos, cp)
        days  = self._calc_hold_days(pos)
        cfg   = self.config

        # ── 止损1: 固定止损 (-8%) ───────────────────────────────────
        if pnl <= -cfg['hard_stop_loss']:
            return ExitSignal(
                pos.code, ExitSignal.HARD_STOP,
                f"触发固定止损: 亏损{pnl*100:.1f}% ≥ {cfg['hard_stop_loss']*100:.0f}%",
                cp, pos.buy_price, '立即清仓', pnl
            )

        # ── 止损2: 时间止损（持有>10天且仍亏损>2%）────────────────────
        if days >= cfg['hard_time_stop_days'] and pnl <= cfg['hard_time_stop_ret']:
            return ExitSignal(
                pos.code, ExitSignal.HARD_STOP,
                f"时间止损: 持有{days}天，收益{pnl*100:.1f}% < {cfg['hard_time_stop_ret']*100:.0f}%",
                cp, pos.buy_price, '立即清仓', pnl
            )
        return None

    def check_trailing_stop(self, pos: Position, cur: Dict) -> Optional[ExitSignal]:
        """移动止盈: 从最高价回撤 > 12% 触发"""
        cp   = float(cur.get('close', pos.buy_price))
        pnl  = self._calc_pnl(pos, cp)
        cfg  = self.config

        # 更新最高价
        if cp > pos.peak_price:
            pos.peak_price = cp

        # 固定止盈 (+20%)
        if pnl >= cfg['take_profit_fixed']:
            return ExitSignal(
                pos.code, ExitSignal.TAKE_PROFIT,
                f"固定止盈: 盈利{pnl*100:.1f}% ≥ {cfg['take_profit_fixed']*100:.0f}%",
                cp, pos.buy_price, '立即清仓', pnl
            )

        # 移动止盈: 从最高点回撤 > 12%
        drawdown = (pos.peak_price - cp) / pos.peak_price
        if pos.peak_price > pos.buy_price * 1.05 and drawdown >= cfg['trailing_stop_pct']:
            return ExitSignal(
                pos.code, ExitSignal.TAKE_PROFIT,
                f"移动止盈: 从最高{pos.peak_price:.2f}回撤{drawdown*100:.1f}% ≥ {cfg['trailing_stop_pct']*100:.0f}%",
                cp, pos.buy_price, '立即清仓', pnl
            )
        return None

    def check_soft_stop(self, pos: Position, cur: Dict) -> Optional[ExitSignal]:
        """Layer2: 软止损（建议减仓50%，次日执行）"""
        cp       = float(cur.get('close', pos.buy_price))
        ma20     = float(cur.get('ma20', cp))
        ai_score = float(cur.get('ai_score', pos.buy_ai_score))
        b_score  = float(cur.get('buy_score', pos.buy_score))
        volat    = float(cur.get('volat_hist_20d', 0.02))
        pnl      = self._calc_pnl(pos, cp)
        days     = self._calc_hold_days(pos)
        cfg      = self.config

        # ── 软止损1: 跌破MA20（趋势转空信号）──────────────────────────
        if cfg['soft_ma20_break'] and cp < ma20 * 0.99:  # 留1%缓冲
            return ExitSignal(
                pos.code, ExitSignal.SOFT_STOP,
                f"跌破MA20: 收盘{cp:.2f} < MA20×0.99={ma20*0.99:.2f}",
                cp, pos.buy_price, '减仓50%', pnl
            )

        # ── 软止损2: AI评分大幅下降（模型信心丧失）──────────────────────
        ai_drop = pos.buy_ai_score - ai_score
        if ai_drop >= cfg['soft_ai_drop']:
            return ExitSignal(
                pos.code, ExitSignal.SOFT_STOP,
                f"AI评分崩塌: 买入时{pos.buy_ai_score:.0f}→当前{ai_score:.0f}（下降{ai_drop:.0f}分）",
                cp, pos.buy_price, '减仓50%', pnl
            )

        # ── 软止损3: 综合评分大幅下降 ─────────────────────────────────
        score_drop = pos.buy_score - b_score
        if score_drop >= cfg['soft_score_drop']:
            return ExitSignal(
                pos.code, ExitSignal.SOFT_STOP,
                f"综合评分下降: 买入{pos.buy_score:.0f}→当前{b_score:.0f}（↓{score_drop:.0f}分）",
                cp, pos.buy_price, '减仓50%', pnl
            )

        # ── 软止损4: 波动率异常（妖股化）─────────────────────────────
        if volat >= cfg['soft_high_vol_sell']:
            return ExitSignal(
                pos.code, ExitSignal.SOFT_STOP,
                f"波动率异常: {volat*100:.1f}% ≥ {cfg['soft_high_vol_sell']*100:.1f}%（妖股风险）",
                cp, pos.buy_price, '减仓50%', pnl
            )

        # ── 软止损5: 时间止盈（持有>5天收益<1%，僵尸股）────────────────
        if days >= cfg['time_profit_days'] and pnl < cfg['time_profit_ret']:
            return ExitSignal(
                pos.code, ExitSignal.SOFT_STOP,
                f"时间止盈: 持有{days}天，收益{pnl*100:.1f}% < {cfg['time_profit_ret']*100:.0f}%（换股）",
                cp, pos.buy_price, '换股（减仓换入更高分股）', pnl
            )
        return None

    def check_watch(self, pos: Position, cur: Dict) -> Optional[ExitSignal]:
        """Layer3: 预警（持续监控，下一日确认）"""
        cp    = float(cur.get('close', pos.buy_price))
        pnl   = self._calc_pnl(pos, cp)
        ma20  = float(cur.get('ma20', cp))
        days  = self._calc_hold_days(pos)

        reasons = []
        # 接近止损线（亏损>5%）
        if pnl <= -0.05:
            reasons.append(f"接近止损(-{abs(pnl)*100:.1f}%)")
        # 接近MA20（距离<2%）
        if cp < ma20 * 1.02 and cp > ma20 * 0.99:
            reasons.append(f"接近MA20（距离{(cp/ma20-1)*100:.1f}%）")
        # 持有超过3天收益<0（时间风险预警）
        if days >= 3 and pnl < 0:
            reasons.append(f"持有{days}天仍亏损")

        if reasons:
            return ExitSignal(
                pos.code, ExitSignal.WATCH,
                "预警: " + "；".join(reasons),
                cp, pos.buy_price, '加密监控，明日复核', pnl
            )
        return None

    def scan_single(self, pos: Position, cur: Dict) -> Optional[ExitSignal]:
        """对单只持仓进行全层扫描（优先级：硬止损>止盈>软止损>预警）"""
        # 容错：cur为空时跳过
        if not cur:
            logger.warning(f"  {pos.code}: 无当日行情数据，跳过巡检")
            return None

        sig = self.check_hard_stop(pos, cur)
        if sig: return sig

        sig = self.check_trailing_stop(pos, cur)
        if sig: return sig

        sig = self.check_soft_stop(pos, cur)
        if sig: return sig

        sig = self.check_watch(pos, cur)
        return sig

    def daily_scan(self, positions: List[Position],
                   current_data: Dict[str, Dict]) -> Dict:
        """
        每日巡检入口
        ════════════════════════════════════════════
        参数:
          positions: 持仓列表（Position对象）
          current_data: {ts_code: {close, ma20, ai_score, buy_score, volat_hist_20d}}
        返回:
          {hard_stops, take_profits, soft_stops, watches, clean_holds, summary}
        ════════════════════════════════════════════
        """
        hard_stops   = []
        take_profits = []
        soft_stops   = []
        watches      = []
        clean_holds  = []

        logger.info(f"\n{'='*60}")
        logger.info(f"📋 每日持仓巡检 ({len(positions)}只持仓) {datetime.now().strftime('%Y-%m-%d')}")
        logger.info(f"{'='*60}")

        for pos in positions:
            code = pos.code
            cur  = current_data.get(code) or current_data.get(code.split('.')[0]) or {}
            sig  = self.scan_single(pos, cur)

            if sig is None:
                clean_holds.append({
                    'code':  code,
                    'pnl_pct': round(self._calc_pnl(pos, float(cur.get('close', pos.buy_price)))*100, 2),
                    'days':  self._calc_hold_days(pos),
                    'status': '✅ 持有安全'
                })
                continue

            d = sig.to_dict()
            if sig.signal_type == ExitSignal.HARD_STOP:
                hard_stops.append(d)
                logger.warning(f"  ❌ HARD_STOP {code}: {sig.reason}")
            elif sig.signal_type == ExitSignal.TAKE_PROFIT:
                take_profits.append(d)
                logger.info(f"  💰 TAKE_PROFIT {code}: {sig.reason}")
            elif sig.signal_type == ExitSignal.SOFT_STOP:
                soft_stops.append(d)
                logger.warning(f"  ⚠️ SOFT_STOP {code}: {sig.reason}")
            elif sig.signal_type == ExitSignal.WATCH:
                watches.append(d)
                logger.info(f"  👀 WATCH {code}: {sig.reason}")

        summary = {
            'scan_date':      datetime.now().strftime('%Y-%m-%d'),
            'total_positions': len(positions),
            'hard_stop_count': len(hard_stops),
            'take_profit_count': len(take_profits),
            'soft_stop_count': len(soft_stops),
            'watch_count':    len(watches),
            'clean_count':    len(clean_holds),
            'action_required': len(hard_stops) + len(take_profits) > 0,
        }

        logger.info(f"\n巡检摘要: 硬止损={len(hard_stops)}, 止盈={len(take_profits)}, "
                    f"软止损={len(soft_stops)}, 预警={len(watches)}, 安全={len(clean_holds)}")
        logger.info("="*60)

        return {
            'hard_stops':   hard_stops,
            'take_profits': take_profits,
            'soft_stops':   soft_stops,
            'watches':      watches,
            'clean_holds':  clean_holds,
            'summary':      summary,
        }


# ── Django视图接口（可选集成到 views.py）─────────────────────────
try:
    from django.views.decorators.csrf import csrf_exempt
    from django.http import JsonResponse
    import json

    exit_manager = ExitManager()

    @csrf_exempt
    def daily_exit_scan(request):
        """
        每日巡检接口
        POST body:
        {
          "positions": [
            {
              "code": "000001",
              "buy_price": 12.5,
              "buy_date": "20250101",
              "shares": 1000,
              "buy_score": 82.0,
              "buy_ai_score": 65.0
            }
          ],
          "current_data": {
            "000001": {
              "close": 13.2,
              "ma20": 12.8,
              "ai_score": 58.0,
              "buy_score": 75.0,
              "volat_hist_20d": 0.025
            }
          }
        }
        """
        try:
            body = json.loads(request.body) if request.body else {}
            raw_positions = body.get('positions', [])
            current_data  = body.get('current_data', {})

            positions = []
            for p in raw_positions:
                try:
                    positions.append(Position(
                        code      = str(p['code']),
                        buy_price = float(p['buy_price']),
                        buy_date  = str(p['buy_date']),
                        shares    = int(p.get('shares', 100)),
                        buy_score = float(p.get('buy_score', 70.0)),
                        buy_ai_score = float(p.get('buy_ai_score', 60.0)),
                    ))
                except (KeyError, ValueError) as pe:
                    logger.warning(f"  持仓数据解析失败: {p} → {pe}")

            if not positions:
                return JsonResponse({'status': 'error', 'message': '无有效持仓数据'})

            report = exit_manager.daily_scan(positions, current_data)
            return JsonResponse({'status': 'success', **report})

        except Exception as e:
            import traceback
            logger.error(f"巡检接口异常: {e}\n{traceback.format_exc()}")
            return JsonResponse({'status': 'error', 'message': str(e)})

except ImportError:
    # 非Django环境，跳过视图定义
    pass


# ── 命令行测试 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    em = ExitManager()

    # 模拟5个持仓场景
    positions = [
        Position("000001", 12.50, "20250115", buy_score=82.0, buy_ai_score=68.0),
        Position("000002", 15.00, "20250110", buy_score=75.0, buy_ai_score=60.0),
        Position("000003", 8.00,  "20250105", buy_score=70.0, buy_ai_score=55.0),
        Position("000004", 20.00, "20250118", buy_score=88.0, buy_ai_score=72.0),
        Position("000005", 5.00,  "20250120", buy_score=65.0, buy_ai_score=50.0),
    ]

    current_data = {
        "000001": {"close": 11.40, "ma20": 12.2, "ai_score": 55.0, "buy_score": 78.0, "volat_hist_20d": 0.028},
        "000002": {"close": 14.85, "ma20": 15.5, "ai_score": 35.0, "buy_score": 55.0, "volat_hist_20d": 0.032},  # AI崩塌
        "000003": {"close": 7.30,  "ma20": 7.80, "ai_score": 50.0, "buy_score": 68.0, "volat_hist_20d": 0.045},  # 止损
        "000004": {"close": 24.00, "ma20": 22.0, "ai_score": 85.0, "buy_score": 90.0, "volat_hist_20d": 0.018},  # 安全
        "000005": {"close": 5.02,  "ma20": 5.10, "ai_score": 48.0, "buy_score": 62.0, "volat_hist_20d": 0.022},  # 预警
    }
    # 000004 设置最高价（模拟移动止盈）
    positions[3].peak_price = 26.0  # 从26元跌到24元，回撤7.7%（<12%，不触发）

    report = em.daily_scan(positions, current_data)

    print("\n══════════ 巡检结果 ══════════")
    for category, label in [("hard_stops","❌硬止损"), ("take_profits","💰止盈"),
                              ("soft_stops","⚠️软止损"), ("watches","👀预警"), ("clean_holds","✅安全")]:
        items = report[category]
        if items:
            print(f"\n{label} ({len(items)}只):")
            for item in items:
                print(f"  {item}")
    print(f"\n{report['summary']}")