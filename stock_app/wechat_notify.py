# -*- coding: utf-8 -*-
"""
私募级微信通知引擎 (WeChat Notification)
==========================================
免费方案：Server酱 (https://sct.ftqq.com/) + PushPlus 双通道
每日免费额度：Server酱 500条，PushPlus 200条

用途：
  from .wechat_notify import WeChatNotifier
  notifier = WeChatNotifier()
  notifier.send_stock_report(stocks, market_timing)

私募标准：
  - 选股结果每日推送（Top 10 + 行业分布）
  - 风险告警实时推送（止损触发/市场大跌/择时变更）
  - 免打扰时段静默
  - 失败自动切换备用通道
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

from .config_v19 import WECHAT_NOTIFY_CONFIG

logger = logging.getLogger(__name__)

# ── 消息模板 ────────────────────────────────────────────────────────
_STOCK_ROW = (
    "{rank}. {name}({code}) | 综合分:{score:.0f} | "
    "涨势:{trend:.0f}抄底:{bottom:.0f} | {sector}"
)

_RISK_ALERT_TEMPLATE = """## {alert_type}
> 触发时间: {time}
> 风险等级: {level}

{detail}

---
*V35 量化选股系统 · 自动推送*"""

_WEEKLY_REPORT_HEADER = """## 本周选股周报
> 报告期间: {date_range}
> 市场状态: {market_regime}
> 择时信号: {timing_signal}

### 市场概况
- 全市场得分中位数: {score_median}
- 选股数量: {selected_count}/{total_count}
- 行业集中度: Top3行业占比 {top3_pct}%

### 本周推荐 Top{top_n}
"""


class WeChatNotifier:
    """微信通知器 — Server酱 + PushPlus 双通道"""

    def __init__(self):
        self.enabled = WECHAT_NOTIFY_CONFIG.get('enable', True)
        self.sendkey = WECHAT_NOTIFY_CONFIG.get('sendkey', '')
        self.quiet_start, self.quiet_end = WECHAT_NOTIFY_CONFIG.get('quiet_hours', [23, 7])
        self.max_stocks = WECHAT_NOTIFY_CONFIG.get('max_stocks_in_msg', 20)

        # PushPlus 备用通道 token（免费注册 pushplus.plus）
        self.pp_token = WECHAT_NOTIFY_CONFIG.get('pushplus_token', '')

        self._last_send = 0.0
        self._send_count_today = 0
        self._today = datetime.now().day

        if not self.sendkey:
            logger.info("微信通知: SendKey 未配置，跳过通知（设置环境变量 SERVERCHAN_SENDKEY 即可启用）")

        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'V35-Quant/3.5 (Private Equity)',
            'Content-Type': 'application/json',
        })

    # ── 免打扰判定 ──────────────────────────────────────────────────
    def _is_quiet_hours(self) -> bool:
        h = datetime.now().hour
        if self.quiet_start > self.quiet_end:
            return h >= self.quiet_start or h < self.quiet_end
        return self.quiet_start <= h < self.quiet_end

    # ── 频率控制 ─────────────────────────────────────────────────────
    def _rate_limit(self, min_interval: float = 3.0) -> bool:
        """两次发送至少间隔 min_interval 秒"""
        now = time.time()
        if now - self._last_send < min_interval:
            return False
        self._last_send = now
        return True

    # ── 核心发送 ─────────────────────────────────────────────────────
    def _send_serverchan(self, title: str, content: str) -> bool:
        """Server酱 主通道"""
        if not self.sendkey:
            return False
        try:
            url = f"https://sctapi.ftqq.com/{self.sendkey}.send"
            resp = self._session.post(url, json={
                'title': title,
                'desp': content,
            }, timeout=10)
            data = resp.json()
            if data.get('code') == 0:
                logger.info(f"微信推送成功 (Server酱): {title}")
                return True
            else:
                logger.warning(f"Server酱返回异常: {data}")
                return False
        except Exception as e:
            logger.error(f"Server酱发送失败: {e}")
            return False

    def _send_pushplus(self, title: str, content: str) -> bool:
        """PushPlus 备用通道"""
        if not self.pp_token:
            return False
        try:
            resp = self._session.post("https://www.pushplus.plus/send", json={
                'token': self.pp_token,
                'title': title,
                'content': content,
                'template': 'markdown',
            }, timeout=10)
            data = resp.json()
            if data.get('code') == 200:
                logger.info(f"微信推送成功 (PushPlus): {title}")
                return True
            return False
        except Exception as e:
            logger.error(f"PushPlus发送失败: {e}")
            return False

    def _send(self, title: str, content: str) -> bool:
        """双通道发送，自动降级"""
        if not self.enabled:
            return False
        if self._is_quiet_hours():
            logger.debug(f"免打扰时段，静默: {title}")
            return False
        if not self._rate_limit():
            logger.debug(f"频率限制，跳过: {title}")
            return False

        # 主通道: Server酱
        if self._send_serverchan(title, content):
            return True
        # 备用通道: PushPlus
        if self._send_pushplus(title, content):
            return True
        logger.warning(f"所有通知通道发送失败: {title}")
        return False

    # ── 公开接口 ─────────────────────────────────────────────────────

    def send_stock_report(
        self,
        stocks: List[Dict],
        market_timing: Optional[Dict] = None,
        stock_pool_name: str = "中证1000",
    ) -> bool:
        """
        每日选股报告推送

        Args:
            stocks: 选股结果列表，每项含 code/name/score/trend_score/bottom_score/sector
            market_timing: 择时信号
            stock_pool_name: 股票池名称
        """
        if not WECHAT_NOTIFY_CONFIG.get('push_daily_signal', True):
            return False

        n = min(len(stocks), self.max_stocks)
        if n == 0:
            return self._send(
                f"V35选股 · {datetime.now().strftime('%m/%d')} · 无符合条件的股票",
                f"> 股票池: {stock_pool_name}\n\n当前市场环境下无满足全部条件的标的，建议观望。"
            )

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        timing_str = "未获取"
        if market_timing:
            regime = market_timing.get('regime', '未知')
            trend_allowed = "允许" if market_timing.get('trend_allowed') else "禁止"
            timing_str = f"{regime} | 趋势信号:{trend_allowed}"

        # 行业统计
        sector_counts: Dict[str, int] = {}
        for s in stocks[:n]:
            sec = s.get('sector', '未知')
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        top_sectors = sorted(sector_counts.items(), key=lambda x: -x[1])[:5]
        sector_lines = "\n".join(f"- {sec}: {cnt}只" for sec, cnt in top_sectors)

        rows = []
        for i, s in enumerate(stocks[:n], 1):
            code = s.get('code', s.get('ts_code', '?'))
            rows.append(_STOCK_ROW.format(
                rank=i,
                name=s.get('name', '?'),
                code=code.split('.')[0] if '.' in str(code) else str(code),
                score=s.get('score', s.get('final_score', 0)),
                trend=s.get('trend_score', 0),
                bottom=s.get('bottom_score', 0),
                sector=s.get('sector', s.get('industry', '?')),
            ))

        content = f"""## 每日选股报告
> 时间: {now_str}
> 股票池: {stock_pool_name}
> 市场择时: {timing_str}

### 行业分布
{sector_lines}

### Top {n} 推荐
{chr(10).join(rows)}

---
*V35 量化选股系统 · 自动推送 · 仅供参考不构成投资建议*"""

        title = f"V35选股 {datetime.now().strftime('%m/%d')} · Top{n} · {stock_pool_name}"
        return self._send(title, content)

    def send_risk_alert(
        self,
        alert_type: str,
        detail: str,
        level: str = "⚠️ 中",
    ) -> bool:
        """
        风险告警推送

        Args:
            alert_type: 告警类型（止损触发/市场大跌/流动性危机等）
            detail: 详细信息
            level: 风险等级
        """
        if not WECHAT_NOTIFY_CONFIG.get('push_risk_alert', True):
            return False

        title = f"🚨 {alert_type} · V35风控"
        content = _RISK_ALERT_TEMPLATE.format(
            alert_type=alert_type,
            time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            level=level,
            detail=detail,
        )
        return self._send(title, content)

    def send_market_timing_change(
        self,
        old_regime: str,
        new_regime: str,
        market_score: float,
        detail: str = "",
    ) -> bool:
        """
        择时信号变更推送

        Args:
            old_regime: 旧市场状态
            new_regime: 新市场状态
            market_score: 市场评分
            detail: 补充说明
        """
        if not WECHAT_NOTIFY_CONFIG.get('push_market_timing', True):
            return False

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        content = f"""## 择时信号变更
> 时间: {now_str}

- 旧状态: {old_regime}
- 新状态: {new_regime}
- 市场评分: {market_score:.2f}
- 趋势交易: {"允许" if market_score > 0 else "禁止"}

{detail}

---
*V35 量化选股系统 · 择时信号*"""

        title = f"择时变更: {old_regime} → {new_regime}"
        return self._send(title, content)

    def send_weekly_report(
        self,
        date_range: str,
        stocks: List[Dict],
        market_regime: str,
        timing_signal: str,
        stats: Optional[Dict] = None,
    ) -> bool:
        """
        每周选股周报推送
        """
        if not WECHAT_NOTIFY_CONFIG.get('push_weekly_report', True):
            return False

        stats = stats or {}
        top_n = min(len(stocks), 10)

        rows = []
        for i, s in enumerate(stocks[:top_n], 1):
            code = s.get('code', s.get('ts_code', '?'))
            rows.append(_STOCK_ROW.format(
                rank=i,
                name=s.get('name', '?'),
                code=str(code).split('.')[0] if '.' in str(code) else str(code),
                score=s.get('score', s.get('final_score', 0)),
                trend=s.get('trend_score', 0),
                bottom=s.get('bottom_score', 0),
                sector=s.get('sector', s.get('industry', '?')),
            ))

        content = _WEEKLY_REPORT_HEADER.format(
            date_range=date_range,
            market_regime=market_regime,
            timing_signal=timing_signal,
            score_median=stats.get('score_median', 'N/A'),
            selected_count=stats.get('selected_count', len(stocks)),
            total_count=stats.get('total_count', '?'),
            top3_pct=stats.get('top3_pct', 'N/A'),
            top_n=top_n,
        ) + "\n".join(rows) + "\n\n---\n*V35 量化选股系统 · 周报 · 仅供参考*"

        title = f"V35周报 {date_range} · Top{top_n}"
        return self._send(title, content)

    def send_simple(self, title: str, content: str) -> bool:
        """通用发送接口"""
        return self._send(title, content)


# ── 模块级便捷函数 ──────────────────────────────────────────────────
_notifier: Optional[WeChatNotifier] = None


def get_notifier() -> WeChatNotifier:
    global _notifier
    if _notifier is None:
        _notifier = WeChatNotifier()
    return _notifier
