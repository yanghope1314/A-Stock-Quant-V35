# -*- coding: utf-8 -*-
"""
交易成本模拟模块（私募级）
功能：
- 印花税、佣金、滑点计算
- 流动性约束检查
- 策略容量估算
- 回测成本调整
===============================================
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """交易成本模型"""

    def __init__(self,
                 stamp_duty=0.001,      # 印花税0.1%（仅卖出）
                 commission=0.0002,     # 佣金0.02%（双向）
                 min_commission=5,       # 最低佣金5元
                 slippage_bps=5,         # 滑点5bp
                 liquidity_limit=0.01):  # 单股持仓 ≤ 日均成交额 * liquidity_limit
        self.stamp_duty = stamp_duty
        self.commission = commission
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps / 10000
        self.liquidity_limit = liquidity_limit
        logger.info(f"💰 交易成本模型初始化: 印花税={stamp_duty:.2%}, 佣金={commission:.3%}, 滑点={slippage_bps}bp")

    def calculate_commission(self, trade_amount):
        """计算佣金"""
        commission = trade_amount * self.commission
        return max(commission, self.min_commission)

    def calculate_slippage(self, price, volume, daily_volume):
        """计算滑点（基础滑点 + 冲击成本）"""
        base = price * self.slippage_bps
        if daily_volume > 0:
            volume_ratio = volume / daily_volume
            impact = price * 0.1 * np.sqrt(volume_ratio)  # 冲击成本模型
        else:
            impact = 0
        return base + impact

    def calculate_buy_cost(self, price, shares, daily_volume):
        """买入成本 = 佣金 + 滑点"""
        amount = price * shares
        commission = self.calculate_commission(amount)
        slippage = self.calculate_slippage(price, shares, daily_volume) * shares
        return commission + slippage

    def calculate_sell_cost(self, price, shares, daily_volume):
        """卖出成本 = 佣金 + 印花税 + 滑点"""
        amount = price * shares
        commission = self.calculate_commission(amount)
        stamp = amount * self.stamp_duty
        slippage = self.calculate_slippage(price, shares, daily_volume) * shares
        return commission + stamp + slippage

    def check_liquidity_constraint(self, df, position_size):
        """
        检查流动性约束：单股持仓 ≤ 日均成交额 * liquidity_limit
        兼容 Tushare 列名（ts_code / vol，以及 amount）
        """
        df = df.copy()
        # 列名适配：ts_code/symbol, vol/volume, amount
        code_col  = 'ts_code'  if 'ts_code'  in df.columns else ('symbol' if 'symbol' in df.columns else None)
        vol_col   = 'vol'      if 'vol'      in df.columns else ('volume' if 'volume' in df.columns else None)
        amt_col   = 'amount'   if 'amount'   in df.columns else None

        if amt_col:
            df['avg_amount'] = df.groupby(code_col)[amt_col].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            ) if code_col else pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        elif vol_col and 'close' in df.columns:
            df['_calc_amount'] = (pd.to_numeric(df[vol_col],   errors='coerce').fillna(0) *
                                  pd.to_numeric(df['close'],   errors='coerce').fillna(0))
            if code_col:
                df['avg_amount'] = df.groupby(code_col)['_calc_amount'].transform(
                    lambda x: x.rolling(20, min_periods=5).mean()
                )
            else:
                df['avg_amount'] = df['_calc_amount']
            df.drop(columns=['_calc_amount'], errors='ignore', inplace=True)
        else:
            logger.warning("⚠️ 缺少成交额数据（amount/vol），无法检查流动性，全部标记通过")
            df['liquidity_ok'] = True
            return df

        df['avg_amount'] = pd.to_numeric(df['avg_amount'], errors='coerce').fillna(0)
        df['liquidity_ok'] = position_size <= (df['avg_amount'] * self.liquidity_limit)
        passed = df['liquidity_ok'].sum()
        logger.info(f"   流动性检查: {passed}/{len(df)} ({passed/max(len(df),1):.1%}) 通过")
        return df

    def apply_to_backtest(self, backtest_results):
        """对回测结果应用交易成本"""
        logger.info("\n💰 应用交易成本至回测结果...")
        raw_returns = backtest_results.get('returns', [])
        trades = backtest_results.get('trades', [])
        if not raw_returns or not trades:
            logger.warning("⚠️ 缺少收益或交易数据")
            return backtest_results

        total_cost = 0
        for trade in trades:
            action = trade.get('action')
            price = trade.get('price', 0)
            shares = trade.get('shares', 0)
            daily_vol = trade.get('daily_volume', shares * 10)
            if action == 'buy':
                cost = self.calculate_buy_cost(price, shares, daily_vol)
            elif action == 'sell':
                cost = self.calculate_sell_cost(price, shares, daily_vol)
            else:
                cost = 0
            total_cost += cost

        capital = backtest_results.get('initial_capital', 1e6)
        cost_drag = total_cost / capital
        adj_returns = [r - cost_drag for r in raw_returns]

        backtest_results['returns_after_cost'] = adj_returns
        backtest_results['transaction_cost'] = total_cost
        backtest_results['cost_drag'] = cost_drag
        logger.info(f"   原始总收益: {sum(raw_returns):.2%}, 成本拖累: {cost_drag:.2%}, 调整后: {sum(adj_returns):.2%}")
        return backtest_results

    def estimate_capacity(self, stock_universe, strategy_turnover=2.0):
        """估算策略容量（亿元）- 兼容Tushare列名"""
        logger.info("\n📊 估算策略容量...")
        code_col = 'ts_code' if 'ts_code' in stock_universe.columns else (
                   'symbol' if 'symbol' in stock_universe.columns else None)
        amt_col  = 'amount' if 'amount' in stock_universe.columns else None
        vol_col  = 'vol'    if 'vol'    in stock_universe.columns else (
                   'volume' if 'volume' in stock_universe.columns else None)

        if amt_col and code_col:
            daily_amounts = stock_universe.groupby(code_col)[amt_col].mean()
        elif vol_col and 'close' in stock_universe.columns and code_col:
            calc = (pd.to_numeric(stock_universe[vol_col],   errors='coerce').fillna(0) *
                    pd.to_numeric(stock_universe['close'],   errors='coerce').fillna(0))
            daily_amounts = calc.groupby(stock_universe[code_col]).mean()
        else:
            logger.warning("⚠️ 无法计算日均成交额，容量估算跳过")
            return 0

        available = (daily_amounts * self.liquidity_limit).sum()
        capacity = available * 252 / strategy_turnover / 1e8  # 亿元
        logger.info(f"   可用流动性: {available/1e8:.2f} 亿/天, 换手率: {strategy_turnover}倍")
        logger.info(f"   策略容量: {capacity:.2f} 亿元")
        return capacity


# 全局实例
transaction_cost_model = TransactionCostModel()


if __name__ == '__main__':
    # 测试
    df = pd.DataFrame({
        'symbol': [f'00000{i}.SZ' for i in range(100)],
        'close': np.random.uniform(10, 100, 100),
        'volume': np.random.uniform(1e6, 1e8, 100),
        'amount': np.random.uniform(1e7, 1e9, 100),
    })
    cost = transaction_cost_model.calculate_buy_cost(50, 10000, 1e6)
    print(f"买入成本: {cost:.2f}")
    df = transaction_cost_model.check_liquidity_constraint(df, position_size=1e7)
    print(f"流动性通过率: {df['liquidity_ok'].mean():.1%}")
    cap = transaction_cost_model.estimate_capacity(df, strategy_turnover=2.0)
    print(f"容量: {cap:.2f}亿")