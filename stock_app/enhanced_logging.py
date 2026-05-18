# -*- coding: utf-8 -*-
"""增强日志系统 - 优化4（支持配置和详细指标）"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any


class EnhancedLogger:
    """
    增强日志记录器
    功能：
    - 记录权重分配（可视化柱状图）
    - 记录模型训练状态
    - 记录运行时间
    - 记录预测得分分布、因子IC等指标
    - 支持通过配置开关启用/禁用
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化
        参数:
            config: 配置字典，应包含 'enable', 'log_weights', 'log_model_status' 等键
                    若不提供，默认启用所有日志
        """
        if config is None:
            config = {'enable': True, 'log_weights': True, 'log_model_status': True}
        self.config = config
        self.enabled = config.get('enable', True)

        # 创建专用logger
        self.logger = logging.getLogger('v19_industrial')
        if not self.logger.handlers:
            # 避免重复添加handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.metrics = {}  # 存储指标，供后续分析

    def log_weights(self, weights: Dict[str, float]):
        """记录权重分配（带柱状图）"""
        if not self.enabled or not self.config.get('log_weights', True):
            return
        self.logger.info("\n" + "=" * 60)
        self.logger.info("⚖️ 动态权重分配")
        self.logger.info("=" * 60)
        for factor, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(weight * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            self.logger.info(f"  {factor:15s} [{bar}] {weight:6.1%}")
        self.metrics['weights'] = weights

    def log_model_status(self, status: Dict[str, bool]):
        """记录模型训练状态"""
        if not self.enabled or not self.config.get('log_model_status', True):
            return
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🤖 模型训练状态")
        self.logger.info("=" * 60)
        for name, trained in status.items():
            icon = "✅" if trained else "❌"
            text = "已训练" if trained else "未训练"
            self.logger.info(f"  {icon} {name:15s}: {text}")
        self.metrics['model_status'] = status

    def log_runtime(self, start: datetime, end: datetime):
        """记录运行时间"""
        if not self.enabled:
            return
        duration = (end - start).total_seconds()
        self.logger.info(f"\n⏱️ 运行时间: {duration:.2f} 秒")
        self.metrics['runtime'] = duration

    def log_prediction_distribution(self, predictions: np.ndarray, name: str = "模型"):
        """记录预测得分的分布（均值、标准差、分位数）"""
        if not self.enabled or not self.config.get('log_prediction_distribution', True):
            return
        if len(predictions) == 0:
            return
        mean = np.mean(predictions)
        std = np.std(predictions)
        q1, q2, q3 = np.percentile(predictions, [25, 50, 75])
        self.logger.info(f"📊 {name} 预测分布: 均值={mean:.4f}, 标准差={std:.4f}, "
                         f"25%={q1:.4f}, 中位数={q2:.4f}, 75%={q3:.4f}")
        key = f"{name}_pred_stats"
        self.metrics[key] = {'mean': mean, 'std': std, 'q1': q1, 'q2': q2, 'q3': q3}

    def log_ic(self, ic_dict: Dict[str, float], period: str = "当日"):
        """记录因子IC"""
        if not self.enabled or not self.config.get('log_ic', True):
            return
        self.logger.info(f"\n📈 {period} 因子IC:")
        for factor, ic in ic_dict.items():
            self.logger.info(f"   {factor:15s}: {ic:+.4f}")
        self.metrics[f'ic_{period}'] = ic_dict

    def log_rule_baseline(self, baseline_scores: np.ndarray, factor_count: int = 0):
        """记录规则基线分布（用于判断规则因子是否有区分度）"""
        if not self.enabled:
            return
        if len(baseline_scores) == 0:
            return
        std = np.std(baseline_scores)
        flag = '✅' if std > 0.05 else '⚠️ 区分度不足'
        self.logger.info(
            f"📐 规则基线分布: 均值={np.mean(baseline_scores):.4f}, "
            f"std={std:.4f}, 范围=[{baseline_scores.min():.4f},{baseline_scores.max():.4f}] "
            f"| 因子数={factor_count} | {flag}"
        )
        self.metrics['rule_baseline'] = {
            'mean': float(np.mean(baseline_scores)),
            'std': float(std),
            'factor_count': factor_count,
        }

    def log_factor_scores(self, factor_data: Dict[str, Any], label: str = "因子得分汇总"):
        """记录各因子的均值/std，快速定位哪个因子为全零"""
        if not self.enabled:
            return
        self.logger.info(f"\n🔍 {label}:")
        for name, series in factor_data.items():
            try:
                vals = np.array(series, dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    self.logger.info(f"   {name:18s}: 空序列")
                    continue
                self.logger.info(
                    f"   {name:18s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                    f"nonzero={np.count_nonzero(vals)}/{len(vals)}"
                )
            except Exception:
                self.logger.info(f"   {name:18s}: 无法计算")

    def get_metrics(self) -> Dict:
        """返回收集的指标"""
        return self.metrics


# 全局实例（主流程可导入后直接使用，也可重新配置）
enhanced_logger = EnhancedLogger()
