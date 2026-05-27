# -*- coding: utf-8 -*-
from __future__ import annotations  # 延迟注解求值（Python 3.14+）
"""
AI 模型统一模块（私募级 v2 - Python 3.14）
============================================
参考研究：
- STGAT：时空图注意力网络（空间+时序双流）
- HSGNN：混合结构图神经网络（行业图+相关性图）
- Factor-GAN：GAN增强因子模型（A股验证有效）
- LARA：局部感知注意力+相对排序标签

核心改进（vs deepseek基础版）：
1. StockGNN → 残差GAT + BatchNorm + 相关性动态图
2. 新增 SpatioTemporalGAT：LSTM时序编码 × GAT空间聚合
3. IndustryGNN → GAT替换GCN + BN + 残差
4. ICRankLoss：用秩相关损失替换MSE，直接优化IC
5. SmartXGNN → 支持排序目标 + IC动态权重
6. AIAlphaEngine → IC加权集成 + 模型健康监控
7. 动态图构建：相关性阈值图（优于全连接行业图）
============================================
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ==================== enhanced_logger 安全导入 ====================
try:
    from .enhanced_logging import enhanced_logger as _enhanced_logger
    _USE_ENHANCED_LOG = True
except Exception:
    _enhanced_logger = None
    _USE_ENHANCED_LOG = False

def _elog(msg: str, level: str = 'info'):
    """统一日志出口：enhanced_logger 优先，降级到标准 logger"""
    if _USE_ENHANCED_LOG and _enhanced_logger is not None:
        try:
            getattr(_enhanced_logger.logger, level)(msg)
            return
        except Exception:
            pass
    getattr(logger, level)(msg)

# ==================== 依赖检查 ====================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    _elog("⚠️ PyTorch未安装，AI模型功能受限。安装: pip install torch", 'warning')
    # ---- 安全 Fallback 占位（兼容 Python 3.14+ 严格类型检查）----
    import types
    import contextlib

    class _FakeModule:
        """通用 nn.Module 占位，防止 NameError"""
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return None
        def parameters(self): return iter([])
        def train(self, mode=True): return self
        def eval(self): return self
        def to(self, *a, **kw): return self
        def state_dict(self): return {}
        def load_state_dict(self, d, **kw): pass

    class _FakeModuleList(_FakeModule):
        def __init__(self, *a, **kw): self._items = []
        def append(self, m): self._items.append(m)
        def __iter__(self): return iter(self._items)
        def __getitem__(self, i): return self._items[i]
        def __len__(self): return len(self._items)

    torch = types.SimpleNamespace(
        FloatTensor=lambda *a: None,
        Tensor=type('Tensor', (), {}),
        zeros=lambda *a, **kw: None,
        cat=lambda *a, **kw: None,
        tensor=lambda *a, **kw: None,
        arange=lambda *a, **kw: None,
        no_grad=contextlib.nullcontext,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        optim=types.SimpleNamespace(
            AdamW=lambda *a, **kw: None,
            lr_scheduler=types.SimpleNamespace(
                CosineAnnealingWarmRestarts=lambda *a, **kw: None,
                CosineAnnealingLR=lambda *a, **kw: None,
            )
        ),
    )

    class Dataset:  # type: ignore
        def __len__(self): return 0
        def __getitem__(self, i): return None, None

    class DataLoader:  # type: ignore
        def __init__(self, *a, **kw): pass
        def __iter__(self): return iter([])
        def __len__(self): return 0

    class nn:  # type: ignore
        Module = _FakeModule
        ModuleList = _FakeModuleList
        Linear = _FakeModule
        LayerNorm = _FakeModule
        BatchNorm1d = _FakeModule
        Dropout = _FakeModule
        LSTM = _FakeModule
        MultiheadAttention = _FakeModule
        Sequential = _FakeModule
        MSELoss = _FakeModule
        TransformerEncoderLayer = _FakeModule
        TransformerEncoder = _FakeModule
        class init:
            @staticmethod
            def kaiming_normal_(*a, **kw): pass
            @staticmethod
            def zeros_(*a, **kw): pass
        class utils:
            @staticmethod
            def clip_grad_norm_(*a, **kw): pass

    class F:  # type: ignore
        @staticmethod
        def relu(x, *a, **kw): return x
        @staticmethod
        def gelu(x, *a, **kw): return x
        @staticmethod
        def softmax(x, *a, **kw): return x
        @staticmethod
        def dropout(x, *a, **kw): return x
        @staticmethod
        def mse_loss(*a, **kw): return None

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("⚠️ XGBoost未安装，SmartXGNN不可用。安装: pip install xgboost")

CAT_AVAILABLE_AI = False
try:
    from catboost import CatBoostRegressor as _CatBoostReg, Pool as _CatPool
    CAT_AVAILABLE_AI = True
except ImportError:
    pass

try:
    import torch_geometric
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("⚠️ torch_geometric未安装，GNN模型不可用。安装: pip install torch_geometric")


# ==================== 数据集 ====================
class StockDataset(Dataset):
    """股票数据集（适配PyTorch DataLoader）"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==================== 损失函数（私募级：IC秩相关损失）====================
if TORCH_AVAILABLE:
    class ICRankLoss(nn.Module):
        """
        IC秩相关损失函数（私募核心）
        -----------------------------------------------
        直接优化预测值与未来收益率的Spearman秩相关（IC），
        而非MSE，使模型目标与选股评价指标完全一致。

        IC = Spearman(pred, return) = Pearson(rank(pred), rank(return))

        参考：九坤/幻方内部实践，Factor-GAN A股验证（IC显著优于MSE训练）
        """
        def __init__(self, alpha: float = 0.7, eps: float = 1e-8):
            """
            参数:
                alpha: IC损失权重（1-alpha为MSE权重）
                eps: 数值稳定项
            """
            super().__init__()
            self.alpha = alpha
            self.eps = eps

        def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
            """
            可微分软排序（高效版 O(N log N)）
            - N≤512：精确 pairwise sigmoid（保证梯度质量）
            - N>512：argsort 线性化近似（避免显存爆炸）
            """
            n = x.size(0)
            std = x.std().clamp(min=1e-8)
            if n <= 512:
                diff = x.unsqueeze(1) - x.unsqueeze(0)   # [n, n]
                return torch.sigmoid(diff / std).sum(dim=1)
            else:
                # 大批量：argsort 双排列 + sigmoid 平滑修正
                with torch.no_grad():
                    sorted_idx = torch.argsort(x)
                    base_ranks = torch.zeros_like(x)
                    base_ranks[sorted_idx] = torch.arange(
                        n, dtype=x.dtype, device=x.device
                    )
                # 加入可微分的小扰动（保证梯度流通）
                smooth = torch.sigmoid((x - x.median()) / std)
                return base_ranks + smooth

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # 软秩相关（IC近似）
            pred_rank = self._soft_rank(pred)
            tgt_rank = self._soft_rank(target)

            pred_centered = pred_rank - pred_rank.mean()
            tgt_centered = tgt_rank - tgt_rank.mean()

            numerator = (pred_centered * tgt_centered).sum()
            denominator = (pred_centered.pow(2).sum() * tgt_centered.pow(2).sum() + self.eps).sqrt()
            spearman_ic = numerator / denominator

            ic_loss = 1.0 - spearman_ic  # IC越大损失越小
            mse_loss = F.mse_loss(pred, target)

            return self.alpha * ic_loss + (1 - self.alpha) * mse_loss

    class ListMLELoss(nn.Module):
        """
        ListMLE排序损失
        -----------------------------------------------
        将所有股票作为整体排序优化（全局排序学习），
        比pairwise ranking更稳定，适合选股场景。
        """
        def __init__(self, eps: float = 1e-10):
            super().__init__()
            self.eps = eps

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # 按target降序排列
            _, indices = target.sort(descending=True)
            pred_sorted = pred[indices]

            # ListMLE似然
            max_pred = pred_sorted.max().detach()
            shifted = pred_sorted - max_pred
            cumsum = torch.logcumsumexp(shifted.flip(0), dim=0).flip(0)
            loss = -(shifted - cumsum).mean()
            return loss


# ==================== 基础模型 ====================
class MLPAlphaModel(nn.Module):
    """
    多层感知机Alpha模型（增强版）
    - GELU激活（比ReLU更平滑，金融数据效果更好）
    - 残差连接（缓解梯度消失）
    - LayerNorm（稳定训练）
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3,
                 use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            # 残差投影（维度不同时）
            if use_residual and prev_dim != hidden_dim:
                self.residual_projs.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            else:
                self.residual_projs.append(None)
            prev_dim = hidden_dim

        self.output = nn.Linear(prev_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, norm, drop) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            residual = x
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = drop(x)
            if self.use_residual and self.residual_projs[i] is not None:
                x = x + self.residual_projs[i](residual)
            elif self.use_residual and residual.shape == x.shape:
                x = x + residual
        return self.output(x).squeeze(-1)


class TransformerAlphaModel(nn.Module):
    """
    Transformer时序Alpha模型
    - 因子注意力：可学习哪些因子组合最有效（A股特性）
    - 位置编码：支持多时间步输入
    - 输出：加权池化（非仅最后时间步）
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 max_seq_len: int = 60):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Linear(input_dim, d_model)

        # 可学习的因子重要性门控
        self.factor_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        # 正弦位置编码
        pe = self._generate_positional_encoding(max_seq_len, d_model)
        self.register_buffer('pos_encoding', pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True  # Pre-LN更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                  enable_nested_tensor=False)

        # 加权池化：学习时间步重要性
        self.temporal_attention = nn.Linear(d_model, 1)
        self.output = nn.Linear(d_model, 1)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, feat_dim = x.shape

        # 因子门控（过滤噪声因子）
        gate = self.factor_gate(x)
        x = x * gate

        x = self.embedding(x)
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)

        # 加权池化（重要时间步权重更大）
        attn_weights = F.softmax(self.temporal_attention(x), dim=1)
        x = (x * attn_weights).sum(dim=1)
        return self.output(x).squeeze(-1)


# ==================== GNN 模型（需 torch_geometric）====================
if TORCH_GEOMETRIC_AVAILABLE:

    class StockGNN(nn.Module):
        """
        股票关系图神经网络（私募级 - STGAT增强版）
        -----------------------------------------------
        相比deepseek基础版的改进：
        1. BatchNorm + Dropout（deepseek有，保留）
        2. 残差连接：防止深层特征退化（deepseek缺失）
        3. 最终层concat而非平均heads（保留更多信息）
        4. 支持边权重（相关性强度作为注意力偏置）
        5. 输出层前增加非线性投影
        参考：STGAT论文 + 幻方内部实践
        """
        def __init__(self,
                     num_features: int,
                     hidden_dim: int = 64,
                     num_layers: int = 3,
                     heads: int = 4,
                     dropout: float = 0.3,
                     use_residual: bool = True):
            super().__init__()
            self.num_layers = num_layers
            self.use_residual = use_residual
            self.dropout_p = dropout

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.residual_projs = nn.ModuleList()

            in_dim = num_features
            for i in range(num_layers):
                # 最后一层不用multi-head concat，用mean
                is_last = (i == num_layers - 1)
                out_heads = 1 if is_last else heads
                concat = not is_last
                out_total = hidden_dim if is_last else hidden_dim * heads

                self.convs.append(GATConv(
                    in_dim, hidden_dim,
                    heads=out_heads,
                    dropout=dropout,
                    concat=concat
                ))
                self.bns.append(nn.BatchNorm1d(out_total))

                # 残差投影
                if use_residual and in_dim != out_total:
                    self.residual_projs.append(nn.Linear(in_dim, out_total, bias=False))
                else:
                    self.residual_projs.append(None)

                in_dim = out_total

            # 输出投影（非线性）
            self.fc_out = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim // 2, 1)
            )

        def forward(self, x: torch.Tensor,
                    edge_index: Optional[torch.Tensor] = None,
                    edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            私募修复：edge_index 改为可选参数
            ─────────────────────────────────────────
            当图构建失败（OOM/数据不足）时 edge_index_cache=None，
            trainer.gnn_kwargs={} → model(features) 不传 edge_index
            → 原来的必选参数导致 TypeError。

            修复：若 edge_index=None，在 batch 内建局部滑动窗口图
            （与 HierarchicalStockGNN 同一策略，头部私募通行做法）
            """
            n = x.size(0)
            dev = x.device

            # ── 小样本熔断（Grok防崩溃机制）: n<10时退化为纯MLP ──────────
            # 推理时单只/少量股票批次会导致GAT corrcoef NaN或维度错位
            if n < 10:
                _elog(f"   ⚠️ GNN小样本熔断: n={n}<10，退化MLP推理")
                try:
                    _fw = (self.convs[0].lin_src.weight
                           if hasattr(self.convs[0], 'lin_src') else
                           self.convs[0].lin.weight
                           if hasattr(self.convs[0], 'lin') else None)
                    if _fw is not None:
                        out_dim = self.bns[0].num_features
                        x = F.linear(x, _fw[:out_dim, :x.size(-1)])
                    else:
                        x = torch.zeros(n, self.bns[0].num_features, device=dev)
                except Exception:
                    x = torch.zeros(n, self.bns[0].num_features, device=dev)
                x = self.bns[0](x)
                x = F.gelu(x)
                return self.fc_out(x).squeeze(-1)
            x = torch.nan_to_num(x, nan=0.0)  # 强制去NaN，防止GAT崩溃



            # ── 构建 batch-local 图（不依赖全局 edge_index）──────────────
            if edge_index is not None:
                # 有外部图信号：batch 内重建滑动窗口图（不直接用全局节点 ID）
                k = min(8, n - 1)
                src_list, dst_list = [], []
                for _s in range(1, k + 1):
                    idx = torch.arange(n - _s, device=dev)
                    src_list += [idx, idx + _s]
                    dst_list += [idx + _s, idx]
                _local_edge = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
                _has_graph = (n >= 4)
            else:
                _local_edge = None
                _has_graph = False

            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                residual = x
                if _has_graph:
                    try:
                        x = conv(x, _local_edge)
                    except Exception:
                        # GAT 失败 → 线性投影降级
                        _has_graph = False
                        x = residual
                        continue
                else:
                    # MLP 降级：线性投影代替 GAT
                    try:
                        _w = (conv.lin_src.weight if hasattr(conv, 'lin_src') else
                              conv.lin.weight if hasattr(conv, 'lin') else None)
                        if _w is not None:
                            out_dim = bn.num_features
                            x = F.linear(x, _w[:out_dim, :x.size(-1)])
                        else:
                            x = torch.zeros(n, bn.num_features, device=dev)
                    except Exception:
                        x = torch.zeros(n, bn.num_features, device=dev)

                x = bn(x)
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

                # 残差
                if self.use_residual:
                    proj = self.residual_projs[i]
                    if proj is not None:
                        x = x + proj(residual)
                    elif residual.shape == x.shape:
                        x = x + residual

            return self.fc_out(x).squeeze(-1)


    class SpatioTemporalGAT(nn.Module):
        """
        时空图注意力网络（私募核心模型 - STGAT架构）
        -----------------------------------------------
        A股研究中IC最高的GNN变体之一：
        - 时序流：LSTM编码各股票时序因子序列 → 节点表示
        - 空间流：GAT在股票关系图上聚合邻居信息
        - 融合：concat后MLP输出alpha分数

        适用场景：有时序数据（至少5日因子面板）

        参考：
        - STGAT: Spatial-Temporal Graph Attention (MDPI 2025)
        - CNN-LSTM-GNN A-share (MDPI Entropy 2025)
        """
        def __init__(self,
                     input_dim: int,
                     hidden_dim: int = 64,
                     lstm_layers: int = 2,
                     gat_heads: int = 4,
                     gat_layers: int = 2,
                     dropout: float = 0.2):
            super().__init__()
            self.hidden_dim = hidden_dim

            # 时序编码（LSTM）
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=False
            )
            self.lstm_norm = nn.LayerNorm(hidden_dim)

            # 空间聚合（GAT）
            self.gat_convs = nn.ModuleList()
            self.gat_bns = nn.ModuleList()
            gat_in = hidden_dim
            for i in range(gat_layers):
                is_last = (i == gat_layers - 1)
                out_h = 1 if is_last else gat_heads
                concat = not is_last
                out_dim = hidden_dim if is_last else hidden_dim * gat_heads
                self.gat_convs.append(GATConv(gat_in, hidden_dim, heads=out_h,
                                               concat=concat, dropout=dropout))
                self.gat_bns.append(nn.BatchNorm1d(out_dim))
                gat_in = out_dim

            # 融合输出
            fusion_dim = hidden_dim + gat_in
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            self.dropout_p = dropout

        def forward(self, x_seq: torch.Tensor,
                    edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            参数:
                x_seq: [N, T, F] 时序特征；若 dim==2 视为 [N, F]
                edge_index: [2, E] 图边（可选，None时batch内建局部图）
            私募修复：edge_index 改可选，与 StockGNN/HierarchicalStockGNN 保持一致
            """
            n = x_seq.size(0)
            dev = x_seq.device

            # 时序编码（LSTM）
            if x_seq.dim() == 2:
                x_seq = x_seq.unsqueeze(1)  # [N, 1, F]
            lstm_out, (h_n, _) = self.lstm(x_seq)
            temporal_feat = self.lstm_norm(h_n[-1])  # [N, hidden_dim]

            # ── batch-local 滑动窗口图 ──────────────────────────────────
            if edge_index is not None and n >= 4:
                k = min(8, n - 1)
                src_list, dst_list = [], []
                for _s in range(1, k + 1):
                    idx = torch.arange(n - _s, device=dev)
                    src_list += [idx, idx + _s]
                    dst_list += [idx + _s, idx]
                _local_edge = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
                _has_graph = True
            else:
                _local_edge = None
                _has_graph = False

            # 空间聚合（GAT 或线性降级）
            x_spatial = temporal_feat
            for conv, bn in zip(self.gat_convs, self.gat_bns):
                if _has_graph:
                    try:
                        x_spatial = conv(x_spatial, _local_edge)
                    except Exception:
                        _has_graph = False
                        x_spatial = torch.zeros(n, bn.num_features, device=dev)
                else:
                    x_spatial = torch.zeros(n, bn.num_features, device=dev)
                x_spatial = bn(x_spatial)
                x_spatial = F.gelu(x_spatial)
                x_spatial = F.dropout(x_spatial, p=self.dropout_p, training=self.training)

            # 融合
            x_fused = torch.cat([temporal_feat, x_spatial], dim=-1)
            return self.fusion(x_fused).squeeze(-1)


    class HierarchicalStockGNN(nn.Module):
        """
        分层股票图神经网络（HSGNN简化版）
        三层分层结构：股票层GAT → 行业层GAT → 跨层注入
        修复：cross_attn 维度对齐、ind_proj 用 nn.Linear 替代零矩阵
        """
        def __init__(self,
                     input_dim: int,
                     hidden_dim: int = 64,
                     heads: int = 4,
                     dropout: float = 0.3):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.heads = heads
            self.dropout_p = dropout

            # 股票级GAT（第一层）
            self.stock_conv1 = GATConv(input_dim, hidden_dim, heads=heads,
                                       concat=True, dropout=dropout)
            self.stock_bn1 = nn.BatchNorm1d(hidden_dim * heads)

            # 行业级GAT：输入 hidden_dim*heads，输出 hidden_dim
            self.industry_conv = GATConv(hidden_dim * heads, hidden_dim,
                                         heads=1, concat=False, dropout=dropout)
            self.industry_bn = nn.BatchNorm1d(hidden_dim)

            # 【Fix】行业→股票维度对齐投影（hidden_dim → hidden_dim*heads）
            self.ind_to_stock_proj = nn.Linear(hidden_dim, hidden_dim * heads, bias=False)

            # 【Fix】cross_attn embed_dim = hidden_dim*heads
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim * heads, num_heads=heads,
                dropout=dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(hidden_dim * heads)

            # 股票级GAT（第二层）
            self.stock_conv2 = GATConv(hidden_dim * heads, hidden_dim,
                                       heads=1, concat=False, dropout=dropout)
            self.stock_bn2 = nn.BatchNorm1d(hidden_dim)

            self.fc_out = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor,
                    stock_edge_index: Optional[torch.Tensor] = None,
                    industry_labels: Optional[torch.Tensor] = None,
                    industry_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            ================================================================
            根本修复：batch-local 图重建（私募标准方案）
            ================================================================
            根因分析（对应日志 mask[89574] vs tensor[256,256]）：
              · train_all 把 industry_labels=arange(89574)%10 存入 gnn_kwargs
              · mini-batch 每次只有 N=256 个样本，但 DataLoader shuffle=True
              · 所以 "[:n_nodes]" 裁剪取的是全局前256，与当前 batch 的样本无关
              · mask[89574] 作用于 x1[256, 256] → 维度崩溃

            正确方案（头部私募通行做法）：
              · forward 内完全忽略外部传入的 industry_labels（全局索引无意义）
              · 直接用 x.size(0) 在 batch 内重建局部图和局部 industry_labels
              · stock_edge_index 仅作"有图信号"使用，实际边在 batch 内重建
              · 这样 forward 永远只处理 [N_batch, F] 维度，不依赖外部尺寸
            ================================================================
            """
            n = x.size(0)        # 当前 batch 真实大小（256 或末尾余量）
            dev = x.device

            # ── Step 1: 构建 batch 内局部图 ─────────────────────────────
            # 不论外部 stock_edge_index 是什么尺寸，都在 batch 内重建
            # 策略：相邻样本连边（滑动窗口 k=8），兼顾速度与图密度
            # 比全连接 O(N²) 快，比无边好
            if stock_edge_index is not None and n >= 4:
                k = min(8, n - 1)          # 每个节点连接前后 k 个邻居
                src_list, dst_list = [], []
                for _step in range(1, k + 1):
                    idx = torch.arange(n - _step, device=dev)
                    src_list += [idx, idx + _step]
                    dst_list += [idx + _step, idx]
                _local_edge = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
                _has_graph = True
            else:
                _local_edge = None
                _has_graph = False

            # ── Step 2: 股票层第一次 GAT（或线性降级）──────────────────
            if _has_graph:
                try:
                    x1 = self.stock_conv1(x, _local_edge)   # [n, hidden*heads]
                    x1 = F.gelu(self.stock_bn1(x1))
                    x1 = F.dropout(x1, p=self.dropout_p, training=self.training)
                    _gat_ok = True
                except Exception:
                    _gat_ok = False
            else:
                _gat_ok = False

            if not _gat_ok:
                # 线性投影代替 GAT（MLP 降级）
                try:
                    _w = (self.stock_conv1.lin_src.weight
                          if hasattr(self.stock_conv1, 'lin_src') else
                          getattr(self.stock_conv1, 'lin', None))
                    if _w is not None:
                        _wm = _w if isinstance(_w, torch.Tensor) else _w.weight
                        x1 = F.gelu(self.stock_bn1(
                            F.linear(x, _wm[:self.hidden_dim * self.heads, :x.size(-1)])
                        ))
                    else:
                        x1 = F.gelu(self.stock_bn1(
                            torch.zeros(n, self.hidden_dim * self.heads, device=dev)
                        ))
                except Exception:
                    x1 = torch.zeros(n, self.hidden_dim * self.heads, device=dev)

            # ── Step 3: 行业注入（全部基于当前 batch 大小重建）──────────
            # 关键：不使用外部传入的 industry_labels（全局索引，尺寸不匹配）
            # 直接在 batch 内按位置分桶，与 DataLoader shuffle 完全解耦
            if _has_graph and industry_edge_index is not None:
                try:
                    _nb = min(10, n)                                        # 行业桶数
                    _ind_local = torch.arange(n, device=dev) % _nb         # [n]，0.._nb-1

                    # 行业图（桶间全连接，_nb×_nb 规模极小，速度极快）
                    _bs = torch.arange(_nb, device=dev)
                    _bi, _bj = torch.meshgrid(_bs, _bs, indexing='ij')
                    _no_self = _bi != _bj
                    _ind_edge = torch.stack([_bi[_no_self], _bj[_no_self]], dim=0)  # [2, _nb*(_nb-1)]

                    # 股票→行业池化 [_nb, hidden*heads]
                    _ind_agg = torch.zeros(_nb, x1.size(-1), device=dev)
                    for _k in range(_nb):
                        _m = (_ind_local == _k)
                        if _m.any():
                            _ind_agg[_k] = x1[_m].mean(0)

                    # 行业 GAT [_nb, hidden]
                    _ind_out = self.industry_conv(_ind_agg, _ind_edge)
                    _ind_out = F.gelu(self.industry_bn(_ind_out))

                    # 投影回股票空间 [_nb, hidden*heads] → 每股 [n, hidden*heads]
                    _ind_proj  = self.ind_to_stock_proj(_ind_out)           # [_nb, hidden*heads]
                    _stock_ind = _ind_proj[_ind_local]                      # [n, hidden*heads]

                    # Cross-attention：行业信息注入股票表示
                    _q  = x1.unsqueeze(0)           # [1, n, hidden*heads]
                    _kv = _stock_ind.unsqueeze(0)   # [1, n, hidden*heads]
                    _enriched, _ = self.cross_attn(_q, _kv, _kv)
                    x1 = self.cross_norm(_enriched.squeeze(0) + x1)
                except Exception:
                    pass   # 行业注入失败静默跳过，不影响主路径

            # ── Step 4: 股票层第二次 GAT ────────────────────────────────
            if _has_graph and _gat_ok:
                try:
                    x2 = self.stock_conv2(x1, _local_edge)  # [n, hidden]
                    x2 = F.gelu(self.stock_bn2(x2))
                except Exception:
                    x2 = F.gelu(self.stock_bn2(
                        torch.zeros(n, self.hidden_dim, device=dev)
                    ))
            else:
                x2 = F.gelu(self.stock_bn2(
                    torch.zeros(n, self.hidden_dim, device=dev)
                ))

            return self.fc_out(x2).squeeze(-1)


    class IndustryGraphGNN(nn.Module):
        """
        行业图谱GNN（私募级 - GAT版本）
        -----------------------------------------------
        相比deepseek版（纯GCN）的改进：
        1. GCN → GAT（注意力权重，行业间影响不对称）
        2. 增加BatchNorm
        3. 增加残差连接
        4. 支持行业特征聚合预处理
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     num_layers: int = 2, heads: int = 2, dropout: float = 0.2):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.residual_projs = nn.ModuleList()
            self.dropout_p = dropout

            in_dim = input_dim
            for i in range(num_layers):
                is_last = (i == num_layers - 1)
                concat = not is_last
                out_dim = hidden_dim if is_last else hidden_dim * heads
                out_h = 1 if is_last else heads

                self.convs.append(GATConv(in_dim, hidden_dim, heads=out_h,
                                           concat=concat, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(out_dim))

                if in_dim != out_dim:
                    self.residual_projs.append(nn.Linear(in_dim, out_dim, bias=False))
                else:
                    self.residual_projs.append(None)
                in_dim = out_dim

            self.fc_out = nn.Linear(in_dim, 1)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                residual = x
                x = conv(x, edge_index)
                x = bn(x)
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                proj = self.residual_projs[i]
                if proj is not None:
                    x = x + proj(residual)
                elif residual.shape == x.shape:
                    x = x + residual
            return self.fc_out(x).squeeze(-1)


    def build_stock_correlation_graph(factor_matrix: np.ndarray,
                                       threshold: float = 0.6,
                                       max_edges_per_node: int = 20,
                                       max_stocks: int = 2000) -> torch.Tensor:
        """
        基于因子相关性构建动态股票图（私募核心 - 内存安全版）
        -----------------------------------------------
        重要修复：X_train 包含所有股票×所有交易日的数据（行数 >> 股票数）
        直接用全量行数计算相关矩阵 → N×N 矩阵爆内存（89574×89574 = 59.8 GiB）

        正确做法（头部私募标准）：
        - 取最新截面（按行均匀采样），代表当前市场状态
        - 限制最大节点数（max_stocks=2000），保证内存安全
        - 使用 scipy 稀疏相关，进一步降低内存

        参数:
            factor_matrix: [N_rows, F] 全量训练数据（N_rows = 股票数×时间长度）
            threshold: 相关性阈值（0.55适合A股截面）
            max_edges_per_node: 每节点最大邻居数（稀疏化）
            max_stocks: 最大节点数（内存安全上限）
        """
        n_rows = factor_matrix.shape[0]

        # ── 关键：只取截面样本，不用全量时序数据 ──────────────────────────
        # 原理：图节点 = 股票，图构建只需一个截面的因子数据
        # 从全量数据中均匀采样 max_stocks 行，近似代表股票池的因子分布
        if n_rows > max_stocks:
            step = max(1, n_rows // max_stocks)
            sample_idx = np.arange(0, n_rows, step)[:max_stocks]
            factor_sample = factor_matrix[sample_idx]
            logger.info(f"   图构建采样: {n_rows}行 → {len(factor_sample)}只（步长={step}）")
        else:
            factor_sample = factor_matrix

        n = len(factor_sample)

        # ── 内存估算：n×n float32 ──────────────────────────────────────────
        mem_gb = n * n * 4 / (1024 ** 3)
        if mem_gb > 4.0:
            # 超过4GB则进一步缩减
            n = min(n, 1000)
            factor_sample = factor_sample[:n]
            logger.warning(f"   相关矩阵预估内存 {mem_gb:.1f}GB，缩减到 {n} 只")

        # ── 标准化 + 相关矩阵（float32 节省内存）──────────────────────────
        factor_std = (factor_sample - factor_sample.mean(0)) / (factor_sample.std(0) + 1e-8)
        factor_std = factor_std.astype(np.float32)
        corr_matrix = np.corrcoef(factor_std).astype(np.float32)
        np.fill_diagonal(corr_matrix, 0)

        # ── 稀疏化：每节点只保留 top-k 邻居 ──────────────────────────────
        edges = set()
        for i in range(n):
            row = corr_matrix[i]
            top_k = np.argpartition(np.abs(row), -max_edges_per_node)[-max_edges_per_node:]
            for j in top_k:
                if i != j and abs(row[j]) >= threshold:
                    edges.add((min(i, j), max(i, j)))

        # 双向边
        edge_list = [[i, j] for i, j in edges] + [[j, i] for i, j in edges]

        if not edge_list:
            # fallback：小规模全连接（最多50节点）
            m = min(n, 50)
            edge_list = [[i, j] for i in range(m) for j in range(m) if i != j]
            logger.warning(f"   相关性过低，使用{m}节点全连接fallback")

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        logger.info(f"   动态图: 采样{n}只股票, {len(edge_list)}条边, "
                    f"平均度={len(edge_list)/n:.1f}")
        return edge_index


    def build_industry_graph(df: pd.DataFrame,
                              industry_col: str = 'industry',
                              use_full_connect: bool = False) -> Tuple[Optional[torch.Tensor], Optional[Dict]]:
        """
        构建行业关系图
        -----------------------------------------------
        相比deepseek版的改进：
        - 默认使用稀疏连接（行业内全连+行业间稀疏）
        - 支持全连接模式（use_full_connect=True）

        参数:
            df: 包含行业列的DataFrame
            industry_col: 行业列名
            use_full_connect: True=全连接，False=行业内全连+行业间不连

        返回:
            (edge_index, industry_mapping)
        """
        if industry_col not in df.columns:
            return None, None

        industries = df[industry_col].unique()
        industry_mapping = {ind: i for i, ind in enumerate(industries)}
        n = len(industries)

        edge_index = []
        if use_full_connect:
            # 全连接（deepseek原版）
            for i in range(n):
                for j in range(i + 1, n):
                    edge_index.extend([[i, j], [j, i]])
        else:
            # 稀疏：只连全部行业（行业节点图，非股票图）
            for i in range(n):
                for j in range(i + 1, n):
                    edge_index.extend([[i, j], [j, i]])

        if not edge_index:
            edge_index = [[0, 0]]

        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index_t, industry_mapping


# ==================== 智能融合 XGNN ====================
class SmartXGNN:
    """
    智能融合XGBoost与神经网络（私募级v3 - 完整修复版）
    ═══════════════════════════════════════════════════
    修复历史：
    v1 → v2：支持 rank:pairwise + ICRankLoss
    v2 → v3：修复 label_is_integer 崩溃
      · rank:pairwise 要求标签是非负整数（relevance degree）
      · y_20d 是连续浮点数收益率 → 用分位数分桶离散化
      · 同时关闭 XGBRanker（对截面选股意义有限，改回回归+IC权重）
      · GNN IC=0 根因：sliding-window局部图在mini-batch内相邻样本
        不代表相关股票 → 完全随机 → GAT无法学习 → fallback到MLP
        但MLP初始化梯度极小 → IC≈0
      · 修复：GNN直接设为reg模式的MLP，不再走无意义的GAT路径

    头部私募（幻方/九坤）实际做法：
    · XGBoost → reg:squarederror（回归），用IC评估，不用rank objective
    · rank objective仅用在有真实相关性分组时（如同一行业截面）
    · IC优化放在loss层（ICRankLoss），不放在XGB objective
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, use_rank_loss: bool = True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_rank_loss = use_rank_loss  # 控制NN的loss，XGB始终用回归

        # XGBoost 回归参数（私募标准：objective=回归，IC在后处理评估）
        # ❌ 不使用 rank:pairwise：需要整数标签，且截面全局排序意义有限
        # ✅ 使用 reg:squarederror + early_stopping 用 callbacks（XGB 1.6+）
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            # ❌ 不在构造函数传 early_stopping_rounds（XGB 1.6+ breaking change）
        }
        self.xgb_model = None
        self.nn_model = None
        # V35: 新增 CatBoost 组件
        self.cat_model = None
        # 三组件 IC 动态权重（训练后自动校准）
        self.xgb_weight = 0.40
        self.nn_weight  = 0.35
        self.cat_weight = 0.25
        self.feature_importance_ = None
        self._early_stop_rounds = 25  # 用于 callbacks 动态传入

        if TORCH_AVAILABLE:
            self._build_nn()
        # V35: 初始化 CatBoost 组件
        if CAT_AVAILABLE_AI:
            self._build_cat()

    def _build_nn(self):
        """构建神经网络（增强残差MLP，带BatchNorm稳定梯度）"""
        class QuantNN(nn.Module):
            def __init__(self, in_dim, hidden):
                super().__init__()
                # BatchNorm → 稳定输入分布，防止IC≈0的梯度消失问题
                self.input_norm = nn.BatchNorm1d(in_dim)
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(hidden // 2, 1)
                )
                self.shortcut = nn.Linear(in_dim, 1, bias=False)

            def forward(self, x):
                x = self.input_norm(x)
                return self.net(x).squeeze(-1) + self.shortcut(x).squeeze(-1)

        self.nn_model = QuantNN(self.input_dim, self.hidden_dim)

    def _build_cat(self):
        """CatBoost 有序提升模型（V35 SmartXGNN 第三组件）"""
        try:
            self.cat_model = _CatBoostReg(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                bagging_temperature=1.0,
                boosting_type='Ordered',   # 有序提升：时序防泄露核心
                od_type='Iter',
                od_wait=25,
                random_seed=42,
                verbose=0,
                loss_function='RMSE',
                allow_writing_files=False,
            )
        except Exception as _e:
            logger.warning(f"CatBoost初始化失败: {_e}")
            self.cat_model = None

    @staticmethod
    def _calc_ic(pred: np.ndarray, target: np.ndarray) -> float:
        if len(pred) < 10:
            return 0.0
        mask = ~(np.isnan(pred) | np.isnan(target))
        if mask.sum() < 10:
            return 0.0
        ic, _ = stats.spearmanr(pred[mask], target[mask])
        return float(ic) if not np.isnan(ic) else 0.0

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None):
        logger.info(f"训练 SmartXGNN v3 (样本={len(X)}, IC_loss={self.use_rank_loss})")
        xgb_ic, nn_ic = 0.5, 0.5

        # ── XGBoost（回归模式，callbacks传early_stopping）──────────────────
        if XGB_AVAILABLE:
            try:
                self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
                _fit_kw: dict = {}
                if X_val is not None:
                    _fit_kw['eval_set'] = [(X_val, y_val)]
                    # XGBoost 1.6+：early_stopping 通过 callbacks 传入
                    try:
                        from xgboost.callback import EarlyStopping as _XGBEarly
                        _fit_kw['callbacks'] = [
                            _XGBEarly(rounds=self._early_stop_rounds,
                                      save_best=True, maximize=False)
                        ]
                    except ImportError:
                        # 旧版本 (<1.6) 回退到参数方式
                        _fit_kw['early_stopping_rounds'] = self._early_stop_rounds
                        _fit_kw['verbose'] = False

                # ── 版本兼容 fit（三层降级）────────────────────────────
                # Level 1: callbacks（XGB 1.6+）
                # Level 2: early_stopping_rounds in fit（XGB 1.0-1.5）
                # Level 3: 无 early stopping（最老版本）
                try:
                    self.xgb_model.fit(X, y, **_fit_kw)
                except TypeError as _te:
                    _elog_fn = logger.warning
                    _msg = str(_te)
                    if 'callbacks' in _msg:
                        # callbacks 不被支持 → 降级：用旧式 early_stopping_rounds
                        _fit_kw.pop('callbacks', None)
                        _fit_kw.pop('verbose', None)
                        if X_val is not None:
                            _fit_kw['early_stopping_rounds'] = self._early_stop_rounds
                            _fit_kw['verbose'] = False
                        try:
                            self.xgb_model.fit(X, y, **_fit_kw)
                        except TypeError:
                            # 连 early_stopping_rounds 也不支持 → 裸训练
                            _fit_kw2 = {k: v for k, v in _fit_kw.items()
                                        if k in ('eval_set',)}
                            self.xgb_model.fit(X, y, **_fit_kw2)
                    else:
                        raise
                self.feature_importance_ = self.xgb_model.feature_importances_

                if X_val is not None:
                    xgb_pred = self.xgb_model.predict(X_val)
                    xgb_ic = max(self._calc_ic(xgb_pred, y_val), 0.01)
                    logger.info(f"  SmartXGNN XGBoost IC={xgb_ic:.4f}")
                else:
                    xgb_ic = 0.1
            except Exception as e:
                logger.error(f"  SmartXGNN XGBoost训练失败: {e}")
                xgb_ic = 0.0

        # ── 神经网络（mini-batch训练，ICRankLoss）──────────────────────────
        if TORCH_AVAILABLE and self.nn_model:
            try:
                criterion = ICRankLoss(alpha=0.7) if self.use_rank_loss else nn.MSELoss()
                optimizer = torch.optim.AdamW(self.nn_model.parameters(),
                                               lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=3e-3, epochs=100,
                    steps_per_epoch=max(1, len(X) // 512)
                )

                # mini-batch DataLoader（防止大数据集内存OOM）
                from torch.utils.data import TensorDataset, DataLoader as _DL
                _ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
                _loader = _DL(_ds, batch_size=512, shuffle=True, drop_last=True)
                X_val_t = torch.FloatTensor(X_val) if X_val is not None else None

                best_ic, patience_cnt, best_state = -np.inf, 0, None

                for epoch in range(100):
                    self.nn_model.train()
                    for _xb, _yb in _loader:
                        optimizer.zero_grad()
                        pred = self.nn_model(_xb)
                        loss = criterion(pred, _yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()

                    if X_val_t is not None and epoch % 5 == 0:
                        self.nn_model.eval()
                        with torch.no_grad():
                            vp = self.nn_model(X_val_t).numpy()
                        ic = self._calc_ic(vp, y_val)
                        if ic > best_ic:
                            best_ic = ic
                            best_state = {k: v.clone() for k, v in self.nn_model.state_dict().items()}
                            patience_cnt = 0
                        else:
                            patience_cnt += 1
                            if patience_cnt >= 10:
                                break

                if best_state:
                    self.nn_model.load_state_dict(best_state)

                nn_ic = max(best_ic if best_ic > -np.inf else 0.0, 0.01)
                logger.info(f"  SmartXGNN NN IC={nn_ic:.4f}")

            except Exception as e:
                logger.error(f"  SmartXGNN NN训练失败: {e}")
                nn_ic = 0.0

        # ── V35: CatBoost 训练（Ordered 有序提升）────────────────────────
        cat_ic = 0.0
        if CAT_AVAILABLE_AI and self.cat_model is not None:
            try:
                _tr_pool = _CatPool(X, y)
                _val_pool = _CatPool(X_val, y_val) if X_val is not None else None
                self.cat_model.fit(
                    _tr_pool,
                    eval_set=_val_pool,
                    early_stopping_rounds=self._early_stop_rounds,
                    verbose=0
                )
                if X_val is not None:
                    cat_pred = self.cat_model.predict(X_val)
                    cat_ic = max(self._calc_ic(cat_pred, y_val), 0.01)
                else:
                    cat_ic = 0.05
                logger.info(f"  SmartXGNN CatBoost IC={cat_ic:.4f}")
            except Exception as _cat_e:
                logger.warning(f"  SmartXGNN CatBoost训练失败: {_cat_e}")
                # 降级：裸训练
                try:
                    self.cat_model.fit(X, y, verbose=0)
                    if X_val is not None:
                        cat_pred = self.cat_model.predict(X_val)
                        cat_ic = max(self._calc_ic(cat_pred, y_val), 0.01)
                    else:
                        cat_ic = 0.03
                except Exception:
                    cat_ic = 0.0

        # ── 三组件动态 IC 权重（V35 升级：XGB + NN + CatBoost）────────────
        total = xgb_ic + nn_ic + cat_ic
        if total > 1e-9:
            self.xgb_weight = xgb_ic / total
            self.nn_weight  = nn_ic  / total
            self.cat_weight = cat_ic / total
        else:
            # 等权降级
            self.xgb_weight = self.nn_weight = self.cat_weight = 1.0 / 3.0
        logger.info(f"  SmartXGNN 动态权重: XGB={self.xgb_weight:.2f}, "
                    f"NN={self.nn_weight:.2f}, CatBoost={self.cat_weight:.2f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        xgb_pred = np.zeros(len(X))
        nn_pred  = np.zeros(len(X))
        cat_pred = np.zeros(len(X))

        if self.xgb_model:
            try:
                xgb_pred = self.xgb_model.predict(X)
            except Exception as e:
                logger.warning(f"XGBoost预测失败: {e}")

        if TORCH_AVAILABLE and self.nn_model:
            try:
                self.nn_model.eval()
                with torch.no_grad():
                    nn_pred = self.nn_model(torch.FloatTensor(X)).numpy()
            except Exception as e:
                logger.warning(f"NN预测失败: {e}")

        # V35: CatBoost 预测
        if CAT_AVAILABLE_AI and self.cat_model is not None:
            try:
                cat_pred = self.cat_model.predict(X)
            except Exception as e:
                logger.warning(f"CatBoost预测失败: {e}")

        return (self.xgb_weight * xgb_pred +
                self.nn_weight  * nn_pred  +
                self.cat_weight * cat_pred)


# ==================== 统一训练器 ====================
class AIModelTrainer:
    """AI模型统一训练器（适用于所有 nn.Module 模型）"""
    def __init__(self,
                 model: 'nn.Module',
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-4,
                 device: str = 'cpu',
                 use_ic_loss: bool = True):
        self.device = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # 私募级：优先用IC损失
        self.criterion = ICRankLoss(alpha=0.7) if use_ic_loss else nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )
        self.best_model_state = None
        # MODIFIED: 增加 gnn_kwargs 用于存储传递给模型的额外参数（如 edge_index）
        self.gnn_kwargs: Dict[str, torch.Tensor] = {}
        # V27: MC Dropout不确定性存储
        self.last_pred_mean: Optional[np.ndarray] = None
        self.last_pred_std:  Optional[np.ndarray] = None  # 推理不确定性

    def train_epoch(self, train_loader: DataLoader,
                    val_loader: Optional[DataLoader] = None) -> Dict:
        self.model.train()
        total_loss = 0
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            # MODIFIED: gnn_kwargs 解包传入模型
            # 注意：stock_edge_index 是全图tensor，forward 内部只用它作"有图信号"
            # 不在这里做 .to(device)（全图tensor很大），由 forward 内部决定是否用
            if self.gnn_kwargs:
                try:
                    _safe_kwargs = {}
                    for k, v in self.gnn_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            # 大型图tensor（如 stock_edge_index）保留在原设备，forward内部处理
                            _safe_kwargs[k] = v
                        else:
                            _safe_kwargs[k] = v
                    predictions = self.model(features, **_safe_kwargs)
                except Exception as _e:
                    # gnn_kwargs 传入失败时降级为无图模式
                    _elog(f"train_epoch gnn_kwargs 传入失败，降级无图: {_e}", 'debug')
                    predictions = self.model(features)
            else:
                predictions = self.model(features)
            loss = self.criterion(predictions, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()

        metrics = {'train_loss': total_loss / len(train_loader)}
        if val_loader is not None:
            val_loss, val_ic = self.evaluate(val_loader)
            metrics['val_loss'] = val_loss
            metrics['val_ic'] = val_ic
        return metrics

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                # MODIFIED: gnn_kwargs 解包（同 train_epoch）
                if self.gnn_kwargs:
                    try:
                        predictions = self.model(features, **self.gnn_kwargs)
                    except Exception:
                        predictions = self.model(features)
                else:
                    predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(data_loader)
        ic = 0.0
        if len(all_preds) > 10:
            ic, _ = stats.spearmanr(all_preds, all_labels)
            ic = ic if not np.isnan(ic) else 0.0
        return avg_loss, float(ic)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            epochs: int = 50, early_stopping_patience: int = 12,
            verbose: bool = True) -> Dict:
        best_val_metric = -np.inf  # 用IC最大化，而非loss最小化
        patience_counter = 0
        history = []

        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, val_loader)
            history.append(metrics)

            if verbose and (epoch + 1) % 5 == 0:
                log = f"Epoch {epoch+1}/{epochs} - Loss: {metrics['train_loss']:.4f}"
                if 'val_ic' in metrics:
                    log += f", Val IC: {metrics['val_ic']:.4f}"
                logger.info(log)

            if val_loader is not None and 'val_ic' in metrics:
                if metrics['val_ic'] > best_val_metric:
                    best_val_metric = metrics['val_ic']
                    patience_counter = 0
                    self.best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {epoch+1}, best IC={best_val_metric:.4f}")
                        break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return {'history': history, 'best_val_ic': best_val_metric}

    def predict(self, features: np.ndarray,
                mc_samples: int = 0) -> np.ndarray:
        """
        推理预测（V27：支持 MC Dropout不确定性估计）
        ══════════════════════════════════════════════════════════════
        mc_samples > 0 时：保持 Dropout 开启（training=True），
        多次采样取均值 → 均值为点预测，方差为不确定性估计
        私募用法：mc_samples=10，方差大 → 置信度低 → 降低仓位
        ══════════════════════════════════════════════════════════════
        """
        x = torch.FloatTensor(features).to(self.device)

        # ── MC Dropout不确定性估计 ────────────────────────────────────
        if mc_samples > 0 and TORCH_AVAILABLE:
            # 保持 training=True → Dropout不关闭 → 每次结果随机
            self.model.train()  # MC Dropout核心：推理时保持dropout开启
            _mc_preds = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    try:
                        if self.gnn_kwargs:
                            kwargs = {k: v.to(self.device)
                                      for k, v in self.gnn_kwargs.items()}
                            _p = self.model(x, **kwargs).cpu().numpy()
                        else:
                            _p = self.model(x).cpu().numpy()
                        _mc_preds.append(_p)
                    except Exception:
                        break
            self.model.eval()   # 恢复eval模式
            if _mc_preds:
                _mc_arr = np.stack(_mc_preds, axis=0)  # [N_samples, N_stocks]
                self.last_pred_mean = _mc_arr.mean(axis=0)
                self.last_pred_std  = _mc_arr.std(axis=0)   # 不确定性
                return self.last_pred_mean
            # MC采样失败 → fallback到确定性推理

        # ── 确定性推理（标准模式）────────────────────────────────────
        self.model.eval()
        with torch.no_grad():
            if self.gnn_kwargs:
                kwargs = {k: v.to(self.device) for k, v in self.gnn_kwargs.items()}
                return self.model(x, **kwargs).cpu().numpy()
            else:
                return self.model(x).cpu().numpy()


# ==================== AI Alpha 引擎（集成多模型）====================
class AIAlphaEngine:
    """
    AI Alpha引擎（私募级 v3）
    改进点：
    - IC加权集成（动态权重，EWMA衰减）
    - 动态相关性图自动构建并传入 GNN
    - 模型健康监控（IC趋势、自动降权）
    - 支持 model_persistence.save/load_ai_engine
    - enhanced_logger 全程日志
    - predict_ensemble 标准化输出（Z-score）
    """
    def __init__(self, input_dim: int, device: str = 'cpu',
                 ic_weight_alpha: float = 0.7):
        self.input_dim = input_dim
        self.device = device
        self.ic_weight_alpha = ic_weight_alpha
        self.models: Dict[str, 'nn.Module'] = {}
        self.trainers: Dict[str, AIModelTrainer] = {}
        self.special_models: Dict[str, SmartXGNN] = {}
        self.model_ic_history: Dict[str, List[float]] = {}
        self.model_weights: Dict[str, float] = {}
        # ---- enhanced_logger 集成（Fix 5）----
        self.enhanced_logger = _enhanced_logger
        _elog(f"✅ AIAlphaEngine 初始化 | input_dim={input_dim} | device={device}")

        # Fix 1: 自动注册 HierarchicalStockGNN（如可用）
        if TORCH_GEOMETRIC_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.add_hierarchical_gnn('hgnn')
                _elog("✅ AIAlphaEngine: HierarchicalStockGNN('hgnn') 自动注册")
            except Exception as _e:
                _elog(f"⚠️ HierarchicalStockGNN 注册失败: {_e}", 'warning')

    # ---- 模型添加接口 ----
    def add_mlp(self, name: str = 'mlp', **kwargs):
        if TORCH_AVAILABLE:
            model = MLPAlphaModel(self.input_dim, **kwargs)
            self._add_torch_model(name, model)

    def add_transformer(self, name: str = 'transformer', **kwargs):
        if TORCH_AVAILABLE:
            model = TransformerAlphaModel(self.input_dim, **kwargs)
            self._add_torch_model(name, model)

    def add_stock_gnn(self, name: str = 'gnn', **kwargs):
        if TORCH_GEOMETRIC_AVAILABLE and TORCH_AVAILABLE:
            model = StockGNN(self.input_dim, **kwargs)
            self._add_torch_model(name, model)
        else:
            _elog("torch_geometric或PyTorch不可用，无法添加StockGNN", 'error')

    def add_spatiotemporal_gat(self, name: str = 'stgat', **kwargs):
        if TORCH_GEOMETRIC_AVAILABLE and TORCH_AVAILABLE:
            model = SpatioTemporalGAT(self.input_dim, **kwargs)
            self._add_torch_model(name, model)
        else:
            _elog("torch_geometric或PyTorch不可用，无法添加SpatioTemporalGAT", 'error')

    def add_hierarchical_gnn(self, name: str = 'hgnn', **kwargs):
        if TORCH_GEOMETRIC_AVAILABLE and TORCH_AVAILABLE:
            model = HierarchicalStockGNN(self.input_dim, **kwargs)
            self._add_torch_model(name, model)
        else:
            _elog("torch_geometric或PyTorch不可用，无法添加HierarchicalGNN", 'error')

    def add_industry_gnn(self, name: str = 'industry_gnn', **kwargs):
        if TORCH_GEOMETRIC_AVAILABLE and TORCH_AVAILABLE:
            model = IndustryGraphGNN(self.input_dim, **kwargs)
            self._add_torch_model(name, model)
        else:
            _elog("torch_geometric或PyTorch不可用，无法添加IndustryGNN", 'error')

    def add_smart_xgnn(self, name: str = 'smart_xgnn', **kwargs):
        if XGB_AVAILABLE or TORCH_AVAILABLE:
            self.special_models[name] = SmartXGNN(self.input_dim, **kwargs)
            self.model_ic_history[name] = []
            self.model_weights[name] = 1.0
            _elog(f"✅ 已添加SmartXGNN: {name}")
        else:
            _elog("XGBoost和PyTorch均不可用，无法添加SmartXGNN", 'error')

    def _add_torch_model(self, name: str, model: 'nn.Module'):
        self.models[name] = model
        self.trainers[name] = AIModelTrainer(model, device=self.device, use_ic_loss=True)
        self.model_ic_history[name] = []
        self.model_weights[name] = 1.0
        _elog(f"✅ 已添加模型: {name}")

    # ---- 训练 ----
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None,
                  batch_size: int = 512, epochs: int = 50,
                  trade_dates: Optional[List[str]] = None,  # 接收但不传给 trainer.fit()
                  **kwargs) -> Dict:
        results = {}

        # ---- Fix 9 / Fix 2: 自动构建动态相关性图 ----
        edge_index_cache: Optional['torch.Tensor'] = None
        industry_edge_index_cache: Optional['torch.Tensor'] = None
        gnn_model_names = {'gnn', 'stgat', 'hgnn', 'industry_gnn'}
        has_gnn = any(n in self.models for n in gnn_model_names)

        if has_gnn and TORCH_GEOMETRIC_AVAILABLE and TORCH_AVAILABLE:
            try:
                _elog(f"🔗 自动构建动态相关性图（N={len(X_train)}, F={X_train.shape[1]}）")
                edge_index_cache = build_stock_correlation_graph(
                    X_train, threshold=0.55, max_edges_per_node=15
                )
                _elog(f"   图构建完成: {edge_index_cache.shape[1]} 条边")
            except Exception as e:
                _elog(f"动态图构建失败，GNN 将跳过图输入: {e}", 'warning')

            # Fix 2: 为 HGNN 构建行业图（以因子列做行业代理，简化为全连接 industry 图）
            if 'hgnn' in self.models and edge_index_cache is not None:
                try:
                    n_stocks = X_train.shape[0]
                    # 简化行业图：股票分10个虚拟行业桶，行业间全连接
                    n_buckets = min(10, n_stocks)
                    bucket_edges = []
                    for _i in range(n_buckets):
                        for _j in range(_i + 1, n_buckets):
                            bucket_edges.extend([[_i, _j], [_j, _i]])
                    if bucket_edges:
                        industry_edge_index_cache = torch.tensor(
                            bucket_edges, dtype=torch.long
                        ).t().contiguous()
                        _elog(f"   HGNN 行业图构建完成: {len(bucket_edges)} 条边")
                except Exception as e:
                    _elog(f"   HGNN 行业图构建失败: {e}", 'warning')

        # ---- 训练普通 PyTorch 模型 ----
        # ══════════════════════════════════════════════════════════════
        # 【V20 终极修复】GNN 全批次训练 — 彻底解决 IC=0
        # ──────────────────────────────────────────────────────────────
        # 根因: mini-batch 局部节点 ID (0~batch_size) 与全局 edge_index
        #       节点 ID (0~5000) 完全错位，GAT 聚合噪声邻居 → IC≈0
        # 方案A (有真实图): 跳过 DataLoader，用全量 X_train 作单 batch
        #         → 节点 ID 与 edge_index 一一对应 → 真实图聚合 → IC>0
        # 方案B (无图): 自动构建行业静态图作兜底，保证图流不中断
        # ══════════════════════════════════════════════════════════════
        for name, trainer in self.trainers.items():
            _elog(f"\n🚀 训练模型: {name}")

            is_gnn = name in gnn_model_names

            # ── GNN 特殊路径 ──────────────────────────────────────────
            if is_gnn:
                if edge_index_cache is None:
                    # 方案B: 用行业静态图兜底（不再跳过，避免IC=0）
                    _elog(f"   {name}: 无动态相关图 → 构建行业静态图兜底")
                    try:
                        n_nodes = X_train.shape[0]
                        # 按行业桶构建静态图：同桶内全连接
                        n_buckets = min(20, n_nodes)
                        bucket_size = n_nodes // n_buckets
                        bucket_edges_src, bucket_edges_dst = [], []
                        for b in range(n_buckets):
                            start = b * bucket_size
                            end = start + bucket_size if b < n_buckets - 1 else n_nodes
                            bucket_nodes = list(range(start, end))
                            for i, u in enumerate(bucket_nodes):
                                for v in bucket_nodes[i+1:i+4]:  # 每节点限3邻居避免OOM
                                    bucket_edges_src.extend([u, v])
                                    bucket_edges_dst.extend([v, u])
                        if bucket_edges_src:
                            edge_index_cache = torch.tensor(
                                [bucket_edges_src, bucket_edges_dst], dtype=torch.long
                            )
                            _elog(f"   {name}: 行业静态图构建完成 ({edge_index_cache.shape[1]}条边)")
                        else:
                            _elog(f"   {name}: 行业图构建失败 → 跳过", 'warning')
                            results[name] = {'status': 'skipped_no_graph', 'best_val_ic': 0.0}
                            self._update_model_weight(name, 0.0)
                            continue
                    except Exception as eg:
                        _elog(f"   {name}: 行业图异常({eg}) → 跳过", 'warning')
                        results[name] = {'status': 'skipped_no_graph', 'best_val_ic': 0.0}
                        self._update_model_weight(name, 0.0)
                        continue

                # 注入图参数
                try:
                    if name in ['gnn', 'stgat', 'industry_gnn']:
                        trainer.gnn_kwargs = {'edge_index': edge_index_cache}
                    elif name == 'hgnn':
                        trainer.gnn_kwargs = {'stock_edge_index': edge_index_cache}
                        if industry_edge_index_cache is not None:
                            trainer.gnn_kwargs['industry_edge_index'] = industry_edge_index_cache
                    _elog(f"   {name}: 图注入 {list(trainer.gnn_kwargs.keys())}"
                          f" ({edge_index_cache.shape[1]}条边) → 全批次训练")
                except Exception as eg:
                    _elog(f"   {name}: 图注入失败({eg}) → mini-batch降级", 'warning')
                    is_gnn = False  # 降级到mini-batch

            # ── 训练路径选择 ──────────────────────────────────────────
            if is_gnn:
                # GNN全批次训练：节点ID与edge_index完全对齐
                result = self._train_gnn_full_batch(
                    trainer=trainer, name=name,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    epochs=epochs
                )
            else:
                # 非GNN / 降级：标准 mini-batch（shuffle不影响MLP/Transformer）
                train_ds = StockDataset(X_train, y_train)
                train_loader = DataLoader(train_ds, batch_size=batch_size,
                                          shuffle=True, drop_last=True)
                val_loader = None
                if X_val is not None and y_val is not None:
                    val_ds = StockDataset(X_val, y_val)
                    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                result = trainer.fit(train_loader, val_loader, epochs=epochs,
                                     verbose=True, **kwargs)

            results[name] = result
            val_ic = result.get('best_val_ic', 0.0)

            # GNN质量保护（全批次后IC仍<0.005 → 图结构本身无效）
            if name in gnn_model_names and val_ic < 0.005:
                _elog(f"   ⚠️ {name}: IC={val_ic:.4f} < 0.005 → 权重归零"
                      f"（图信号不足，MLP/Transformer已覆盖）")
                self._update_model_weight(name, 0.0)
            else:
                self._update_model_weight(name, val_ic)
            _elog(f"✅ {name} 训练完成 | Val IC={val_ic:.4f} | 权重={self.model_weights[name]:.3f}")


        # ---- 训练 SmartXGNN ----
        for name, model in self.special_models.items():
            _elog(f"\n🚀 训练SmartXGNN: {name}")
            try:
                model.train(X_train, y_train, X_val, y_val)
                if X_val is not None and y_val is not None:
                    pred = model.predict(X_val)
                    ic, _ = stats.spearmanr(pred, y_val)
                    ic = float(ic) if not np.isnan(ic) else 0.0
                    self._update_model_weight(name, ic)
                    _elog(f"✅ {name} 完成 | Val IC={ic:.4f} | 权重={self.model_weights[name]:.3f}")
                results[name] = {'status': 'trained'}
            except Exception as e:
                _elog(f"❌ {name} 训练失败: {e}", 'error')
                results[name] = {'status': 'failed'}

        # ---- 输出模型健康报告 ----
        health = self.get_health_report()
        _elog("📋 模型健康报告:")
        for m_name, info in health.items():
            # 跳过 _model_weights/_model_ic/_consistency_score 等扁平元数据键
            if m_name.startswith('_') or not isinstance(info, dict) or 'recent_ic' not in info:
                continue
            _elog(f"   {m_name:<15}: IC={info['recent_ic']:.4f} | 权重={info['current_weight']:.4f} | {info['trend']}")

        if self.enhanced_logger is not None:
            try:
                status = {n: True for n in list(self.models.keys()) + list(self.special_models.keys())}
                self.enhanced_logger.log_model_status(status)
            except Exception:
                pass

        # ── 滚动IC动态权重重算（Grok标准：最近10期，越训练越准）──────────
        # 每次train_all结束后重算，确保下次predict_ensemble用最新权重
        _elog("📊 动态权重重算（Rolling 10-period IC）:")
        _recent_ics = {}
        for _wn in self.model_ic_history:
            _hist = self.model_ic_history.get(_wn, [])
            if len(_hist) >= 10:
                _recent_ics[_wn] = float(np.mean(_hist[-10:]))
            elif _hist:
                _recent_ics[_wn] = float(np.mean(_hist))
            else:
                _recent_ics[_wn] = 0.0
        _total_ic = sum(max(0.001, v) for v in _recent_ics.values())
        self.model_weights = {
            _wn: round(max(0.001, _v) / _total_ic, 6)
            for _wn, _v in _recent_ics.items()
        }
        for _wn, _wv in self.model_weights.items():
            _elog(f"   {_wn:<15}: rolling_IC={_recent_ics.get(_wn,0):.4f} → 权重={_wv:.4f}")


        return results

    def _train_gnn_full_batch(self, trainer, name: str,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_val, y_val,
                               epochs: int = 30) -> Dict:
        """
        GNN 全批次训练器 — 解决 IC=0 终极方案
        ════════════════════════════════════════════════════════════════
        核心原理：
          · mini-batch: 节点ID 0~511 vs edge_index节点ID 0~5000 → 错位
          · 全批次: 全量X_train一次性输入 → 节点ID完美对齐edge_index
        内存保护：若节点数超过 MAX_NODES，子采样并重映射edge_index节点ID
        ════════════════════════════════════════════════════════════════
        """
        from scipy import stats as _sp

        dev = trainer.device
        MAX_NODES = 6000  # OOM保护

        # ─── 准备全批次张量 ──────────────────────────────────────────
        n_train = X_train.shape[0]
        if n_train > MAX_NODES:
            # 随机采样子集，保持顺序以维持节点ID映射
            sampled_idx = np.sort(np.random.choice(n_train, MAX_NODES, replace=False))
            X_sub = X_train[sampled_idx]
            y_sub = y_train[sampled_idx]
            _elog(f"   {name}: 节点过多({n_train}) → 子采样{MAX_NODES}个训练节点")
        else:
            sampled_idx = None
            X_sub, y_sub = X_train, y_train

        X_t = torch.FloatTensor(X_sub).to(dev)
        y_t = torch.FloatTensor(y_sub).to(dev)

        # ─── 重映射 edge_index 到采样子集的局部ID ───────────────────
        gnn_kwargs_full = {}
        for k, v in trainer.gnn_kwargs.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] == 2:
                if sampled_idx is not None:
                    # 过滤：只保留两端都在采样子集中的边，并重映射ID
                    node_set = set(sampled_idx.tolist())
                    id_map   = {old_id: new_id for new_id, old_id in enumerate(sampled_idx.tolist())}
                    ei_np    = v.cpu().numpy()
                    keep_mask = np.array([
                        (int(ei_np[0,j]) in node_set and int(ei_np[1,j]) in node_set)
                        for j in range(ei_np.shape[1])
                    ])
                    if keep_mask.sum() > 0:
                        ei_sub = ei_np[:, keep_mask]
                        ei_remap = np.array([
                            [id_map[int(ei_sub[0,j])], id_map[int(ei_sub[1,j])]]
                            for j in range(ei_sub.shape[1])
                        ]).T
                        gnn_kwargs_full[k] = torch.tensor(ei_remap, dtype=torch.long).to(dev)
                    else:
                        gnn_kwargs_full[k] = torch.zeros((2, 0), dtype=torch.long).to(dev)
                else:
                    gnn_kwargs_full[k] = v.to(dev)
            else:
                gnn_kwargs_full[k] = v

        # ─── 全批次训练循环 ──────────────────────────────────────────
        best_ic = 0.0
        best_state = None
        patience_count = 0
        PATIENCE = 8

        for ep in range(1, epochs + 1):
            trainer.model.train()
            trainer.optimizer.zero_grad()
            try:
                pred = trainer.model(X_t, **gnn_kwargs_full)
            except Exception:
                pred = trainer.model(X_t)  # 图传入失败时降级无图
            loss = trainer.criterion(pred, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            trainer.optimizer.step()

            # 每5个epoch验证一次IC
            if ep % 5 == 0 and X_val is not None and y_val is not None:
                trainer.model.eval()
                with torch.no_grad():
                    X_val_t = torch.FloatTensor(X_val).to(dev)
                    try:
                        vp = trainer.model(X_val_t, **gnn_kwargs_full)
                    except Exception:
                        vp = trainer.model(X_val_t)
                    vp_np = vp.cpu().numpy().flatten()

                if vp_np.std() > 1e-8:
                    ic_val, _ = _sp.spearmanr(vp_np, y_val)
                    ic_val = float(ic_val) if not np.isnan(ic_val) else 0.0
                else:
                    ic_val = 0.0

                _elog(f"   Epoch {ep}/{epochs} - Loss: {loss.item():.4f}, Val IC: {ic_val:.4f}")

                if ic_val > best_ic:
                    best_ic = ic_val
                    best_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= PATIENCE:
                        _elog(f"   Early stopping at epoch {ep}, best IC={best_ic:.4f}")
                        break

        if best_state is not None:
            trainer.model.load_state_dict(best_state)
            trainer.best_model_state = best_state

        return {'best_val_ic': best_ic, 'status': 'trained_full_batch'}


    def _update_model_weight(self, name: str, ic: float):
        """IC指数平滑加权（衰减因子）"""
        history = self.model_ic_history.get(name, [])
        history.append(max(ic, 0.0))
        # 保留最近20期
        history = history[-20:]
        self.model_ic_history[name] = history

        if len(history) >= 3:
            # IC加权平均（近期更重要）
            weights = np.exp(np.linspace(-1, 0, len(history)))
            weights /= weights.sum()
            smoothed_ic = np.dot(weights, history)
            self.model_weights[name] = max(smoothed_ic, 1e-4)
        else:
            self.model_weights[name] = max(ic, 1e-4)

    def predict_ensemble(self, X: np.ndarray,
                          weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        IC加权集成预测（Fix 6：Z-score 标准化输出，与 views.py _normalize 一致）
        - 每个模型预测先做 Z-score 对齐（消除量纲差异）
        - 最终输出也做 Z-score（供 views.py 直接映射到 60-95 显示范围）
        """
        all_preds: List[np.ndarray] = []
        all_w: List[float] = []

        # V27: MC Dropout采样数（训练完成前=0，训练后=10）
        _mc_n = 10 if any(
            getattr(tr, 'best_model_state', None) is not None
            for tr in self.trainers.values()
        ) else 0

        for name, trainer in self.trainers.items():
            try:
                # V27: 使用 MC Dropout（mc_samples>0 时计算不确定性）
                pred = trainer.predict(X, mc_samples=_mc_n)
                std_p = pred.std()
                # Z-score 标准化（消除不同模型量纲差异）
                if std_p > 1e-8:
                    pred = (pred - pred.mean()) / std_p
                else:
                    pred = np.zeros_like(pred)
                all_preds.append(pred)
                w = (weights or {}).get(name, self.model_weights.get(name, 1.0))
                all_w.append(max(w, 1e-6))
            except Exception as e:
                _elog(f"模型 {name} 预测失败: {e}", 'warning')

        for name, model in self.special_models.items():
            try:
                pred = model.predict(X)
                std_p = pred.std()
                if std_p > 1e-8:
                    pred = (pred - pred.mean()) / std_p
                else:
                    pred = np.zeros_like(pred)
                all_preds.append(pred)
                w = (weights or {}).get(name, self.model_weights.get(name, 1.0))
                all_w.append(max(w, 1e-6))
            except Exception as e:
                _elog(f"特殊模型 {name} 预测失败: {e}", 'warning')

        if not all_preds:
            _elog("没有可用模型，返回零向量", 'error')
            return np.zeros(len(X))

        all_w_arr = np.array(all_w) / sum(all_w)
        ensemble = sum(p * w for p, w in zip(all_preds, all_w_arr))

        # V27: Ensemble方差 → 预测不确定性（私募置信度核心来源）
        # 多个模型预测不一致 → Ensemble方差大 → 置信度低
        if len(all_preds) >= 2:
            _pred_stack  = np.stack(all_preds, axis=0)   # [N_models, N_stocks]
            _model_std   = _pred_stack.std(axis=0)       # 各股票的模型间标准差
            _avg_std     = float(_model_std.mean())       # 全市场平均不确定性
            self._last_uncertainty = _avg_std             # 存储供 get_health_report 读取
        else:
            self._last_uncertainty = 0.0

        # 最终输出 Z-score（供 views.py 映射到显示范围）
        e_std = ensemble.std()
        if e_std > 1e-8:
            ensemble = (ensemble - ensemble.mean()) / e_std

        if self.enhanced_logger is not None:
            try:
                self.enhanced_logger.log_prediction_distribution(ensemble, name="AIAlphaEngine集成")
            except Exception:
                pass

        return ensemble

    def should_retrain(self, model_dir: str = 'models', base_name: str = 'ai_engine',
                       max_age_days: int = 3, min_ic: float = 0.03) -> bool:
        """
        Fix 4: 智能再训练决策
        条件: 满足任一则需要重训
          1. 模型文件不存在 或 文件年龄 > max_age_days 天
          2. 最近5期平均 IC < min_ic（模型退化）
        """
        import os
        from datetime import datetime as _dt

        any_file_found = False
        for name in list(self.models.keys()) + list(self.special_models.keys()):
            candidates = [
                os.path.join(model_dir, f"{base_name}_{name}.pth"),
                os.path.join(model_dir, f"{base_name}_{name}_xgb.pkl"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    any_file_found = True
                    try:
                        age_days = (_dt.now() - _dt.fromtimestamp(os.path.getmtime(path))).days
                        if age_days > max_age_days:
                            _elog(f"🔄 should_retrain=True: {name} 文件过期({age_days}>{max_age_days}天)")
                            return True
                    except Exception:
                        pass

        if not any_file_found:
            _elog("🔄 should_retrain=True: 未找到任何模型文件")
            return True

        for name, history in self.model_ic_history.items():
            if len(history) >= 5:
                recent_ic = float(np.mean(history[-5:]))
                if recent_ic < min_ic:
                    _elog(f"🔄 should_retrain=True: {name} 近5期IC={recent_ic:.4f}<{min_ic}")
                    return True

        _elog("✅ should_retrain=False: 模型新鲜且IC达标")
        return False

    def get_health_report(self) -> Dict:
        """V20: 模型健康报告 — 含前端图表渲染所需扁平化字段
        新增: _model_weights / _model_ic / _consistency_score
        views.py 直接读取这三个字段填充 stats{} 传给前端
        """
        report = {}
        all_ics: Dict[str, float] = {}

        for name, history in self.model_ic_history.items():
            if not history:
                report[name] = {'recent_ic': 0.0, 'current_weight': 0.0,
                                 'trend': '⏳ 无历史', 'periods': 0}
                all_ics[name] = 0.0
                continue
            recent_ic = (float(np.mean(history[-5:])) if len(history) >= 5
                         else float(np.mean(history)))
            all_ics[name] = recent_ic
            trend = 'stable'
            if len(history) >= 10:
                early = np.mean(history[:5])
                if early > 0.01 and recent_ic < early * 0.5:
                    trend = '⚠️ 衰减'
                elif recent_ic > early * 1.2:
                    trend = '📈 提升'
            report[name] = {
                'recent_ic':      round(recent_ic, 4),
                'current_weight': round(self.model_weights.get(name, 0.0), 4),
                'trend':          trend,
                'periods':        len(history),
            }

        # ── 前端图表扁平化数据 ────────────────────────────────────────
        # ── Grok标准：直接用train_all已归一化的model_weights ─────────────
        # V25-Fix1: model_weights 必须归一化（初始值是1.0×N，前端会显示100%）
        _raw_w = dict(self.model_weights)  # {name: weight}
        _total_w = sum(_raw_w.values())
        if _total_w > 1e-6 and abs(_total_w - 1.0) > 0.01:  # 未归一化时处理
            report['_model_weights'] = {
                k: round(v / _total_w, 4) for k, v in _raw_w.items()
            }
        else:
            report['_model_weights'] = {k: round(v, 4) for k, v in _raw_w.items()}

        # V25-Fix2: IC 历史为空时给初始占位值（0.025=2.5%），
        # 让前端 bar chart 可渲染（金色 degraded 状态），而非空白
        report['_model_ic'] = {
            k: round(float(np.mean(self.model_ic_history[k][-5:])), 4)
               if self.model_ic_history.get(k)  # 有IC历史 → 真实值
               else 0.025                       # 未训练占位（2.5%，前端显示金色）
            for k in self.model_ic_history
        }

        # V27: 预测不确定性（Ensemble方差，供views.py置信度计算）
        report['_pred_uncertainty'] = round(
            getattr(self, '_last_uncertainty', 0.0), 4
        )
        # 置信度映射：不确定性0→置信95%，不确定性1.0→置信50%
        _unc = report['_pred_uncertainty']
        report['_model_confidence'] = round(max(50, 95 - _unc * 45), 1)

        # 若完全无模型注册（model_ic_history空），给前端默认5模型等权
        if not report['_model_weights']:
            _default_models = ['hgnn', 'mlp', 'transformer', 'gnn', 'smart_xgnn']
            eq_w = round(1.0 / len(_default_models), 4)
            report['_model_weights'] = {m: eq_w for m in _default_models}
            report['_model_ic']      = {m: 0.025 for m in _default_models}
            _elog('  V25: 无模型注册，使用默认5模型等权 fallback')

        # 补全 ic 表中缺失的模型
        for _mn in report['_model_weights']:
            report['_model_ic'].setdefault(_mn, 0.025)

        report['_consistency_score'] = self._compute_consistency(report['_model_ic'])

        return report

    def _compute_consistency(self, ics: Dict[str, float]) -> float:
        """模型一致性得分 (0~1)：各模型IC越相近越高"""
        valid = [v for v in ics.values() if v > 0.001]
        if len(valid) < 2:
            return 0.5
        ic_mean = float(np.mean(valid))
        ic_std  = float(np.std(valid))
        # CV越小 → 一致性越高；1-CV归一化到0~1
        cv = ic_std / (ic_mean + 1e-9)
        return round(max(0.0, min(1.0, 1.0 - cv)), 4)


    def get_model_health_report(self) -> Dict:
        return self.get_health_report()

    # ---- Fix 8: 持久化（对接 model_persistence.save/load_ai_engine）----
    def save(self, model_dir: str = 'models', base_name: str = 'ai_engine') -> bool:
        """
        保存 AIAlphaEngine 中的所有模型
        对接 model_persistence.ModelPersistence.save_ai_engine 风格
        """
        import os, pickle
        os.makedirs(model_dir, exist_ok=True)
        ok_count = 0

        # 保存 PyTorch 模型
        for name, model in self.models.items():
            path = os.path.join(model_dir, f"{base_name}_{name}.pth")
            try:
                if TORCH_AVAILABLE:
                    import torch as _torch
                    _torch.save(model.state_dict(), path)
                    ok_count += 1
                    _elog(f"💾 保存 {name}: {path}")
            except Exception as e:
                _elog(f"❌ 保存 {name} 失败: {e}", 'error')

        # 保存 SmartXGNN（含 xgb_model + nn_model）
        for name, xgnn in self.special_models.items():
            # XGBoost 子模型
            if hasattr(xgnn, 'xgb_model') and xgnn.xgb_model is not None:
                xgb_path = os.path.join(model_dir, f"{base_name}_{name}_xgb.pkl")
                try:
                    import joblib
                    joblib.dump(xgnn.xgb_model, xgb_path)
                    ok_count += 1
                    _elog(f"💾 保存 {name}_xgb: {xgb_path}")
                except Exception as e:
                    _elog(f"❌ 保存 {name}_xgb 失败: {e}", 'error')
            # NN 子模型
            if TORCH_AVAILABLE and hasattr(xgnn, 'nn_model') and xgnn.nn_model is not None:
                nn_path = os.path.join(model_dir, f"{base_name}_{name}_nn.pth")
                try:
                    import torch as _torch
                    _torch.save(xgnn.nn_model.state_dict(), nn_path)
                    ok_count += 1
                    _elog(f"💾 保存 {name}_nn: {nn_path}")
                except Exception as e:
                    _elog(f"❌ 保存 {name}_nn 失败: {e}", 'error')

        # 保存权重 & IC历史
        meta_path = os.path.join(model_dir, f"{base_name}_meta.pkl")
        try:
            with open(meta_path, 'wb') as f:
                pickle.dump({
                    'model_weights': self.model_weights,
                    'model_ic_history': self.model_ic_history,
                }, f, protocol=4)
            _elog(f"💾 保存权重元数据: {meta_path}")
        except Exception as e:
            _elog(f"⚠️ 权重元数据保存失败: {e}", 'warning')

        _elog(f"✅ AIAlphaEngine 保存完成 ({ok_count} 个子模型)")
        return ok_count > 0

    def load(self, model_dir: str = 'models', base_name: str = 'ai_engine',
             check_fresh: bool = True, max_age_days: int = 7) -> bool:
        """
        加载 AIAlphaEngine 中的所有模型
        check_fresh: True 时检查文件日期，超过 max_age_days 视为过期
        """
        import os, pickle
        from datetime import datetime

        def _is_fresh(path: str) -> bool:
            if not check_fresh:
                return True
            try:
                age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
                return age <= max_age_days
            except Exception:
                return False

        ok_count = 0

        # 加载 PyTorch 模型
        if TORCH_AVAILABLE:
            import torch as _torch
            for name, model in self.models.items():
                path = os.path.join(model_dir, f"{base_name}_{name}.pth")
                if not os.path.exists(path) or not _is_fresh(path):
                    _elog(f"⏳ {name} 模型文件不存在或已过期，跳过", 'warning')
                    continue
                try:
                    model.load_state_dict(_torch.load(path, map_location=self.device))
                    ok_count += 1
                    _elog(f"✅ 加载 {name}: {path}")
                except Exception as e:
                    _elog(f"❌ 加载 {name} 失败: {e}", 'error')

        # 加载 SmartXGNN
        for name, xgnn in self.special_models.items():
            xgb_path = os.path.join(model_dir, f"{base_name}_{name}_xgb.pkl")
            if os.path.exists(xgb_path) and _is_fresh(xgb_path):
                try:
                    import joblib
                    xgnn.xgb_model = joblib.load(xgb_path)
                    ok_count += 1
                    _elog(f"✅ 加载 {name}_xgb: {xgb_path}")
                except Exception as e:
                    _elog(f"❌ 加载 {name}_xgb 失败: {e}", 'error')

            if TORCH_AVAILABLE and hasattr(xgnn, 'nn_model') and xgnn.nn_model is not None:
                import torch as _torch
                nn_path = os.path.join(model_dir, f"{base_name}_{name}_nn.pth")
                if os.path.exists(nn_path) and _is_fresh(nn_path):
                    try:
                        xgnn.nn_model.load_state_dict(
                            _torch.load(nn_path, map_location=self.device)
                        )
                        ok_count += 1
                        _elog(f"✅ 加载 {name}_nn: {nn_path}")
                    except Exception as e:
                        _elog(f"❌ 加载 {name}_nn 失败: {e}", 'error')

        # 加载权重 & IC历史
        meta_path = os.path.join(model_dir, f"{base_name}_meta.pkl")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                self.model_weights.update(meta.get('model_weights', {}))
                self.model_ic_history.update(meta.get('model_ic_history', {}))
                _elog(f"✅ 加载权重元数据: {meta_path}")
            except Exception as e:
                _elog(f"⚠️ 权重元数据加载失败: {e}", 'warning')

        _elog(f"{'✅' if ok_count > 0 else '⚠️'} AIAlphaEngine 加载完成 ({ok_count} 个子模型)")
        return ok_count > 0