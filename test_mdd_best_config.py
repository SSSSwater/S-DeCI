import argparse
import random
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

import numpy as np
import torch

from exp.exp_classification_CV import Exp_Main


METRIC_NAMES = ("accuracy", "precision", "recall", "macro_f1", "roc_auc", "train_seconds")
RESULT_METRIC_COLUMNS = ("accuracy", "precision", "recall", "macro_f1", "auc", "roc_auc")


def infer_site_count(data_path, protocol):
    pattern = f"*_{protocol}_features_timeseries.mat"
    sites = set()
    for path in Path(data_path).rglob(pattern):
        match = re.search(r"_(s\d+)[_-]", path.name.lower())
        sites.add(match.group(1) if match else "unknown")
    return max(len(sites), 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行当前最佳 MDD 配置的独立测试入口（S-DeCI 四模块）。"
    )

    g = parser.add_argument_group("== 数据与任务 ==")
    g.add_argument("--data", default="MDD", help="数据集名称")
    g.add_argument("--data-path", default="dataset/MDD", help="数据集目录（相对路径相对本文件）")
    g.add_argument("--data-type", default="TS", choices=("TS", "FC"), help="输入类型：TS 原始时间序列 / FC 功能连接矩阵")
    g.add_argument("--protocol", default="AAL116", help="脑分区模板，决定 ROI 数（channel）")
    g.add_argument("--channel", type=int, default=116, help="ROI/脑区数量（多变量通道数）")
    g.add_argument("--seq-len", type=int, default=230, help="输入时间长度；短于此的样本会被过滤")
    g.add_argument("--classes", type=int, default=2, help="分类类别数")
    g.add_argument("--kfold", type=int, default=5, help="K 折交叉验证折数")
    g.add_argument("--max-folds", type=int, default=5, help="实际运行的最大折数；=kfold 时才写入 result.xlsx")
    g.add_argument("--iterations", type=int, default=1, help="重复整套交叉验证的次数（多 seed 求均值）")

    g = parser.add_argument_group("== 训练与优化 ==")
    g.add_argument("--train-epochs", type=int, default=50, help="训练 epoch 数")
    g.add_argument("--batch-size", type=int, default=32, help="batch 大小")
    g.add_argument("--learning-rate", type=float, default=1e-3, help="主优化器学习率")
    g.add_argument("--lradj", default="cosine", choices=("constant", "cosine", "type1", "type2", "type3", "type4"), help="学习率调整策略")
    g.add_argument("--weight-decay", type=float, default=1e-2, help="权重衰减（小样本抗过拟合）")
    g.add_argument("--dropout", type=float, default=0.2, help="全局 dropout 比例")
    g.add_argument("--d-model", type=int, default=32, help="模型隐藏维度")
    g.add_argument("--loss", default="bce", help="损失：mse/bce/weighted_bce/weighted_mse（二分类）或 ce（多分类）")
    g.add_argument("--binary-positive-weight", type=float, default=1.0, help="weighted_bce/mse 正类权重（按类别比可设 ~1.96）")
    g.add_argument("--class-loss-weighting", default="sqrt_batch_balanced", choices=("none", "batch_balanced", "sqrt_batch_balanced", "logit_adjusted", "sqrt_logit_adjusted"), help="分类 CE 的类别均衡方式；MDD 5fold/50epoch 当前默认 sqrt_batch_balanced 更稳")
    g.add_argument("--class-label-smoothing", type=float, default=0.0, help="分类交叉熵 label smoothing；ABIDE 等小样本可用于降低后期过拟合置信度")
    g.add_argument("--class-logit-adjust-tau", type=float, default=1.0, help="logit_adjusted 类别先验校正强度")
    g.add_argument("--class-prior-alignment-weight", type=float, default=0.0, help="训练期约束最终概率均值贴近当前batch类别比例；用于缓解raw 0.5下正类预测过少")
    g.add_argument("--use-final-logit-calibration", type=int, default=0, help="是否学习最终logit尺度/偏置；1fold测试为负效果，默认关闭")
    g.add_argument("--patience", type=int, default=50, help="early stopping 容忍轮数")
    g.add_argument("--seed", type=int, default=2024, help="随机种子")
    g.add_argument("--use-model-ema", type=int, default=1, help="是否对权重做 EMA（降方差，配 cosine 更稳）")
    g.add_argument("--use-norm", type=int, default=1, help="是否启用可逆输入归一化")

    g = parser.add_argument_group("== 模块开关（消融）==")
    g.add_argument("--use-deci-module1", type=int, default=1, help="模块1（DeCI/ALFF 特征提取）开关")
    g.add_argument("--use-causal-module2", type=int, default=1, help="模块2（因果图学习）开关")
    g.add_argument("--use-hyperbolic-modules34", type=int, default=1, help="模块3/4（双曲 HGCN+HPEC）总开关")
    g.add_argument("--use-hgcn-module3", type=int, default=1, help="模块3 单独开关；MDD 当前最佳默认开启")
    g.add_argument("--use-hpec-module4", type=int, default=1, help="模块4 单独开关；默认开启，用于验证完整四模块路线")

    g = parser.add_argument_group("== 模块1 特征 ==")
    g.add_argument("--module1-feature-mode", default="alff", choices=("alff", "deci", "raw"), help="模块1 特征：alff 低频生理特征 / deci 旧Cycle分解 / raw 原序列投影")
    g.add_argument("--module1-alff-time-weight", type=float, default=0.2, help="模块1 ALFF描述子与低频重构时序特征的融合权重")
    g.add_argument("--module1-random-crop", type=int, default=1, help="训练期是否对长序列随机时间窗裁剪")
    g.add_argument("--module1-temporal-dropout", type=float, default=0.03, help="训练期随机置零部分时间点的比例")
    g.add_argument("--module1-roi-dropout", type=float, default=0.02, help="训练期随机置零部分脑区的比例")
    g.add_argument("--module1-denoise-loss-weight", type=float, default=0.0, help="模块1 去噪辅助损失权重，0 关闭")
    g.add_argument("--module1-temporal-stats-weight", type=float, default=0.0, help="模块1 时序统计残差融合权重，0 关闭")

    g = parser.add_argument_group("== 模块2 因果 ==")
    g.add_argument("--causal-learning-target", default="temporal_sem", choices=("static_feature", "temporal_sem"), help="模块2 学习目标：temporal_sem 时序SEM / static_feature 旧静态")
    g.add_argument("--causal-graph-method", default="nts_notears", choices=("nts_notears", "attn_nts_notears", "dagma_logdet", "dag_sampling"), help="因果图方法：nts_notears / attn_* 注意力引导 / 后两者旧对照")
    g.add_argument("--temporal-lag-order", type=int, default=5, help="时序SEM 历史滞后阶数")
    g.add_argument("--temporal-causal-init-logit", type=float, default=-4.0, help="模块2时序因果边初始化logit，越小初始图越稀疏")
    g.add_argument("--temporal-sem-input-norm", default="time_zscore", choices=("none", "time_zscore", "zscore", "batch_zscore"), help="模块2时序SEM输入归一化方式")
    g.add_argument("--temporal-prediction-loss-mode", default="bold_alff", choices=("mse", "huber", "bold_alff"), help="模块2预测损失：bold_alff=稳健预测+动态变化+低频趋势+时序相关")
    g.add_argument("--temporal-pred-huber-delta", type=float, default=1.0, help="模块2 Huber 预测损失的 delta，降低 BOLD 异常点影响")
    g.add_argument("--lambda-temporal-pred-delta", type=float, default=0.2, help="模块2时间差分预测损失权重，用于保留 BOLD 动态变化")
    g.add_argument("--lambda-temporal-pred-lowfreq", type=float, default=0.2, help="模块2低频趋势预测损失权重，用于贴合 ALFF/fALFF 低频信号")
    g.add_argument("--lambda-temporal-pred-corr", type=float, default=0.05, help="模块2 ROI 时序相关损失权重，用于保留波形相位/形状")
    g.add_argument("--temporal-lowfreq-kernel-size", type=int, default=9, help="模块2低频趋势移动平均窗口，奇数更稳定")
    g.add_argument("--temporal-a0-sparse-ratio", type=float, default=0.2, help="A0 同时刻残余依赖在稀疏损失中的相对权重")
    g.add_argument("--temporal-a0-scale", type=float, default=0.03, help="A0 同时刻残余修正的最大相对尺度；默认保持弱作用")
    g.add_argument("--temporal-prediction-target-mode", default="innovation", choices=("innovation", "next_value"), help="模块2预测目标：innovation 学习相对上一时刻的变化，next_value 直接预测下一时刻")
    g.add_argument("--lambda-temporal-pred", type=float, default=1.0, help="时序预测损失权重")
    g.add_argument("--lambda-temporal-sparse", type=float, default=0.0005, help="因果图稀疏正则权重")
    g.add_argument("--lambda-temporal-smooth", type=float, default=0.0001, help="跨滞后平滑正则权重")
    g.add_argument("--lambda-temporal-group-sparse", type=float, default=0.0, help="按 lag 组稀疏整条因果边的权重；MDD sweep 中未带来额外收益，默认关闭")
    g.add_argument("--lambda-temporal-lag-hierarchy", type=float, default=0.0, help="鼓励近 lag 优先、抑制远 lag 偶然边的层级正则权重；MDD sweep 中未带来额外收益，默认关闭")
    g.add_argument("--temporal-candidate-parent-topk", type=int, default=4, help="每个目标节点预筛候选父节点数；MDD 5fold/50epoch sweep 中 topk=4 不降分类指标且图更稀疏有向")
    g.add_argument("--temporal-sample-graph-delta-scale", type=float, default=0.15, help="模块2样本级时序残差图A_delta的强度上限")
    g.add_argument("--temporal-sample-lag-graph-mode", default="abs", choices=("abs", "positive", "signed_abs"), help="样本级lag图的符号处理方式")
    g.add_argument("--temporal-sample-graph-rank", type=int, default=4, help="样本级时序残差图的低秩rank")
    g.add_argument("--temporal-decoder-activation", default="identity", choices=("sigmoid", "tanh", "gelu", "identity", "none"), help="模块2局部预测decoder的激活函数；零均值BOLD默认保留符号")
    g.add_argument("--temporal-dagma-warmup-epochs", type=int, default=5, help="模块2 A0 DAG约束预热epoch数")
    g.add_argument("--temporal-dagma-barrier-epochs", type=int, default=20, help="模块2 DAGMA barrier调度epoch数")
    g.add_argument("--temporal-attention-heads", type=int, default=1, help="attn_nts_notears的注意力头数")
    g.add_argument("--temporal-attention-head-dim", type=int, default=4, help="attn_nts_notears每个注意力头维度")
    g.add_argument("--temporal-attention-dropout", type=float, default=0.1, help="attn_nts_notears注意力dropout")
    g.add_argument("--temporal-attention-graph-scale", type=float, default=1.0, help="attn_nts_notears输出分类图的缩放系数")
    g.add_argument("--causal-learning-rate", type=float, default=5e-4, help="模块2 因果图参数单独学习率")
    g.add_argument("--lambda-causal-dag", type=float, default=0.0001, help="DAG 无环约束损失权重")
    g.add_argument("--lambda-causal-l1", type=float, default=0.00001, help="因果图 L1 稀疏权重")
    g.add_argument("--lambda-causal-stability", type=float, default=0.0, help="模块2跨batch图稳定性正则权重")
    g.add_argument("--classification-graph-source", default="causal_soft_masked_fc", choices=("blend", "learned", "causal", "sample_correlation", "fc", "residual_blend", "topk_blend", "gated_fc", "causal_masked_fc", "causal_soft_masked_fc", "gated_fc_centered", "gated_fc_signed"), help="分类用图来源：causal_soft_masked_fc=因果候选边保留FC权重、非候选边保留少量FC底噪（MDD默认）；gated_fc=样本FC按学习因果图门控；causal_masked_fc=只在因果候选边上保留样本FC边权")
    g.add_argument("--module2-sample-correlation-blend", type=float, default=0.75, help="分类图里样本FC与学习因果图的混合比例(0~1)")
    g.add_argument("--module2-graph-residual-alpha", type=float, default=0.10, help="因果图作为FC方向残差的强度；causal_soft_masked_fc 下表示非候选边保留的FC底噪比例")
    g.add_argument("--detach-module2-graph-for-classification", type=int, default=0, help="是否阻断分类loss回传到模块2图参数")
    g.add_argument("--freeze-causal-after-epoch", type=int, default=-1, help="到指定 epoch 后冻结模块2因果图学习；-1 表示不冻结")
    g.add_argument("--use-sample-graph-residual", type=int, default=1, help="是否启用样本级残差图 A_delta")
    g.add_argument("--sample-graph-delta-scale", type=float, default=0.05, help="[旧/静态]样本级残差图A_delta的强度上限")
    g.add_argument("--sample-graph-hidden-dim", type=int, default=0, help="[旧/静态]样本级残差图MLP隐藏维度，0为自动")
    g.add_argument("--lambda-sample-graph-l1", type=float, default=0.0, help="样本级残差图L1正则权重")
    g.add_argument("--lambda-sample-graph-deviation", type=float, default=0.0, help="样本级残差图偏离共享图的约束权重")
    g.add_argument("--causal-edge-dropout", type=float, default=0.25, help="模块3训练期对因果边/分类图边的dropout比例")
    g.add_argument("--sample-correlation-mode", default="abs", choices=("abs", "positive", "raw"), help="样本相关矩阵作图时负值处理：abs/positive/raw")
    g.add_argument("--use-sample-correlation-when-module2-disabled", type=int, default=1, help="模块2关闭但模块3开启时用样本相关矩阵作邻接")
    # 模块2 旧方法（保留作对照）
    g.add_argument("--causal-feature-source", default="sum", choices=("sum", "last"), help="[旧]静态路 Cycle 特征来源：sum/last")
    g.add_argument("--causal-init-logit", type=float, default=-1.0, help="[旧]静态因果邻接 logit 初始化")
    g.add_argument("--lambda-causal-recon", type=float, default=0.01, help="[旧]静态路重构损失权重")

    g = parser.add_argument_group("== 模块3 HGCN / LP ==")
    g.add_argument("--hgcn-hidden-dim", type=int, default=64, help="模块3 输出/分类特征维度")
    g.add_argument("--hgcn-layers", type=int, default=1, help="HGCN 图卷积层数")
    g.add_argument("--hgcn-dropout", type=float, default=0.6, help="模块3 切空间特征 dropout")
    g.add_argument("--use-multi-hop-causal-encoding", type=int, default=0, help="是否用模块2有向图的多阶可达性编码增强模块3输入")
    g.add_argument("--causal-reachability-hops", type=int, default=2, help="多阶因果可达性最大hop数，建议1至3")
    g.add_argument("--causal-reachability-scale", type=float, default=0.25, help="多阶因果编码的残差注入强度")
    g.add_argument("--hgcn-residual-alpha", type=float, default=0.35, help="HGCN 输出与输入双曲特征残差混合比")
    g.add_argument("--hgcn-graph-readout-alpha", type=float, default=0.0, help="模块3 readout 中图/FC门控加权均值的参与比例；0为纯均值，越大越依赖节点权重")
    g.add_argument("--hgcn-readout-mode", default="mean_std", choices=("mean_std", "causal_weighted_mean_std", "graph_weighted_mean_std", "node_stats", "network_stats", "causal_attention", "causal_subnetwork", "network_gated_node_stats"), help="模块3 读出：mean_std为默认主线；causal_weighted_mean_std用因果图软权重计算均值+标准差")
    g.add_argument("--hgcn-causal-attention-heads", type=int, default=4, help="模块3 causal_attention readout 的注意力头数")
    g.add_argument("--hgcn-causal-attention-graph-weight", type=float, default=0.5, help="模块3 causal_attention 中因果图节点先验的权重")
    g.add_argument("--hgcn-causal-subnetwork-count", type=int, default=4, help="causal_subnetwork 读出时自动抽取的因果子网络数量")
    g.add_argument("--hgcn-causal-subnetwork-topk", type=int, default=12, help="每个因果子网络保留的ROI数量")
    g.add_argument("--hgcn-causal-subnetwork-weight", type=float, default=0.5, help="因果子网络读出融合到全局双曲中心点的权重")
    g.add_argument("--hgcn-einstein-readout-weight", type=float, default=0.0, help="模块3用Poincare Einstein midpoint校正全局双曲中心点方向的权重；0关闭")
    g.add_argument("--use-causal-role-readout", type=int, default=0, help="是否按有向图读取global/source/sink/hub四个双曲中心")
    g.add_argument("--causal-role-temperature", type=float, default=1.0, help="因果角色ROI权重的softmax温度")
    g.add_argument("--hgcn-network-gate-strength", type=float, default=0.35, help="network_gated_node_stats 中样本FC网络强度对ROI池化权重的调节幅度")
    g.add_argument("--hgcn-use-graph-degree-encoding", type=int, default=0, help="是否把模块2/分类图的入度和出度作为ROI位置编码注入模块3输入")
    g.add_argument("--hgcn-graph-degree-encoding-weight", type=float, default=0.1, help="模块3图度数位置编码权重")
    g.add_argument("--hgcn-use-radius-head", type=int, default=0, help="是否启用模块3可学习径向门控；5fold未带来稳定收益，默认关闭")
    g.add_argument("--hgcn-radius-min-ratio", type=float, default=0.25, help="模块3径向门控下界占 backclip 半径比例")
    g.add_argument("--use-brain-network-prior", type=int, default=0, help="模块3 是否启用 AAL 脑网络先验 readout")
    g.add_argument("--use-hgcn-radial-calibration", type=int, default=0, help="是否校准模块3双曲中心点半径，防止后期贴近原点")
    g.add_argument("--hgcn-radial-min", type=float, default=0.25, help="模块3半径校准下界")
    g.add_argument("--hgcn-radial-max", type=float, default=0.75, help="模块3半径校准上界")
    g.add_argument("--hgcn-fc-inject-weight", type=float, default=0.0, help="把网络级FC生物标志注入模块3 z_tangent 的权重；默认关闭，避免欧氏特征直接污染双曲切空间")
    g.add_argument("--module34-film-weight", type=float, default=0.0, help="用FC/网络连接条件对模块3双曲中心点做FiLM调制；0表示关闭")
    g.add_argument("--module34-film-max-scale", type=float, default=0.25, help="FiLM尺度调制的最大幅度")
    g.add_argument("--module34-film-shift-norm", type=float, default=0.20, help="FiLM平移项最大范数，避免污染双曲空间")
    g.add_argument("--hyperbolic-logit-residual-weight", type=float, default=0.5, help="双曲原型evidence进入最终logit融合的权重；dual_consensus 下表示模块3/4最低占比")
    g.add_argument("--hyperbolic-residual-fusion-mode", default="dual_consensus", choices=("residual", "logit_blend", "binary_margin", "dual_consensus", "dual_margin_consensus"), help="模块3/4与欧氏局部结构evidence的logit融合方式；dual_margin_consensus 只融合正负类margin方向证据")
    g.add_argument("--hyperbolic-residual-norm", default="temperature", choices=("sample", "temperature", "none"), help="双曲evidence增量归一化：temperature保留HPEC证据强弱（MDD默认）；sample逐样本归一化容易放大弱证据噪声")
    g.add_argument("--hyperbolic-residual-temperature", type=float, default=1.0, help="hyperbolic-residual-norm=temperature 时的温度")
    g.add_argument("--hyperbolic-residual-margin-gain", type=float, default=0.0, help="按模块4原型概率margin放大双曲evidence强度；0表示关闭")
    g.add_argument("--hyperbolic-residual-margin-max-scale", type=float, default=2.0, help="双曲evidence按margin放大的最大倍率")
    g.add_argument("--use-hyperbolic-residual-bias", type=int, default=1, help="是否给模块3/4双曲evidence添加可学习类别偏置，用于训练内校准正负类证据")
    g.add_argument("--keep-gcn-fallback-with-hyperbolic", type=int, default=1, help="开模块3/4时仍构建欧氏局部结构分支，用于与双曲原型evidence做logit级融合")
    g.add_argument("--lorentz-layers", type=int, default=1, help="[LP] 有向Lorentz图卷积层数")
    g.add_argument("--lorentz-curvature", type=float, default=1.0, help="[LP] Lorentz/Poincaré 曲率")
    g.add_argument("--lorentz-dropout", type=float, default=0.3, help="[LP] 图卷积 dropout")
    g.add_argument("--lorentz-alpha-out-init", type=float, default=0.5, help="[LP] 出边聚合分支初始权重")
    g.add_argument("--lorentz-message-residual-weight", type=float, default=0.0, help="[LP] 原始邻域消息残差权重；完整测试为负效果，默认关闭")
    g.add_argument("--lorentz-message-gate-init", type=float, default=0.5, help="[LP] 因果图消息直通门初值；调大可避免双曲图消息在早期被压没")
    g.add_argument("--lorentz-centroid-message-weight", type=float, default=0.0, help="[LP] 使用Lorentz centroid几何消息替代部分切空间value消息的比例")
    g.add_argument("--lorentz-max-tangent-norm", type=float, default=1.0, help="[LP] Lorentz/Poincaré桥接前切空间最大范数")
    g.add_argument("--lp-use-network-readout", type=int, default=1, help="[LP] 是否按AAL功能网络先汇总再生成全局双曲点")
    g.add_argument("--lp-network-readout-blend", type=float, default=0.75, help="[LP] 网络级readout相对全脑readout的融合权重")
    g.add_argument("--lp-stats-update-gate-init", type=float, default=0.0, help="[LP] 统计读出增强门控初值；0 为中性小步残差，避免默认全量统计残差推满半径")
    g.add_argument("--lp-use-dynamic-radius", type=int, default=1, help="[LP] 是否使用样本自适应双曲半径；默认开启，避免所有样本被压到同一双曲半径")
    g.add_argument("--lp-dynamic-radius-min-ratio", type=float, default=0.15, help="[LP] 动态半径下限占 max_tangent_norm 的比例")
    g.add_argument("--lp-dynamic-radius-max-ratio", type=float, default=0.95, help="[LP] 动态半径上限占 max_tangent_norm 的比例")
    g.add_argument("--lp-dynamic-radius-source", default="norm", choices=("norm", "graph_context"), help="[LP] 动态半径来源：norm 原范数策略 / graph_context 图结构上下文策略")
    g.add_argument("--lp-input-residual-weight", type=float, default=0.0, help="[LP] readout 前保留 Lorentz lifting 初始节点表示的权重")
    g.add_argument("--lp-poincare-readout-weight", type=float, default=0.0, help="[LP] 庞加莱Einstein midpoint readout校正权重；短测未带来收益，默认关闭")
    g.add_argument("--lp-mac-clip-mode", default="soft", choices=("hard", "soft"), help="[LP] MAC半径处理：hard强制安全环带，soft仅限制越界并保留小半径差异")
    g.add_argument("--module34-supcon-loss-weight", type=float, default=0.0, help="模块3/4切空间监督对比损失权重")
    g.add_argument("--module34-supcon-temperature", type=float, default=0.2, help="模块3/4切空间监督对比温度")
    g.add_argument("--module34-center-loss-weight", type=float, default=0.0, help="模块3 z_tangent 类中心分离损失权重；0表示关闭")
    g.add_argument("--module34-center-margin", type=float, default=0.5, help="模块3 类中心之间允许的最大余弦相似度")
    g.add_argument("--module34-center-intra-weight", type=float, default=1.0, help="模块3 类内靠近中心项权重")
    g.add_argument("--module34-center-inter-weight", type=float, default=1.0, help="模块3 类间中心分离项权重")
    g.add_argument("--module34-branch-ce-loss-weight", type=float, default=0.2, help="训练期约束模块3/4自身logits可分类的辅助CE权重，避免完整模型绕过模块3/4")
    g.add_argument("--module34-branch-ce-decay-epochs", type=int, default=0, help="模块3/4分支CE在前N个epoch线性衰减；0表示不调度")
    g.add_argument("--module34-branch-ce-min-ratio", type=float, default=1.0, help="模块3/4分支CE衰减后的最低比例，避免完全关闭模块监督")
    g = parser.add_argument_group("== 模块4 HPEC ==")
    g.add_argument("--hpec-classification-mode", default="energy_primary", choices=("energy_primary", "prototype_primary", "energy_prototype_residual", "distance_prototype", "tangent_primary", "tangent_prototype", "feature_fusion", "energy_calibrated"), help="模块4分类策略：energy_primary 用HPEC能量作主边界；distance_prototype 用双曲距离多原型证据作边界")
    g.add_argument("--hpec-energy-mode", default="busemann", choices=("cone", "busemann"), help="能量模式：busemann 边界理想原型（MDD 5fold/50epoch 当前最好）/ cone 蕴含锥")
    g.add_argument("--hpec-loss-mode", default="energy_ce", choices=("margin", "energy_ce"), help="模块4 主损失：margin / energy_ce")
    g.add_argument("--hpec-evidence-weight", type=float, default=1.25, help="energy_primary中prototype similarity证据叠加到HPEC energy logits的权重；MDD 5fold/50epoch 当前 1.25 最好")
    g.add_argument("--hpec-avoid-busemann-double-count", type=int, default=0, help="Busemann模式下避免energy logits和prototype similarity使用同一Busemann证据重复叠加；完整MDD实验中默认保留叠加效果更好")
    g.add_argument("--hpec-prototypes-per-class", type=int, default=2, help="每类原型数")
    g.add_argument("--hpec-proto-temperature", type=float, default=0.6, help="多原型相似度/soft-min 温度")
    g.add_argument("--hpec-busemann-temperature", type=float, default=2.0, help="Busemann 多原型聚合温度；MDD 5fold/50epoch 中 2.0 的 Macro-F1/AUC 更好，表示更平滑地融合多原型证据")
    g.add_argument("--hpec-busemann-point-radius", type=float, default=0.0, help="Busemann打分前把样本投到固定切空间半径；0表示保留原半径，>0时更强调类别方向而非半径置信度")
    g.add_argument("--hpec-busemann-radius-gate-weight", type=float, default=0.0, help="Busemann分数的样本半径门控强度；0关闭，>0时用z半径轻量调节方向证据")
    g.add_argument("--hpec-busemann-radius-gate-center", type=float, default=0.3, help="Busemann半径门控中心；半径高于该值时增强证据，低于该值时减弱证据")
    g.add_argument("--hpec-busemann-class-bias-weight", type=float, default=0.0, help="Busemann 每类边界偏置强度；>0时允许模块4学习每类horosphere偏移，默认0保持当前最佳")
    g.add_argument("--hpec-prototype-energy-blend", type=float, default=0.0, help="prototype_primary 中融合HPEC energy logits的比例；短测负效果，默认0")
    g.add_argument("--hpec-prototype-logit-mode", default="normalized", choices=("normalized", "margin_preserving"), help="prototype_primary 的logits处理：normalized为旧的逐样本归一化；margin_preserving保留原型相似度差的幅度")
    g.add_argument("--hpec-prototype-logit-scale", type=float, default=1.0, help="margin_preserving模式下原型logits整体缩放")
    g.add_argument("--hpec-residual-calibration", default="batch_margin", choices=("none", "batch_margin", "tanh_margin", "running_batch_margin", "hybrid_batch_running_margin", "train_class_margin"), help="模块4校准；train_class_margin只用训练折标签估计固定边界")
    g.add_argument("--hpec-residual-calibration-scale", type=float, default=0.5, help="batch_margin校准后的模块4残差margin尺度")
    g.add_argument("--hpec-residual-calibration-momentum", type=float, default=0.05, help="running_batch_margin 的训练期EMA动量")
    g.add_argument("--hpec-residual-calibration-batch-weight", type=float, default=0.5, help="hybrid_batch_running_margin 测试期当前batch统计的混合权重")
    g.add_argument("--hpec-prototype-residual-weight", type=float, default=0.2, help="energy_prototype_residual 模式下 prototype 残差权重")
    g.add_argument("--hpec-margin", type=float, default=0.5, help="HPEC margin loss 的间隔")
    g.add_argument("--hpec-distance-weight", type=float, default=0.5, help="能量中 Poincaré 距离项权重")
    g.add_argument("--hpec-data-init", type=int, default=0, help="是否用首个训练 batch 初始化 HPEC 原型；MDD 5fold/50epoch 当前最佳为0，保留均匀分散的原型先验")
    g.add_argument("--hpec-trainable-prototypes", type=int, default=1, help="原型是否直接参与梯度训练；MDD 5fold/50epoch 当前最佳为1，让Busemann方向原型随分类目标微调")
    g.add_argument("--hpec-prototype-lr-scale", type=float, default=1.0, help="HPEC原型/边界偏置的学习率缩放；<1表示慢速原型更新，默认1保持当前最佳完整实验行为")
    g.add_argument("--hpec-prototype-parameterization", default="poincare_point", choices=("poincare_point", "tangent_direction"), help="HPEC原型参数化：poincare_point为现有Poincare点；tangent_direction为固定半径、只学习Busemann理想方向")
    g.add_argument("--hpec-use-sinkhorn-ema", type=int, default=1, help="是否用Sinkhorn均衡分配+EMA慢更新多原型")
    g.add_argument("--hpec-prototype-update-mode", default="reliable_tp_ema", choices=("reliable_tp_ema", "epoch_reliable_frechet_ema", "sinkhorn_ema", "none"), help="原型更新模式：epoch_reliable_frechet_ema按完整epoch执行独立流形EMA")
    g.add_argument("--hpec-reliable-confidence-threshold", type=float, default=0.70, help="可靠TP原型更新的真实类最小预测概率")
    g.add_argument("--hpec-reliable-view-consistency-threshold", type=float, default=0.55, help="启用互补视图时可靠TP要求的最小切空间余弦一致性")
    g.add_argument("--hpec-reliable-min-samples", type=int, default=2, help="一个类内原型执行可靠EMA所需的最少TP样本数")
    g.add_argument("--hpec-reliable-weight-floor", type=float, default=0.05, help="epoch流形EMA中低置信样本的最小连续权重")
    g.add_argument("--hpec-epoch-frechet-steps", type=int, default=3, help="epoch原型目标中心的Karcher迭代次数")
    g.add_argument("--hpec-ema-start-epoch", type=int, default=20, help="可靠TP原型EMA从第几个epoch开始；先让模块3/4形成初始表征")
    g.add_argument("--hpec-ema-update-epochs", type=int, default=-1, help="可靠TP原型EMA持续多少epoch；负数表示warm-up后全程更新")
    g.add_argument("--hpec-ema-alpha", type=float, default=0.995, help="HPEC原型EMA历史保留系数，越大更新越慢")
    g.add_argument("--hpec-ema-anchor-weight", type=float, default=0.10, help="可靠TP原型EMA保留初始分散方向的比例")
    g.add_argument("--hpec-hgcn-logit-blend", type=float, default=0.20, help="最终预测中融合HGCN切空间logits的权重")
    g.add_argument("--hpec-network-energy-weight", type=float, default=0.0, help="模块4中网络级HPEC能量相对全局HPEC能量的融合权重；完整测试为负效果，默认关闭")
    g.add_argument("--hpec-network-energy-mode", default="attention_mean", choices=("attention_mean", "class_softmin"), help="网络级HPEC能量聚合：attention_mean为旧平均；class_softmin为每类选择最支持的脑网络")
    g.add_argument("--hpec-network-energy-temperature", type=float, default=0.5, help="class_softmin 聚合温度，越小越偏向单个高支持子网络")
    g.add_argument("--hpec-network-energy-prior-weight", type=float, default=1.0, help="class_softmin 中模块3网络注意力先验权重")
    g.add_argument("--hpec-network-energy-normalize", type=int, default=1, help="class_softmin 前是否按样本/类别对子网络能量做相对归一化，避免退化为平均池化")
    g.add_argument("--hpec-network-selector-sharpness", type=float, default=1.0, help="class_softmin 子网络选择锐度，越大越强调最支持当前类别的因果子网络")
    g.add_argument("--hpec-energy-loss-weight", type=float, default=0.2, help="HPEC energy 辅助损失权重，保证模块4能量边界直接接受监督")
    g.add_argument("--hpec-energy-ce-margin", type=float, default=0.0, help="真实类别HPEC energy CE的附加间隔；0关闭")
    g.add_argument("--hpec-causal-role-energy-weight", type=float, default=0.0, help="因果角色HPEC能量相对全局能量的融合比例")
    g.add_argument("--hpec-prototype-ce-loss-weight", type=float, default=0.05, help="模块4多原型相似度CE辅助损失权重；MDD 5fold/50epoch 中 0.05 比 0.1 更稳，减少与 Busemann energy 的监督拉扯")
    g.add_argument("--hpec-z-radius-loss-weight", type=float, default=0.0, help="z 半径下界正则权重；默认关闭，避免额外几何正则和分类目标拉扯")
    g.add_argument("--hpec-z-min-radius", type=float, default=0.18, help="z_global 在 Poincare 球中的半径下界，0 表示退化为固定目标半径正则")
    g.add_argument("--hpec-input-radius-min", type=float, default=0.0, help="模块4输入半径校准下界；0表示不校准")
    g.add_argument("--hpec-input-radius-max", type=float, default=0.0, help="模块4输入半径校准上界；0表示不校准")
    g.add_argument("--hpec-input-tangent-noise-std", type=float, default=0.0, help="训练期给模块4输入切空间点加入轻量噪声；0关闭，推理期始终不加")
    g.add_argument("--hpec-prototype-min-radius-loss-weight", type=float, default=0.0, help="prototype 半径下界正则权重；默认关闭，保留作结构正则消融")
    g.add_argument("--hpec-prototype-min-radius", type=float, default=0.18, help="prototype 在 Poincare 球中的半径下界")
    g.add_argument("--hpec-prototype-separation-loss-weight", type=float, default=0.0, help="原型间分离正则权重；默认关闭，保留作结构正则消融")
    g.add_argument("--hpec-teacher-distill-weight", type=float, default=0.0, help="可选 distillation 校准消融权重；默认关闭，仅测试小样本 logits 校准是否有帮助")
    g.add_argument("--hpec-teacher-distill-mode", default="kl", choices=("kl", "centered_kl", "margin_mse"), help="HPEC distillation 校准方式：kl完整软标签 / centered_kl去除共同偏置 / margin_mse只校准正负类margin")
    g.add_argument("--hyperbolic-residual-source", default="prototype", choices=("prototype", "tangent"), help="最终双曲 evidence 来源：prototype=模块4logits；tangent=模块3切空间分类头，模块4作原型约束/门控")
    g.add_argument("--use-hyperbolic-residual-gate", type=int, default=0, help="是否按模块4置信度动态门控双曲残差；consensus完整5fold为负效果，默认关闭")
    g.add_argument("--hyperbolic-residual-gate-mode", default="margin", choices=("margin", "agreement", "consensus"), help="双曲残差门控模式：margin旧模式 / agreement方向一致放大 / consensus双方都有margin且一致时增强HPEC")
    g.add_argument("--hyperbolic-residual-gate-min", type=float, default=0.20, help="双曲残差门控下限，保证模块3/4不被完全旁路")
    g.add_argument("--hyperbolic-residual-gate-max", type=float, default=0.80, help="双曲残差门控上限，避免弱证据冲坏FC主边界")
    g.add_argument("--hyperbolic-residual-gate-gain", type=float, default=2.0, help="双曲残差门控对prototype margin的敏感度")
    g.add_argument("--hpec-prototype-min-radius-ratio", type=float, default=0.6, help="原型切空间半径下限相对 hpec_prototype_radius 的比例，防止原型塌到原点")
    g.add_argument("--hpec-prototype-max-radius-ratio", type=float, default=1.4, help="原型切空间半径上限相对 hpec_prototype_radius 的比例，防止原型跑到边界")

    g = parser.add_argument_group("== GCN fallback / FC 分支 ==")
    g.add_argument("--gcn-fallback-hidden-dim", type=int, default=32, help="GCN-fallback 隐藏维度")
    g.add_argument("--gcn-fallback-layers", type=int, default=1, help="GCN-fallback 层数")
    g.add_argument("--gcn-fallback-dropout", type=float, default=0.5, help="GCN-fallback dropout")
    g.add_argument("--gcn-fallback-readout-mode", default="mean_std", choices=("mean", "attention", "mean_max", "mean_std"), help="GCN-fallback 节点读出方式；mean_std 保留脑区表示的均值与离散度")
    g.add_argument("--gcn-fallback-directional-propagation", type=int, default=1, help="是否分别编码因果图的入边、出边及方向差异")
    g.add_argument("--gcn-fallback-use-graph-stats", type=int, default=1, help="GCN-fallback 是否把因果图统计作为额外分类特征")
    g.add_argument("--gcn-fallback-graph-stats-mode", default="causal", choices=("basic", "causal"), help="GCN-fallback 图统计模式：causal 加入方向性、出入度差异和top边强度")
    g.add_argument("--gcn-fallback-graph-stats-input", default="normalized", choices=("normalized", "raw"), help="GCN-fallback 图统计来源：normalized 更稳，raw 保留原始边强度作对照")
    g.add_argument("--gcn-fallback-edge-readout-topk", type=int, default=0, help="GCN-fallback 额外读取每个样本图中最强 top-k 边的端点特征；0 关闭")
    g.add_argument("--gcn-graph-branch-ce-loss-weight", type=float, default=0.0, help="GCN 图分支辅助CE权重；默认0，仅诊断图分支是否能独立分类")
    g.add_argument("--gcn-fc-branch-ce-loss-weight", type=float, default=0.0, help="FC readout分支辅助CE权重；默认0，仅诊断FC分支是否能独立分类")
    g.add_argument("--use-fc-readout-branch", type=int, default=1, help="是否启用FC生物标志分支（直送分类器）")
    g.add_argument("--fc-readout-mode", default="network", choices=("network", "upper_tri", "both"), help="FC分支特征：network 8×8网络FC(~72维) / upper_tri 全边 / both")
    g.add_argument("--fc-readout-dropout", type=float, default=0.5, help="FC分支 MLP dropout")
    g.add_argument("--fc-readout-edge-dropout", type=float, default=0.0, help="训练期随机丢弃 upper-tri FC 边特征的比例；仅对 upper_tri/both 生效，用于抑制全边FC过拟合")

    g = parser.add_argument_group("== 多站点校正 ==")
    g.add_argument("--time-series-harmonization", default="site_zscore", choices=("none", "site_zscore"), help="输入时序多站点校正；site_zscore 仅用训练集估计站点统计")
    g.add_argument("--use-site-adversarial", type=int, default=0, help="[多站点]站点对抗head开关")
    g.add_argument("--lambda-site-adversarial", type=float, default=0.02, help="[多站点]站点对抗损失权重")
    g.add_argument("--site-grl-lambda", type=float, default=1.0, help="[多站点]梯度反转强度")
    g.add_argument("--site-adversarial-dropout", type=float, default=0.1, help="[多站点]站点对抗头 dropout")

    g = parser.add_argument_group("== 可视化与日志 ==")
    g.add_argument("--visualize-causal", type=int, default=0, help="是否保存模块2/3中间量热图与 t-SNE")
    g.add_argument("--causal-vis-dir", default="outputs/mdd_best_config_causal", help="可视化/诊断输出目录")
    g.add_argument("--result-file", default="result.xlsx", help="结果汇总 xlsx 路径")
    g.add_argument("--use-tensorboard", type=int, default=1, help="是否写 TensorBoard 标量")
    g.add_argument("--tensorboard-dir", default="outputs/tensorboard", help="TensorBoard 日志根目录")
    g.add_argument("--tensorboard-run-name", default=None, help="TensorBoard run 名；默认用 setting")
    g.add_argument("--tensorboard-disable-smoke-runs", type=int, default=1, help="run 名称包含 smoke 时是否跳过 TensorBoard 记录")
    g.add_argument("--print-metric-every", type=int, default=10, help="每隔多少 epoch 打印一次指标")
    g.add_argument("--print-data-info", type=int, default=0, help="是否打印数据加载信息")

    g = parser.add_argument_group("== 运行环境 ==")
    g.add_argument("--use-gpu", type=int, default=int(torch.cuda.is_available()), help="是否用GPU")
    g.add_argument("--gpu", type=int, default=0, help="主GPU编号")
    g.add_argument("--checkpoints", default="checkpoints", help="checkpoint 目录")
    g.add_argument("--del-weight", type=int, default=1, help="测试后是否删除权重以省空间")
    g.add_argument("--keep-weight", action="store_true", help="保留权重（覆盖 --del-weight）")
    return parser.parse_args()
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args(cli_args):
    root = Path(__file__).resolve().parent
    data_path = Path(cli_args.data_path)
    if not data_path.is_absolute():
        data_path = root / data_path

    checkpoints = Path(cli_args.checkpoints)
    if not checkpoints.is_absolute():
        checkpoints = root / checkpoints
    checkpoints.mkdir(parents=True, exist_ok=True)

    use_gpu = bool(cli_args.use_gpu and torch.cuda.is_available())
    module34_default = int(bool(cli_args.use_hyperbolic_modules34))
    use_hgcn_module3 = (
        module34_default
        if cli_args.use_hgcn_module3 is None
        else int(bool(cli_args.use_hgcn_module3))
    )
    use_hpec_module4 = (
        module34_default
        if cli_args.use_hpec_module4 is None
        else int(bool(cli_args.use_hpec_module4))
    )
    if use_hpec_module4 and not use_hgcn_module3:
        raise ValueError("--use-hpec-module4 1 requires --use-hgcn-module3 1")
    prototype_update_mode = cli_args.hpec_prototype_update_mode if use_hpec_module4 else "none"
    cli_args.hpec_prototype_update_mode = prototype_update_mode

    return SimpleNamespace(
        Method="DL",
        model="S-DeCI",
        data=cli_args.data,
        data_path=str(data_path),
        data_type=cli_args.data_type,
        protocol=cli_args.protocol,
        kfold=cli_args.kfold,
        max_folds=cli_args.max_folds,
        iterations=cli_args.iterations,
        checkpoints=str(checkpoints),
        del_weight=not cli_args.keep_weight and bool(cli_args.del_weight),
        seq_len=cli_args.seq_len,
        channel=cli_args.channel,
        classes=cli_args.classes,
        d_model=cli_args.d_model,
        layer=1,
        n_head=8,
        dropout=cli_args.dropout,
        module1_random_crop=cli_args.module1_random_crop,
        module1_feature_mode=cli_args.module1_feature_mode,
        module1_tr=2.0,
        module1_alff_low_hz=0.01,
        module1_alff_high_hz=0.08,
        module1_alff_time_weight=cli_args.module1_alff_time_weight,
        module1_temporal_dropout=cli_args.module1_temporal_dropout,
        module1_roi_dropout=cli_args.module1_roi_dropout,
        module1_denoise_loss_weight=cli_args.module1_denoise_loss_weight,
        module1_temporal_stats_weight=cli_args.module1_temporal_stats_weight,
        use_norm=cli_args.use_norm,
        time_series_harmonization=cli_args.time_series_harmonization,
        site_harmonization_min_samples=2,
        use_site_adversarial=cli_args.use_site_adversarial,
        lambda_site_adversarial=cli_args.lambda_site_adversarial,
        site_grl_lambda=cli_args.site_grl_lambda,
        site_adversarial_dropout=cli_args.site_adversarial_dropout,
        site_count=infer_site_count(data_path, cli_args.protocol),
        use_deci_module1=cli_args.use_deci_module1,
        use_causal_module2=cli_args.use_causal_module2,
        causal_feature_source=cli_args.causal_feature_source,
        causal_graph_method=cli_args.causal_graph_method,
        causal_learning_target=cli_args.causal_learning_target,
        temporal_lag_order=cli_args.temporal_lag_order,
        temporal_causal_init_logit=cli_args.temporal_causal_init_logit,
        temporal_sem_input_norm=cli_args.temporal_sem_input_norm,
        temporal_sample_graph_delta_scale=cli_args.temporal_sample_graph_delta_scale,
        temporal_sample_lag_graph_mode=cli_args.temporal_sample_lag_graph_mode,
        temporal_sample_graph_rank=cli_args.temporal_sample_graph_rank,
        temporal_prediction_loss_mode=cli_args.temporal_prediction_loss_mode,
        temporal_pred_huber_delta=cli_args.temporal_pred_huber_delta,
        lambda_temporal_pred_delta=cli_args.lambda_temporal_pred_delta,
        lambda_temporal_pred_lowfreq=cli_args.lambda_temporal_pred_lowfreq,
        lambda_temporal_pred_corr=cli_args.lambda_temporal_pred_corr,
        temporal_lowfreq_kernel_size=cli_args.temporal_lowfreq_kernel_size,
        temporal_a0_sparse_ratio=cli_args.temporal_a0_sparse_ratio,
        temporal_a0_scale=cli_args.temporal_a0_scale,
        temporal_prediction_target_mode=cli_args.temporal_prediction_target_mode,
        lambda_temporal_pred=cli_args.lambda_temporal_pred,
        lambda_temporal_sparse=cli_args.lambda_temporal_sparse,
        lambda_temporal_smooth=cli_args.lambda_temporal_smooth,
        lambda_temporal_group_sparse=cli_args.lambda_temporal_group_sparse,
        lambda_temporal_lag_hierarchy=cli_args.lambda_temporal_lag_hierarchy,
        temporal_candidate_parent_topk=cli_args.temporal_candidate_parent_topk,
        temporal_decoder_activation=cli_args.temporal_decoder_activation,
        temporal_dagma_warmup_epochs=cli_args.temporal_dagma_warmup_epochs,
        temporal_dagma_barrier_epochs=cli_args.temporal_dagma_barrier_epochs,
        temporal_attention_heads=cli_args.temporal_attention_heads,
        temporal_attention_head_dim=cli_args.temporal_attention_head_dim,
        temporal_attention_dropout=cli_args.temporal_attention_dropout,
        temporal_attention_graph_scale=cli_args.temporal_attention_graph_scale,
        causal_input_norm="none",
        causal_init_logit=cli_args.causal_init_logit,
        causal_learning_rate=cli_args.causal_learning_rate,
        causal_graph_hidden_dim=0,
        dag_sampling_temperature=1.0,
        dag_sampling_noise=0.0,
        dag_sampling_sinkhorn_iters=20,
        dag_sampling_hard=1,
        causal_threshold=0.05,
        detach_causal_input=1,
        lambda_causal_recon=cli_args.lambda_causal_recon,
        lambda_causal_dag=cli_args.lambda_causal_dag,
        lambda_causal_l1=cli_args.lambda_causal_l1,
        use_sample_graph_residual=cli_args.use_sample_graph_residual,
        module2_sample_correlation_blend=cli_args.module2_sample_correlation_blend,
        module2_graph_residual_alpha=cli_args.module2_graph_residual_alpha,
        detach_module2_graph_for_classification=cli_args.detach_module2_graph_for_classification,
        freeze_causal_after_epoch=cli_args.freeze_causal_after_epoch,
        classification_graph_source=cli_args.classification_graph_source,
        sample_graph_delta_scale=cli_args.sample_graph_delta_scale,
        sample_graph_hidden_dim=cli_args.sample_graph_hidden_dim,
        lambda_sample_graph_l1=cli_args.lambda_sample_graph_l1,
        lambda_sample_graph_deviation=cli_args.lambda_sample_graph_deviation,
        causal_loss_schedule="constant",
        causal_dag_warmup_epochs=0,
        causal_l1_warmup_epochs=0,
        sample_graph_reg_warmup_epochs=0,
        dagma_logdet_s=1.0,
        dagma_logdet_margin=0.1,
        causal_analytic_margin=0.1,
        causal_analytic_power_iters=5,
        use_hyperbolic_modules34=int(bool(use_hgcn_module3 or use_hpec_module4)),
        use_hgcn_module3=use_hgcn_module3,
        use_hpec_module4=use_hpec_module4,
        hgcn_hidden_dim=cli_args.hgcn_hidden_dim,
        hgcn_fc_inject_weight=cli_args.hgcn_fc_inject_weight,
        module34_film_weight=cli_args.module34_film_weight,
        module34_film_max_scale=cli_args.module34_film_max_scale,
        module34_film_shift_norm=cli_args.module34_film_shift_norm,
        hgcn_fc_anchor_norm_target=0.5,
        hgcn_fc_anchor_gate_init=-1.5,
        hgcn_layers=cli_args.hgcn_layers,
        hgcn_curvature=1.0,
        hgcn_backclip_radius=1.0,
        hgcn_dropout=cli_args.hgcn_dropout,
        use_multi_hop_causal_encoding=cli_args.use_multi_hop_causal_encoding,
        causal_reachability_hops=cli_args.causal_reachability_hops,
        causal_reachability_scale=cli_args.causal_reachability_scale,
        hgcn_residual_alpha=cli_args.hgcn_residual_alpha,
        hgcn_graph_readout_alpha=cli_args.hgcn_graph_readout_alpha,
        hgcn_delta_readout_alpha=0.0,
        hgcn_readout_mode=cli_args.hgcn_readout_mode,
        hgcn_causal_attention_heads=cli_args.hgcn_causal_attention_heads,
        hgcn_causal_attention_graph_weight=cli_args.hgcn_causal_attention_graph_weight,
        hgcn_causal_subnetwork_count=cli_args.hgcn_causal_subnetwork_count,
        hgcn_causal_subnetwork_topk=cli_args.hgcn_causal_subnetwork_topk,
        hgcn_causal_subnetwork_weight=cli_args.hgcn_causal_subnetwork_weight,
        hgcn_einstein_readout_weight=cli_args.hgcn_einstein_readout_weight,
        use_causal_role_readout=cli_args.use_causal_role_readout,
        causal_role_temperature=cli_args.causal_role_temperature,
        hgcn_network_gate_strength=cli_args.hgcn_network_gate_strength,
        hgcn_use_graph_degree_encoding=cli_args.hgcn_use_graph_degree_encoding,
        hgcn_graph_degree_encoding_weight=cli_args.hgcn_graph_degree_encoding_weight,
        hgcn_use_radius_head=cli_args.hgcn_use_radius_head,
        hgcn_radius_min_ratio=cli_args.hgcn_radius_min_ratio,
        use_hgcn_radial_calibration=cli_args.use_hgcn_radial_calibration,
        hgcn_radial_min=cli_args.hgcn_radial_min,
        hgcn_radial_max=cli_args.hgcn_radial_max,
        hgcn_add_self_loop=1,
        hgcn_adjacency_normalization="row",
        lorentz_layers=cli_args.lorentz_layers,
        lorentz_curvature=cli_args.lorentz_curvature,
        lorentz_dropout=cli_args.lorentz_dropout,
        lorentz_alpha_out_init=cli_args.lorentz_alpha_out_init,
        lorentz_message_residual_weight=cli_args.lorentz_message_residual_weight,
        lorentz_message_gate_init=cli_args.lorentz_message_gate_init,
        lorentz_centroid_message_weight=cli_args.lorentz_centroid_message_weight,
        lorentz_max_tangent_norm=cli_args.lorentz_max_tangent_norm,
        lp_use_network_readout=cli_args.lp_use_network_readout,
        lp_network_readout_blend=cli_args.lp_network_readout_blend,
        lp_stats_update_gate_init=cli_args.lp_stats_update_gate_init,
        lp_use_dynamic_radius=cli_args.lp_use_dynamic_radius,
        lp_dynamic_radius_min_ratio=cli_args.lp_dynamic_radius_min_ratio,
        lp_dynamic_radius_max_ratio=cli_args.lp_dynamic_radius_max_ratio,
        lp_dynamic_radius_source=cli_args.lp_dynamic_radius_source,
        lp_input_residual_weight=cli_args.lp_input_residual_weight,
        lp_poincare_readout_weight=cli_args.lp_poincare_readout_weight,
        lp_mac_clip_mode=cli_args.lp_mac_clip_mode,
        module34_supcon_loss_weight=cli_args.module34_supcon_loss_weight,
        module34_supcon_temperature=cli_args.module34_supcon_temperature,
        module34_center_loss_weight=cli_args.module34_center_loss_weight,
        module34_center_margin=cli_args.module34_center_margin,
        module34_center_intra_weight=cli_args.module34_center_intra_weight,
        module34_center_inter_weight=cli_args.module34_center_inter_weight,
        module34_branch_ce_loss_weight=cli_args.module34_branch_ce_loss_weight,
        module34_branch_ce_decay_epochs=cli_args.module34_branch_ce_decay_epochs,
        module34_branch_ce_min_ratio=cli_args.module34_branch_ce_min_ratio,
        module34_geo_dtype="auto",
        use_brain_network_prior=cli_args.use_brain_network_prior,
        fc_residual_weight=0.0,
        fc_network_residual_weight=0.0,
        fc_residual_norm_target=1.0,
        use_fc_residual_gate=1,
        fc_residual_gate_init=-2.0,
        causal_edge_dropout=cli_args.causal_edge_dropout,
        lambda_causal_stability=cli_args.lambda_causal_stability,
        class_loss_weighting=cli_args.class_loss_weighting,
        class_logit_adjust_tau=cli_args.class_logit_adjust_tau,
        class_prior_alignment_weight=cli_args.class_prior_alignment_weight,
        class_label_smoothing=cli_args.class_label_smoothing,
        use_final_logit_calibration=cli_args.use_final_logit_calibration,
        hpec_hgcn_logit_blend=cli_args.hpec_hgcn_logit_blend,
        hpec_classification_mode=cli_args.hpec_classification_mode,
        hpec_network_energy_weight=cli_args.hpec_network_energy_weight,
        hpec_network_energy_mode=cli_args.hpec_network_energy_mode,
        hpec_network_energy_temperature=cli_args.hpec_network_energy_temperature,
        hpec_network_energy_prior_weight=cli_args.hpec_network_energy_prior_weight,
        hpec_network_energy_normalize=cli_args.hpec_network_energy_normalize,
        hpec_network_selector_sharpness=cli_args.hpec_network_selector_sharpness,
        hpec_evidence_weight=cli_args.hpec_evidence_weight,
        hpec_avoid_busemann_double_count=cli_args.hpec_avoid_busemann_double_count,
        hpec_logit_temperature=0.5,
        hpec_prototype_energy_blend=cli_args.hpec_prototype_energy_blend,
        hpec_prototype_logit_mode=cli_args.hpec_prototype_logit_mode,
        hpec_prototype_logit_scale=cli_args.hpec_prototype_logit_scale,
        hpec_residual_calibration=cli_args.hpec_residual_calibration,
        hpec_residual_calibration_scale=cli_args.hpec_residual_calibration_scale,
        hpec_residual_calibration_momentum=cli_args.hpec_residual_calibration_momentum,
        hpec_residual_calibration_batch_weight=cli_args.hpec_residual_calibration_batch_weight,
        hpec_prototype_residual_weight=cli_args.hpec_prototype_residual_weight,
        hpec_gate_init=-2.5,
        hpec_energy_loss_weight=cli_args.hpec_energy_loss_weight,
        hpec_energy_ce_margin=cli_args.hpec_energy_ce_margin,
        hpec_causal_role_energy_weight=cli_args.hpec_causal_role_energy_weight,
        hpec_prototype_ce_loss_weight=cli_args.hpec_prototype_ce_loss_weight,
        hpec_teacher_distill_weight=cli_args.hpec_teacher_distill_weight,
        hpec_teacher_distill_temperature=2.0,
        hpec_teacher_distill_mode=cli_args.hpec_teacher_distill_mode,
        hpec_teacher_detach=1,
        hpec_z_radius_loss_weight=cli_args.hpec_z_radius_loss_weight,
        hpec_z_radius_target=0.35,
        hpec_z_min_radius=cli_args.hpec_z_min_radius,
        hpec_input_radius_min=cli_args.hpec_input_radius_min,
        hpec_input_radius_max=cli_args.hpec_input_radius_max,
        hpec_input_tangent_noise_std=cli_args.hpec_input_tangent_noise_std,
        hpec_prototype_min_radius_loss_weight=cli_args.hpec_prototype_min_radius_loss_weight,
        hpec_prototype_min_radius=cli_args.hpec_prototype_min_radius,
        hpec_prototype_separation_loss_weight=cli_args.hpec_prototype_separation_loss_weight,
        hpec_prototype_separation_max_cos=0.35,
        keep_gcn_fallback_with_hyperbolic=cli_args.keep_gcn_fallback_with_hyperbolic,
        hyperbolic_logit_residual_weight=cli_args.hyperbolic_logit_residual_weight,
        hyperbolic_residual_fusion_mode=cli_args.hyperbolic_residual_fusion_mode,
        hyperbolic_residual_norm=cli_args.hyperbolic_residual_norm,
        hyperbolic_residual_temperature=cli_args.hyperbolic_residual_temperature,
        hyperbolic_residual_margin_gain=cli_args.hyperbolic_residual_margin_gain,
        hyperbolic_residual_margin_max_scale=cli_args.hyperbolic_residual_margin_max_scale,
        use_hyperbolic_residual_bias=cli_args.use_hyperbolic_residual_bias,
        hyperbolic_residual_source=cli_args.hyperbolic_residual_source,
        use_hyperbolic_residual_gate=cli_args.use_hyperbolic_residual_gate,
        hyperbolic_residual_gate_mode=cli_args.hyperbolic_residual_gate_mode,
        hyperbolic_residual_gate_min=cli_args.hyperbolic_residual_gate_min,
        hyperbolic_residual_gate_max=cli_args.hyperbolic_residual_gate_max,
        hyperbolic_residual_gate_gain=cli_args.hyperbolic_residual_gate_gain,
        use_sample_correlation_when_module2_disabled=cli_args.use_sample_correlation_when_module2_disabled,
        sample_correlation_mode=cli_args.sample_correlation_mode,
        gcn_fallback_hidden_dim=cli_args.gcn_fallback_hidden_dim,
        gcn_fallback_layers=cli_args.gcn_fallback_layers,
        gcn_fallback_dropout=cli_args.gcn_fallback_dropout,
        gcn_fallback_add_self_loop=1,
        gcn_fallback_adjacency_normalization="row",
        gcn_fallback_use_graph_stats=cli_args.gcn_fallback_use_graph_stats,
        gcn_fallback_graph_stats_mode=cli_args.gcn_fallback_graph_stats_mode,
        gcn_fallback_graph_stats_input=cli_args.gcn_fallback_graph_stats_input,
        gcn_fallback_readout_mode=cli_args.gcn_fallback_readout_mode,
        gcn_fallback_directional_propagation=cli_args.gcn_fallback_directional_propagation,
        gcn_fallback_edge_readout_topk=cli_args.gcn_fallback_edge_readout_topk,
        gcn_fallback_input_residual_weight=0.0,
        gcn_graph_branch_ce_loss_weight=cli_args.gcn_graph_branch_ce_loss_weight,
        gcn_fc_branch_ce_loss_weight=cli_args.gcn_fc_branch_ce_loss_weight,
        use_fc_readout_branch=cli_args.use_fc_readout_branch,
        fc_readout_mode=cli_args.fc_readout_mode,
        fc_readout_hidden_dim=64,
        fc_readout_embed_dim=64,
        fc_readout_dropout=cli_args.fc_readout_dropout,
        fc_readout_fisher_z=1,
        fc_readout_edge_dropout=cli_args.fc_readout_edge_dropout,
        hpec_prototype_radius=0.3,
        hpec_cone_k=0.1,
        hpec_margin=cli_args.hpec_margin,
        hpec_prototypes_per_class=cli_args.hpec_prototypes_per_class,
        hpec_proto_temperature=cli_args.hpec_proto_temperature,
        hpec_distance_weight=cli_args.hpec_distance_weight,
        hpec_energy_scale=1.0,
        hpec_energy_mode=cli_args.hpec_energy_mode,
        hpec_loss_mode=cli_args.hpec_loss_mode,
        hpec_busemann_temperature=cli_args.hpec_busemann_temperature,
        hpec_busemann_point_radius=cli_args.hpec_busemann_point_radius,
        hpec_busemann_radius_gate_weight=cli_args.hpec_busemann_radius_gate_weight,
        hpec_busemann_radius_gate_center=cli_args.hpec_busemann_radius_gate_center,
        hpec_busemann_class_bias_weight=cli_args.hpec_busemann_class_bias_weight,
        hpec_data_init=cli_args.hpec_data_init,
        hpec_use_sinkhorn_ema=cli_args.hpec_use_sinkhorn_ema,
        hpec_prototype_update_mode=prototype_update_mode,
        hpec_reliable_confidence_threshold=cli_args.hpec_reliable_confidence_threshold,
        hpec_reliable_view_consistency_threshold=cli_args.hpec_reliable_view_consistency_threshold,
        hpec_reliable_min_samples=cli_args.hpec_reliable_min_samples,
        hpec_reliable_weight_floor=cli_args.hpec_reliable_weight_floor,
        hpec_epoch_frechet_steps=cli_args.hpec_epoch_frechet_steps,
        hpec_sinkhorn_epsilon=0.05,
        hpec_sinkhorn_iters=3,
        hpec_ema_alpha=cli_args.hpec_ema_alpha,
        hpec_ema_anchor_weight=cli_args.hpec_ema_anchor_weight,
        hpec_ema_start_epoch=cli_args.hpec_ema_start_epoch,
        hpec_ema_update_epochs=cli_args.hpec_ema_update_epochs,
        hpec_intra_class_max_cos=0.25,
        hpec_prototype_min_radius_ratio=cli_args.hpec_prototype_min_radius_ratio,
        hpec_prototype_max_radius_ratio=cli_args.hpec_prototype_max_radius_ratio,
        hpec_trainable_prototypes=cli_args.hpec_trainable_prototypes,
        hpec_prototype_lr_scale=cli_args.hpec_prototype_lr_scale,
        hpec_init_steps=500,
        hpec_eps=1e-7,
        mac_min_radius=0.05,
        mac_max_radius=0.98,
        hbr_safe_radius=2.0,
        hbr_loss_weight=0.0,
        visualize_causal=cli_args.visualize_causal,
        causal_vis_dir=str(root / cli_args.causal_vis_dir),
        visualize_every=0,
        factor=1,
        moving_avg=25,
        top_k=5,
        num_kernels=2,
        decomp_method="moving_avg",
        down_sampling_layers=3,
        down_sampling_window=2,
        down_sampling_method="avg",
        seg_len=24,
        small_kernel_merged=False,
        stem_ratio=6,
        downsample_ratio=2,
        ffn_ratio=2,
        patch_size=16,
        patch_stride=8,
        num_blocks=[1, 1, 1],
        large_size=[31, 29, 27],
        small_size=[5, 5, 5],
        dims=[64, 64, 64],
        dw_dims=[64, 64, 64],
        single_channel=False,
        patch_len_list="12,24,48",
        no_inter_attn=False,
        activation="gelu",
        output_attention=False,
        seed=cli_args.seed,
        num_workers=0,
        pin_memory=1,
        persistent_workers=1,
        prefetch_factor=2,
        train_epochs=cli_args.train_epochs,
        batch_size=cli_args.batch_size,
        patience=cli_args.patience,
        learning_rate=cli_args.learning_rate,
        weight_decay=cli_args.weight_decay,
        use_model_ema=cli_args.use_model_ema,
        model_ema_decay=0.995,
        loss=cli_args.loss,
        lradj=cli_args.lradj,
        early_stop_metric="best_macro_f1",
        use_best_threshold=1,
        binary_positive_weight=cli_args.binary_positive_weight,
        print_process=0,
        print_metric_every=cli_args.print_metric_every,
        print_data_info=cli_args.print_data_info,
        use_tensorboard=cli_args.use_tensorboard,
        tensorboard_dir=cli_args.tensorboard_dir,
        tensorboard_run_name=cli_args.tensorboard_run_name,
        tensorboard_disable_smoke_runs=cli_args.tensorboard_disable_smoke_runs,
        use_gpu=use_gpu,
        gpu=cli_args.gpu,
        gpu_idx=[cli_args.gpu],
        use_multi_gpu=False,
        devices=str(cli_args.gpu),
        device_ids=[cli_args.gpu],
    )


def setting_name(args):
    return (
        f"{args.data}_data_type_{args.data_type}_protocol_{args.protocol}_"
        f"kfold_{args.kfold}_model_{args.model}_bs_{args.batch_size}_"
        f"lr_{args.learning_rate}_dp_{args.dropout}_dm_{args.d_model}_seq_{args.seq_len}_"
        f"m1_{args.use_deci_module1}_m2_{args.use_causal_module2}_"
        f"m3_{args.use_hgcn_module3}_m4_{args.use_hpec_module4}_target_{args.causal_learning_target}_"
        f"lag_{args.temporal_lag_order}_tpred_{args.lambda_temporal_pred}_"
        f"graph_{args.causal_graph_method}_adj_{args.sample_correlation_mode}_"
        f"blend_{args.module2_sample_correlation_blend}_loss_{args.loss}"
    )


def format_metrics(metrics):
    parts = []
    for name, value in zip(METRIC_NAMES, metrics):
        if name == "train_seconds":
            parts.append(f"{name}: {float(value):.2f}s")
        else:
            parts.append(f"{name}: {float(value):.4f}")
    return ", ".join(parts)


def _column_name(index):
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _sheet_name(cli_args):
    raw_name = f"{cli_args.data}_{cli_args.protocol}_{cli_args.train_epochs}ep"
    invalid = '[]:*?/\\'
    for char in invalid:
        raw_name = raw_name.replace(char, "-")
    return raw_name[:31]


def _read_result_sheets(path):
    if not path.exists():
        return {}
    try:
        import xml.etree.ElementTree as ET

        with ZipFile(path, "r") as zf:
            names = zf.namelist()
            shared = []
            if "xl/sharedStrings.xml" in names:
                root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in root.findall("a:si", ns):
                    texts = [node.text or "" for node in si.findall(".//a:t", ns)]
                    shared.append("".join(texts))

            ns = {
                "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
                "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
            }
            workbook = ET.fromstring(zf.read("xl/workbook.xml"))
            workbook_rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rid_to_target = {
                rel.attrib["Id"]: rel.attrib["Target"]
                for rel in workbook_rels.findall("rel:Relationship", ns)
            }
            sheet_entries = []
            for sheet_node in workbook.findall(".//a:sheets/a:sheet", ns):
                sheet_name = sheet_node.attrib["name"]
                rid = sheet_node.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                target = rid_to_target.get(rid)
                if target:
                    sheet_entries.append((sheet_name, "xl/" + target.lstrip("/")))
            if not sheet_entries and "xl/worksheets/sheet1.xml" in names:
                sheet_entries = [("results", "xl/worksheets/sheet1.xml")]

            sheets = {}
            for sheet_name, sheet_path in sheet_entries:
                if sheet_path not in names:
                    continue
                sheet = ET.fromstring(zf.read(sheet_path))
                rows = []
                for row in sheet.findall(".//a:sheetData/a:row", ns):
                    values = []
                    for cell in row.findall("a:c", ns):
                        value_node = cell.find("a:v", ns)
                        if value_node is None:
                            values.append("")
                            continue
                        value = value_node.text or ""
                        if cell.get("t") == "s":
                            value = shared[int(value)]
                        else:
                            try:
                                value = float(value)
                                if value.is_integer():
                                    value = int(value)
                            except ValueError:
                                pass
                        values.append(value)
                    rows.append(values)
                sheets[sheet_name] = rows
            return sheets
    except Exception:
        backup = path.with_suffix(path.suffix + ".bak")
        path.replace(backup)
        print(f"Existing result file could not be parsed; backed up to: {backup}")
        return {}


def _read_result_rows(path):
    sheets = _read_result_sheets(path)
    if not sheets:
        return []
    return next(iter(sheets.values()))


def _sheet_xml(rows):
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row):
            ref = f"{_column_name(col_idx)}{row_idx}"
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{float(value):.10g}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="s"><v>{_write_xlsx.string_index(value)}</v></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    max_col = max((len(row) for row in rows), default=1) - 1
    dimension = f"A1:{_column_name(max_col)}{max(len(rows), 1)}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/><sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )


def _write_xlsx(path, sheets):
    if isinstance(sheets, list):
        sheets = {"results": sheets}
    strings = []
    string_to_index = {}

    def string_index(value):
        text = str(value)
        if text not in string_to_index:
            string_to_index[text] = len(strings)
            strings.append(text)
        return string_to_index[text]

    _write_xlsx.string_index = string_index
    sheet_xmls = {}
    for sheet_name, rows in sheets.items():
        sheet_xmls[sheet_name] = _sheet_xml(rows)

    shared_items = "".join(f"<si><t>{escape(text)}</t></si>" for text in strings)
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(strings)}" uniqueCount="{len(strings)}">{shared_items}</sst>'
    )
    sheet_nodes = []
    workbook_rel_nodes = [
        '<Relationship Id="rIdShared" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
    ]
    content_sheet_overrides = []
    for idx, (sheet_name, _) in enumerate(sheet_xmls.items(), start=1):
        rid = f"rId{idx}"
        sheet_nodes.append(
            f'<sheet name="{escape(sheet_name)}" sheetId="{idx}" r:id="{rid}"/>'
        )
        workbook_rel_nodes.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
        content_sheet_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(sheet_nodes)}</sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(workbook_rel_nodes)}</Relationships>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f'{"".join(content_sheet_overrides)}'
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        '</Types>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        for idx, sheet_xml in enumerate(sheet_xmls.values(), start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml)
        zf.writestr("xl/sharedStrings.xml", shared_xml)


def _write_xlsx_legacy(path, rows):
    strings = []
    string_to_index = {}

    def string_index(value):
        text = str(value)
        if text not in string_to_index:
            string_to_index[text] = len(strings)
            strings.append(text)
        return string_to_index[text]

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row):
            ref = f"{_column_name(col_idx)}{row_idx}"
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{float(value):.10g}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="s"><v>{string_index(value)}</v></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    shared_items = "".join(f"<si><t>{escape(text)}</t></si>" for text in strings)
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(strings)}" uniqueCount="{len(strings)}">{shared_items}</sst>'
    )
    max_col = max((len(row) for row in rows), default=1) - 1
    dimension = f"A1:{_column_name(max_col)}{max(len(rows), 1)}"
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/><sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="results" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
        '</Relationships>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        '</Types>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/sharedStrings.xml", shared_xml)


def _old_rows_to_new_rows(rows):
    if not rows:
        return []
    old_header = rows[0]
    old_index = {str(name): idx for idx, name in enumerate(old_header)}
    required = {"accuracy", "precision", "recall", "macro_f1", "auc"}
    if not required.issubset(old_index):
        return rows
    headers = _result_headers()
    new_rows = [headers]
    for row in rows[1:]:
        def get(name, default=""):
            idx = old_index.get(name)
            return row[idx] if idx is not None and idx < len(row) else default

        converted = []
        for column in headers:
            if column in RESULT_METRIC_COLUMNS:
                converted.append(_metric_percent(get(column)))
            else:
                converted.append(get(column))
        new_rows.append(converted)
    return new_rows


def _metric_percent(value):
    if value in ("", None):
        return ""
    text = str(value)
    if text.endswith("%"):
        try:
            return f"{float(text[:-1]):.2f}%"
        except ValueError:
            return text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return text
    if abs(number) <= 1.0:
        number *= 100.0
    return f"{number:.2f}%"


def _result_headers():
    return [
        "module1",
        "module2",
        "module3",
        "module4",
        "module1_feature",
        "module1_alff_time_weight",
        "causal_target",
        "causal_graph_method",
        "graph_source",
        "graph_blend",
        "graph_residual_alpha",
        "detach_graph_cls",
        "class_loss_weighting",
        "class_label_smoothing",
        "class_logit_adjust_tau",
        "lradj",
        "learning_rate",
        "weight_decay",
        "loss",
        "binary_positive_weight",
        "temporal_lag_order",
        "temporal_prediction_loss_mode",
        "temporal_pred_huber_delta",
        "lambda_temporal_pred_delta",
        "lambda_temporal_pred_lowfreq",
        "lambda_temporal_pred_corr",
        "temporal_lowfreq_kernel_size",
        "temporal_a0_sparse_ratio",
        "temporal_a0_scale",
        "temporal_prediction_target_mode",
        "lambda_temporal_pred",
        "lambda_temporal_sparse",
        "lambda_temporal_smooth",
        "lambda_temporal_group_sparse",
        "lambda_temporal_lag_hierarchy",
        "temporal_candidate_parent_topk",
        "lambda_causal_dag",
        "gcn_fallback_readout",
        "gcn_fallback_directional_propagation",
        "gcn_fallback_use_graph_stats",
        "gcn_fallback_graph_stats_mode",
        "gcn_fallback_graph_stats_input",
        "gcn_fallback_edge_readout_topk",
        "gcn_graph_branch_ce_weight",
        "gcn_fc_branch_ce_weight",
        "use_fc_readout_branch",
        "fc_readout_mode",
        "fc_readout_dropout",
        "fc_readout_edge_dropout",
        "hgcn_fc_inject_weight",
        "module34_film_weight",
        "module34_film_max_scale",
        "module34_film_shift_norm",
        "hyperbolic_logit_residual_weight",
        "hyperbolic_residual_fusion",
        "hyperbolic_residual_gate_mode",
        "keep_gcn_fallback",
        "hgcn_hidden",
        "hgcn_dropout",
        "causal_reachability_enabled",
        "causal_reachability_hops",
        "causal_reachability_scale",
        "hgcn_graph_readout_alpha",
        "hgcn_readout_mode",
        "hgcn_causal_attn_heads",
        "hgcn_causal_attn_graph_weight",
        "hgcn_causal_subnetwork_count",
        "hgcn_causal_subnetwork_topk",
        "hgcn_causal_subnetwork_weight",
        "hgcn_einstein_readout_weight",
        "causal_role_readout",
        "causal_role_temperature",
        "hgcn_network_gate_strength",
        "hgcn_use_graph_degree_encoding",
        "hgcn_graph_degree_encoding_weight",
        "lorentz_layers",
        "lorentz_alpha_out",
        "lorentz_msg_res",
        "lorentz_centroid_msg",
        "lorentz_max_tangent_norm",
        "lp_network_readout",
        "lp_network_readout_blend",
        "lp_stats_gate_init",
        "lp_dynamic_radius",
        "lp_radius_min_ratio",
        "lp_radius_max_ratio",
        "lp_radius_source",
        "lp_input_residual",
        "lp_mac_clip_mode",
        "module34_supcon_weight",
        "module34_supcon_temp",
        "module34_center_weight",
        "module34_center_margin",
        "module34_center_intra",
        "module34_center_inter",
        "module34_branch_ce_weight",
        "hpec_energy_mode",
        "hpec_cls_mode",
        "hpec_loss_mode",
        "hpec_evidence_weight",
        "hpec_avoid_busemann_double_count",
        "hpec_proto_per_class",
        "hpec_proto_temperature",
        "hpec_busemann_temperature",
        "hpec_busemann_point_radius",
        "hpec_busemann_radius_gate_weight",
        "hpec_busemann_radius_gate_center",
        "hpec_busemann_class_bias_weight",
        "hpec_prototype_ce_loss_weight",
        "hpec_proto_logit_mode",
        "hpec_proto_logit_scale",
        "hpec_residual_calibration",
        "hpec_residual_calibration_scale",
        "hpec_residual_calibration_momentum",
        "hpec_residual_calibration_batch_weight",
        "hpec_proto_residual_weight",
        "hpec_hgcn_logit_blend",
        "hpec_network_energy_weight",
        "hpec_network_energy_mode",
        "hpec_network_energy_temp",
        "hpec_network_energy_prior",
        "hpec_network_energy_norm",
        "hpec_network_selector_sharp",
        "hpec_energy_loss_weight",
        "hpec_energy_ce_margin",
        "hpec_causal_role_energy_weight",
        "hpec_z_radius_loss_weight",
        "hpec_input_tangent_noise_std",
        "hpec_proto_sep_loss_weight",
        "hpec_teacher_distill_weight",
        "hpec_teacher_distill_mode",
        "hpec_data_init",
        "hpec_prototype_update_mode",
        "hpec_reliable_confidence_threshold",
        "hpec_reliable_view_consistency_threshold",
        "hpec_reliable_min_samples",
        "hpec_reliable_weight_floor",
        "hpec_epoch_frechet_steps",
        "hpec_ema_start_epoch",
        "hpec_trainable_prototypes",
        "hpec_prototype_lr_scale",
        "hpec_prototype_parameterization",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "auc",
        "train_seconds",
    ]

def _normalize_metric_columns(rows):
    if not rows:
        return rows
    header = [str(item) for item in rows[0]]
    metric_indices = [
        idx
        for idx, column in enumerate(header)
        if column in RESULT_METRIC_COLUMNS
    ]
    normalized = [rows[0]]
    for row in rows[1:]:
        new_row = list(row)
        for idx in metric_indices:
            if idx < len(new_row):
                new_row[idx] = _metric_percent(new_row[idx])
        normalized.append(new_row)
    return normalized

def _legacy_sheet_name(rows):
    if len(rows) < 2:
        return "legacy"
    header = {str(name): idx for idx, name in enumerate(rows[0])}
    row = rows[1]
    def get(name, default=""):
        idx = header.get(name)
        return row[idx] if idx is not None and idx < len(row) else default
    data = get("data", "Result")
    epochs = get("epochs", "")
    return f"{data}_legacy_{epochs}ep"[:31]


def _read_result_sheets_with_migration(path):
    sheets = _read_result_sheets(path)
    migrated = {}
    current_headers = _result_headers()
    for name, rows in sheets.items():
        if rows and "kfold" in [str(item) for item in rows[0]]:
            migrated[_legacy_sheet_name(rows)] = _normalize_metric_columns(_old_rows_to_new_rows(rows))
        elif rows and [str(item) for item in rows[0]] != current_headers:
            old_header = [str(item) for item in rows[0]]
            old_index = {column: idx for idx, column in enumerate(old_header)}
            aligned_rows = [current_headers]
            for row in rows[1:]:
                aligned_rows.append(
                    [
                        _metric_percent(row[old_index[column]])
                        if column in RESULT_METRIC_COLUMNS and column in old_index and old_index[column] < len(row)
                        else (
                            row[old_index[column]]
                            if column in old_index and old_index[column] < len(row)
                            else ""
                        )
                        for column in current_headers
                    ]
                )
            migrated[name] = _normalize_metric_columns(aligned_rows)
        else:
            migrated[name] = _normalize_metric_columns(rows)
    return migrated


def save_result_row(cli_args, metrics, setting):
    result_path = Path(cli_args.result_file)
    if not result_path.is_absolute():
        result_path = Path(__file__).resolve().parent / result_path
    sheet_name = _sheet_name(cli_args)
    headers = _result_headers()
    sheets = _read_result_sheets_with_migration(result_path)
    rows = sheets.get(sheet_name, [])
    if not rows:
        rows = [headers]
    metric_values = list(metrics)
    metric_cells = [_metric_percent(value) for value in metric_values[:5]]
    train_seconds = metric_values[5] if len(metric_values) > 5 else ""
    if train_seconds not in ("", None):
        try:
            train_seconds = f"{float(train_seconds):.2f}"
        except (TypeError, ValueError):
            train_seconds = str(train_seconds)
    row = [
        cli_args.use_deci_module1,
        cli_args.use_causal_module2,
        int(cli_args.use_hgcn_module3) if cli_args.use_hgcn_module3 is not None else cli_args.use_hyperbolic_modules34,
        int(cli_args.use_hpec_module4) if cli_args.use_hpec_module4 is not None else cli_args.use_hyperbolic_modules34,
        cli_args.module1_feature_mode,
        cli_args.module1_alff_time_weight,
        cli_args.causal_learning_target,
        cli_args.causal_graph_method,
        cli_args.classification_graph_source,
        cli_args.module2_sample_correlation_blend,
        cli_args.module2_graph_residual_alpha,
        cli_args.detach_module2_graph_for_classification,
        cli_args.class_loss_weighting,
        cli_args.class_label_smoothing,
        cli_args.class_logit_adjust_tau,
        cli_args.lradj,
        cli_args.learning_rate,
        cli_args.weight_decay,
        cli_args.loss,
        cli_args.binary_positive_weight,
        cli_args.temporal_lag_order,
        cli_args.temporal_prediction_loss_mode,
        cli_args.temporal_pred_huber_delta,
        cli_args.lambda_temporal_pred_delta,
        cli_args.lambda_temporal_pred_lowfreq,
        cli_args.lambda_temporal_pred_corr,
        cli_args.temporal_lowfreq_kernel_size,
        cli_args.temporal_a0_sparse_ratio,
        cli_args.temporal_a0_scale,
        cli_args.temporal_prediction_target_mode,
        cli_args.lambda_temporal_pred,
        cli_args.lambda_temporal_sparse,
        cli_args.lambda_temporal_smooth,
        cli_args.lambda_temporal_group_sparse,
        cli_args.lambda_temporal_lag_hierarchy,
        cli_args.temporal_candidate_parent_topk,
        cli_args.lambda_causal_dag,
        cli_args.gcn_fallback_readout_mode,
        cli_args.gcn_fallback_directional_propagation,
        cli_args.gcn_fallback_use_graph_stats,
        cli_args.gcn_fallback_graph_stats_mode,
        cli_args.gcn_fallback_graph_stats_input,
        cli_args.gcn_fallback_edge_readout_topk,
        cli_args.gcn_graph_branch_ce_loss_weight,
        cli_args.gcn_fc_branch_ce_loss_weight,
        cli_args.use_fc_readout_branch,
        cli_args.fc_readout_mode,
        cli_args.fc_readout_dropout,
        cli_args.fc_readout_edge_dropout,
        cli_args.hgcn_fc_inject_weight,
        cli_args.module34_film_weight,
        cli_args.module34_film_max_scale,
        cli_args.module34_film_shift_norm,
        cli_args.hyperbolic_logit_residual_weight,
        cli_args.hyperbolic_residual_fusion_mode,
        cli_args.hyperbolic_residual_gate_mode,
        cli_args.keep_gcn_fallback_with_hyperbolic,
        cli_args.hgcn_hidden_dim,
        cli_args.hgcn_dropout,
        cli_args.use_multi_hop_causal_encoding,
        cli_args.causal_reachability_hops,
        cli_args.causal_reachability_scale,
        cli_args.hgcn_graph_readout_alpha,
        cli_args.hgcn_readout_mode,
        cli_args.hgcn_causal_attention_heads,
        cli_args.hgcn_causal_attention_graph_weight,
        cli_args.hgcn_causal_subnetwork_count,
        cli_args.hgcn_causal_subnetwork_topk,
        cli_args.hgcn_causal_subnetwork_weight,
        cli_args.hgcn_einstein_readout_weight,
        cli_args.use_causal_role_readout,
        cli_args.causal_role_temperature,
        cli_args.hgcn_network_gate_strength,
        cli_args.hgcn_use_graph_degree_encoding,
        cli_args.hgcn_graph_degree_encoding_weight,
        cli_args.lorentz_layers,
        cli_args.lorentz_alpha_out_init,
        cli_args.lorentz_message_residual_weight,
        cli_args.lorentz_centroid_message_weight,
        cli_args.lorentz_max_tangent_norm,
        cli_args.lp_use_network_readout,
        cli_args.lp_network_readout_blend,
        cli_args.lp_stats_update_gate_init,
        cli_args.lp_use_dynamic_radius,
        cli_args.lp_dynamic_radius_min_ratio,
        cli_args.lp_dynamic_radius_max_ratio,
        cli_args.lp_dynamic_radius_source,
        cli_args.lp_input_residual_weight,
        cli_args.lp_mac_clip_mode,
        cli_args.module34_supcon_loss_weight,
        cli_args.module34_supcon_temperature,
        cli_args.module34_center_loss_weight,
        cli_args.module34_center_margin,
        cli_args.module34_center_intra_weight,
        cli_args.module34_center_inter_weight,
        cli_args.module34_branch_ce_loss_weight,
        cli_args.hpec_energy_mode,
        cli_args.hpec_classification_mode,
        cli_args.hpec_loss_mode,
        cli_args.hpec_evidence_weight,
        cli_args.hpec_avoid_busemann_double_count,
        cli_args.hpec_prototypes_per_class,
        cli_args.hpec_proto_temperature,
        cli_args.hpec_busemann_temperature,
        cli_args.hpec_busemann_point_radius,
        cli_args.hpec_busemann_radius_gate_weight,
        cli_args.hpec_busemann_radius_gate_center,
        cli_args.hpec_busemann_class_bias_weight,
        cli_args.hpec_prototype_ce_loss_weight,
        cli_args.hpec_prototype_logit_mode,
        cli_args.hpec_prototype_logit_scale,
        cli_args.hpec_residual_calibration,
        cli_args.hpec_residual_calibration_scale,
        cli_args.hpec_residual_calibration_momentum,
        cli_args.hpec_residual_calibration_batch_weight,
        cli_args.hpec_prototype_residual_weight,
        cli_args.hpec_hgcn_logit_blend,
        cli_args.hpec_network_energy_weight,
        cli_args.hpec_network_energy_mode,
        cli_args.hpec_network_energy_temperature,
        cli_args.hpec_network_energy_prior_weight,
        cli_args.hpec_network_energy_normalize,
        cli_args.hpec_network_selector_sharpness,
        cli_args.hpec_energy_loss_weight,
        cli_args.hpec_energy_ce_margin,
        cli_args.hpec_causal_role_energy_weight,
        cli_args.hpec_z_radius_loss_weight,
        cli_args.hpec_input_tangent_noise_std,
        cli_args.hpec_prototype_separation_loss_weight,
        cli_args.hpec_teacher_distill_weight,
        cli_args.hpec_teacher_distill_mode,
        cli_args.hpec_data_init,
        cli_args.hpec_prototype_update_mode,
        cli_args.hpec_reliable_confidence_threshold,
        cli_args.hpec_reliable_view_consistency_threshold,
        cli_args.hpec_reliable_min_samples,
        cli_args.hpec_reliable_weight_floor,
        cli_args.hpec_epoch_frechet_steps,
        cli_args.hpec_ema_start_epoch,
        cli_args.hpec_trainable_prototypes,
        cli_args.hpec_prototype_lr_scale,
        cli_args.hpec_prototype_parameterization,
        *metric_cells,
        train_seconds,
    ]
    rows.append(row)
    sheets[sheet_name] = rows
    _write_xlsx(result_path, sheets)
    print(f"5-fold result saved to: {result_path} [{sheet_name}]")


def _manifest_path(cli_args):
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs" / "training_records"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cli_args.data))
    return output_dir / f"{dataset}_{timestamp}_{os.getpid()}.json"


def _metric_dict(metrics):
    return {name: float(value) for name, value in zip(METRIC_NAMES, metrics)}


def create_training_manifest(cli_args):
    path = _manifest_path(cli_args)
    record = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "entry_script": Path(sys.argv[0]).name,
        "command": list(sys.argv),
        "parameters": vars(cli_args),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path, record


def finalize_training_manifest(path, record, status, metrics=None, error=None):
    record["status"] = status
    record["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if metrics is not None:
        record["metrics"] = _metric_dict(metrics)
    if error:
        record["error"] = str(error)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    cli_args = parse_args()
    manifest_path, manifest = create_training_manifest(cli_args)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    iteration_metrics = []

    print(f"Standalone {cli_args.data} best-config test:")
    for repeat in range(max(1, int(cli_args.iterations))):
        seed_everything(cli_args.seed + repeat)
        args = build_args(cli_args)
        run_tag = f"run_{repeat + 1}" if cli_args.iterations > 1 else "run"
        print(
            f"  [{run_tag}] data={args.data}, path={args.data_path}, protocol={args.protocol}, "
            f"folds={args.kfold}, max_folds={args.max_folds}, epochs={args.train_epochs}, "
            f"batch_size={args.batch_size}, lr={args.learning_rate}, device={'cuda' if args.use_gpu else 'cpu'}"
        )
        print(
            "  modules: "
            f"module1={args.use_deci_module1}, module2={args.use_causal_module2}, "
            f"module3(HGCN)={args.use_hgcn_module3}, module4(HPEC)={args.use_hpec_module4}"
        )

        exp = Exp_Main(args)
        setting = f"{setting_name(args)}_{run_tag}"
        metrics = exp.kf_train(setting)
        iteration_metrics.append(metrics)

    print(f"Standalone {cli_args.data} best-config test completed.")
    if len(iteration_metrics) == 1:
        final_metrics = iteration_metrics[0]
        print(format_metrics(final_metrics))
        if int(cli_args.max_folds) == int(cli_args.kfold):
            save_result_row(cli_args, final_metrics, setting_name(build_args(cli_args)))
    else:
        stacked = np.asarray(iteration_metrics, dtype=float)
        final_metrics = stacked.mean(axis=0)
        std_metrics = stacked.std(axis=0)
        print("Mean: " + format_metrics(final_metrics))
        print("Std : " + format_metrics(std_metrics))
        if int(cli_args.max_folds) == int(cli_args.kfold):
            save_result_row(cli_args, final_metrics, setting_name(build_args(cli_args)))
    manifest["settings"] = [setting_name(build_args(cli_args))]
    manifest["iteration_count"] = len(iteration_metrics)
    finalize_training_manifest(manifest_path, manifest, "completed", final_metrics)


if __name__ == "__main__":
    main()
