import argparse
import torch
from exp.exp_classification_CV import Exp_Main
import random
import numpy as np
import os
import re
from pathlib import Path
from tqdm import tqdm
import psutil

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true/false/1/0/yes/no")

def use_cpus(gpus: list, cpus_per_gpu: int):
        cpus = []
        for gpu in gpus:
            cpus.extend(list(range(gpu* cpus_per_gpu, (gpu+1)* cpus_per_gpu)))
        p = psutil.Process()
        p.cpu_affinity(cpus)
        print("A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(len(cpus)))
        
def seed_set(seed=2024):
    random.seed(seed)   
    np.random.seed(seed)   
    torch.manual_seed(seed)   

def infer_site_count(data_path, protocol):
    pattern = f"*_{protocol}_features_timeseries.mat"
    sites = set()
    for path in Path(data_path).rglob(pattern):
        match = re.search(r"_(s\d+)[_-]", path.name.lower())
        sites.add(match.group(1) if match else "unknown")
    return max(len(sites), 1)

def normalize_s_deci_module_args(args):
    if args.model != "S-DeCI":
        return args
    args.use_deci_module1 = int(bool(args.use_deci_module1))
    args.use_causal_module2 = int(bool(args.use_causal_module2))
    module34_default = int(bool(args.use_hyperbolic_modules34))
    args.use_hgcn_module3 = (
        module34_default
        if args.use_hgcn_module3 is None
        else int(bool(args.use_hgcn_module3))
    )
    args.use_hpec_module4 = (
        module34_default
        if args.use_hpec_module4 is None
        else int(bool(args.use_hpec_module4))
    )
    if args.use_hpec_module4 and not args.use_hgcn_module3:
        raise ValueError("--use_hpec_module4 1 requires --use_hgcn_module3 1")
    args.use_hyperbolic_modules34 = int(bool(args.use_hgcn_module3 or args.use_hpec_module4))
    if not args.use_hpec_module4:
        args.hpec_prototype_update_mode = "none"
    args.use_sample_correlation_when_module2_disabled = int(
        bool(args.use_sample_correlation_when_module2_disabled)
    )
    # 低层模块 2 参数收敛为内部默认值，避免日常训练命令被实现细节淹没。
    module2_defaults = {
        "causal_graph_hidden_dim": 0,
        "dag_sampling_temperature": 1.0,
        "dag_sampling_noise": 0.0,
        "dag_sampling_sinkhorn_iters": 20,
        "dag_sampling_hard": 1,
        "detach_causal_input": 1,
        "lambda_causal_stability": 0.0,
        "causal_edge_dropout": 0.0,
        "sample_graph_delta_scale": 0.05,
        "sample_graph_hidden_dim": 0,
        "lambda_sample_graph_l1": 0.0,
        "lambda_sample_graph_deviation": 0.0,
        "causal_loss_schedule": "constant",
        "causal_dag_warmup_epochs": 0,
        "causal_l1_warmup_epochs": 0,
        "sample_graph_reg_warmup_epochs": 0,
        "dagma_logdet_s": 1.0,
        "dagma_logdet_margin": 0.1,
        "causal_analytic_margin": 0.1,
        "causal_analytic_power_iters": 5,
        "temporal_sem_input_norm": "time_zscore",
        "temporal_sample_graph_delta_scale": 0.15,
        "temporal_sample_graph_rank": 4,
        "lambda_temporal_pred": 1.0,
        "lambda_temporal_sparse": 0.0005,
        "lambda_temporal_smooth": 0.0001,
        "lambda_temporal_group_sparse": 0.0,
        "lambda_temporal_lag_hierarchy": 0.0,
        "temporal_prediction_loss_mode": "bold_alff",
        "temporal_pred_huber_delta": 1.0,
        "lambda_temporal_pred_delta": 0.2,
        "lambda_temporal_pred_lowfreq": 0.2,
        "lambda_temporal_pred_corr": 0.05,
        "temporal_lowfreq_kernel_size": 9,
        "temporal_a0_sparse_ratio": 0.2,
        "temporal_a0_scale": 0.03,
        "temporal_prediction_target_mode": "innovation",
        "temporal_candidate_parent_topk": 4,
        "temporal_decoder_activation": "identity",
        "temporal_dagma_warmup_epochs": 5,
        "temporal_dagma_barrier_epochs": 20,
        "temporal_reg_warmup_epochs": 0,
        "temporal_attention_heads": 2,
        "temporal_attention_head_dim": 8,
        "temporal_attention_dropout": 0.0,
        "temporal_attention_graph_scale": 1.0,
        "classification_graph_source": "causal_soft_masked_fc",
        "module2_graph_residual_alpha": 0.10,
        "detach_module2_graph_for_classification": 0,
    }
    module1_defaults = {
        "module1_random_crop": 0,
        "module1_feature_mode": "alff",
        "module1_tr": 2.0,
        "module1_alff_low_hz": 0.01,
        "module1_alff_high_hz": 0.08,
        "module1_alff_time_weight": 0.2,
        "module1_temporal_dropout": 0.0,
        "module1_roi_dropout": 0.0,
        "module1_denoise_loss_weight": 0.0,
        "module1_temporal_stats_weight": 0.0,
    }
    module3_defaults = {
        "hgcn_residual_alpha": 0.35,
        "hgcn_einstein_readout_weight": 0.0,
        "use_causal_role_readout": 0,
        "causal_role_temperature": 1.0,
        "hgcn_fc_inject_weight": 0.0,
        "hgcn_fc_anchor_norm_target": 0.5,
        "hgcn_fc_anchor_gate_init": -1.5,
        "module34_geo_dtype": "auto",
        "keep_gcn_fallback_with_hyperbolic": 0,
        "hyperbolic_logit_residual_weight": 0.0,
        "hpec_teacher_distill_weight": 0.0,
        "hpec_teacher_distill_temperature": 2.0,
        "hpec_teacher_detach": 1,
        "hpec_z_radius_loss_weight": 0.0,
        "hpec_ema_start_epoch": 0,
        "hpec_prototype_lr_scale": 1.0,
        "hpec_z_radius_target": 0.3,
        "hpec_prototype_separation_loss_weight": 0.0,
        "hpec_prototype_separation_max_cos": 0.35,
        "hpec_prototype_ce_loss_weight": 0.05,
        "hpec_energy_ce_margin": 0.0,
        "hpec_causal_role_energy_weight": 0.0,
        "hpec_evidence_weight": 1.25,
        "hpec_avoid_busemann_double_count": 0,
    }
    for name, value in {**module1_defaults, **module2_defaults, **module3_defaults}.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args

def train(ii,args):
    setting = '{}_data_type_{}_protocol_{}_kfold_{}_model_{}_bs_{}_lr_{}_dp_{}_dm_{}_seq_{}'.format(
                    args.data,
                    args.data_type,
                    args.protocol,
                    args.kfold,
                    args.model,
                    args.batch_size,
                    args.learning_rate,
                    args.dropout,
                    args.d_model,
                    args.seq_len,
                    )

    exp = Exp(args)  # set experiments
    print(f'Start K-Fold Cross Validation Training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    avg_metric=exp.kf_train(setting) if args.Method == 'DL'  else exp.kf_ML(setting)
    print(f'End Training : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    
    return avg_metric
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='用于 fMRI 分类的通用时间序列骨干训练入口')
    root_parser = parser

    # ==================== 基础配置 ====================
    parser = root_parser.add_argument_group('==================== 基础配置 ====================')
    parser.add_argument('--Method', type=str, default='DL',
                        help='选择训练方法：ML 表示机器学习模型（如 SVM/svm、RF/rf），DL 表示深度学习模型')
    parser.add_argument('--model', type=str, default='S-DeCI',
                        help='模型名称，可选如 [Linear, iTransformer, DeCI, S-DeCI]')

    # ==================== 数据加载 ====================
    parser = root_parser.add_argument_group('==================== 数据加载 ====================')
    parser.add_argument('--data', type=str, default='Abide', help='使用的数据集，可选如 [PPMI, Mātai, Neurocon, Taowu, Abide, MDD]')
    parser.add_argument('--data_path', type=str, default='dataset/Abide', help='数据集文件或目录路径')
    parser.add_argument('--data_type', type=str, default='TS', 
                        help='数据类型，可选 [TS: 原始时间序列, FC: 功能连接矩阵]')
    parser.add_argument('--protocol', type=str, default="AAL116",
                        help='脑模板/分区协议，可选 [schaefer100, AAL116, harvard48, ward100, kmeans100]；同时决定 ROI 数量（即多变量时间序列的 channel 数）')
    parser.add_argument('--kfold', type=int, default=5, help='K-Fold 交叉验证的折数')
    parser.add_argument('--max_folds', type=int, default=0,
                        help='最多实际运行多少个 fold；0 表示运行全部 fold，调参时可设为 1')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型 checkpoint 保存目录')
    parser.add_argument('--del_weight', type=bool, default=True, help='是否在测试后删除模型权重；为 True 时删除以节省空间')

    # ==================== 分类任务 ====================
    parser = root_parser.add_argument_group('==================== 分类任务 ====================')
    parser.add_argument('--seq_len', type=int, default=120,
                        help='fMRI 输入时间长度，可参考 [PPMI: 210, Mātai: 200, Neurocon: 137, Taowu: 239, Abide: 300, MDD: 230]；使用 FC 时 seq_len 通常等于 channel')
    parser.add_argument('--channel', type=int, default=116,
                        help='fMRI ROI 数量，对应多变量时间序列中的 channel/variate；可参考 [schaefer100: 100, AAL116: 116, harvard48: 48, ward100: 100, kmeans100: 100]')
    parser.add_argument('--classes', type=int, default=2,
                        help='分类类别数，可参考 [PPMI: 4, Mātai: 2, Neurocon: 2, Taowu: 2, Abide: 2, MDD: 2]')

    # ==================== 通用模型结构 ====================
    parser = root_parser.add_argument_group('==================== 通用模型结构 ====================')
    parser.add_argument('--d_model', type=int, default=64, help='模型隐藏维度')
    parser.add_argument('--layer', type=int, default=1, help='时间序列骨干网络层数')
    parser.add_argument('--n_head', type=int, default=8, help='多头注意力的 head 数量')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout 比例')
    parser.add_argument('--module1_random_crop', type=int, default=0,
                        help='模块 1 训练期是否对长于 seq_len 的时序执行随机时间窗裁剪')
    parser.add_argument('--module1_feature_mode', type=str, default='alff',
                        choices=['alff', 'deci', 'raw'],
                        help='模块 1 特征模式：alff 使用 ALFF/fALFF 低频生理特征，deci 使用旧 DeCI/Cycle 分解，raw 直接投影原时序')
    parser.add_argument('--module1_tr', type=float, default=2.0,
                        help='模块 1 计算 ALFF/fALFF 时使用的 TR，单位为秒')
    parser.add_argument('--module1_alff_low_hz', type=float, default=0.01,
                        help='模块 1 ALFF/fALFF 低频带下界，单位 Hz')
    parser.add_argument('--module1_alff_high_hz', type=float, default=0.08,
                        help='模块 1 ALFF/fALFF 低频带上界，单位 Hz')
    parser.add_argument('--module1_alff_time_weight', type=float, default=0.2,
                        help='模块 1 低频带通重构时序投影的融合权重')
    parser.add_argument('--module1_temporal_dropout', type=float, default=0.0,
                        help='模块 1 训练期 temporal dropout 比例，随机置零部分时间点')
    parser.add_argument('--module1_roi_dropout', type=float, default=0.0,
                        help='模块 1 训练期 ROI dropout 比例，随机置零部分脑区输入')
    parser.add_argument('--module1_denoise_loss_weight', type=float, default=0.0,
                        help='模块 1 去噪辅助损失权重，0 表示关闭')
    parser.add_argument('--module1_temporal_stats_weight', type=float, default=0.0,
                        help='模块 1 时序统计残差权重；大于 0 时把每个 ROI 的均值、波动和一阶差分统计投影后融合到节点特征')
    parser.add_argument('--use_norm', type=int, default=1, help='是否使用可逆归一化/Reversible Normalization')
    parser.add_argument('--time_series_harmonization', type=str, default='none',
                        choices=['none', 'site_zscore'],
                        help='输入时序层面的多站点 harmonization；site_zscore 表示每个 fold 只用训练集估计站点级 ROI 均值/方差')
    parser.add_argument('--site_harmonization_min_samples', type=int, default=2,
                        help='估计单个站点 harmonization 统计量所需的最小训练样本数，不足时回退到训练集全局统计')
    parser.add_argument('--use_site_adversarial', type=int, default=0,
                        help='是否启用站点对抗 head，让 z_global 难以预测采集站点；当前默认关闭，建议先查看站点-标签分布后再做消融')
    parser.add_argument('--lambda_site_adversarial', type=float, default=0.02,
                        help='站点对抗损失权重，建议从较小值开始')
    parser.add_argument('--site_grl_lambda', type=float, default=1.0,
                        help='gradient reversal layer 的反向梯度强度')
    parser.add_argument('--site_adversarial_dropout', type=float, default=0.1,
                        help='站点对抗分类头 dropout')

    # ==================== S-DeCI 模块开关 ====================
    parser = root_parser.add_argument_group('==================== S-DeCI 模块开关 ====================')
    parser.add_argument('--use_deci_module1', type=int, default=1,
                        help='是否在 S-DeCI 中启用模块 1 DeCI/Cycle 分解；0 表示直接投影原始时间序列为 d_model 维节点特征')
    parser.add_argument('--use_causal_module2', type=int, default=1,
                        help='是否在 S-DeCI 中启用模块 2 因果图学习器')
    parser.add_argument('--causal_feature_source', type=str, default='sum',
                        choices=['sum', 'last'], help='S-DeCI 中模块 2 使用的 Cycle feature 来源：sum 表示多层求和，last 表示最后一层')
    parser.add_argument('--causal_graph_method', type=str, default='nts_notears',
                        choices=['nts_notears', 'attn_nts_notears', 'dagma_logdet', 'dag_sampling'], help='S-DeCI 模块 2 因果图学习结构：nts_notears 为时间序列 NTS-NOTEARS，attn_nts_notears 为 Attention-guided Temporal NTS-NOTEARS；静态 DAGMA / DAG-Sampling 仅作 legacy 对照')
    parser.add_argument('--causal_learning_target', type=str, default='temporal_sem',
                        choices=['static_feature', 'temporal_sem'], help='模块 2 学习目标：static_feature 使用节点特征重构，temporal_sem 使用时间序列预测式 SEM')
    parser.add_argument('--temporal_lag_order', type=int, default=5,
                        help='temporal_sem 模式使用的历史时间滞后阶数')
    parser.add_argument('--temporal_reg_warmup_epochs', type=int, default=0,
                        help='temporal SEM 中 DAG、稀疏、平滑和样本图正则的 warmup epoch 数；0 表示不调度')
    parser.add_argument('--temporal_decoder_activation', type=str, default='identity',
                        choices=['tanh', 'sigmoid', 'gelu', 'identity'],
                        help='temporal NTS-NOTEARS decoder 的隐藏激活函数')
    parser.add_argument('--temporal_prediction_loss_mode', type=str, default='bold_alff',
                        choices=['mse', 'huber', 'bold_alff'],
                        help='Module2 temporal prediction loss mode: mse/huber/bold_alff')
    parser.add_argument('--temporal_pred_huber_delta', type=float, default=1.0,
                        help='Module2 Huber delta for robust BOLD prediction loss')
    parser.add_argument('--lambda_temporal_pred_delta', type=float, default=0.2,
                        help='Module2 first-order temporal-delta prediction loss weight')
    parser.add_argument('--lambda_temporal_pred_lowfreq', type=float, default=0.2,
                        help='Module2 low-frequency trend prediction loss weight for ALFF/fALFF')
    parser.add_argument('--lambda_temporal_pred_corr', type=float, default=0.05,
                        help='Module2 temporal waveform correlation loss weight')
    parser.add_argument('--temporal_lowfreq_kernel_size', type=int, default=9,
                        help='Module2 low-frequency moving-average kernel size')
    parser.add_argument('--temporal_a0_sparse_ratio', type=float, default=0.2,
                        help='Module2 A0 contemporaneous residual graph sparsity ratio')
    parser.add_argument('--temporal_a0_scale', type=float, default=0.03,
                        help='Module2 A0 contemporaneous correction scale')
    parser.add_argument('--temporal_prediction_target_mode', type=str, default='innovation',
                        choices=['innovation', 'next_value'],
                        help='Module2 predicts BOLD innovation or the next value directly')
    parser.add_argument('--temporal_attention_heads', type=int, default=2,
                        help='Attention-guided Temporal NTS-NOTEARS 使用的 attention head 数')
    parser.add_argument('--temporal_attention_head_dim', type=int, default=8,
                        help='Attention-guided Temporal NTS-NOTEARS 每个 head 的隐藏维度')
    parser.add_argument('--temporal_attention_dropout', type=float, default=0.0,
                        help='Attention-guided Temporal NTS-NOTEARS 中 attention 权重的 dropout 比例')
    parser.add_argument('--temporal_attention_graph_scale', type=float, default=1.0,
                        help='Attention-guided Temporal NTS-NOTEARS 传给模块3分类图的尺度系数；1 表示使用原始聚合图')
    parser.add_argument('--causal_input_norm', type=str, default='none',
                        choices=['none', 'feature_zscore', 'batch_node_zscore'],
                        help='S-DeCI 模块 2 输入特征归一化方式；none 保持旧行为')
    parser.add_argument('--causal_init_logit', type=float, default=-2.0,
                        help='S-DeCI 模块 2 因果邻接矩阵 logit 的初始化值')
    parser.add_argument('--causal_learning_rate', type=float, default=1e-2,
                        help='S-DeCI 模块 2 因果图参数的单独优化学习率')
    parser.add_argument('--causal_threshold', type=float, default=0.15,
                        help='可视化二值因果邻接矩阵时使用的阈值')
    parser.add_argument('--lambda_causal_recon', type=float, default=0.02,
                        help='S-DeCI 模块 2 reconstruction loss 权重')
    parser.add_argument('--lambda_causal_dag', type=float, default=0.001,
                        help='S-DeCI 模块 2 DAG acyclicity loss 权重')
    parser.add_argument('--lambda_causal_l1', type=float, default=0.0001,
                        help='S-DeCI 模块 2 L1 sparsity loss 权重')
    parser.add_argument('--lambda_causal_stability', type=float, default=0.0,
                        help='S-DeCI 模块 2 因果图稳定性 loss 权重')
    parser.add_argument('--causal_edge_dropout', type=float, default=0.0,
                        help='模块 3 训练期因果图边 dropout 比例')
    parser.add_argument('--use_sample_graph_residual', type=int, default=0,
                        help='是否启用样本级残差图 A_delta；0 表示仅使用共享因果图 A_shared')
    parser.add_argument('--module2_graph_residual_alpha', type=float, default=0.10,
                        help='classification_graph_source=residual_blend 时，模块 2 因果图作为 FC 方向残差的强度')
    parser.add_argument('--classification_graph_source', type=str, default='causal_soft_masked_fc',
                        choices=['blend', 'learned', 'causal', 'sample_correlation', 'fc', 'residual_blend', 'topk_blend', 'gated_fc', 'causal_masked_fc', 'causal_soft_masked_fc', 'gated_fc_centered', 'gated_fc_signed'],
                        help='分类用图来源：causal_soft_masked_fc 表示因果候选边保留 FC 权重、非候选边只保留少量 FC 背景')
    parser.add_argument('--detach_module2_graph_for_classification', type=int, default=0,
                        help='是否阻断分类 loss 回传到模块 2 图参数；1 表示模块 2 图仅由时序因果损失学习')
    parser.add_argument('--use_hyperbolic_modules34', type=int, default=0,
                        help='是否联合启用 S-DeCI 模块 3 HGCN 与模块 4 HPEC；0 表示使用普通 GCN fallback 分类路径')

    # ==================== S-DeCI 模块 3 HGCN ====================
    parser = root_parser.add_argument_group('==================== S-DeCI 模块 3 HGCN ====================')
    parser.add_argument('--use_hgcn_module3', type=int, default=None,
                        help='模块 3 HGCN 开关；不填写时跟随 use_hyperbolic_modules34，总开关仅作为兼容默认值')
    parser.add_argument('--hgcn_hidden_dim', type=int, default=128,
                        help='S-DeCI 模块 3 输出的 z_global / 切空间分类特征维度')
    parser.add_argument('--hgcn_layers', type=int, default=1,
                        help='S-DeCI 模块 3 HGCN 图卷积层数')
    parser.add_argument('--hgcn_curvature', type=float, default=1.0,
                        help='S-DeCI 模块 3 Poincare Ball 曲率参数')
    parser.add_argument('--hgcn_backclip_radius', type=float, default=1.0,
                        help='S-DeCI 模块 3 Backclip 限幅半径')
    parser.add_argument('--hgcn_dropout', type=float, default=0.0,
                        help='S-DeCI 模块 3 HGCN 切空间特征 dropout 比例')
    parser.add_argument('--use_multi_hop_causal_encoding', type=int, default=0,
                        help='是否在模块3输入加入模块2有向因果图的多阶可达性编码')
    parser.add_argument('--causal_reachability_hops', type=int, default=2,
                        help='多阶因果可达性最大 hop 数，建议 1-3')
    parser.add_argument('--causal_reachability_scale', type=float, default=0.25,
                        help='多阶因果编码以残差注入模块3输入的强度')
    parser.add_argument('--hgcn_residual_alpha', type=float, default=0.35,
                        help='S-DeCI 模块 3 HGCN 输出与输入双曲特征的 residual 混合比例')
    parser.add_argument('--hgcn_fc_inject_weight', type=float, default=0.0,
                        help='模块 3 将 FC readout embedding 注入双曲切空间的权重；0 表示关闭该注入路径')
    parser.add_argument('--hgcn_fc_anchor_norm_target', type=float, default=0.5,
                        help='FC anchor 注入切空间前的目标范数；控制 FC 证据进入双曲表示的幅度')
    parser.add_argument('--hgcn_fc_anchor_gate_init', type=float, default=-1.5,
                        help='FC anchor 门控的初始 logit，越小表示训练初期越保守')
    parser.add_argument('--keep_gcn_fallback_with_hyperbolic', type=int, default=0,
                        help='启用模块 3/4 时是否保留 GCN fallback logits 作为主分类输出')
    parser.add_argument('--hyperbolic_logit_residual_weight', type=float, default=0.0,
                        help='双曲原型 evidence 进入最终 logit 融合的权重；参数名沿用 residual，语义是双视角证据融合')
    parser.add_argument('--hyperbolic_residual_fusion_mode', type=str, default='residual',
                        choices=['residual', 'logit_blend', 'binary_margin', 'dual_consensus', 'dual_margin_consensus'],
                        help='模块3/4与GCN fallback的logit融合方式；dual_margin_consensus只融合正负类margin方向证据')
    parser.add_argument('--class_loss_weighting', type=str, default='none',
                        choices=['none', 'batch_balanced', 'sqrt_batch_balanced', 'logit_adjusted', 'sqrt_logit_adjusted'],
                        help='分类交叉熵的类别均衡方式；MDD 等类别不均衡数据可尝试 sqrt_batch_balanced')
    parser.add_argument('--class_logit_adjust_tau', type=float, default=1.0,
                        help='logit_adjusted 类别先验校正强度')
    parser.add_argument('--fc_residual_weight', type=float, default=0.0,
                        help='模块 3 图级中心点融合样本 FC 节点强度残差的权重')
    parser.add_argument('--hpec_hgcn_logit_blend', type=float, default=0.1,
                        help='模块 4 最终预测中融合 HGCN 切空间 logits 的权重')
    parser.add_argument('--hgcn_add_self_loop', type=int, default=1,
                        help='S-DeCI 模块 3 图传播时是否加入 self-loop')
    parser.add_argument('--hgcn_adjacency_normalization', type=str, default='row',
                        choices=['row', 'sym', 'none'], help='S-DeCI 模块 3 因果邻接矩阵归一化方式')
    parser.add_argument('--use_brain_network_prior', type=int, default=1,
                        help='模块 3 是否启用 MDD 文献脑网络先验 readout；AAL116 下会聚合 DMN、fronto-limbic、control、salience 等网络')
    parser.add_argument('--use_sample_correlation_when_module2_disabled', type=int, default=1,
                        help='当 S-DeCI 关闭模块 2 但启用模块 3 时，是否加载每个样本对应的相关系数矩阵作为 HGCN adjacency')
    parser.add_argument('--sample_correlation_mode', type=str, default='abs',
                        choices=['abs', 'positive', 'raw'], help='样本相关矩阵作为图结构时的负值处理方式：abs 取绝对值，positive 仅保留正相关，raw 保留原值')
    parser.add_argument('--module34_geo_dtype', type=str, default='auto',
                        choices=['auto', 'float32', 'float64'],
                        help='模块 3/4 几何计算精度；auto 保持训练速度，float64 更稳但更慢')
    parser.add_argument('--use_causal_role_readout', type=int, default=0,
                        help='是否按有向图读取global/source/sink/hub四个双曲中心')
    parser.add_argument('--causal_role_temperature', type=float, default=1.0,
                        help='因果角色ROI权重的softmax温度')

    # ==================== S-DeCI 模块 4 HPEC ====================
    parser = root_parser.add_argument_group('==================== S-DeCI 模块 4 HPEC ====================')
    parser.add_argument('--use_hpec_module4', type=int, default=None,
                        help='模块 4 HPEC 开关；不填写时跟随 use_hyperbolic_modules34；开启模块 4 必须同时开启模块 3')
    parser.add_argument('--hpec_prototype_radius', type=float, default=0.3,
                        help='HPEC 类别 prototype 在 Poincare Ball 中的初始化半径')
    parser.add_argument('--hpec_cone_k', type=float, default=0.1,
                        help='HPEC cone aperture/psi 的 K 参数')
    parser.add_argument('--hpec_margin', type=float, default=0.5,
                        help='HPEC energy loss 中非真实类别的 margin')
    parser.add_argument('--hpec_prototypes_per_class', type=int, default=2,
                        help='HPEC 每个类别使用的 prototype 数量；1 表示单 prototype 回退路径')
    parser.add_argument('--hpec_proto_temperature', type=float, default=0.6,
                        help='HPEC 多 prototype 相似度和 soft-min 聚合使用的温度')
    parser.add_argument('--hpec_distance_weight', type=float, default=0.5,
                        help='HPEC energy 中 Poincare distance 项的权重；0 表示仅使用 cone violation')
    parser.add_argument('--hpec_energy_scale', type=float, default=1.0,
                        help='HPEC energy 矩阵缩放系数')
    parser.add_argument('--hpec_energy_mode', type=str, default='busemann',
                        choices=['cone', 'busemann'],
                        help='模块 4 energy 模式：cone 使用 HPEC cone violation；busemann 使用 ideal prototype 的 Busemann score')
    parser.add_argument('--hpec_loss_mode', type=str, default='energy_ce',
                        choices=['margin', 'energy_ce'],
                        help='模块 4 主损失：margin 使用 HPEC margin loss，energy_ce 使用负能量交叉熵')
    parser.add_argument('--hpec_evidence_weight', type=float, default=1.25,
                        help='HPEC energy logits 中叠加 prototype similarity 证据的权重；MDD 5fold/50epoch 当前 1.25 最好')
    parser.add_argument('--hpec_prototype_ce_loss_weight', type=float, default=0.05,
                        help='模块 4 多原型相似度 CE 辅助损失权重；MDD 5fold/50epoch 当前默认 0.05')
    parser.add_argument('--hpec_energy_ce_margin', type=float, default=0.0,
                        help='真实类别HPEC energy CE的附加间隔；0关闭')
    parser.add_argument('--hpec_causal_role_energy_weight', type=float, default=0.0,
                        help='因果角色HPEC能量相对全局能量的融合比例')
    parser.add_argument('--hpec_avoid_busemann_double_count', type=int, default=0,
                        help='Busemann 模式下避免 energy logits 与 prototype similarity 使用同一证据重复叠加；MDD 默认保留叠加')
    parser.add_argument('--hpec_residual_calibration', type=str, default='batch_margin',
                        choices=['none', 'batch_margin', 'tanh_margin', 'running_batch_margin', 'hybrid_batch_running_margin', 'train_class_margin'],
                        help='模块4双曲证据进入最终融合前的无标签margin校准方式')
    parser.add_argument('--hpec_residual_calibration_scale', type=float, default=0.5,
                        help='模块4margin校准后的尺度')
    parser.add_argument('--hpec_residual_calibration_momentum', type=float, default=0.05,
                        help='running/hybrid校准的训练期EMA动量')
    parser.add_argument('--hpec_residual_calibration_batch_weight', type=float, default=0.5,
                        help='hybrid校准在测试期使用当前batch统计的权重')
    parser.add_argument('--hpec_teacher_distill_weight', type=float, default=0.0,
                        help='可选 distillation 校准消融权重；默认关闭，仅用于测试小样本 logits 校准是否有帮助')
    parser.add_argument('--hpec_teacher_distill_temperature', type=float, default=2.0,
                        help='HPEC distillation 校准的 softmax 温度')
    parser.add_argument('--hpec_teacher_detach', type=int, default=1,
                        help='是否阻断参考 logits 的梯度；默认 1 表示参考分布只用于校准消融')
    parser.add_argument('--hpec_z_radius_loss_weight', type=float, default=0.0,
                        help='模块 4 防止 z_global 退回双曲球中心的半径正则权重')
    parser.add_argument('--hpec_z_radius_target', type=float, default=0.3,
                        help='模块 4 z_global 半径正则的目标 Poincare 半径')
    parser.add_argument('--hpec_prototype_separation_loss_weight', type=float, default=0.0,
                        help='模块 4 prototype 方向分离正则权重，用于避免多原型挤在同一方向')
    parser.add_argument('--hpec_prototype_separation_max_cos', type=float, default=0.35,
                        help='prototype 分离正则允许的最大余弦相似度')
    parser.add_argument('--module34_branch_ce_decay_epochs', type=int, default=0,
                        help='模块3/4分支CE在前N个epoch线性衰减；0表示不调度')
    parser.add_argument('--module34_branch_ce_min_ratio', type=float, default=1.0,
                        help='模块3/4分支CE衰减后的最低比例，避免完全关闭模块监督')
    parser.add_argument('--hpec_busemann_temperature', type=float, default=2.0,
                        help='Busemann energy 聚合多个 prototype 时的温度')
    parser.add_argument('--hpec_busemann_radius_gate_weight', type=float, default=0.0,
                        help='Busemann radius gate weight; 0 disables this calibration')
    parser.add_argument('--hpec_busemann_radius_gate_center', type=float, default=0.3,
                        help='Busemann radius gate center')
    parser.add_argument('--hpec_data_init', type=int, default=0,
                        help='是否用首个训练 batch 的类别中心 warm-start HPEC prototype')
    parser.add_argument('--hpec_prototype_update_mode', type=str, default='reliable_tp_ema',
                        choices=['reliable_tp_ema', 'sinkhorn_ema', 'none'],
                        help='HPEC原型更新：可靠TP EMA、Sinkhorn legacy对照或冻结')
    parser.add_argument('--hpec_reliable_confidence_threshold', type=float, default=0.70,
                        help='可靠TP原型更新的真实类别最小预测概率')
    parser.add_argument('--hpec_reliable_min_samples', type=int, default=2,
                        help='一个类内prototype执行可靠EMA所需的最少TP样本数')
    parser.add_argument('--hpec_ema_alpha', type=float, default=0.995,
                        help='HPEC prototype EMA 更新的历史保留系数')
    parser.add_argument('--hpec_ema_anchor_weight', type=float, default=0.15,
                        help='HPEC prototype EMA 更新时保留初始分散方向的权重')
    parser.add_argument('--hpec_ema_start_epoch', type=int, default=20,
                        help='可靠TP原型EMA开始的epoch，先让模块3/4形成初始表征')
    parser.add_argument('--hpec_ema_update_epochs', type=int, default=-1,
                        help='HPEC prototype EMA更新持续epoch数；负数表示warm-up后全程更新')
    parser.add_argument('--hpec_trainable_prototypes', type=int, default=1,
                        help='HPEC prototype 是否参与训练；0 表示固定 prototype')
    parser.add_argument('--hpec_prototype_lr_scale', type=float, default=1.0,
                        help='HPEC prototype/边界偏置学习率缩放；<1 表示慢速原型更新')
    parser.add_argument('--hpec_prototype_parameterization', type=str, default='poincare_point',
                        choices=['poincare_point', 'tangent_direction'],
                        help='HPEC prototype 参数化：poincare_point 为现有 Poincare 点；tangent_direction 为固定半径、只学习 Busemann 理想方向')
    parser.add_argument('--hpec_init_steps', type=int, default=500,
                        help='HPEC hyperspherical separation prototype 初始化步数')
    parser.add_argument('--hpec_eps', type=float, default=1e-7,
                        help='HPEC 角度、孔径和除法计算中的数值稳定 eps')
    # ==================== S-DeCI GCN fallback ====================
    parser = root_parser.add_argument_group('==================== S-DeCI GCN fallback ====================')
    parser.add_argument('--gcn_fallback_hidden_dim', type=int, default=64,
                        help='S-DeCI 模块 3/4 关闭时普通 GCN fallback 的隐藏维度')
    parser.add_argument('--gcn_fallback_layers', type=int, default=1,
                        help='S-DeCI 模块 3/4 关闭时普通 GCN fallback 的图卷积层数')
    parser.add_argument('--gcn_fallback_dropout', type=float, default=0.0,
                        help='S-DeCI 普通 GCN fallback 的 dropout 比例')
    parser.add_argument('--gcn_fallback_add_self_loop', type=int, default=1,
                        help='S-DeCI 普通 GCN fallback 是否给 adjacency 添加 self-loop')
    parser.add_argument('--gcn_fallback_adjacency_normalization', type=str, default='row',
                        choices=['row', 'sym', 'none'], help='S-DeCI 普通 GCN fallback 的 adjacency 归一化方式')
    parser.add_argument('--gcn_fallback_readout_mode', type=str, default='mean_std',
                        choices=['mean', 'attention', 'mean_max', 'mean_std'],
                        help='S-DeCI 模块 3/4 关闭时 GCN fallback 的节点读出方式')
    parser.add_argument('--gcn_fallback_input_residual_weight', type=float, default=0.0,
                        help='S-DeCI 模块 3/4 关闭时 GCN fallback 保留原始节点特征的残差强度')
    parser.add_argument('--gcn_fallback_directional_propagation', type=int, default=1,
                        help='是否分别编码因果图的入边、出边及方向差异')
    parser.add_argument('--gcn_fallback_edge_readout_topk', type=int, default=0,
                        help='S-DeCI GCN fallback 额外读取每个样本最强 top-k 边端点特征；0 表示关闭')

    # ==================== 可视化与诊断 ====================
    parser = root_parser.add_argument_group('==================== 可视化与诊断 ====================')
    parser.add_argument('--visualize_causal', type=int, default=1,
                        help='是否保存 S-DeCI 模块 2/3 的中间量 heatmap')
    parser.add_argument('--causal_vis_dir', type=str, default='outputs/s_deci_causal',
                        help='S-DeCI 因果学习中间量可视化图片保存目录')
    parser.add_argument('--visualize_every', type=int, default=0,
                        help='已废弃参数；当前因果可视化在每个 fold 训练结束后保存一次')
    
    # ==================== 其他骨干模型参数 ====================
    parser = root_parser.add_argument_group('==================== 其他骨干模型参数 ====================')
    parser.add_argument('--factor', type=int, default=1, help='注意力 factor 参数')
    parser.add_argument('--moving_avg', type=int, default=25, help='seasonal-trend decomposition 中移动平均核大小')
    
    #TimesNet
    parser.add_argument('--top_k', type=int, default=5, help='TimesBlock 使用的 top_k 参数')
    parser.add_argument('--num_kernels', type=int, default=2, help='Inception 模块使用的 kernel 数量')
    
    
    # TimeMixer
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='序列分解方法，仅支持 moving_avg 或 dft_decomp；使用 dft_decomp 时需要相应调整 top_k')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='下采样层数')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='下采样窗口大小')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='下采样方法，仅支持 avg、max、conv')
    
    # SegRNN
    parser.add_argument('--seg_len', type=int, default=24,
                        help='SegRNN 中 segment-wise iteration 的片段长度')
    
    # ModernTCN
    parser.add_argument('--small_kernel_merged', type=bool, default=False, help='small_kernel 是否已经完成结构重参数合并')
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio 参数')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio 参数')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio 参数')
    parser.add_argument('--patch_size', type=int, default=16, help='patch 大小')
    parser.add_argument('--patch_stride', type=int, default=8, help='patch 步长')

    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1], help='每个 stage 的 block 数量')
    parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27], help='大卷积核大小')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5], help='结构重参数化使用的小卷积核大小')
    parser.add_argument('--dims', nargs='+',type=int, default=[64,64,64], help='每个 stage 的 d_model 维度')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[64,64,64], help='每个 stage 中 depthwise conv 的维度')
    
    #Medformer
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="Medformer 是否使用 single channel patching",
        default=False,
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="12,24,48",
        help="Medformer 使用的 patch length 列表",
    )
    parser.add_argument(
        "--no_inter_attn",
        action="store_true",
        help="是否关闭 encoder 中的 inter-attention；使用该参数表示不使用 inter-attention",
        default=False,
    )
    parser.add_argument("--activation", type=str, default="gelu", help="激活函数类型")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="是否输出 encoder 中的 attention",
    )
    

    # ==================== 优化与日志 ====================
    parser = root_parser.add_argument_group('==================== 优化与日志 ====================')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--iterations', type=int, default=5,
                        help='重复运行完整交叉验证的次数；调参时可设为 1')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker 数量；当前 Windows + 内存预加载数据默认 0 更快，可手动试 2/4')
    parser.add_argument('--pin_memory', type=int, default=1, help='使用 GPU 时是否启用 DataLoader pinned memory')
    parser.add_argument('--persistent_workers', type=int, default=1, help='num_workers > 0 时是否复用 DataLoader worker')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='每个 DataLoader worker 预取的 batch 数')
    parser.add_argument('--train_epochs', type=int, default=50, help='训练 epoch 数')
    parser.add_argument('--batch_size', type=int, default=8, help='训练输入 batch size')
    parser.add_argument('--patience', type=int, default=50, help='early stopping 的 patience')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='优化器学习率')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='损失函数；二分类可选 MSE/mse、BCE/bce、weighted_bce/wbce，多分类通常使用 CE/ce')
    parser.add_argument('--binary_positive_weight', type=float, default=1.0,
                        help='二分类 weighted_bce 中正类样本的损失权重，MDD 可按 control/patient 比例设为约 1.96')
    parser.add_argument('--early_stop_metric', type=str, default='accuracy',
                        choices=['accuracy', 'acc', 'precision', 'recall', 'macro_f1', 'f1', 'roc_auc', 'auc', 'best_macro_f1', 'best_f1'],
                        help='保存最佳 checkpoint 使用的验证指标；best_macro_f1 表示按验证集最佳阈值后的 macro F1 保存')
    parser.add_argument('--use_best_threshold', type=int, default=1,
                        help='二分类最终评估是否使用训练过程中在验证集上选出的最佳阈值；0 表示固定使用 0.5')
    parser.add_argument('--lradj', type=str, default='constant', help='学习率调整策略')
    parser.add_argument('--print_process', type=int, default=0, help='是否打印每个 epoch 的训练过程')
    parser.add_argument('--print_metric_every', type=int, default=10,
                        help='每隔多少个 epoch 打印 loss 和训练/验证指标；0 表示仅由 print_process 控制')
    parser.add_argument('--print_data_info', type=int, default=0, help='是否打印数据加载信息')

    parser.add_argument('--use_tensorboard', type=int, default=1, help='是否启用 TensorBoard 标量日志')
    parser.add_argument('--tensorboard_dir', type=str, default='outputs/tensorboard', help='TensorBoard 日志根目录')
    parser.add_argument('--tensorboard_run_name', type=str, default=None, help='TensorBoard run 名称；默认使用 setting')
    parser.add_argument('--tensorboard_disable_smoke_runs', type=int, default=1, help='run 名称包含 smoke 时是否跳过 TensorBoard 记录')

    # ==================== GPU 与 CPU 绑定 ====================
    parser = root_parser.add_argument_group('==================== GPU 与 CPU 绑定 ====================')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='是否使用 GPU')
    parser.add_argument('--gpu', type=int, default=0, help='使用的主 GPU 编号')
    parser.add_argument('--torch_fast_math', type=str2bool, default=True, help='是否启用 cuDNN benchmark / TF32 加速')
    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0], help="需要绑定 CPU affinity 的 GPU 编号列表")
    parser.add_argument("--bind_cpu_affinity", type=str2bool, default=False, help="是否按 GPU 编号绑定 CPU 核；默认关闭以使用全部 CPU 逻辑核")
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多 GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='多 GPU 设备编号列表')


    args = root_parser.parse_args()
    args = normalize_s_deci_module_args(args)
    if args.model == "S-DeCI":
        args.site_count = infer_site_count(args.data_path, args.protocol)
    seed_set(args.seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.torch_fast_math and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # For server 1, set cpus_per_gpu to 12
    #For server 2, set cpus_per_gpu to 24
    if args.bind_cpu_affinity:
        use_cpus(gpus=args.gpu_idx, cpus_per_gpu=8)
    
    print('Args in experiment:')
    print(args)
    Exp = Exp_Main
    avg_metrics=[]
    means=[]
    stds=[]
    
    iteration=args.iterations
    for i in tqdm(range(iteration)):
        print(f'The {i+1}-th {args.kfold} CE training begin>>>>>>>>>>>>>>>>>>>\n')
        avg_metrics.append(train(1,args))
        print(f'The {i+1}-th {args.kfold} CE training end<<<<<<<<<<<<<<<<<<<<<\n')
    means=[np.mean([avg_metrics[i][j] for i in range(iteration)]) for j in range(5)]
    stds=[np.std([avg_metrics[i][j] for i in range(iteration)]) for j in range(5)]
    print(f'Mean accuracy: {means[0]:.4f}, precision: {means[1]:.4f},recall: {means[2]:.4f}, macro_f1: {means[3]:.4f}, roc_auc: {means[4]:.4f}')
    print(f'Std accuracy: {stds[0]:.4f}, precision: {stds[1]:.4f},recall: {stds[2]:.4f}, macro_f1: {stds[3]:.4f}, roc_auc: {stds[4]:.4f}')
