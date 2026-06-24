import argparse
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from exp.exp_classification_CV import Exp_Main


METRIC_NAMES = ("accuracy", "precision", "recall", "macro_f1", "roc_auc")


def infer_site_count(data_path, protocol):
    pattern = f"*_{protocol}_features_timeseries.mat"
    sites = set()
    for path in Path(data_path).rglob(pattern):
        match = re.search(r"_(s\d+)[_-]", path.name.lower())
        sites.add(match.group(1) if match else "unknown")
    return max(len(sites), 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行 ABIDE 指定序列长度的完整四模块 S-DeCI 测试配置。"
    )
    parser.add_argument("--data", default="Abide")
    parser.add_argument("--data-path", default="dataset/Abide")
    parser.add_argument("--data-type", default="TS", choices=("TS", "FC"))
    parser.add_argument("--protocol", default="AAL116")
    parser.add_argument("--channel", type=int, default=116)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7e-2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--module1-random-crop", type=int, default=1)
    parser.add_argument("--module1-feature-mode", default="alff", choices=("alff", "deci", "raw"))
    parser.add_argument("--module1-tr", type=float, default=2.0)
    parser.add_argument("--module1-alff-low-hz", type=float, default=0.01)
    parser.add_argument("--module1-alff-high-hz", type=float, default=0.08)
    parser.add_argument("--module1-alff-time-weight", type=float, default=0.2)
    parser.add_argument("--module1-temporal-dropout", type=float, default=0.12)
    parser.add_argument("--module1-roi-dropout", type=float, default=0.08)
    parser.add_argument("--module1-denoise-loss-weight", type=float, default=0.0)
    parser.add_argument("--module1-temporal-stats-weight", type=float, default=0.0)
    parser.add_argument("--use-norm", type=int, default=1)
    parser.add_argument("--time-series-harmonization", default="none", choices=("none", "site_zscore"))
    parser.add_argument("--site-harmonization-min-samples", type=int, default=2)
    parser.add_argument("--use-site-adversarial", type=int, default=0)
    parser.add_argument("--lambda-site-adversarial", type=float, default=0.02)
    parser.add_argument("--site-grl-lambda", type=float, default=1.0)
    parser.add_argument("--site-adversarial-dropout", type=float, default=0.1)
    parser.add_argument("--use-site-modulation", type=int, default=0)
    parser.add_argument("--site-modulation-dim", type=int, default=16)
    parser.add_argument("--site-modulation-strength", type=float, default=0.1)
    parser.add_argument("--lambda-site-modulation-reg", type=float, default=0.001)

    # ABIDE 主路径默认启用完整四模块；GCN fallback 仅作为显式消融对照。
    parser.add_argument("--use-deci-module1", type=int, default=1)
    parser.add_argument("--use-causal-module2", type=int, default=1)
    parser.add_argument("--use-hyperbolic-modules34", type=int, default=0)
    parser.add_argument("--use-hgcn-module3", type=int, default=None)
    parser.add_argument("--sample-correlation-mode", default="abs", choices=("abs", "positive", "raw"))
    parser.add_argument("--gcn-fallback-hidden-dim", type=int, default=16)
    parser.add_argument("--gcn-fallback-layers", type=int, default=1)
    parser.add_argument("--gcn-fallback-dropout", type=float, default=0.2)
    parser.add_argument("--gcn-fallback-add-self-loop", type=int, default=1)
    parser.add_argument("--gcn-fallback-adjacency-normalization", default="row", choices=("row", "sym", "none"))
    parser.add_argument("--use-sample-correlation-when-module2-disabled", type=int, default=1)

    # 保留模块 2/3/4 参数，便于后续直接在本脚本中做消融。
    parser.add_argument("--causal-feature-source", default="sum", choices=("sum", "last"))
    parser.add_argument("--causal-graph-method", default="nts_notears", choices=("nts_notears", "dagma_logdet", "dag_sampling"))
    parser.add_argument("--causal-learning-target", default="temporal_sem", choices=("static_feature", "temporal_sem"))
    parser.add_argument("--temporal-lag-order", type=int, default=2)
    parser.add_argument("--temporal-causal-init-logit", type=float, default=-3.5)
    parser.add_argument("--temporal-sample-graph-delta-scale", type=float, default=0.01)
    parser.add_argument("--temporal-sample-graph-rank", type=int, default=4)
    parser.add_argument("--lambda-temporal-pred", type=float, default=1.0)
    parser.add_argument("--lambda-temporal-sparse", type=float, default=0.0005)
    parser.add_argument("--lambda-temporal-smooth", type=float, default=0.0002)
    parser.add_argument("--lambda-temporal-counterfactual", type=float, default=0.0)
    parser.add_argument("--temporal-counterfactual-edges", type=int, default=4)
    parser.add_argument("--temporal-counterfactual-temperature", type=float, default=0.1)
    parser.add_argument("--temporal-counterfactual-interval", type=int, default=1)
    parser.add_argument("--temporal-counterfactual-baseline", default="zero", choices=("zero", "shuffle"))
    parser.add_argument("--temporal-dagma-warmup-epochs", type=int, default=5)
    parser.add_argument("--temporal-dagma-barrier-epochs", type=int, default=20)
    parser.add_argument("--temporal-reg-warmup-epochs", type=int, default=8)
    parser.add_argument("--causal-input-norm", default="none", choices=("none", "feature_zscore", "batch_node_zscore"))
    parser.add_argument("--causal-init-logit", type=float, default=-1.0)
    parser.add_argument("--causal-learning-rate", type=float, default=5e-4)
    parser.add_argument("--causal-threshold", type=float, default=0.05)
    parser.add_argument("--lambda-causal-recon", type=float, default=0.01)
    parser.add_argument("--lambda-causal-dag", type=float, default=0.0001)
    parser.add_argument("--lambda-causal-l1", type=float, default=0.00005)
    parser.add_argument("--lambda-causal-stability", type=float, default=0.002)
    parser.add_argument("--use-sample-graph-residual", type=int, default=0)
    parser.add_argument("--module2-sample-correlation-blend", type=float, default=0.75)
    parser.add_argument("--classification-graph-source", default="blend", choices=("blend", "learned", "causal", "sample_correlation", "fc"))
    parser.add_argument("--lambda-sample-graph-l1", type=float, default=0.0001)
    parser.add_argument("--lambda-sample-graph-deviation", type=float, default=0.001)
    parser.add_argument("--hgcn-hidden-dim", type=int, default=64)
    parser.add_argument("--hgcn-layers", type=int, default=1)
    parser.add_argument("--hgcn-curvature", type=float, default=1.0)
    parser.add_argument("--hgcn-backclip-radius", type=float, default=1.0)
    parser.add_argument("--hgcn-dropout", type=float, default=0.25)
    parser.add_argument("--hgcn-residual-alpha", type=float, default=0.35)
    parser.add_argument("--hgcn-graph-readout-alpha", type=float, default=0.0)
    parser.add_argument("--hgcn-delta-readout-alpha", type=float, default=0.0)
    parser.add_argument("--hgcn-readout-mode", default="network_stats", choices=("node_stats", "network_stats"))
    parser.add_argument("--fc-residual-weight", type=float, default=0.2)
    parser.add_argument("--causal-edge-dropout", type=float, default=0.25)
    parser.add_argument("--lambda-hgcn-radius-reg", type=float, default=0.001)
    parser.add_argument("--hgcn-radius-target", type=float, default=0.85)
    parser.add_argument("--lambda-hgcn-cls-aux", type=float, default=0.3)
    parser.add_argument("--lambda-hgcn-view-consistency", type=float, default=0.1)
    parser.add_argument("--hpec-hgcn-logit-blend", type=float, default=0.0)
    parser.add_argument("--lambda-hgcn-supcon", type=float, default=0.0)
    parser.add_argument("--hgcn-supcon-temperature", type=float, default=0.2)
    parser.add_argument("--hgcn-add-self-loop", type=int, default=1)
    parser.add_argument("--hgcn-adjacency-normalization", default="row", choices=("row", "sym", "none"))
    parser.add_argument("--use-brain-network-prior", type=int, default=1)
    parser.add_argument("--use-hpec-module4", type=int, default=None)
    parser.add_argument("--hpec-prototype-radius", type=float, default=0.3)
    parser.add_argument("--hpec-cone-k", type=float, default=0.1)
    parser.add_argument("--hpec-margin", type=float, default=0.25)
    parser.add_argument("--hpec-prototypes-per-class", type=int, default=2)
    parser.add_argument("--hpec-proto-temperature", type=float, default=0.35)
    parser.add_argument("--hpec-distance-weight", type=float, default=0.35)
    parser.add_argument("--hpec-energy-scale", type=float, default=1.0)
    parser.add_argument("--hpec-energy-mode", default="cone", choices=("cone", "busemann"))
    parser.add_argument("--hpec-loss-mode", default="energy_ce", choices=("margin", "energy_ce"))
    parser.add_argument("--hpec-busemann-temperature", type=float, default=1.0)
    parser.add_argument("--hpec-data-init", type=int, default=1)
    parser.add_argument("--lambda-hpec-mle", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-pcl", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-pal", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-radius-reg", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-diversity", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-hsic", type=float, default=0.01)
    parser.add_argument("--lambda-hpec-intra-orthogonal", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-inter-margin", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-ce-aux", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-energy-loss", type=float, default=0.0)
    parser.add_argument("--hpec-use-sinkhorn-ema", type=int, default=1)
    parser.add_argument("--hpec-sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--hpec-sinkhorn-iters", type=int, default=3)
    parser.add_argument("--hpec-ema-alpha", type=float, default=0.995)
    parser.add_argument("--hpec-ema-anchor-weight", type=float, default=0.15)
    parser.add_argument("--hpec-ema-update-epochs", type=int, default=2)
    parser.add_argument("--hpec-trainable-prototypes", type=int, default=0)
    parser.add_argument("--hpec-init-steps", type=int, default=500)
    parser.add_argument("--hpec-eps", type=float, default=1e-7)

    parser.add_argument("--use-gpu", type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--torch-fast-math", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--persistent-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--del-weight", type=int, default=1)
    parser.add_argument("--print-process", type=int, default=0)
    parser.add_argument("--print-metric-every", type=int, default=5)
    parser.add_argument("--print-data-info", type=int, default=0)
    parser.add_argument("--use-best-threshold", type=int, default=1)
    parser.add_argument("--visualize-causal", type=int, default=1)
    parser.add_argument("--causal-vis-dir", default="outputs/abide_best_config_causal")
    parser.add_argument("--visualize-every", type=int, default=10)
    parser.add_argument("--keep-weight", action="store_true")
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
        layer=cli_args.layer,
        n_head=8,
        dropout=cli_args.dropout,
        module1_random_crop=cli_args.module1_random_crop,
        module1_feature_mode=cli_args.module1_feature_mode,
        module1_tr=cli_args.module1_tr,
        module1_alff_low_hz=cli_args.module1_alff_low_hz,
        module1_alff_high_hz=cli_args.module1_alff_high_hz,
        module1_alff_time_weight=cli_args.module1_alff_time_weight,
        module1_temporal_dropout=cli_args.module1_temporal_dropout,
        module1_roi_dropout=cli_args.module1_roi_dropout,
        module1_denoise_loss_weight=cli_args.module1_denoise_loss_weight,
        module1_temporal_stats_weight=cli_args.module1_temporal_stats_weight,
        use_norm=cli_args.use_norm,
        time_series_harmonization=cli_args.time_series_harmonization,
        site_harmonization_min_samples=cli_args.site_harmonization_min_samples,
        use_site_adversarial=cli_args.use_site_adversarial,
        lambda_site_adversarial=cli_args.lambda_site_adversarial,
        site_grl_lambda=cli_args.site_grl_lambda,
        site_adversarial_dropout=cli_args.site_adversarial_dropout,
        use_site_modulation=cli_args.use_site_modulation,
        site_modulation_dim=cli_args.site_modulation_dim,
        site_modulation_strength=cli_args.site_modulation_strength,
        lambda_site_modulation_reg=cli_args.lambda_site_modulation_reg,
        site_count=infer_site_count(data_path, cli_args.protocol),
        use_deci_module1=cli_args.use_deci_module1,
        use_causal_module2=cli_args.use_causal_module2,
        causal_feature_source=cli_args.causal_feature_source,
        causal_graph_method=cli_args.causal_graph_method,
        causal_learning_target=cli_args.causal_learning_target,
        temporal_lag_order=cli_args.temporal_lag_order,
        temporal_causal_init_logit=cli_args.temporal_causal_init_logit,
        temporal_sem_input_norm="time_zscore",
        temporal_sample_graph_delta_scale=cli_args.temporal_sample_graph_delta_scale,
        temporal_sample_graph_rank=cli_args.temporal_sample_graph_rank,
        lambda_temporal_pred=cli_args.lambda_temporal_pred,
        lambda_temporal_sparse=cli_args.lambda_temporal_sparse,
        lambda_temporal_smooth=cli_args.lambda_temporal_smooth,
        lambda_temporal_counterfactual=cli_args.lambda_temporal_counterfactual,
        temporal_counterfactual_edges=cli_args.temporal_counterfactual_edges,
        temporal_counterfactual_temperature=cli_args.temporal_counterfactual_temperature,
        temporal_counterfactual_interval=cli_args.temporal_counterfactual_interval,
        temporal_counterfactual_baseline=cli_args.temporal_counterfactual_baseline,
        temporal_dagma_warmup_epochs=cli_args.temporal_dagma_warmup_epochs,
        temporal_dagma_barrier_epochs=cli_args.temporal_dagma_barrier_epochs,
        temporal_reg_warmup_epochs=cli_args.temporal_reg_warmup_epochs,
        causal_input_norm=cli_args.causal_input_norm,
        causal_init_logit=cli_args.causal_init_logit,
        causal_learning_rate=cli_args.causal_learning_rate,
        causal_graph_hidden_dim=0,
        dag_sampling_temperature=1.0,
        dag_sampling_noise=0.0,
        dag_sampling_sinkhorn_iters=20,
        dag_sampling_hard=1,
        causal_threshold=cli_args.causal_threshold,
        detach_causal_input=1,
        lambda_causal_recon=cli_args.lambda_causal_recon,
        lambda_causal_dag=cli_args.lambda_causal_dag,
        lambda_causal_l1=cli_args.lambda_causal_l1,
        lambda_causal_stability=cli_args.lambda_causal_stability,
        use_sample_graph_residual=cli_args.use_sample_graph_residual,
        module2_sample_correlation_blend=cli_args.module2_sample_correlation_blend,
        classification_graph_source=cli_args.classification_graph_source,
        sample_graph_delta_scale=0.05,
        sample_graph_hidden_dim=0,
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
        hgcn_layers=cli_args.hgcn_layers,
        hgcn_curvature=cli_args.hgcn_curvature,
        hgcn_backclip_radius=cli_args.hgcn_backclip_radius,
        hgcn_dropout=cli_args.hgcn_dropout,
        hgcn_residual_alpha=cli_args.hgcn_residual_alpha,
        hgcn_graph_readout_alpha=cli_args.hgcn_graph_readout_alpha,
        hgcn_delta_readout_alpha=cli_args.hgcn_delta_readout_alpha,
        hgcn_readout_mode=cli_args.hgcn_readout_mode,
        fc_residual_weight=cli_args.fc_residual_weight,
        causal_edge_dropout=cli_args.causal_edge_dropout,
        lambda_hgcn_radius_reg=cli_args.lambda_hgcn_radius_reg,
        hgcn_radius_target=cli_args.hgcn_radius_target,
        lambda_hgcn_cls_aux=cli_args.lambda_hgcn_cls_aux,
        lambda_hgcn_view_consistency=cli_args.lambda_hgcn_view_consistency,
        hpec_hgcn_logit_blend=cli_args.hpec_hgcn_logit_blend,
        lambda_hgcn_supcon=cli_args.lambda_hgcn_supcon,
        hgcn_supcon_temperature=cli_args.hgcn_supcon_temperature,
        hgcn_add_self_loop=cli_args.hgcn_add_self_loop,
        hgcn_adjacency_normalization=cli_args.hgcn_adjacency_normalization,
        use_brain_network_prior=cli_args.use_brain_network_prior,
        use_sample_correlation_when_module2_disabled=cli_args.use_sample_correlation_when_module2_disabled,
        sample_correlation_mode=cli_args.sample_correlation_mode,
        gcn_fallback_hidden_dim=cli_args.gcn_fallback_hidden_dim,
        gcn_fallback_layers=cli_args.gcn_fallback_layers,
        gcn_fallback_dropout=cli_args.gcn_fallback_dropout,
        gcn_fallback_add_self_loop=cli_args.gcn_fallback_add_self_loop,
        gcn_fallback_adjacency_normalization=cli_args.gcn_fallback_adjacency_normalization,
        hpec_prototype_radius=cli_args.hpec_prototype_radius,
        hpec_cone_k=cli_args.hpec_cone_k,
        hpec_margin=cli_args.hpec_margin,
        hpec_prototypes_per_class=cli_args.hpec_prototypes_per_class,
        hpec_proto_temperature=cli_args.hpec_proto_temperature,
        hpec_distance_weight=cli_args.hpec_distance_weight,
        hpec_energy_scale=cli_args.hpec_energy_scale,
        hpec_energy_mode=cli_args.hpec_energy_mode,
        hpec_loss_mode=cli_args.hpec_loss_mode,
        hpec_busemann_temperature=cli_args.hpec_busemann_temperature,
        hpec_data_init=cli_args.hpec_data_init,
        lambda_hpec_mle=cli_args.lambda_hpec_mle,
        lambda_hpec_pcl=cli_args.lambda_hpec_pcl,
        lambda_hpec_pal=cli_args.lambda_hpec_pal,
        lambda_hpec_radius_reg=cli_args.lambda_hpec_radius_reg,
        lambda_hpec_diversity=cli_args.lambda_hpec_diversity,
        lambda_hpec_hsic=cli_args.lambda_hpec_hsic,
        lambda_hpec_intra_orthogonal=cli_args.lambda_hpec_intra_orthogonal,
        lambda_hpec_inter_margin=cli_args.lambda_hpec_inter_margin,
        lambda_hpec_ce_aux=cli_args.lambda_hpec_ce_aux,
        lambda_hpec_energy_loss=cli_args.lambda_hpec_energy_loss,
        hpec_use_sinkhorn_ema=cli_args.hpec_use_sinkhorn_ema,
        hpec_sinkhorn_epsilon=cli_args.hpec_sinkhorn_epsilon,
        hpec_sinkhorn_iters=cli_args.hpec_sinkhorn_iters,
        hpec_ema_alpha=cli_args.hpec_ema_alpha,
        hpec_ema_anchor_weight=cli_args.hpec_ema_anchor_weight,
        hpec_ema_update_epochs=cli_args.hpec_ema_update_epochs,
        hpec_trainable_prototypes=cli_args.hpec_trainable_prototypes,
        hpec_init_steps=cli_args.hpec_init_steps,
        hpec_eps=cli_args.hpec_eps,
        visualize_causal=cli_args.visualize_causal,
        causal_vis_dir=str(root / cli_args.causal_vis_dir),
        visualize_every=cli_args.visualize_every,
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
        num_workers=cli_args.num_workers,
        pin_memory=cli_args.pin_memory,
        persistent_workers=cli_args.persistent_workers,
        prefetch_factor=cli_args.prefetch_factor,
        train_epochs=cli_args.train_epochs,
        batch_size=cli_args.batch_size,
        patience=cli_args.patience,
        learning_rate=cli_args.learning_rate,
        loss=cli_args.loss,
        lradj="constant",
        early_stop_metric="macro_f1",
        use_best_threshold=cli_args.use_best_threshold,
        binary_positive_weight=1.0,
        print_process=cli_args.print_process,
        print_metric_every=cli_args.print_metric_every,
        print_data_info=cli_args.print_data_info,
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
        f"m3_{args.use_hgcn_module3}_m4_{args.use_hpec_module4}_adj_{args.sample_correlation_mode}_"
        f"full_sdeci_hpec_{args.hpec_prototypes_per_class}proto_loss_{args.loss}"
    )


def format_metrics(metrics):
    return ", ".join(f"{name}: {float(value):.4f}" for name, value in zip(METRIC_NAMES, metrics))


def main():
    cli_args = parse_args()
    if cli_args.torch_fast_math and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    iteration_metrics = []

    print(f"Standalone ABIDE-{cli_args.seq_len} test configuration:")
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
            f"module3(HGCN)={args.use_hgcn_module3}, module4(HPEC)={args.use_hpec_module4}, "
            f"path={'HGCN+HPEC' if args.use_hpec_module4 else ('HGCN-only' if args.use_hgcn_module3 else 'GCN fallback')}, "
            f"adjacency={args.sample_correlation_mode} correlation when fallback/blend"
        )

        exp = Exp_Main(args)
        metrics = exp.kf_train(f"{setting_name(args)}_{run_tag}")
        iteration_metrics.append(metrics)

    print(f"Standalone ABIDE-{cli_args.seq_len} test completed.")
    if len(iteration_metrics) == 1:
        print(format_metrics(iteration_metrics[0]))
    else:
        stacked = np.asarray(iteration_metrics, dtype=float)
        mean_metrics = stacked.mean(axis=0)
        std_metrics = stacked.std(axis=0)
        print("Mean: " + format_metrics(mean_metrics))
        print("Std : " + format_metrics(std_metrics))


if __name__ == "__main__":
    main()
