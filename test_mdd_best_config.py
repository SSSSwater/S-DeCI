import argparse
import random
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

import numpy as np
import torch

from exp.exp_classification_CV import Exp_Main


METRIC_NAMES = ("accuracy", "precision", "recall", "macro_f1", "roc_auc")
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
        description="Run the current best-known MDD configuration as a standalone test."
    )
    parser.add_argument("--data", default="MDD")
    parser.add_argument("--data-path", default="dataset/MDD")
    parser.add_argument("--data-type", default="TS", choices=("TS", "FC"))
    parser.add_argument("--protocol", default="AAL116")
    parser.add_argument("--channel", type=int, default=116)
    parser.add_argument("--seq-len", type=int, default=230)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lradj", default="constant", choices=("constant", "cosine", "type1", "type2", "type3", "type4"))
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-model-ema", type=int, default=0)
    parser.add_argument("--model-ema-decay", type=float, default=0.995)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--module1-random-crop", type=int, default=1)
    parser.add_argument("--module1-feature-mode", default="alff", choices=("alff", "deci", "raw"))
    parser.add_argument("--module1-tr", type=float, default=2.0)
    parser.add_argument("--module1-alff-low-hz", type=float, default=0.01)
    parser.add_argument("--module1-alff-high-hz", type=float, default=0.08)
    parser.add_argument("--module1-alff-time-weight", type=float, default=0.2)
    parser.add_argument("--module1-temporal-dropout", type=float, default=0.03)
    parser.add_argument("--module1-roi-dropout", type=float, default=0.02)
    parser.add_argument("--module1-denoise-loss-weight", type=float, default=0.0)
    parser.add_argument("--module1-temporal-stats-weight", type=float, default=0.0)
    parser.add_argument("--use-norm", type=int, default=1)
    parser.add_argument("--time-series-harmonization", default="site_zscore", choices=("none", "site_zscore"))
    parser.add_argument("--site-harmonization-min-samples", type=int, default=2)
    parser.add_argument("--use-site-adversarial", type=int, default=0)
    parser.add_argument("--lambda-site-adversarial", type=float, default=0.02)
    parser.add_argument("--site-grl-lambda", type=float, default=1.0)
    parser.add_argument("--site-adversarial-dropout", type=float, default=0.1)
    parser.add_argument("--use-deci-module1", type=int, default=1)
    parser.add_argument("--use-causal-module2", type=int, default=1)
    parser.add_argument("--use-hyperbolic-modules34", type=int, default=1)
    parser.add_argument("--use-hgcn-module3", type=int, default=None)
    parser.add_argument("--sample-correlation-mode", default="abs", choices=("abs", "positive", "raw"))
    parser.add_argument("--gcn-fallback-hidden-dim", type=int, default=64)
    parser.add_argument("--gcn-fallback-layers", type=int, default=1)
    parser.add_argument("--gcn-fallback-dropout", type=float, default=0.0)
    parser.add_argument("--gcn-fallback-add-self-loop", type=int, default=1)
    parser.add_argument("--gcn-fallback-adjacency-normalization", default="row", choices=("row", "sym", "none"))
    parser.add_argument("--use-sample-correlation-when-module2-disabled", type=int, default=1)
    parser.add_argument("--causal-feature-source", default="sum", choices=("sum", "last"))
    parser.add_argument("--causal-graph-method", default="nts_notears", choices=("nts_notears", "dagma_logdet", "dag_sampling"))
    parser.add_argument("--causal-learning-target", default="temporal_sem", choices=("static_feature", "temporal_sem"))
    parser.add_argument("--temporal-lag-order", type=int, default=2)
    parser.add_argument("--temporal-causal-init-logit", type=float, default=-4.0)
    parser.add_argument("--temporal-sample-graph-delta-scale", type=float, default=0.02)
    parser.add_argument("--temporal-sample-graph-rank", type=int, default=4)
    parser.add_argument("--lambda-temporal-pred", type=float, default=1.0)
    parser.add_argument("--lambda-temporal-sparse", type=float, default=0.0005)
    parser.add_argument("--lambda-temporal-smooth", type=float, default=0.0001)
    parser.add_argument("--lambda-temporal-counterfactual", type=float, default=0.0)
    parser.add_argument("--temporal-counterfactual-edges", type=int, default=4)
    parser.add_argument("--temporal-counterfactual-temperature", type=float, default=0.1)
    parser.add_argument("--temporal-counterfactual-interval", type=int, default=1)
    parser.add_argument("--temporal-counterfactual-baseline", default="zero", choices=("zero", "shuffle"))
    parser.add_argument("--temporal-dagma-warmup-epochs", type=int, default=5)
    parser.add_argument("--temporal-dagma-barrier-epochs", type=int, default=20)
    parser.add_argument("--causal-input-norm", default="none", choices=("none", "feature_zscore", "batch_node_zscore"))
    parser.add_argument("--causal-init-logit", type=float, default=-1.0)
    parser.add_argument("--causal-learning-rate", type=float, default=5e-4)
    parser.add_argument("--causal-threshold", type=float, default=0.05)
    parser.add_argument("--lambda-causal-recon", type=float, default=0.01)
    parser.add_argument("--lambda-causal-dag", type=float, default=0.0001)
    parser.add_argument("--lambda-causal-l1", type=float, default=0.00001)
    parser.add_argument("--use-sample-graph-residual", type=int, default=0)
    parser.add_argument("--module2-sample-correlation-blend", type=float, default=0.75)
    parser.add_argument("--classification-graph-source", default="blend", choices=("blend", "learned", "causal", "sample_correlation", "fc"))
    parser.add_argument("--lambda-sample-graph-l1", type=float, default=0.0)
    parser.add_argument("--lambda-sample-graph-deviation", type=float, default=0.0)
    parser.add_argument("--hgcn-hidden-dim", type=int, default=64)
    parser.add_argument("--hgcn-layers", type=int, default=1)
    parser.add_argument("--hgcn-curvature", type=float, default=1.0)
    parser.add_argument("--hgcn-backclip-radius", type=float, default=1.0)
    parser.add_argument("--hgcn-dropout", type=float, default=0.3)
    parser.add_argument("--hgcn-residual-alpha", type=float, default=0.35)
    parser.add_argument("--hgcn-graph-readout-alpha", type=float, default=0.0)
    parser.add_argument("--hgcn-delta-readout-alpha", type=float, default=0.0)
    parser.add_argument("--hgcn-readout-mode", default="node_stats", choices=("node_stats", "network_stats"))
    parser.add_argument("--hgcn-add-self-loop", type=int, default=1)
    parser.add_argument("--hgcn-adjacency-normalization", default="row", choices=("row", "sym", "none"))
    parser.add_argument("--use-brain-network-prior", type=int, default=0)
    parser.add_argument("--fc-residual-weight", type=float, default=0.02)
    parser.add_argument("--fc-network-residual-weight", type=float, default=0.0)
    parser.add_argument("--fc-residual-norm-target", type=float, default=0.0)
    parser.add_argument("--use-fc-residual-gate", type=int, default=1)
    parser.add_argument("--fc-residual-gate-init", type=float, default=-2.0)
    parser.add_argument("--causal-edge-dropout", type=float, default=0.25)
    parser.add_argument("--lambda-hgcn-radius-reg", type=float, default=0.0)
    parser.add_argument("--hgcn-radius-target", type=float, default=0.95)
    parser.add_argument("--lambda-hgcn-cls-aux", type=float, default=0.0)
    parser.add_argument(
        "--class-loss-weighting",
        default="batch_balanced",
        choices=("none", "batch_balanced", "sqrt_batch_balanced"),
    )
    parser.add_argument("--class-label-smoothing", type=float, default=0.0)
    parser.add_argument("--lambda-hgcn-view-consistency", type=float, default=0.0)
    parser.add_argument("--hpec-hgcn-logit-blend", type=float, default=0.0)
    parser.add_argument("--hpec-evidence-weight", type=float, default=0.5)
    parser.add_argument("--hpec-logit-temperature", type=float, default=0.5)
    parser.add_argument("--hpec-gate-init", type=float, default=-2.5)
    parser.add_argument("--lambda-hgcn-supcon", type=float, default=0.0)
    parser.add_argument("--hgcn-supcon-temperature", type=float, default=0.2)
    parser.add_argument("--use-hpec-module4", type=int, default=None)
    parser.add_argument("--hpec-prototype-radius", type=float, default=0.3)
    parser.add_argument("--hpec-cone-k", type=float, default=0.1)
    parser.add_argument("--hpec-margin", type=float, default=0.5)
    parser.add_argument("--hpec-prototypes-per-class", type=int, default=2)
    parser.add_argument("--hpec-proto-temperature", type=float, default=0.6)
    parser.add_argument("--hpec-distance-weight", type=float, default=0.5)
    parser.add_argument("--hpec-energy-scale", type=float, default=1.0)
    parser.add_argument("--hpec-energy-mode", default="cone", choices=("cone", "busemann"))
    parser.add_argument("--hpec-loss-mode", default="energy_ce", choices=("margin", "energy_ce"))
    parser.add_argument("--hpec-busemann-temperature", type=float, default=1.0)
    parser.add_argument("--hpec-data-init", type=int, default=1)
    parser.add_argument("--lambda-hpec-mle", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-pcl", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-pal", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-radius-reg", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-diversity", type=float, default=0.01)
    parser.add_argument("--lambda-hpec-hsic", type=float, default=0.005)
    parser.add_argument("--lambda-hpec-intra-orthogonal", type=float, default=0.005)
    parser.add_argument("--lambda-hpec-inter-margin", type=float, default=0.03)
    parser.add_argument("--lambda-hpec-class-center-margin", type=float, default=0.03)
    parser.add_argument("--lambda-hpec-anchor", type=float, default=0.02)
    parser.add_argument("--lambda-hpec-ce-aux", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-energy-loss", type=float, default=0.0)
    parser.add_argument("--hpec-use-sinkhorn-ema", type=int, default=1)
    parser.add_argument("--hpec-sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--hpec-sinkhorn-iters", type=int, default=3)
    parser.add_argument("--hpec-ema-alpha", type=float, default=0.995)
    parser.add_argument("--hpec-ema-anchor-weight", type=float, default=0.25)
    parser.add_argument("--hpec-ema-update-epochs", type=int, default=5)
    parser.add_argument("--hpec-sample-margin", type=float, default=0.3)
    parser.add_argument("--hpec-intra-class-max-cos", type=float, default=0.25)
    parser.add_argument("--hpec-inter-class-max-cos", type=float, default=0.0)
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
    parser.add_argument("--print-metric-every", type=int, default=10)
    parser.add_argument("--print-data-info", type=int, default=0)
    parser.add_argument("--visualize-causal", type=int, default=1)
    parser.add_argument("--causal-vis-dir", default="outputs/mdd_best_config_causal")
    parser.add_argument("--visualize-every", type=int, default=20)
    parser.add_argument("--result-file", default="result.xlsx")
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
        hgcn_add_self_loop=cli_args.hgcn_add_self_loop,
        hgcn_adjacency_normalization=cli_args.hgcn_adjacency_normalization,
        use_brain_network_prior=cli_args.use_brain_network_prior,
        fc_residual_weight=cli_args.fc_residual_weight,
        fc_network_residual_weight=cli_args.fc_network_residual_weight,
        fc_residual_norm_target=cli_args.fc_residual_norm_target,
        use_fc_residual_gate=cli_args.use_fc_residual_gate,
        fc_residual_gate_init=cli_args.fc_residual_gate_init,
        causal_edge_dropout=cli_args.causal_edge_dropout,
        lambda_hgcn_radius_reg=cli_args.lambda_hgcn_radius_reg,
        hgcn_radius_target=cli_args.hgcn_radius_target,
        lambda_hgcn_cls_aux=cli_args.lambda_hgcn_cls_aux,
        class_loss_weighting=cli_args.class_loss_weighting,
        class_label_smoothing=cli_args.class_label_smoothing,
        lambda_hgcn_view_consistency=cli_args.lambda_hgcn_view_consistency,
        hpec_hgcn_logit_blend=cli_args.hpec_hgcn_logit_blend,
        hpec_evidence_weight=cli_args.hpec_evidence_weight,
        hpec_logit_temperature=cli_args.hpec_logit_temperature,
        hpec_gate_init=cli_args.hpec_gate_init,
        lambda_hgcn_supcon=cli_args.lambda_hgcn_supcon,
        hgcn_supcon_temperature=cli_args.hgcn_supcon_temperature,
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
        lambda_hpec_class_center_margin=cli_args.lambda_hpec_class_center_margin,
        lambda_hpec_anchor=cli_args.lambda_hpec_anchor,
        lambda_hpec_ce_aux=cli_args.lambda_hpec_ce_aux,
        lambda_hpec_energy_loss=cli_args.lambda_hpec_energy_loss,
        hpec_use_sinkhorn_ema=cli_args.hpec_use_sinkhorn_ema,
        hpec_sinkhorn_epsilon=cli_args.hpec_sinkhorn_epsilon,
        hpec_sinkhorn_iters=cli_args.hpec_sinkhorn_iters,
        hpec_ema_alpha=cli_args.hpec_ema_alpha,
        hpec_ema_anchor_weight=cli_args.hpec_ema_anchor_weight,
        hpec_ema_update_epochs=cli_args.hpec_ema_update_epochs,
        hpec_sample_margin=cli_args.hpec_sample_margin,
        hpec_intra_class_max_cos=cli_args.hpec_intra_class_max_cos,
        hpec_inter_class_max_cos=cli_args.hpec_inter_class_max_cos,
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
        weight_decay=cli_args.weight_decay,
        use_model_ema=cli_args.use_model_ema,
        model_ema_decay=cli_args.model_ema_decay,
        loss=cli_args.loss,
        lradj=cli_args.lradj,
        early_stop_metric="best_macro_f1",
        use_best_threshold=1,
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
        f"m3_{args.use_hgcn_module3}_m4_{args.use_hpec_module4}_target_{args.causal_learning_target}_"
        f"lag_{args.temporal_lag_order}_tpred_{args.lambda_temporal_pred}_"
        f"cf_{args.lambda_temporal_counterfactual}_"
        f"graph_{args.causal_graph_method}_adj_{args.sample_correlation_mode}_"
        f"blend_{args.module2_sample_correlation_blend}_loss_{args.loss}"
    )


def format_metrics(metrics):
    return ", ".join(f"{name}: {float(value):.4f}" for name, value in zip(METRIC_NAMES, metrics))


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

        new_rows.append(
            [
                get("module1"),
                get("module2"),
                get("module3"),
                get("module4"),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                get("fc_residual"),
                get("fc_network_residual", 0.0),
                get("fc_gate_init"),
                get("gated_residual"),
                get("hgcn_hidden"),
                get("hgcn_dropout"),
                get("hpec_proto_per_class"),
                get("hpec_loss_mode"),
                _metric_percent(get("accuracy")),
                _metric_percent(get("precision")),
                _metric_percent(get("recall")),
                _metric_percent(get("macro_f1")),
                _metric_percent(get("auc")),
            ]
        )
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
        "causal_target",
        "causal_graph_method",
        "graph_blend",
        "classification_graph_source",
        "class_loss_weighting",
        "class_label_smoothing",
        "lradj",
        "learning_rate",
        "weight_decay",
        "lambda_temporal_sparse",
        "lambda_temporal_counterfactual",
        "temporal_counterfactual_edges",
        "temporal_counterfactual_baseline",
        "lambda_causal_dag",
        "lambda_hgcn_supcon",
        "lambda_hpec_mle",
        "lambda_hpec_pcl",
        "lambda_hpec_pal",
        "lambda_hpec_energy_loss",
        "lambda_hpec_hsic",
        "lambda_hpec_inter_margin",
        "lambda_hpec_class_center_margin",
        "lambda_hpec_anchor",
        "hpec_sample_margin",
        "hpec_intra_class_max_cos",
        "hpec_inter_class_max_cos",
        "hpec_hgcn_logit_blend",
        "hpec_evidence_weight",
        "hpec_logit_temperature",
        "hpec_gate_init",
        "lambda_hpec_ce_aux",
        "fc_residual",
        "fc_network_residual",
        "fc_gate_init",
        "gated_residual",
        "hgcn_hidden",
        "hgcn_dropout",
        "hpec_proto_per_class",
        "hpec_loss_mode",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "auc",
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
    row = [
        cli_args.use_deci_module1,
        cli_args.use_causal_module2,
        int(cli_args.use_hgcn_module3) if cli_args.use_hgcn_module3 is not None else cli_args.use_hyperbolic_modules34,
        int(cli_args.use_hpec_module4) if cli_args.use_hpec_module4 is not None else cli_args.use_hyperbolic_modules34,
        cli_args.module1_feature_mode,
        cli_args.causal_learning_target,
        cli_args.causal_graph_method,
        cli_args.module2_sample_correlation_blend,
        cli_args.classification_graph_source,
        cli_args.class_loss_weighting,
        cli_args.class_label_smoothing,
        cli_args.lradj,
        cli_args.learning_rate,
        cli_args.weight_decay,
        cli_args.lambda_temporal_sparse,
        cli_args.lambda_temporal_counterfactual,
        cli_args.temporal_counterfactual_edges,
        cli_args.temporal_counterfactual_baseline,
        cli_args.lambda_causal_dag,
        cli_args.lambda_hgcn_supcon,
        cli_args.lambda_hpec_mle,
        cli_args.lambda_hpec_pcl,
        cli_args.lambda_hpec_pal,
        cli_args.lambda_hpec_energy_loss,
        cli_args.lambda_hpec_hsic,
        cli_args.lambda_hpec_inter_margin,
        cli_args.lambda_hpec_class_center_margin,
        cli_args.lambda_hpec_anchor,
        cli_args.hpec_sample_margin,
        cli_args.hpec_intra_class_max_cos,
        cli_args.hpec_inter_class_max_cos,
        cli_args.hpec_hgcn_logit_blend,
        cli_args.hpec_evidence_weight,
        cli_args.hpec_logit_temperature,
        cli_args.hpec_gate_init,
        cli_args.lambda_hpec_ce_aux,
        cli_args.fc_residual_weight,
        cli_args.fc_network_residual_weight,
        cli_args.fc_residual_gate_init,
        cli_args.use_fc_residual_gate,
        cli_args.hgcn_hidden_dim,
        cli_args.hgcn_dropout,
        cli_args.hpec_prototypes_per_class,
        cli_args.hpec_loss_mode,
        *[_metric_percent(value) for value in metrics],
    ]
    rows.append(row)
    sheets[sheet_name] = rows
    _write_xlsx(result_path, sheets)
    print(f"5-fold result saved to: {result_path} [{sheet_name}]")


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
        print(format_metrics(iteration_metrics[0]))
        if int(cli_args.max_folds) == int(cli_args.kfold):
            save_result_row(cli_args, iteration_metrics[0], setting_name(build_args(cli_args)))
    else:
        stacked = np.asarray(iteration_metrics, dtype=float)
        mean_metrics = stacked.mean(axis=0)
        std_metrics = stacked.std(axis=0)
        print("Mean: " + format_metrics(mean_metrics))
        print("Std : " + format_metrics(std_metrics))
        if int(cli_args.max_folds) == int(cli_args.kfold):
            save_result_row(cli_args, mean_metrics, setting_name(build_args(cli_args)))


if __name__ == "__main__":
    main()
