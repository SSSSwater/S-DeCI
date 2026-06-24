from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.DeCI_Layer import DeCI_Block
from layers.causal_graph_learner import (
    CausalGraphLearner,
    normalized_dag_loss,
    threshold_adjacency,
)
from layers.temporal_sem_causal_learner import TemporalSEMCausalLearner
from layers.hyperbolic_gcn_layer import Module3HGCNReadout, _aal116_network_masks
from layers.hpec_energy_layer import HPECPrototypeEnergy
from layers.gcn_fallback_layer import ModuleGCNFallback
from utils.tensor_visualization import visualize_tensors


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reverse(x, lambd=1.0):
    return _GradientReverse.apply(x, lambd)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.use_deci_module1 = bool(getattr(configs, "use_deci_module1", 1))
        default_module1_mode = "alff" if self.use_deci_module1 else "raw"
        self.module1_feature_mode = str(
            getattr(configs, "module1_feature_mode", default_module1_mode)
        ).lower()
        if self.module1_feature_mode in ("physio", "physiological", "alff_falff"):
            self.module1_feature_mode = "alff"
        if self.module1_feature_mode not in ("deci", "alff", "raw"):
            raise ValueError(
                f"Unsupported module1_feature_mode={self.module1_feature_mode!r}. "
                "Use 'alff', 'deci' or 'raw'."
            )
        self.Variate_Embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.deci_blocks = nn.ModuleList(
            [DeCI_Block(configs) for _ in range(configs.layer)]
            if self.use_deci_module1 and self.module1_feature_mode == "deci"
            else []
        )

        self.out_dim = 1 if configs.classes == 2 else configs.classes
        self.use_causal_module2 = bool(getattr(configs, "use_causal_module2", 1))
        self.causal_learning_target = getattr(configs, "causal_learning_target", "temporal_sem")
        if self.causal_learning_target == "temporal_nts_notears":
            self.causal_learning_target = "temporal_sem"
        if self.causal_learning_target not in ("static_feature", "temporal_sem"):
            raise ValueError(
                f"Unsupported causal_learning_target={self.causal_learning_target!r}. "
                "Use 'temporal_sem' for temporal NTS-NOTEARS or 'static_feature' for legacy debug."
            )
        requested_hyperbolic = getattr(configs, "use_hyperbolic_modules34", None)
        default_hyperbolic = bool(requested_hyperbolic) if requested_hyperbolic is not None else True
        self.use_hgcn_module3 = bool(getattr(configs, "use_hgcn_module3", default_hyperbolic))
        self.use_hpec_module4 = bool(getattr(configs, "use_hpec_module4", default_hyperbolic))
        if self.use_hpec_module4 and not self.use_hgcn_module3:
            raise ValueError("use_hpec_module4=1 requires use_hgcn_module3=1.")
        self.use_hyperbolic_modules34 = self.use_hgcn_module3 or self.use_hpec_module4
        self.causal_feature_source = getattr(configs, "causal_feature_source", "sum")
        self.lambda_causal_recon = float(getattr(configs, "lambda_causal_recon", 0.02))
        self.lambda_causal_dag = float(getattr(configs, "lambda_causal_dag", 0.001))
        self.lambda_causal_l1 = float(getattr(configs, "lambda_causal_l1", 0.0001))
        self.lambda_sample_graph_l1 = float(getattr(configs, "lambda_sample_graph_l1", 0.0))
        self.lambda_sample_graph_deviation = float(getattr(configs, "lambda_sample_graph_deviation", 0.0))
        self.causal_loss_schedule = getattr(configs, "causal_loss_schedule", "constant")
        self.causal_dag_warmup_epochs = int(getattr(configs, "causal_dag_warmup_epochs", 0))
        self.causal_l1_warmup_epochs = int(getattr(configs, "causal_l1_warmup_epochs", 0))
        self.sample_graph_reg_warmup_epochs = int(getattr(configs, "sample_graph_reg_warmup_epochs", 0))
        self.causal_threshold = float(getattr(configs, "causal_threshold", 0.05))
        self.causal_vis_dir = getattr(configs, "causal_vis_dir", "outputs/s_deci_causal")
        self.detach_causal_input = bool(getattr(configs, "detach_causal_input", 1))
        self.module1_temporal_dropout = float(getattr(configs, "module1_temporal_dropout", 0.0))
        self.module1_roi_dropout = float(getattr(configs, "module1_roi_dropout", 0.0))
        self.module1_denoise_loss_weight = float(getattr(configs, "module1_denoise_loss_weight", 0.0))
        self.module1_temporal_stats_weight = float(getattr(configs, "module1_temporal_stats_weight", 0.0))
        self.module1_tr = float(getattr(configs, "module1_tr", 2.0))
        self.module1_alff_low_hz = float(getattr(configs, "module1_alff_low_hz", 0.01))
        self.module1_alff_high_hz = float(getattr(configs, "module1_alff_high_hz", 0.08))
        self.module1_alff_time_weight = float(getattr(configs, "module1_alff_time_weight", 0.2))
        self.module1_alff_eps = float(getattr(configs, "module1_alff_eps", 1e-6))
        self.lambda_causal_stability = float(getattr(configs, "lambda_causal_stability", 0.0))
        self.lambda_hgcn_radius_reg = float(getattr(configs, "lambda_hgcn_radius_reg", 0.0))
        self.hgcn_radius_target = float(getattr(configs, "hgcn_radius_target", 0.95))
        self.lambda_hgcn_cls_aux = float(getattr(configs, "lambda_hgcn_cls_aux", 0.0))
        self.class_loss_weighting = str(
            getattr(configs, "class_loss_weighting", "none")
        ).lower()
        self.class_label_smoothing = float(getattr(configs, "class_label_smoothing", 0.0))
        self.lambda_hgcn_view_consistency = float(getattr(configs, "lambda_hgcn_view_consistency", 0.0))
        self.hpec_hgcn_logit_blend = float(getattr(configs, "hpec_hgcn_logit_blend", 0.1))
        self.hpec_evidence_weight = float(getattr(configs, "hpec_evidence_weight", 0.5))
        self.hpec_logit_temperature = float(getattr(configs, "hpec_logit_temperature", 0.5))
        self.hpec_gate_init = float(getattr(configs, "hpec_gate_init", -2.5))
        self.fc_residual_weight = float(getattr(configs, "fc_residual_weight", 0.0))
        self.fc_network_residual_weight = float(getattr(configs, "fc_network_residual_weight", 0.0))
        self.fc_residual_norm_target = float(getattr(configs, "fc_residual_norm_target", 1.0))
        self.use_fc_residual_gate = bool(getattr(configs, "use_fc_residual_gate", 0))
        self.fc_residual_gate_init = float(getattr(configs, "fc_residual_gate_init", -2.0))
        self.lambda_hgcn_supcon = float(getattr(configs, "lambda_hgcn_supcon", 0.0))
        self.hgcn_supcon_temperature = float(getattr(configs, "hgcn_supcon_temperature", 0.2))
        self.current_epoch = 0
        self.lambda_hpec_mle = float(getattr(configs, "lambda_hpec_mle", 0.0))
        self.lambda_hpec_pcl = float(getattr(configs, "lambda_hpec_pcl", 0.0))
        self.lambda_hpec_pal = float(getattr(configs, "lambda_hpec_pal", 0.0))
        self.lambda_hpec_radius_reg = float(getattr(configs, "lambda_hpec_radius_reg", 0.0))
        self.lambda_hpec_diversity = float(getattr(configs, "lambda_hpec_diversity", 0.0))
        self.lambda_hpec_hsic = float(getattr(configs, "lambda_hpec_hsic", 0.0))
        self.lambda_hpec_intra_orthogonal = float(getattr(configs, "lambda_hpec_intra_orthogonal", 0.0))
        self.lambda_hpec_inter_margin = float(getattr(configs, "lambda_hpec_inter_margin", 0.0))
        self.lambda_hpec_class_center_margin = float(getattr(configs, "lambda_hpec_class_center_margin", 0.0))
        self.lambda_hpec_anchor = float(getattr(configs, "lambda_hpec_anchor", 0.0))
        self.lambda_hpec_ce_aux = float(getattr(configs, "lambda_hpec_ce_aux", 0.0))
        self.lambda_hpec_energy_loss = float(getattr(configs, "lambda_hpec_energy_loss", 0.0))
        self.hpec_use_sinkhorn_ema = bool(getattr(configs, "hpec_use_sinkhorn_ema", 1))
        self.hpec_sinkhorn_epsilon = float(getattr(configs, "hpec_sinkhorn_epsilon", 0.05))
        self.hpec_sinkhorn_iters = int(getattr(configs, "hpec_sinkhorn_iters", 3))
        self.hpec_ema_alpha = float(getattr(configs, "hpec_ema_alpha", 0.995))
        self.hpec_ema_anchor_weight = float(getattr(configs, "hpec_ema_anchor_weight", 0.1))
        self.hpec_ema_update_epochs = int(getattr(configs, "hpec_ema_update_epochs", 5))
        self.use_site_adversarial = bool(getattr(configs, "use_site_adversarial", 0))
        self.use_site_modulation = bool(getattr(configs, "use_site_modulation", 0))
        self.site_count = int(getattr(configs, "site_count", 1))
        self.lambda_site_adversarial = float(getattr(configs, "lambda_site_adversarial", 0.0))
        self.site_grl_lambda = float(getattr(configs, "site_grl_lambda", 1.0))
        self.site_modulation_strength = float(getattr(configs, "site_modulation_strength", 0.1))
        self.lambda_site_modulation_reg = float(getattr(configs, "lambda_site_modulation_reg", 0.001))
        self.use_sample_correlation_when_module2_disabled = bool(
            getattr(
                configs,
                "use_sample_correlation_when_module2_disabled",
                int(not self.use_causal_module2),
            )
        )
        self.module2_sample_correlation_blend = float(
            getattr(configs, "module2_sample_correlation_blend", 0.0)
        )
        self.classification_graph_source = str(
            getattr(configs, "classification_graph_source", "blend")
        ).lower()
        self.sample_correlation_mode = getattr(configs, "sample_correlation_mode", "abs")
        self._forward_count = 0

        # 轻量时序统计残差：保留每个 ROI 的低频幅值、波动和一阶动态变化，避免 Cycle 分解过度抹平判别信息。
        if self.use_causal_module2 and self.causal_learning_target == "temporal_sem":
            self.causal_learner = TemporalSEMCausalLearner(
                n_nodes=configs.channel,
                lag_order=int(getattr(configs, "temporal_lag_order", 3)),
                init_logit=float(getattr(configs, "temporal_causal_init_logit", -4.0)),
                input_norm=getattr(configs, "temporal_sem_input_norm", "time_zscore"),
                use_sample_graph_residual=bool(getattr(configs, "use_sample_graph_residual", 0)),
                sample_graph_delta_scale=float(getattr(configs, "temporal_sample_graph_delta_scale", 0.02)),
                sample_graph_rank=int(getattr(configs, "temporal_sample_graph_rank", 4)),
                lambda_pred=float(getattr(configs, "lambda_temporal_pred", 1.0)),
                lambda_dag=float(getattr(configs, "lambda_causal_dag", 0.001)),
                lambda_sparse=float(getattr(configs, "lambda_temporal_sparse", getattr(configs, "lambda_causal_l1", 0.0005))),
                lambda_smooth=float(getattr(configs, "lambda_temporal_smooth", 0.0001)),
                lambda_counterfactual=float(getattr(configs, "lambda_temporal_counterfactual", 0.0)),
                counterfactual_edges=int(getattr(configs, "temporal_counterfactual_edges", 4)),
                counterfactual_temperature=float(getattr(configs, "temporal_counterfactual_temperature", 0.1)),
                counterfactual_interval=int(getattr(configs, "temporal_counterfactual_interval", 1)),
                counterfactual_baseline=getattr(configs, "temporal_counterfactual_baseline", "zero"),
                lambda_sample_l1=float(getattr(configs, "lambda_sample_graph_l1", 0.0)),
                lambda_sample_deviation=float(getattr(configs, "lambda_sample_graph_deviation", 0.0)),
                dagma_warmup_epochs=int(getattr(configs, "temporal_dagma_warmup_epochs", 5)),
                dagma_barrier_epochs=int(getattr(configs, "temporal_dagma_barrier_epochs", 20)),
                reg_warmup_epochs=int(getattr(configs, "temporal_reg_warmup_epochs", 5)),
                graph_hidden_dim=int(getattr(configs, "temporal_graph_hidden_dim", 8)),
            )
        elif self.use_causal_module2:
            # Legacy static feature NOTEARS path. The formal module 2 path is
            # temporal NTS-NOTEARS above, because lagged prediction gives edge
            # direction information that static feature DAG learning cannot.
            self.causal_learner = CausalGraphLearner(
                n_nodes=configs.channel,
                feature_dim=configs.d_model,
                init_logit=float(getattr(configs, "causal_init_logit", -2.0)),
                # 外部统一通过 causal_graph_method 切换 NTS-NOTEARS、DAGMA log-det
                # 与 Differentiable-DAG-Sampling；旧配置仍保持 analytic DAG 约束。
                dag_method="analytic",
                analytic_margin=float(getattr(configs, "causal_analytic_margin", 0.1)),
                analytic_power_iters=int(getattr(configs, "causal_analytic_power_iters", 5)),
                graph_hidden_dim=getattr(configs, "causal_graph_hidden_dim", None),
                graph_method=getattr(configs, "causal_graph_method", "nts_notears"),
                dag_sampling_temperature=float(getattr(configs, "dag_sampling_temperature", 1.0)),
                dag_sampling_noise=float(getattr(configs, "dag_sampling_noise", 0.0)),
                dag_sampling_sinkhorn_iters=int(getattr(configs, "dag_sampling_sinkhorn_iters", 20)),
                dag_sampling_hard=bool(getattr(configs, "dag_sampling_hard", 1)),
                causal_input_norm=getattr(configs, "causal_input_norm", "none"),
                use_sample_graph_residual=bool(getattr(configs, "use_sample_graph_residual", 0)),
                sample_graph_delta_scale=float(getattr(configs, "sample_graph_delta_scale", 0.05)),
                sample_graph_hidden_dim=getattr(configs, "sample_graph_hidden_dim", None),
                dagma_logdet_s=float(getattr(configs, "dagma_logdet_s", 1.0)),
                dagma_logdet_margin=float(getattr(configs, "dagma_logdet_margin", 0.1)),
            )
        else:
            self.causal_learner = None

        if self.use_hgcn_module3:
            if not self.use_causal_module2 and not self.use_sample_correlation_when_module2_disabled:
                raise ValueError(
                    "use_hgcn_module3=1 with use_causal_module2=0 requires "
                    "use_sample_correlation_when_module2_disabled=1."
                )
            self.hgcn_hidden_dim = int(getattr(configs, "hgcn_hidden_dim", 128))
            self.hgcn_module3 = Module3HGCNReadout(
                input_dim=configs.d_model,
                hidden_dim=self.hgcn_hidden_dim,
                num_layers=int(getattr(configs, "hgcn_layers", 1)),
                curvature=float(getattr(configs, "hgcn_curvature", 1.0)),
                backclip_radius=float(getattr(configs, "hgcn_backclip_radius", 1.0)),
                dropout=float(getattr(configs, "hgcn_dropout", getattr(configs, "dropout", 0.0))),
                add_self_loop=bool(getattr(configs, "hgcn_add_self_loop", 1)),
                adjacency_normalization=getattr(configs, "hgcn_adjacency_normalization", "row"),
                sample_correlation_mode=getattr(configs, "sample_correlation_mode", "abs"),
                use_brain_network_prior=bool(getattr(configs, "use_brain_network_prior", 1)),
                causal_edge_dropout=float(getattr(configs, "causal_edge_dropout", 0.0)),
                residual_alpha=float(getattr(configs, "hgcn_residual_alpha", 0.35)),
                graph_readout_alpha=float(getattr(configs, "hgcn_graph_readout_alpha", 0.0)),
                delta_readout_alpha=float(getattr(configs, "hgcn_delta_readout_alpha", 0.0)),
                readout_mode=getattr(configs, "hgcn_readout_mode", "node_stats"),
            )
            self.hgcn_classifier = nn.Linear(self.hgcn_hidden_dim, self.out_dim)
            self.hpec_logit_gate = nn.Parameter(torch.tensor(self.hpec_gate_init, dtype=torch.float32))
            self.fc_residual_projection = nn.Sequential(
                nn.LayerNorm(configs.channel),
                nn.Linear(configs.channel, self.hgcn_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.0))),
                nn.Linear(self.hgcn_hidden_dim, self.hgcn_hidden_dim),
            )
            groups, _ = _aal116_network_masks()
            network_count = len(groups)
            self.fc_network_residual_projection = nn.Sequential(
                nn.LayerNorm(network_count * network_count),
                nn.Linear(network_count * network_count, self.hgcn_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.0))),
                nn.Linear(self.hgcn_hidden_dim, self.hgcn_hidden_dim),
            )
            if int(getattr(configs, "channel", 0)) == 116:
                masks = torch.zeros(network_count, 116, dtype=torch.float32)
                for group_idx, (_, indices) in enumerate(groups):
                    masks[group_idx, indices] = 1.0
                masks = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
            else:
                masks = torch.zeros(0, int(getattr(configs, "channel", 0)), dtype=torch.float32)
            self.register_buffer("fc_network_masks", masks)
            if self.use_fc_residual_gate:
                self.fc_residual_gate = nn.Sequential(
                    nn.LayerNorm(self.hgcn_hidden_dim * 2),
                    nn.Linear(self.hgcn_hidden_dim * 2, self.hgcn_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hgcn_hidden_dim, self.hgcn_hidden_dim),
                )
                nn.init.zeros_(self.fc_residual_gate[-1].weight)
                nn.init.constant_(self.fc_residual_gate[-1].bias, self.fc_residual_gate_init)
            else:
                self.fc_residual_gate = None
        else:
            self.hgcn_hidden_dim = None
            self.hgcn_module3 = None
            self.hgcn_classifier = None
            self.fc_residual_projection = None
            self.fc_network_residual_projection = None
            self.register_buffer("fc_network_masks", torch.zeros(0, int(getattr(configs, "channel", 0)), dtype=torch.float32))
            self.fc_residual_gate = None
        self.fc_residual_norm = nn.LayerNorm(self.hgcn_hidden_dim) if self.hgcn_hidden_dim is not None else None

        if self.use_hpec_module4:
            # 模块 4 使用模块 3 的 z_global 与类别 prototype 计算 HPEC energy。
            self.hpec_module4 = HPECPrototypeEnergy(
                num_classes=configs.classes,
                embedding_dim=self.hgcn_hidden_dim,
                manifold=self.hgcn_module3.manifold,
                prototype_radius=float(getattr(configs, "hpec_prototype_radius", 0.3)),
                cone_k=float(getattr(configs, "hpec_cone_k", 0.1)),
                margin=float(getattr(configs, "hpec_margin", 1.0)),
                prototypes_per_class=int(getattr(configs, "hpec_prototypes_per_class", 1)),
                proto_temperature=float(getattr(configs, "hpec_proto_temperature", 0.2)),
                trainable_prototypes=bool(getattr(configs, "hpec_trainable_prototypes", 0)),
                init_steps=int(getattr(configs, "hpec_init_steps", 500)),
                distance_weight=float(getattr(configs, "hpec_distance_weight", 0.0)),
                energy_scale=float(getattr(configs, "hpec_energy_scale", 1.0)),
                energy_mode=getattr(configs, "hpec_energy_mode", "cone"),
                loss_mode=getattr(configs, "hpec_loss_mode", "margin"),
                busemann_temperature=float(getattr(configs, "hpec_busemann_temperature", 1.0)),
                data_init=bool(getattr(configs, "hpec_data_init", 0)),
                prototype_radius_reg_target=float(
                    getattr(
                        configs,
                        "hpec_prototype_radius_reg_target",
                        getattr(configs, "hpec_prototype_radius", 0.3),
                    )
                ),
                use_sinkhorn_ema=self.hpec_use_sinkhorn_ema,
                sinkhorn_epsilon=self.hpec_sinkhorn_epsilon,
                sinkhorn_iters=self.hpec_sinkhorn_iters,
                ema_alpha=self.hpec_ema_alpha,
                ema_anchor_weight=self.hpec_ema_anchor_weight,
                sample_margin=float(getattr(configs, "hpec_sample_margin", 0.2)),
                intra_class_max_cos=float(getattr(configs, "hpec_intra_class_max_cos", 0.35)),
                inter_class_max_cos=float(getattr(configs, "hpec_inter_class_max_cos", 0.0)),
                eps=float(getattr(configs, "hpec_eps", 1e-7)),
                seed=int(getattr(configs, "seed", 2024)),
            )
        else:
            self.hpec_module4 = None

        if self.use_site_adversarial and self.use_hgcn_module3 and self.site_count > 1:
            # 站点对抗头使用 z_tangent 预测 site；通过 GRL 反向让主表征去站点化。
            self.site_classifier = nn.Sequential(
                nn.Linear(self.hgcn_hidden_dim, self.hgcn_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "site_adversarial_dropout", 0.1))),
                nn.Linear(self.hgcn_hidden_dim // 2, self.site_count),
            )
        else:
            self.site_classifier = None

        if self.use_site_modulation and self.site_count > 1:
            site_embedding_dim = int(getattr(configs, "site_modulation_dim", 16))
            # 站点 FiLM 只做小幅校准，用来减少采集风格差异对模块2/3的干扰。
            self.site_embedding = nn.Embedding(self.site_count, site_embedding_dim)
            self.site_modulator = nn.Sequential(
                nn.Linear(site_embedding_dim, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, configs.d_model * 2),
            )
            nn.init.zeros_(self.site_modulator[-1].weight)
            nn.init.zeros_(self.site_modulator[-1].bias)
        else:
            self.site_embedding = None
            self.site_modulator = None

        # 轻量时序统计残差：零初始化最后一层，使新分支从完全不影响原模型开始学习。
        self.temporal_stats_projection = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        nn.init.zeros_(self.temporal_stats_projection[-1].weight)
        nn.init.zeros_(self.temporal_stats_projection[-1].bias)
        self.alff_feature_projection = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, configs.d_model),
            nn.GELU(),
            nn.Dropout(float(getattr(configs, "dropout", 0.0))),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.alff_band_time_projection = nn.Linear(configs.seq_len, configs.d_model)

        if not self.use_hgcn_module3:
            self.gcn_fallback = ModuleGCNFallback(
                input_dim=configs.d_model,
                hidden_dim=int(getattr(configs, "gcn_fallback_hidden_dim", configs.d_model)),
                out_dim=self.out_dim,
                num_layers=int(getattr(configs, "gcn_fallback_layers", 1)),
                dropout=float(getattr(configs, "gcn_fallback_dropout", getattr(configs, "dropout", 0.0))),
                add_self_loop=bool(getattr(configs, "gcn_fallback_add_self_loop", 1)),
                adjacency_normalization=getattr(
                    configs,
                    "gcn_fallback_adjacency_normalization",
                    getattr(configs, "hgcn_adjacency_normalization", "row"),
                ),
                sample_correlation_mode=getattr(configs, "sample_correlation_mode", "abs"),
            )
        else:
            self.gcn_fallback = None

        self._reset_causal_cache()

    def _reset_causal_cache(self):
        self.latest_cycle_features = None
        self.latest_node_features = None
        self.latest_node_feature_source = None
        self.latest_causal_output = None
        self.latest_module3_output = None
        self.latest_module4_output = None
        self.latest_gcn_fallback_output = None
        self.latest_primary_loss = None
        self.latest_prototype_aux_loss = None
        self.latest_site_adversarial_loss = None
        self.latest_site_modulation_reg_loss = None
        self.latest_hgcn_aux_logits = None
        self.latest_prediction_logits = None
        self.latest_module1_denoise_weighted_in_causal_aux = False
        self.latest_sample_correlation_adjacency = None
        self.latest_classification_adjacency = None
        self.latest_temporal_series = None
        self.latest_temporal_series_source = None
        self.latest_module1_clean_series = None
        self.latest_module1_noisy_series = None
        self.latest_module1_clean_features = None
        self.latest_module1_noisy_features = None
        self.latest_module1_temporal_stats_features = None
        self.latest_module1_alff_descriptor = None
        self.latest_module1_band_limited_series = None
        if self.use_hpec_module4:
            self.latest_graph_path = "hgcn_hpec"
        elif self.use_hgcn_module3:
            self.latest_graph_path = "hgcn_only"
        else:
            self.latest_graph_path = "gcn_fallback"
        self.latest_aux_losses = {}

    def _temporal_stats_features(self, x_enc):
        if self.module1_temporal_stats_weight <= 0:
            return None
        diff = x_enc[:, 1:, :] - x_enc[:, :-1, :]
        stats = torch.stack(
            [
                x_enc.mean(dim=1),
                x_enc.std(dim=1, unbiased=False),
                diff.std(dim=1, unbiased=False),
                diff.abs().mean(dim=1),
            ],
            dim=-1,
        )
        return self.temporal_stats_projection(stats)

    def _module1_alff_features(self, x_enc):
        """Extract ROI-wise ALFF/fALFF-style physiological descriptors.

        The input is expected as [B, T, N].  We use rFFT along the time axis,
        keep the slow BOLD band, and project compact per-ROI descriptors to
        d_model.  This gives module 1 a physiological bias instead of letting a
        fully learnable decomposition chase high-frequency scanner noise.
        """

        if x_enc.ndim != 3:
            raise ValueError(f"Expected x_enc [B, T, N], got {tuple(x_enc.shape)}.")
        time_len = x_enc.shape[1]
        if time_len <= 1:
            x_node_time = x_enc.transpose(1, 2)
            self.latest_module1_alff_descriptor = None
            self.latest_module1_band_limited_series = x_enc
            return self.Variate_Embedding(x_node_time), "raw projected (short series)"

        centered = x_enc - x_enc.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=1)
        amplitude = spectrum.abs()
        power = amplitude.pow(2)
        freqs = torch.fft.rfftfreq(
            time_len,
            d=max(self.module1_tr, self.module1_alff_eps),
            device=x_enc.device,
        )
        band_mask = (freqs >= self.module1_alff_low_hz) & (freqs <= self.module1_alff_high_hz)
        non_dc_mask = freqs > 0
        if not torch.any(band_mask):
            band_mask = non_dc_mask
        if not torch.any(band_mask):
            band_mask = torch.ones_like(freqs, dtype=torch.bool)

        band_amp = amplitude[:, band_mask, :]
        band_power = power[:, band_mask, :]
        total_power = power[:, non_dc_mask, :] if torch.any(non_dc_mask) else power
        band_freqs = freqs[band_mask].to(x_enc.dtype)

        alff = band_amp.mean(dim=1)
        band_power_sum = band_power.sum(dim=1)
        total_power_sum = total_power.sum(dim=1).clamp_min(self.module1_alff_eps)
        falff = band_power_sum / total_power_sum
        band_std = band_amp.std(dim=1, unbiased=False)
        dominant_idx = band_amp.argmax(dim=1)
        dominant_freq = band_freqs[dominant_idx]
        slow_power_ratio = falff
        temporal_std = centered.std(dim=1, unbiased=False)

        descriptor = torch.stack(
            [
                torch.log1p(alff),
                falff,
                torch.log1p(band_std),
                dominant_freq,
                slow_power_ratio,
                temporal_std,
            ],
            dim=-1,
        )
        descriptor = torch.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)
        descriptor_features = self.alff_feature_projection(descriptor)

        filtered_spectrum = torch.zeros_like(spectrum)
        filtered_spectrum[:, band_mask, :] = spectrum[:, band_mask, :]
        band_limited = torch.fft.irfft(filtered_spectrum, n=time_len, dim=1)
        band_time_features = self.alff_band_time_projection(band_limited.transpose(1, 2))

        self.latest_module1_alff_descriptor = descriptor
        self.latest_module1_band_limited_series = band_limited
        node_features = descriptor_features + self.module1_alff_time_weight * band_time_features
        return node_features, "ALFF/fALFF physiological"

    def _apply_site_modulation(self, node_features, site_label):
        if self.site_modulator is None or site_label is None:
            self.latest_site_modulation_reg_loss = None
            return node_features
        site_label = site_label.to(node_features.device).long().reshape(-1)
        if site_label.numel() != node_features.shape[0]:
            self.latest_site_modulation_reg_loss = None
            return node_features
        site_code = self.site_embedding(site_label)
        gamma_beta = self.site_modulator(site_code)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        strength = max(self.site_modulation_strength, 0.0)
        gamma = torch.tanh(gamma).unsqueeze(1) * strength
        beta = torch.tanh(beta).unsqueeze(1) * strength
        modulated = node_features * (1.0 + gamma) + beta
        reg_loss = gamma.pow(2).mean() + beta.pow(2).mean()
        weighted_reg = self.lambda_site_modulation_reg * reg_loss
        self.latest_site_modulation_reg_loss = weighted_reg
        self.latest_aux_losses.update(
            {
                "site_modulation_reg_loss": reg_loss,
                "site_modulation_reg_weighted_loss": weighted_reg,
            }
        )
        base_aux = self.latest_aux_losses.get("causal_aux_loss")
        self.latest_aux_losses["causal_aux_loss"] = (
            weighted_reg if base_aux is None else base_aux + weighted_reg
        )
        return modulated

    def set_train_epoch(self, epoch):
        self.current_epoch = int(epoch)
        if self.causal_learner is not None and hasattr(self.causal_learner, "set_epoch"):
            self.causal_learner.set_epoch(epoch)

    def _scheduled_weight(self, base_weight, warmup_epochs):
        if self.causal_loss_schedule == "constant" or warmup_epochs <= 0:
            return float(base_weight)
        if self.causal_loss_schedule != "warmup":
            raise ValueError(
                f"Unsupported causal_loss_schedule={self.causal_loss_schedule!r}. "
                "Use 'constant' or 'warmup'."
            )
        scale = min(max((self.current_epoch + 1) / float(warmup_epochs), 0.0), 1.0)
        return float(base_weight) * scale

    def _aggregate_cycle_features(self, cycle_features):
        if not cycle_features:
            return None
        if self.causal_feature_source == "last":
            return cycle_features[-1]
        if self.causal_feature_source != "sum":
            raise ValueError(
                f"Unsupported causal_feature_source={self.causal_feature_source!r}. "
                "Use 'sum' or 'last'."
            )
        return sum(cycle_features)

    def _apply_module1_input_perturbation(self, x_enc):
        """训练期对模块 1 输入做轻量去噪增强，验证/测试保持原样。"""

        if not self.training:
            return x_enc
        if self.module1_temporal_dropout <= 0 and self.module1_roi_dropout <= 0:
            return x_enc

        x_noisy = x_enc
        if self.module1_temporal_dropout > 0:
            keep_prob = max(1.0 - self.module1_temporal_dropout, 1e-6)
            mask = torch.rand(
                x_enc.shape[0],
                x_enc.shape[1],
                1,
                device=x_enc.device,
                dtype=x_enc.dtype,
            ).lt(keep_prob)
            x_noisy = x_noisy * mask / keep_prob
        if self.module1_roi_dropout > 0:
            keep_prob = max(1.0 - self.module1_roi_dropout, 1e-6)
            mask = torch.rand(
                x_enc.shape[0],
                1,
                x_enc.shape[2],
                device=x_enc.device,
                dtype=x_enc.dtype,
            ).lt(keep_prob)
            x_noisy = x_noisy * mask / keep_prob
        return x_noisy

    def _extract_module1_features(self, x_enc):
        x_node_time = x_enc.transpose(1, 2)
        x_embed = self.Variate_Embedding(x_node_time)
        if self.use_deci_module1 and self.module1_feature_mode == "alff":
            seasonals = []
            node_features, feature_source = self._module1_alff_features(x_enc)
        elif self.use_deci_module1 and self.module1_feature_mode == "deci":
            res = x_embed
            seasonals = []
            cycle_feature_list = []
            for deci_block in self.deci_blocks:
                _, seasonal, res, _, seasonal_feature = deci_block(res, return_features=True)
                seasonals.append(seasonal)
                cycle_feature_list.append(seasonal_feature)
            node_features = self._aggregate_cycle_features(cycle_feature_list)
            feature_source = "Cycle/seasonal"
        else:
            seasonals = []
            node_features = x_embed
            feature_source = "raw projected"
        stats_features = self._temporal_stats_features(x_enc)
        if stats_features is not None:
            node_features = node_features + self.module1_temporal_stats_weight * stats_features
            self.latest_module1_temporal_stats_features = stats_features
            feature_source = f"{feature_source}+temporal_stats"
        return node_features, seasonals, feature_source

    def _module1_denoise_loss(self, noisy_features, clean_features):
        if self.module1_denoise_loss_weight <= 0 or clean_features is None or noisy_features is None:
            return None
        loss = F.mse_loss(noisy_features, clean_features.detach())
        weighted_loss = self.module1_denoise_loss_weight * loss
        self.latest_module1_denoise_weighted_in_causal_aux = False
        self.latest_aux_losses.update(
            {
                "module1_denoise_loss": loss,
                "module1_denoise_weighted_loss": weighted_loss,
            }
        )
        return weighted_loss

    def _causal_stability_loss(self, causal_output):
        if self.lambda_causal_stability <= 0 or causal_output is None:
            return None
        if causal_output.a_delta is not None:
            stability_loss = causal_output.a_delta.pow(2).mean()
        elif causal_output.a_effective.ndim == 3:
            centered = causal_output.a_effective - causal_output.a_effective.mean(dim=0, keepdim=True)
            stability_loss = centered.pow(2).mean()
        else:
            stability_loss = causal_output.a_effective.sum() * 0.0
        weighted_loss = self.lambda_causal_stability * stability_loss
        self.latest_aux_losses.update(
            {
                "causal_stability_loss": stability_loss,
                "causal_stability_weighted_loss": weighted_loss,
            }
        )
        return weighted_loss

    def _run_temporal_sem_module2(self, temporal_series):
        if (
            not self.use_causal_module2
            or self.causal_learning_target != "temporal_sem"
            or self.causal_learner is None
            or temporal_series is None
        ):
            return None

        # temporal SEM 使用节点时间序列预测未来时间片，loss 不使用真实因果图监督。
        causal_input = temporal_series.detach() if self.detach_causal_input else temporal_series
        causal_output = self.causal_learner(causal_input)
        aux_loss, aux_parts = self.causal_learner.compute_losses(causal_output)
        module1_denoise_weighted = self.latest_aux_losses.get("module1_denoise_weighted_loss")
        if module1_denoise_weighted is not None:
            aux_loss = aux_loss + module1_denoise_weighted
            self.latest_module1_denoise_weighted_in_causal_aux = True
            aux_parts["module1_denoise_loss"] = self.latest_aux_losses.get(
                "module1_denoise_loss",
                module1_denoise_weighted.detach() * 0.0,
            )
            aux_parts["module1_denoise_weighted_loss"] = module1_denoise_weighted
        self.latest_causal_output = causal_output
        self.latest_aux_losses.update(aux_parts)
        stability_loss = self._causal_stability_loss(causal_output)
        if stability_loss is not None:
            aux_loss = aux_loss + stability_loss
            self.latest_aux_losses["causal_aux_loss"] = aux_loss
        return causal_output

    def _run_causal_module2(self, node_features):
        if not self.use_causal_module2 or self.causal_learner is None or node_features is None:
            return None
        if self.causal_learning_target == "temporal_sem":
            return self._run_temporal_sem_module2(self.latest_temporal_series)

        # 默认切断 causal loss 到模块 1 的梯度；模块 2 根据当前节点特征学习 causal graph。
        causal_target = node_features.detach() if self.detach_causal_input else node_features
        causal_output = self.causal_learner(causal_target)
        recon_target = (
            causal_output.normalized_input
            if causal_output.normalized_input is not None
            else causal_target
        )
        recon_loss = F.mse_loss(causal_output.c_hat, recon_target)
        dag_loss = normalized_dag_loss(causal_output.dag_penalty, self.causal_learner.n_nodes)
        l1_loss = self.causal_learner.first_layer_l1_loss()
        sample_l1_loss = self.causal_learner.sample_graph_l1_loss(causal_output.a_delta)
        sample_deviation_loss = self.causal_learner.sample_graph_deviation_loss(
            causal_output.a_effective,
            causal_output.a_shared,
        )
        dag_weight = self._scheduled_weight(self.lambda_causal_dag, self.causal_dag_warmup_epochs)
        l1_weight = self._scheduled_weight(self.lambda_causal_l1, self.causal_l1_warmup_epochs)
        sample_l1_weight = self._scheduled_weight(
            self.lambda_sample_graph_l1,
            self.sample_graph_reg_warmup_epochs,
        )
        sample_deviation_weight = self._scheduled_weight(
            self.lambda_sample_graph_deviation,
            self.sample_graph_reg_warmup_epochs,
        )
        aux_loss = (
            self.lambda_causal_recon * recon_loss
            + dag_weight * dag_loss
            + l1_weight * l1_loss
            + sample_l1_weight * sample_l1_loss
            + sample_deviation_weight * sample_deviation_loss
        )
        module1_denoise_weighted = self.latest_aux_losses.get("module1_denoise_weighted_loss")
        if module1_denoise_weighted is not None:
            aux_loss = aux_loss + module1_denoise_weighted
            self.latest_module1_denoise_weighted_in_causal_aux = True

        # 缓存中间量，训练循环读取 loss，可视化函数读取 Cycle/C_hat/A 等诊断量。
        self.latest_cycle_features = node_features
        self.latest_causal_output = causal_output
        aux_update = {
            "causal_recon_loss": recon_loss,
            "causal_dag_loss": dag_loss,
            "causal_l1_loss": l1_loss,
            "sample_graph_l1_loss": sample_l1_loss,
            "sample_graph_deviation_loss": sample_deviation_loss,
            "causal_recon_weighted_loss": self.lambda_causal_recon * recon_loss,
            "causal_dag_weighted_loss": dag_weight * dag_loss,
            "causal_l1_weighted_loss": l1_weight * l1_loss,
            "sample_graph_l1_weighted_loss": sample_l1_weight * sample_l1_loss,
            "sample_graph_deviation_weighted_loss": sample_deviation_weight * sample_deviation_loss,
            "causal_aux_loss": aux_loss,
        }
        if module1_denoise_weighted is not None:
            aux_update["module1_denoise_loss"] = self.latest_aux_losses.get(
                "module1_denoise_loss",
                module1_denoise_weighted.detach() * 0.0,
            )
            aux_update["module1_denoise_weighted_loss"] = module1_denoise_weighted
        self.latest_aux_losses.update(aux_update)
        for key, value in causal_output.dag_metadata.items():
            if isinstance(value, (int, float)):
                self.latest_aux_losses[f"causal_meta_{key}"] = torch.as_tensor(
                    value,
                    dtype=aux_loss.dtype,
                    device=aux_loss.device,
                )
        stability_loss = self._causal_stability_loss(causal_output)
        if stability_loss is not None:
            aux_loss = aux_loss + stability_loss
            self.latest_aux_losses["causal_aux_loss"] = aux_loss
        return causal_output

    def _resolve_graph_adjacency(self, causal_output, correlation_matrix=None, module_name="graph path"):
        if causal_output is not None:
            if self.classification_graph_source in ("sample_correlation", "fc"):
                if correlation_matrix is None:
                    raise RuntimeError(
                        f"{module_name} requires correlation_matrix when "
                        "classification_graph_source='sample_correlation'."
                    )
                sample_adjacency = self._prepare_sample_correlation_adjacency(
                    correlation_matrix,
                    device=causal_output.a_effective.device,
                    dtype=causal_output.a_effective.dtype,
                )
                self.latest_sample_correlation_adjacency = sample_adjacency
                self.latest_classification_adjacency = sample_adjacency
                return sample_adjacency, True
            if self.classification_graph_source not in ("blend", "learned", "causal"):
                raise ValueError(
                    f"Unsupported classification_graph_source={self.classification_graph_source!r}. "
                    "Use 'blend', 'learned' or 'sample_correlation'."
                )
            adjacency = causal_output.a_effective
            if self.classification_graph_source in ("learned", "causal"):
                self.latest_classification_adjacency = adjacency
                return adjacency, False
            if (
                correlation_matrix is not None
                and self.module2_sample_correlation_blend > 0.0
            ):
                # 模块2启用时保留样本 FC 作为结构先验，再由学习到的因果图提供有向修正。
                sample_adjacency = self._prepare_sample_correlation_adjacency(
                    correlation_matrix,
                    device=adjacency.device,
                    dtype=adjacency.dtype,
                )
                learned_adjacency = adjacency
                if learned_adjacency.ndim == 2 and sample_adjacency.ndim == 3:
                    learned_adjacency = learned_adjacency.unsqueeze(0).expand_as(sample_adjacency)
                blend = min(max(self.module2_sample_correlation_blend, 0.0), 1.0)
                adjacency = (1.0 - blend) * learned_adjacency + blend * sample_adjacency
                self.latest_sample_correlation_adjacency = sample_adjacency
            self.latest_classification_adjacency = adjacency
            return adjacency, False
        if correlation_matrix is None:
            raise RuntimeError(
                f"{module_name} requires sample correlation adjacency when module 2 is disabled. "
                "Provide correlation_matrix or enable use_causal_module2."
            )
        self.latest_sample_correlation_adjacency = correlation_matrix
        self.latest_classification_adjacency = correlation_matrix
        return correlation_matrix, True

    def _prepare_sample_correlation_adjacency(self, correlation_matrix, device, dtype):
        adjacency = torch.nan_to_num(
            correlation_matrix.to(device=device, dtype=dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.sample_correlation_mode == "abs":
            return adjacency.abs()
        if self.sample_correlation_mode == "positive":
            return adjacency.clamp_min(0.0)
        if self.sample_correlation_mode == "raw":
            return adjacency
        raise ValueError(
            f"Unsupported sample_correlation_mode={self.sample_correlation_mode!r}. "
            "Use 'abs', 'positive' or 'raw'."
        )

    def _run_hgcn_module3(self, cycle_features, causal_output, correlation_matrix=None):
        if not self.use_hgcn_module3:
            return None
        if cycle_features is None:
            raise RuntimeError("Module 3 requires node feature.")

        adjacency, is_sample_correlation = self._resolve_graph_adjacency(
            causal_output,
            correlation_matrix=correlation_matrix,
            module_name="Module 3",
        )

        # 模块 3 使用 Cycle feature C 与图结构；模块 2 开启时图来自可微 A_learned，
        # 模块 2 关闭时图来自样本相关矩阵，退化为模块 3 自身的 HGCN 设计。
        module3_output = self.hgcn_module3(
            cycle_features,
            adjacency,
            is_sample_correlation=is_sample_correlation,
        )
        if (
            (self.fc_residual_weight > 0 or self.fc_network_residual_weight > 0)
            and correlation_matrix is not None
            and self.fc_residual_projection is not None
        ):
            # 使用样本 FC 的节点平均连接强度作为图级残差信号，补充 HGCN readout
            # 在小样本 ABIDE 上可能丢失的整体功能连接判别信息。
            fc_adjacency = self._prepare_sample_correlation_adjacency(
                correlation_matrix,
                device=cycle_features.device,
                dtype=cycle_features.dtype,
            )
            fc_strength = fc_adjacency.mean(dim=-1)
            fc_residual = self.fc_residual_projection(fc_strength)
            if (
                self.fc_network_residual_weight > 0
                and self.fc_network_residual_projection is not None
                and self.fc_network_masks.numel() > 0
                and fc_adjacency.shape[-1] == self.fc_network_masks.shape[-1]
            ):
                # 将样本图压缩为 AAL 功能网络之间的连接模式，作为比 ROI 平均强度更稳定的图级线索。
                masks = self.fc_network_masks.to(device=fc_adjacency.device, dtype=fc_adjacency.dtype)
                network_fc = torch.einsum("gn,bnm,hm->bgh", masks, fc_adjacency, masks)
                network_residual = self.fc_network_residual_projection(network_fc.flatten(start_dim=1))
                fc_residual = fc_residual + self.fc_network_residual_weight * network_residual
                self.latest_aux_losses["fc_network_residual_norm_mean"] = network_residual.norm(dim=-1).mean()
            if self.fc_residual_norm_target > 0 and self.fc_residual_norm is not None:
                fc_residual = self.fc_residual_norm(fc_residual)
                residual_norm = fc_residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                fc_residual = fc_residual * (self.fc_residual_norm_target / residual_norm)
            if self.fc_residual_gate is not None:
                gate_input = torch.cat([module3_output.z_tangent, fc_residual], dim=-1)
                fc_gate = torch.sigmoid(self.fc_residual_gate(gate_input))
                residual_update = self.fc_residual_weight * fc_gate * fc_residual
                self.latest_aux_losses["fc_residual_gate_mean"] = fc_gate.mean()
                self.latest_aux_losses["fc_residual_gate_max"] = fc_gate.max()
            else:
                residual_update = self.fc_residual_weight * fc_residual
            z_tangent = module3_output.z_tangent + residual_update
            backclip_radius = float(getattr(self.hgcn_module3.backclip, "radius", 1.0))
            if backclip_radius > 0:
                norm = z_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                z_tangent = z_tangent * torch.clamp(backclip_radius / norm, max=1.0)
            z_global = self.hgcn_module3.manifold.expmap0(z_tangent, dim=-1, project=True)
            z_global = self.hgcn_module3.manifold.projx(z_global, dim=-1)
            z_tangent = self.hgcn_module3.manifold.logmap0(z_global, dim=-1)
            module3_output.z_global = z_global
            module3_output.z_tangent = z_tangent
            self.latest_aux_losses["fc_residual_norm_mean"] = fc_residual.norm(dim=-1).mean()
            self.latest_aux_losses["fc_residual_update_norm_mean"] = residual_update.norm(dim=-1).mean()
        self.latest_module3_output = module3_output
        z_radius = torch.linalg.norm(module3_output.z_global, dim=-1)
        z_tangent_norm = torch.linalg.norm(module3_output.z_tangent, dim=-1)
        self.latest_aux_losses.update(
            {
                "z_radius_mean": z_radius.mean(),
                "z_radius_max": z_radius.max(),
                "z_tangent_norm_mean": z_tangent_norm.mean(),
            }
        )
        if self.lambda_hgcn_radius_reg > 0:
            radius_loss = F.relu(z_radius - self.hgcn_radius_target).pow(2).mean()
            weighted_loss = self.lambda_hgcn_radius_reg * radius_loss
            base_aux = self.latest_aux_losses.get("causal_aux_loss")
            self.latest_aux_losses.update(
                {
                    "hgcn_radius_loss": radius_loss,
                    "hgcn_radius_weighted_loss": weighted_loss,
                    "causal_aux_loss": weighted_loss if base_aux is None else base_aux + weighted_loss,
                }
            )
        return module3_output

    def _compute_hgcn_view_consistency(
        self,
        clean_features,
        causal_output,
        correlation_matrix,
        noisy_module3_output,
    ):
        if (
            self.lambda_hgcn_view_consistency <= 0
            or clean_features is None
            or noisy_module3_output is None
            or self.hgcn_module3 is None
        ):
            return None

        adjacency, is_sample_correlation = self._resolve_graph_adjacency(
            causal_output,
            correlation_matrix=correlation_matrix,
            module_name="Module 3 consistency",
        )
        if torch.is_tensor(adjacency):
            adjacency = adjacency.detach()

        # 干净视图只作为稳定目标，不反向更新；扰动视图继续接收主分类梯度。
        was_training = self.hgcn_module3.training
        try:
            self.hgcn_module3.eval()
            with torch.no_grad():
                clean_output = self.hgcn_module3(
                    clean_features.detach(),
                    adjacency,
                    is_sample_correlation=is_sample_correlation,
                )
        finally:
            self.hgcn_module3.train(was_training)

        noisy_z = F.normalize(noisy_module3_output.z_tangent, p=2, dim=-1)
        clean_z = F.normalize(clean_output.z_tangent.detach(), p=2, dim=-1)
        consistency_loss = (1.0 - (noisy_z * clean_z).sum(dim=-1)).mean()
        weighted_loss = self.lambda_hgcn_view_consistency * consistency_loss
        base_aux = self.latest_aux_losses.get("causal_aux_loss")
        self.latest_aux_losses.update(
            {
                "hgcn_view_consistency_loss": consistency_loss,
                "hgcn_view_consistency_weighted_loss": weighted_loss,
                "causal_aux_loss": weighted_loss if base_aux is None else base_aux + weighted_loss,
            }
        )
        return weighted_loss

    def _run_gcn_fallback(self, node_features, causal_output, correlation_matrix=None):
        if self.gcn_fallback is None:
            return None
        if node_features is None:
            raise RuntimeError("GCN fallback requires node feature.")
        adjacency, is_sample_correlation = self._resolve_graph_adjacency(
            causal_output,
            correlation_matrix=correlation_matrix,
            module_name="GCN fallback",
        )
        # 模块 3/4 关闭时退化为普通 GCN，仍复用模块 2 或样本相关矩阵提供的图结构。
        gcn_output = self.gcn_fallback(
            node_features,
            adjacency,
            is_sample_correlation=is_sample_correlation,
        )
        self.latest_gcn_fallback_output = gcn_output
        return gcn_output

    def _run_hpec_module4(self, module3_output):
        if not self.use_hpec_module4:
            return None
        if module3_output is None:
            raise RuntimeError("Module 4 requires module 3 z_global output.")

        # HPEC 默认使用 Poincare Ball 中的 z_global，而 z_tangent 继续作为调试/t-SNE 表示。
        module4_output = self.hpec_module4(module3_output.z_global)
        self.latest_module4_output = module4_output
        return module4_output

    def _compose_prediction_logits(self, module4_output, module3_output):
        """组合模块 4 energy logits 与模块 3 切空间 logits，作为最终分类输出。"""

        if module4_output is None:
            return None
        hpec_logits = -module4_output.energy_matrix
        if self.hpec_evidence_weight > 0 and hasattr(module4_output, "prototype_similarity"):
            temperature = max(self.hpec_logit_temperature, 1e-6)
            evidence_logits = torch.logsumexp(
                module4_output.prototype_similarity / temperature,
                dim=-1,
            ) * temperature
            evidence_logits = evidence_logits - evidence_logits.mean(dim=-1, keepdim=True)
            hpec_logits = hpec_logits + self.hpec_evidence_weight * evidence_logits
        if module3_output is not None and self.hgcn_classifier is not None:
            hgcn_logits = self.hgcn_classifier(module3_output.z_tangent)
            if (
                hgcn_logits.ndim == 2
                and hgcn_logits.shape[-1] == 1
                and hpec_logits.ndim == 2
                and hpec_logits.shape[-1] == 2
            ):
                # 二分类模型原本使用单 logit；与 HPEC 的两类 energy 融合时，
                # 转为 [class0_logit, class1_logit]，避免形状不一致时退回纯 HPEC。
                hgcn_logits = torch.cat([torch.zeros_like(hgcn_logits), hgcn_logits], dim=-1)
            self.latest_hgcn_aux_logits = hgcn_logits
            if hgcn_logits.shape == hpec_logits.shape:
                if self.hpec_hgcn_logit_blend > 0:
                    # HPEC 在小样本上不再接管分类头，只作为轻量 energy 校准项。
                    centered_hpec = hpec_logits - hpec_logits.mean(dim=-1, keepdim=True)
                    scale = centered_hpec.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                    gate = torch.sigmoid(self.hpec_logit_gate) * self.hpec_hgcn_logit_blend
                    logits = hgcn_logits + gate * (centered_hpec / scale)
                    self.latest_aux_losses["hpec_logit_gate"] = gate.detach()
                else:
                    logits = hgcn_logits
            else:
                logits = hpec_logits
        else:
            logits = hpec_logits
        self.latest_prediction_logits = logits
        return logits

    def get_aux_loss(self):
        aux_loss = self.latest_aux_losses.get("causal_aux_loss")
        module1_loss = self.latest_aux_losses.get("module1_denoise_weighted_loss")
        if module1_loss is not None and not self.latest_module1_denoise_weighted_in_causal_aux:
            return module1_loss if aux_loss is None else aux_loss + module1_loss
        return aux_loss

    def get_aux_losses(self):
        return self.latest_aux_losses

    def compute_site_adversarial_loss(self, site_labels):
        if (
            self.site_classifier is None
            or self.latest_module3_output is None
            or site_labels is None
            or self.lambda_site_adversarial <= 0.0
        ):
            self.latest_site_adversarial_loss = None
            return None
        site_labels = site_labels.to(self.latest_module3_output.z_tangent.device).long().reshape(-1)
        if site_labels.numel() != self.latest_module3_output.z_tangent.shape[0]:
            self.latest_site_adversarial_loss = None
            return None
        reversed_feature = gradient_reverse(self.latest_module3_output.z_tangent, self.site_grl_lambda)
        site_logits = self.site_classifier(reversed_feature)
        site_loss = F.cross_entropy(site_logits, site_labels)
        weighted_loss = self.lambda_site_adversarial * site_loss
        self.latest_site_adversarial_loss = weighted_loss
        self.latest_aux_losses.update(
            {
                "site_adversarial_loss": site_loss,
                "site_adversarial_weighted_loss": weighted_loss,
            }
        )
        return weighted_loss

    def compute_primary_loss(self, labels):
        if self.latest_module4_output is None:
            return None
        if self.training and self.latest_module3_output is not None:
            # 首个训练 batch 上按真实标签做一次 prototype warm-start；推理/测试阶段不触碰标签。
            self.hpec_module4.maybe_initialize_from_batch(
                self.latest_module3_output.z_global,
                labels,
            )
            if self.hpec_ema_update_epochs < 0 or self.current_epoch < self.hpec_ema_update_epochs:
                sinkhorn_stats = self.hpec_module4.update_prototypes_with_sinkhorn_ema(
                    self.latest_module3_output.z_global,
                    labels,
                )
                if sinkhorn_stats:
                    self.latest_aux_losses.update(sinkhorn_stats)
            self.latest_module4_output = self.hpec_module4(self.latest_module3_output.z_global)
            self._compose_prediction_logits(self.latest_module4_output, self.latest_module3_output)
        hpec_energy_loss = self.hpec_module4.loss(
            self.latest_module4_output.energy_matrix,
            labels,
        )
        label_index = labels.long().reshape(-1)
        if self.latest_prediction_logits is not None:
            final_ce_loss = self._classification_ce_loss(
                self.latest_prediction_logits,
                label_index,
            )
            self.latest_primary_loss = final_ce_loss
            if self.lambda_hpec_energy_loss > 0:
                self.latest_primary_loss = (
                    self.latest_primary_loss
                    + self.lambda_hpec_energy_loss * hpec_energy_loss
                )
            self.latest_aux_losses.update(
                {
                    "hpec_final_ce_loss": final_ce_loss,
                    "hpec_energy_loss": hpec_energy_loss,
                    "hpec_energy_weighted_loss": self.lambda_hpec_energy_loss * hpec_energy_loss,
                }
            )
        else:
            self.latest_primary_loss = hpec_energy_loss
        if self.lambda_hpec_ce_aux > 0:
            ce_loss = self._classification_ce_loss(
                -self.latest_module4_output.energy_matrix,
                label_index,
            )
            weighted_ce_loss = self.lambda_hpec_ce_aux * ce_loss
            self.latest_primary_loss = self.latest_primary_loss + weighted_ce_loss
            self.latest_aux_losses.update(
                {
                    "hpec_ce_aux_loss": ce_loss,
                    "hpec_ce_aux_weighted_loss": weighted_ce_loss,
                }
            )
        if (
            self.lambda_hgcn_cls_aux > 0
            and self.latest_module3_output is not None
            and self.hgcn_classifier is not None
        ):
            hgcn_logits = self.hgcn_classifier(self.latest_module3_output.z_tangent)
            self.latest_hgcn_aux_logits = hgcn_logits
            if hgcn_logits.shape[-1] == 1:
                hgcn_prob = torch.sigmoid(hgcn_logits.reshape(-1))
                hgcn_loss = F.binary_cross_entropy(hgcn_prob, label_index.to(hgcn_prob.dtype))
            else:
                hgcn_loss = self._classification_ce_loss(hgcn_logits, label_index)
            weighted_hgcn_loss = self.lambda_hgcn_cls_aux * hgcn_loss
            self.latest_primary_loss = self.latest_primary_loss + weighted_hgcn_loss
            self.latest_aux_losses.update(
                {
                    "hgcn_cls_aux_loss": hgcn_loss,
                    "hgcn_cls_aux_weighted_loss": weighted_hgcn_loss,
                }
            )
        if self.lambda_hgcn_supcon > 0 and self.latest_module3_output is not None:
            supcon_loss = self._supervised_contrastive_loss(
                self.latest_module3_output.z_tangent,
                label_index,
                temperature=self.hgcn_supcon_temperature,
            )
            weighted_supcon_loss = self.lambda_hgcn_supcon * supcon_loss
            self.latest_primary_loss = self.latest_primary_loss + weighted_supcon_loss
            self.latest_aux_losses.update(
                {
                    "hgcn_supcon_loss": supcon_loss,
                    "hgcn_supcon_weighted_loss": weighted_supcon_loss,
                }
            )
        self._compute_prototype_aux_loss(labels)
        return self.latest_primary_loss

    def _classification_ce_loss(self, logits, labels):
        """按配置计算分类 CE；MDD 小样本默认可用 batch 内类别均衡。"""

        labels = labels.long().reshape(-1)
        if logits.ndim != 2 or logits.shape[-1] <= 1:
            return F.cross_entropy(logits, labels, label_smoothing=self.class_label_smoothing)
        if self.class_loss_weighting in ("batch_balanced", "balanced", "inverse_freq"):
            counts = torch.bincount(labels, minlength=logits.shape[-1]).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            if torch.count_nonzero(counts).item() <= 1:
                return F.cross_entropy(logits, labels, label_smoothing=self.class_label_smoothing)
            weights = counts.sum() / (counts.clamp_min(1.0) * logits.shape[-1])
            weights = weights / weights.mean().clamp_min(1e-8)
            return F.cross_entropy(
                logits,
                labels,
                weight=weights,
                label_smoothing=self.class_label_smoothing,
            )
        if self.class_loss_weighting in ("sqrt_batch_balanced", "sqrt_balanced"):
            counts = torch.bincount(labels, minlength=logits.shape[-1]).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            if torch.count_nonzero(counts).item() <= 1:
                return F.cross_entropy(logits, labels, label_smoothing=self.class_label_smoothing)
            weights = torch.sqrt(counts.sum() / (counts.clamp_min(1.0) * logits.shape[-1]))
            weights = weights / weights.mean().clamp_min(1e-8)
            return F.cross_entropy(
                logits,
                labels,
                weight=weights,
                label_smoothing=self.class_label_smoothing,
            )
        return F.cross_entropy(logits, labels, label_smoothing=self.class_label_smoothing)

    def _supervised_contrastive_loss(self, features, labels, temperature=0.2):
        features = F.normalize(features, p=2, dim=-1)
        labels = labels.long().reshape(-1)
        batch_size = features.shape[0]
        if batch_size <= 1:
            return features.sum() * 0.0
        logits = torch.matmul(features, features.T) / max(float(temperature), 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        positive_mask = labels[:, None].eq(labels[None, :]) & (~self_mask)
        if not torch.any(positive_mask):
            return features.sum() * 0.0
        logits_masked = logits.masked_fill(self_mask, -1e9)
        log_prob = logits - torch.logsumexp(logits_masked, dim=1, keepdim=True)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        loss_per_sample = -(log_prob * positive_mask.to(log_prob.dtype)).sum(dim=1) / positive_count
        valid = positive_mask.any(dim=1)
        return loss_per_sample[valid].mean()

    def _compute_prototype_aux_loss(self, labels):
        if self.latest_module3_output is None or self.hpec_module4 is None:
            self.latest_prototype_aux_loss = None
            return None

        # 多 prototype loss 在 logmap0(z_global) 的切空间中计算，
        # 作为 HPEC energy loss 之外的可选结构约束。
        losses = self.hpec_module4.prototype_losses(
            self.latest_module3_output.z_global,
            labels,
        )
        mle_loss = losses["hpec_mle_loss"]
        pcl_loss = losses["hpec_pcl_loss"]
        pal_loss = losses["hpec_pal_loss"]
        prototype_aux_loss = (
            self.lambda_hpec_mle * mle_loss
            + self.lambda_hpec_pcl * pcl_loss
            + self.lambda_hpec_pal * pal_loss
        )
        hpec_radius_loss = losses.get("hpec_radius_loss")
        hpec_diversity_loss = losses.get("hpec_diversity_loss")
        hpec_hsic_loss = losses.get("hpec_hsic_loss")
        hpec_intra_orthogonal_loss = losses.get("hpec_intra_orthogonal_loss")
        hpec_inter_margin_loss = losses.get("hpec_inter_margin_loss")
        hpec_class_center_margin_loss = losses.get("hpec_class_center_margin_loss")
        hpec_anchor_loss = losses.get("hpec_anchor_loss")
        if hpec_radius_loss is not None:
            prototype_aux_loss = prototype_aux_loss + self.lambda_hpec_radius_reg * hpec_radius_loss
        if hpec_diversity_loss is not None:
            prototype_aux_loss = prototype_aux_loss + self.lambda_hpec_diversity * hpec_diversity_loss
        if hpec_hsic_loss is not None:
            prototype_aux_loss = prototype_aux_loss + self.lambda_hpec_hsic * hpec_hsic_loss
        if hpec_intra_orthogonal_loss is not None:
            prototype_aux_loss = prototype_aux_loss + self.lambda_hpec_intra_orthogonal * hpec_intra_orthogonal_loss
        if hpec_inter_margin_loss is not None:
            prototype_aux_loss = prototype_aux_loss + self.lambda_hpec_inter_margin * hpec_inter_margin_loss
        if hpec_class_center_margin_loss is not None:
            prototype_aux_loss = (
                prototype_aux_loss
                + self.lambda_hpec_class_center_margin * hpec_class_center_margin_loss
            )
        if hpec_anchor_loss is not None:
            prototype_aux_loss = prototype_aux_loss + self.lambda_hpec_anchor * hpec_anchor_loss
        self.latest_prototype_aux_loss = prototype_aux_loss
        aux_update = {
            "hpec_mle_loss": mle_loss,
            "hpec_pcl_loss": pcl_loss,
            "hpec_pal_loss": pal_loss,
            "hpec_prototype_aux_loss": prototype_aux_loss,
        }
        if hpec_radius_loss is not None:
            aux_update["hpec_radius_loss"] = hpec_radius_loss
            aux_update["hpec_radius_weighted_loss"] = self.lambda_hpec_radius_reg * hpec_radius_loss
        if hpec_diversity_loss is not None:
            aux_update["hpec_diversity_loss"] = hpec_diversity_loss
            aux_update["hpec_diversity_weighted_loss"] = self.lambda_hpec_diversity * hpec_diversity_loss
        if hpec_hsic_loss is not None:
            aux_update["hpec_hsic_loss"] = hpec_hsic_loss
            aux_update["hpec_hsic_weighted_loss"] = self.lambda_hpec_hsic * hpec_hsic_loss
        if hpec_intra_orthogonal_loss is not None:
            aux_update["hpec_intra_orthogonal_loss"] = hpec_intra_orthogonal_loss
            aux_update["hpec_intra_orthogonal_weighted_loss"] = (
                self.lambda_hpec_intra_orthogonal * hpec_intra_orthogonal_loss
            )
        if hpec_inter_margin_loss is not None:
            aux_update["hpec_inter_margin_loss"] = hpec_inter_margin_loss
            aux_update["hpec_inter_margin_weighted_loss"] = (
                self.lambda_hpec_inter_margin * hpec_inter_margin_loss
            )
        if hpec_class_center_margin_loss is not None:
            aux_update["hpec_class_center_margin_loss"] = hpec_class_center_margin_loss
            aux_update["hpec_class_center_margin_weighted_loss"] = (
                self.lambda_hpec_class_center_margin * hpec_class_center_margin_loss
            )
        if hpec_anchor_loss is not None:
            aux_update["hpec_anchor_loss"] = hpec_anchor_loss
            aux_update["hpec_anchor_weighted_loss"] = self.lambda_hpec_anchor * hpec_anchor_loss
        if "prototype_cos_abs_mean" in losses:
            aux_update["prototype_cos_abs_mean"] = losses["prototype_cos_abs_mean"]
        if "prototype_cos_abs_max" in losses:
            aux_update["prototype_cos_abs_max"] = losses["prototype_cos_abs_max"]
        if "prototype_same_class_cos_max" in losses:
            aux_update["prototype_same_class_cos_max"] = losses["prototype_same_class_cos_max"]
        if "hpec_sinkhorn_assignment_entropy" in losses:
            aux_update["hpec_sinkhorn_assignment_entropy"] = losses["hpec_sinkhorn_assignment_entropy"]
        if "hpec_sinkhorn_usage_min" in losses:
            aux_update["hpec_sinkhorn_usage_min"] = losses["hpec_sinkhorn_usage_min"]
        if "hpec_sinkhorn_usage_max" in losses:
            aux_update["hpec_sinkhorn_usage_max"] = losses["hpec_sinkhorn_usage_max"]
        self.latest_aux_losses.update(aux_update)
        return prototype_aux_loss

    def get_primary_loss(self):
        return self.latest_primary_loss

    def get_latest_prediction(self):
        if self.latest_prediction_logits is not None:
            if self.latest_prediction_logits.ndim > 1 and self.latest_prediction_logits.shape[-1] == 1:
                return (torch.sigmoid(self.latest_prediction_logits.reshape(-1)) > 0.5).long()
            return torch.argmax(self.latest_prediction_logits, dim=-1)
        if self.latest_module4_output is None:
            return None
        return self.latest_module4_output.prediction

    def get_latest_probabilities(self):
        if self.latest_prediction_logits is not None:
            if self.latest_prediction_logits.ndim > 1 and self.latest_prediction_logits.shape[-1] == 1:
                return torch.sigmoid(self.latest_prediction_logits)
            return torch.softmax(self.latest_prediction_logits, dim=-1)
        if self.latest_module4_output is None:
            return None
        return self.latest_module4_output.probability

    def visualize_causal_intermediates(
        self,
        save_path=None,
        output_dir=None,
        prefix="s_deci_causal",
        batch_index=0,
        threshold=None,
        labels=None,
        predictions=None,
    ):
        if self.latest_cycle_features is None:
            raise RuntimeError("No causal intermediates are available. Run forward() first.")

        items = [
            self.latest_node_features if self.latest_node_features is not None else self.latest_cycle_features,
        ]
        titles = [
            f"{self.latest_node_feature_source or 'node'} feature",
        ]
        if self.latest_temporal_series is not None:
            items.append(self.latest_temporal_series)
            titles.append(f"{self.latest_temporal_series_source or 'temporal'} series")
        if self.latest_module1_clean_features is not None:
            items.append(self.latest_module1_clean_features)
            titles.append("Module1 clean node features")
        if self.latest_module1_noisy_features is not None:
            items.append(self.latest_module1_noisy_features)
            titles.append("Module1 denoised/augmented node features")
        if self.latest_module1_temporal_stats_features is not None:
            items.append(self.latest_module1_temporal_stats_features)
            titles.append("Module1 temporal stats residual features")
        if self.latest_module1_alff_descriptor is not None:
            items.append(self.latest_module1_alff_descriptor)
            titles.append("Module1 ALFF/fALFF descriptors")
        if self.latest_module1_band_limited_series is not None:
            items.append(self.latest_module1_band_limited_series)
            titles.append("Module1 band-limited low-frequency series")
        if self.latest_causal_output is not None:
            threshold = self.causal_threshold if threshold is None else threshold
            causal_output = self.latest_causal_output
            a_effective = causal_output.a_effective
            adjacency_binary = threshold_adjacency(a_effective, threshold=threshold)
            adjacency_direction_delta = a_effective - a_effective.transpose(-1, -2)
            if self.causal_learning_target == "temporal_sem":
                pred_error = causal_output.x_hat - causal_output.target
                a_lag_mean = causal_output.a_lag.mean(dim=0)
                items.extend(
                    [
                        causal_output.target,
                        causal_output.x_hat,
                        pred_error,
                        causal_output.a0,
                        causal_output.a_lag,
                        a_lag_mean,
                        causal_output.a_shared,
                        a_effective,
                        adjacency_binary,
                        adjacency_direction_delta,
                    ]
                )
                titles.extend(
                    [
                        "Temporal NTS-NOTEARS target",
                        "Temporal NTS-NOTEARS X_hat",
                        "Temporal NTS-NOTEARS prediction error",
                        "Temporal NTS-NOTEARS A0 residual graph",
                        "Temporal NTS-NOTEARS A_lag per lag",
                        "Temporal NTS-NOTEARS A_lag_mean primary graph",
                        "Temporal NTS-NOTEARS A_shared primary graph",
                        "A_effective shared+delta",
                        "A_effective_binary",
                        "A_effective - A_effective.T",
                    ]
                )
            else:
                recon_target = (
                    causal_output.normalized_input
                    if causal_output.normalized_input is not None
                    else self.latest_node_features
                )
                reconstruction_error = causal_output.c_hat - recon_target
                items.extend(
                    [
                        causal_output.normalized_input if causal_output.normalized_input is not None else recon_target,
                        causal_output.c_hat,
                        reconstruction_error,
                        causal_output.a_shared,
                        a_effective,
                        adjacency_binary,
                        adjacency_direction_delta,
                    ]
                )
                titles.extend(
                    [
                        "Module2 normalized input C",
                        "C_hat reconstruction",
                        "C_hat - node feature",
                        "A_shared learned graph",
                        "A_effective shared+delta",
                        "A_effective_binary",
                        "A_effective - A_effective.T",
                    ]
                )
            if causal_output.a_delta is not None:
                items.append(causal_output.a_delta)
                titles.append("A_delta sample residual graph")
            if self.latest_classification_adjacency is not None:
                items.append(self.latest_classification_adjacency)
                titles.append("A_cls adjacency used by classifier")
        elif self.latest_sample_correlation_adjacency is not None:
            items.append(self.latest_sample_correlation_adjacency)
            titles.append("Sample correlation adjacency")

        if self.latest_module3_output is not None:
            module3_output = self.latest_module3_output
            items.extend(
                [
                    module3_output.c_clipped,
                    module3_output.h0,
                    module3_output.h_gcn,
                    module3_output.z_global,
                    module3_output.z_tangent,
                    module3_output.node_attention,
                    module3_output.network_summary,
                    module3_output.network_attention.reshape(1, -1),
                    module3_output.normalized_adjacency,
                ]
            )
            titles.extend(
                [
                    "Module3 C_clipped",
                    "Module3 H0 Poincare",
                    "Module3 H_gcn",
                    "Module3 z_global",
                    "Module3 logmap0(z_global)",
                    "Module3 node attention weights",
                    "Module3 MDD-prior network summary",
                    "Module3 MDD-prior network attention",
                    "Module3 normalized A",
                ]
            )

        if self.latest_gcn_fallback_output is not None:
            gcn_output = self.latest_gcn_fallback_output
            items.extend(
                [
                    gcn_output.normalized_adjacency,
                    gcn_output.h_gcn,
                    gcn_output.readout,
                    gcn_output.logits,
                ]
            )
            titles.extend(
                [
                    "GCN fallback normalized adjacency",
                    "GCN fallback hidden",
                    "GCN fallback readout",
                    "GCN fallback logits",
                ]
            )

        if self.latest_module4_output is not None:
            module4_output = self.latest_module4_output
            items.extend(
                [
                    module4_output.prototypes,
                    module4_output.angle_matrix,
                    module4_output.aperture,
                    module4_output.energy_per_proto,
                    module4_output.prototype_similarity,
                    module4_output.energy_matrix,
                    module4_output.probability,
                    module4_output.prototype_assignment.reshape(-1, 1),
                    module4_output.prediction.reshape(-1, 1),
                ]
            )
            titles.extend(
                [
                    f"Module4 HPEC prototypes CxKxD {tuple(module4_output.prototypes.shape)}",
                    "Module4 angle matrix",
                    "Module4 psi aperture",
                    "Module4 prototype-level energy",
                    "Module4 prototype similarity",
                    "Module4 energy matrix",
                    "Module4 softmax(-energy)",
                    "Module4 prototype assignment",
                    "Module4 predicted labels",
                ]
            )

        if labels is not None:
            # 标签转成 [B, 1] 竖列显示，便于和 z_global 的 batch 维逐行对比。
            label_tensor = labels.detach().cpu() if hasattr(labels, "detach") else torch.as_tensor(labels)
            items.append(label_tensor.reshape(-1, 1))
            titles.append("Ground truth labels (not model input)")

        if predictions is not None:
            # 预测标签来自刚才的无标签 forward，仅用于和真实标签做可视化对照。
            pred_tensor = (
                predictions.detach().cpu()
                if hasattr(predictions, "detach")
                else torch.as_tensor(predictions)
            )
            items.append(pred_tensor.reshape(-1, 1))
            titles.append("Predicted labels")

        if save_path is None:
            base_dir = Path(output_dir or self.causal_vis_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            save_path = base_dir / f"{prefix}_{self._forward_count:06d}.png"

        return visualize_tensors(
            *items,
            titles=titles,
            save_path=save_path,
            show=False,
            batch_index=batch_index,
        )

    def forward(self, x_enc, correlation_matrix=None, site_label=None):
        self._reset_causal_cache()
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # temporal SEM 优先使用模块 1 训练期增强后的时序；分类路径使用同一视图提取节点特征。
        clean_temporal_series = x_enc
        augmented_temporal_series = self._apply_module1_input_perturbation(x_enc)
        self.latest_module1_clean_series = clean_temporal_series
        self.latest_module1_noisy_series = augmented_temporal_series
        self.latest_temporal_series = augmented_temporal_series
        self.latest_temporal_series_source = (
            "module1 augmented temporal series"
            if augmented_temporal_series is not clean_temporal_series
            else "normalized raw temporal series"
        )

        clean_features = None
        if self.training and (
            self.module1_denoise_loss_weight > 0
            or self.lambda_hgcn_view_consistency > 0
        ):
            with torch.no_grad():
                clean_features, _, _ = self._extract_module1_features(clean_temporal_series)
            self.latest_module1_clean_features = clean_features

        node_features, seasonals, feature_source = self._extract_module1_features(augmented_temporal_series)
        node_features = self._apply_site_modulation(node_features, site_label)
        self.latest_node_feature_source = feature_source
        self.latest_module1_noisy_features = node_features
        if clean_features is None:
            self.latest_module1_clean_features = node_features

        self.latest_node_features = node_features
        self.latest_cycle_features = node_features
        self._module1_denoise_loss(node_features, clean_features)
        causal_output = self._run_causal_module2(node_features)
        module3_output = self._run_hgcn_module3(
            node_features,
            causal_output,
            correlation_matrix=correlation_matrix,
        )
        if self.training:
            self._compute_hgcn_view_consistency(
                clean_features,
                causal_output,
                correlation_matrix,
                module3_output,
            )
        module4_output = self._run_hpec_module4(module3_output)
        gcn_output = self._run_gcn_fallback(
            node_features,
            causal_output,
            correlation_matrix=correlation_matrix,
        )

        self._forward_count += 1

        if module4_output is not None:
            y_hat = self._compose_prediction_logits(module4_output, module3_output)
        elif module3_output is not None:
            # 当前阶段不接模块 4，直接用 z_global 的切空间表示作为分类依据。
            y_hat = self.hgcn_classifier(module3_output.z_tangent)
            self.latest_prediction_logits = y_hat
        elif gcn_output is not None:
            y_hat = gcn_output.logits
            self.latest_prediction_logits = y_hat
        else:
            y_hat = sum(seasonals)
        if self.out_dim == 1 and module4_output is None:
            y_hat = torch.sigmoid(y_hat)
        return y_hat
