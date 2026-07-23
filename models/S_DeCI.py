from dataclasses import replace
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
from layers.attention_temporal_causal_learner import AttentionGuidedTemporalCausalLearner
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

        # 启用模块3/4时统一使用 C 维 logits，避免 binary 单 logit 与 HPEC 两类 energy
        # 在 residual/蒸馏/指标计算中反复转换导致语义漂移。
        self.out_dim = int(configs.classes) if bool(getattr(configs, "use_hyperbolic_modules34", 1)) else (
            1 if configs.classes == 2 else configs.classes
        )
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
        self.causal_graph_method = str(getattr(configs, "causal_graph_method", "nts_notears")).lower()
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
        self.class_loss_weighting = str(
            getattr(configs, "class_loss_weighting", "none")
        ).lower()
        self.class_label_smoothing = float(getattr(configs, "class_label_smoothing", 0.0))
        self.class_logit_adjust_tau = float(getattr(configs, "class_logit_adjust_tau", 1.0))
        self.class_prior_alignment_weight = float(
            getattr(configs, "class_prior_alignment_weight", 0.0)
        )
        self.gcn_graph_branch_ce_loss_weight = float(
            getattr(configs, "gcn_graph_branch_ce_loss_weight", 0.0)
        )
        self.gcn_fc_branch_ce_loss_weight = float(
            getattr(configs, "gcn_fc_branch_ce_loss_weight", 0.0)
        )
        self.use_final_logit_calibration = bool(getattr(configs, "use_final_logit_calibration", 0))
        self.hpec_hgcn_logit_blend = float(getattr(configs, "hpec_hgcn_logit_blend", 0.1))
        self.hpec_classification_mode = str(
            getattr(configs, "hpec_classification_mode", "energy_primary")
        ).lower()
        if self.hpec_classification_mode not in (
            "energy_primary",
            "prototype_primary",
            "energy_prototype_residual",
            "distance_prototype",
            "tangent_primary",
            "tangent_prototype",
            "feature_fusion",
            "energy_calibrated",
        ):
            raise ValueError(
                "hpec_classification_mode must be 'energy_primary', 'prototype_primary', "
                "'energy_prototype_residual', 'distance_prototype', 'tangent_primary', 'tangent_prototype', "
                "'feature_fusion' or 'energy_calibrated'."
            )
        self.hpec_network_energy_weight = float(getattr(configs, "hpec_network_energy_weight", 0.0))
        self.hpec_network_energy_mode = str(
            getattr(configs, "hpec_network_energy_mode", "attention_mean")
        ).lower()
        if self.hpec_network_energy_mode not in ("attention_mean", "class_softmin"):
            raise ValueError(
                "hpec_network_energy_mode must be 'attention_mean' or 'class_softmin'."
            )
        self.hpec_network_energy_temperature = float(
            getattr(configs, "hpec_network_energy_temperature", 0.5)
        )
        self.hpec_network_energy_prior_weight = float(
            getattr(configs, "hpec_network_energy_prior_weight", 1.0)
        )
        self.hpec_network_energy_normalize = int(
            getattr(configs, "hpec_network_energy_normalize", 1)
        )
        self.hpec_network_selector_sharpness = float(
            getattr(configs, "hpec_network_selector_sharpness", 1.0)
        )
        self.hpec_evidence_weight = float(getattr(configs, "hpec_evidence_weight", 0.5))
        self.hpec_avoid_busemann_double_count = bool(
            getattr(configs, "hpec_avoid_busemann_double_count", 1)
        )
        self.hpec_logit_temperature = float(getattr(configs, "hpec_logit_temperature", 0.5))
        self.hpec_prototype_energy_blend = float(getattr(configs, "hpec_prototype_energy_blend", 0.0))
        self.hpec_prototype_residual_weight = float(
            getattr(configs, "hpec_prototype_residual_weight", 0.2)
        )
        self.hpec_prototype_logit_mode = str(
            getattr(configs, "hpec_prototype_logit_mode", "normalized")
        ).lower()
        if self.hpec_prototype_logit_mode not in ("normalized", "margin_preserving"):
            raise ValueError(
                "hpec_prototype_logit_mode must be 'normalized' or 'margin_preserving'."
            )
        self.hpec_prototype_logit_scale = float(getattr(configs, "hpec_prototype_logit_scale", 1.0))
        self.hpec_residual_calibration = str(
            getattr(configs, "hpec_residual_calibration", "none")
        ).lower()
        if self.hpec_residual_calibration not in (
            "none",
            "batch_margin",
            "tanh_margin",
            "running_batch_margin",
            "hybrid_batch_running_margin",
            "train_class_margin",
        ):
            raise ValueError(
                "hpec_residual_calibration must be 'none', 'batch_margin', "
                "'tanh_margin', 'running_batch_margin', 'hybrid_batch_running_margin' "
                "or 'train_class_margin'."
            )
        self.hpec_residual_calibration_scale = float(
            getattr(configs, "hpec_residual_calibration_scale", 1.0)
        )
        self.hpec_residual_calibration_momentum = float(
            getattr(configs, "hpec_residual_calibration_momentum", 0.05)
        )
        self.hpec_residual_calibration_batch_weight = float(
            getattr(configs, "hpec_residual_calibration_batch_weight", 0.5)
        )
        # running_batch_margin 用训练期 HPEC margin 统计做测试期校准，避免测试 batch 临时统计影响预测。
        self.register_buffer("hpec_residual_running_margin_mean", torch.zeros(1), persistent=True)
        self.register_buffer("hpec_residual_running_margin_std", torch.ones(1), persistent=True)
        self.register_buffer("hpec_residual_running_margin_ready", torch.zeros(1), persistent=True)
        self.register_buffer(
            "hpec_residual_class_margin_mean",
            torch.zeros(max(int(getattr(configs, "classes", 2)), 2)),
            persistent=True,
        )
        self.register_buffer(
            "hpec_residual_class_margin_ready",
            torch.zeros(max(int(getattr(configs, "classes", 2)), 2)),
            persistent=True,
        )
        self.hpec_gate_init = float(getattr(configs, "hpec_gate_init", -2.5))
        self.hpec_energy_loss_weight = float(getattr(configs, "hpec_energy_loss_weight", 0.0))
        self.hpec_energy_ce_margin = max(
            float(getattr(configs, "hpec_energy_ce_margin", 0.0)), 0.0
        )
        self.hpec_prototype_ce_loss_weight = float(
            getattr(configs, "hpec_prototype_ce_loss_weight", 0.0)
        )
        self.hpec_teacher_distill_weight = float(getattr(configs, "hpec_teacher_distill_weight", 0.0))
        self.hpec_teacher_distill_temperature = float(
            getattr(configs, "hpec_teacher_distill_temperature", 2.0)
        )
        self.hpec_teacher_distill_mode = str(
            getattr(configs, "hpec_teacher_distill_mode", "kl")
        ).lower()
        if self.hpec_teacher_distill_mode not in ("kl", "centered_kl", "margin_mse"):
            raise ValueError(
                "hpec_teacher_distill_mode must be 'kl', 'centered_kl' or 'margin_mse'."
            )
        self.hpec_teacher_detach = bool(getattr(configs, "hpec_teacher_detach", 1))
        self.hpec_z_radius_loss_weight = float(getattr(configs, "hpec_z_radius_loss_weight", 0.0))
        self.hpec_z_radius_target = float(
            getattr(configs, "hpec_z_radius_target", getattr(configs, "hpec_prototype_radius", 0.3))
        )
        self.hpec_z_min_radius = float(getattr(configs, "hpec_z_min_radius", 0.0))
        self.hpec_input_radius_min = float(getattr(configs, "hpec_input_radius_min", 0.0))
        self.hpec_input_radius_max = float(getattr(configs, "hpec_input_radius_max", 0.0))
        self.hpec_input_tangent_noise_std = max(
            float(getattr(configs, "hpec_input_tangent_noise_std", 0.0)),
            0.0,
        )
        self.hpec_prototype_min_radius_loss_weight = float(
            getattr(configs, "hpec_prototype_min_radius_loss_weight", 0.0)
        )
        self.hpec_prototype_min_radius = float(
            getattr(configs, "hpec_prototype_min_radius", getattr(configs, "hpec_prototype_radius", 0.3) * 0.6)
        )
        self.hpec_prototype_separation_loss_weight = float(
            getattr(configs, "hpec_prototype_separation_loss_weight", 0.0)
        )
        self.hpec_prototype_separation_max_cos = float(
            getattr(configs, "hpec_prototype_separation_max_cos", 0.35)
        )
        self.module34_supcon_loss_weight = float(getattr(configs, "module34_supcon_loss_weight", 0.0))
        self.module34_supcon_temperature = float(getattr(configs, "module34_supcon_temperature", 0.2))
        self.module34_center_loss_weight = float(getattr(configs, "module34_center_loss_weight", 0.0))
        self.module34_center_margin = float(getattr(configs, "module34_center_margin", 0.5))
        self.module34_center_intra_weight = float(getattr(configs, "module34_center_intra_weight", 1.0))
        self.module34_center_inter_weight = float(getattr(configs, "module34_center_inter_weight", 1.0))
        self.module34_branch_ce_loss_weight = float(
            getattr(configs, "module34_branch_ce_loss_weight", 0.0)
        )
        self.module34_branch_ce_decay_epochs = int(
            getattr(configs, "module34_branch_ce_decay_epochs", 0)
        )
        self.module34_branch_ce_min_ratio = float(
            getattr(configs, "module34_branch_ce_min_ratio", 1.0)
        )
        self.use_causal_role_readout = bool(
            getattr(configs, "use_causal_role_readout", 0)
        )
        self.hpec_causal_role_energy_weight = min(
            max(float(getattr(configs, "hpec_causal_role_energy_weight", 0.0)), 0.0),
            1.0,
        )
        self.hyperbolic_logit_residual_weight = float(getattr(configs, "hyperbolic_logit_residual_weight", 0.0))
        self.hyperbolic_residual_fusion_mode = str(
            getattr(configs, "hyperbolic_residual_fusion_mode", "residual")
        ).lower()
        if self.hyperbolic_residual_fusion_mode not in ("residual", "logit_blend", "binary_margin", "dual_consensus", "dual_margin_consensus"):
            raise ValueError(
                "hyperbolic_residual_fusion_mode must be 'residual', 'logit_blend', "
                "'binary_margin', 'dual_consensus' or 'dual_margin_consensus'."
            )
        self.hyperbolic_residual_source = str(getattr(configs, "hyperbolic_residual_source", "prototype")).lower()
        if self.hyperbolic_residual_source not in ("prototype", "tangent"):
            raise ValueError("hyperbolic_residual_source must be 'prototype' or 'tangent'.")
        self.hyperbolic_residual_norm = str(getattr(configs, "hyperbolic_residual_norm", "sample")).lower()
        if self.hyperbolic_residual_norm not in ("sample", "temperature", "none"):
            raise ValueError("hyperbolic_residual_norm must be 'sample', 'temperature' or 'none'.")
        self.hyperbolic_residual_temperature = float(getattr(configs, "hyperbolic_residual_temperature", 1.0))
        self.hyperbolic_residual_margin_gain = float(getattr(configs, "hyperbolic_residual_margin_gain", 0.0))
        self.hyperbolic_residual_margin_max_scale = float(
            getattr(configs, "hyperbolic_residual_margin_max_scale", 2.0)
        )
        self.use_hyperbolic_residual_bias = bool(getattr(configs, "use_hyperbolic_residual_bias", 0))
        self.use_hyperbolic_residual_gate = bool(getattr(configs, "use_hyperbolic_residual_gate", 0))
        self.hyperbolic_residual_gate_mode = str(
            getattr(configs, "hyperbolic_residual_gate_mode", "margin")
        ).lower()
        if self.hyperbolic_residual_gate_mode not in ("margin", "agreement", "consensus"):
            raise ValueError("hyperbolic_residual_gate_mode must be 'margin', 'agreement' or 'consensus'.")
        self.hyperbolic_residual_gate_min = float(getattr(configs, "hyperbolic_residual_gate_min", 0.20))
        self.hyperbolic_residual_gate_max = float(getattr(configs, "hyperbolic_residual_gate_max", 0.80))
        if self.hyperbolic_residual_gate_max < self.hyperbolic_residual_gate_min:
            self.hyperbolic_residual_gate_max = self.hyperbolic_residual_gate_min
        self.hyperbolic_residual_gate_gain = float(getattr(configs, "hyperbolic_residual_gate_gain", 2.0))
        self.hyperbolic_residual_gate_bias = float(getattr(configs, "hyperbolic_residual_gate_bias", 0.0))
        self.keep_gcn_fallback_with_hyperbolic = bool(getattr(configs, "keep_gcn_fallback_with_hyperbolic", 0))
        self.module34_film_weight = float(getattr(configs, "module34_film_weight", 0.0))
        self.module34_film_max_scale = float(getattr(configs, "module34_film_max_scale", 0.25))
        self.module34_film_shift_norm = float(getattr(configs, "module34_film_shift_norm", 0.20))
        self.use_hgcn_radial_calibration = bool(getattr(configs, "use_hgcn_radial_calibration", 0))
        self.hgcn_radial_min = float(getattr(configs, "hgcn_radial_min", 0.25))
        self.hgcn_radial_max = float(getattr(configs, "hgcn_radial_max", 0.75))
        self.fc_residual_weight = float(getattr(configs, "fc_residual_weight", 0.0))
        self.fc_network_residual_weight = float(getattr(configs, "fc_network_residual_weight", 0.0))
        self.fc_residual_norm_target = float(getattr(configs, "fc_residual_norm_target", 1.0))
        self.use_fc_residual_gate = bool(getattr(configs, "use_fc_residual_gate", 0))
        self.fc_residual_gate_init = float(getattr(configs, "fc_residual_gate_init", -2.0))
        self.current_epoch = 0
        self.hpec_use_sinkhorn_ema = bool(getattr(configs, "hpec_use_sinkhorn_ema", 1))
        default_prototype_update_mode = "reliable_tp_ema" if self.use_hpec_module4 else "none"
        self.hpec_prototype_update_mode = str(
            getattr(configs, "hpec_prototype_update_mode", default_prototype_update_mode)
        ).lower()
        if self.hpec_prototype_update_mode not in (
            "reliable_tp_ema",
            "sinkhorn_ema",
            "none",
        ):
            raise ValueError(
                "hpec_prototype_update_mode must be 'reliable_tp_ema', "
                "'sinkhorn_ema' or 'none'."
            )
        self.hpec_reliable_confidence_threshold = float(
            getattr(configs, "hpec_reliable_confidence_threshold", 0.70)
        )
        self.hpec_reliable_min_samples = max(
            int(getattr(configs, "hpec_reliable_min_samples", 2)), 1
        )
        self.hpec_sinkhorn_epsilon = float(getattr(configs, "hpec_sinkhorn_epsilon", 0.05))
        self.hpec_sinkhorn_iters = int(getattr(configs, "hpec_sinkhorn_iters", 3))
        self.hpec_ema_alpha = float(getattr(configs, "hpec_ema_alpha", 0.995))
        self.hpec_ema_anchor_weight = float(getattr(configs, "hpec_ema_anchor_weight", 0.1))
        self.hpec_ema_update_epochs = int(getattr(configs, "hpec_ema_update_epochs", 5))
        self.hpec_ema_start_epoch = max(int(getattr(configs, "hpec_ema_start_epoch", 0)), 0)
        # BrainCL 衍生训练期开关：默认关闭，避免未验证的新分支改变既有最佳路线。
        self.use_multi_hop_causal_encoding = bool(
            getattr(configs, "use_multi_hop_causal_encoding", 0)
        )
        self.causal_reachability_hops = max(
            int(getattr(configs, "causal_reachability_hops", 2)), 1
        )
        self.causal_reachability_scale = max(
            float(getattr(configs, "causal_reachability_scale", 0.25)), 0.0
        )
        if self.use_multi_hop_causal_encoding and not self.use_hgcn_module3:
            raise ValueError(
                "因果显著性互补学习和多阶因果编码要求启用模块 3 HGCN。"
            )
        if self.hpec_prototype_update_mode == "reliable_tp_ema" and not self.use_hpec_module4:
            raise ValueError("reliable_tp_ema 要求启用模块 4 HPEC。")
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
        self.module2_graph_residual_alpha = float(
            getattr(configs, "module2_graph_residual_alpha", 0.10)
        )
        self.detach_module2_graph_for_classification = bool(
            getattr(configs, "detach_module2_graph_for_classification", 0)
        )
        self.freeze_causal_after_epoch = int(getattr(configs, "freeze_causal_after_epoch", -1))
        self.classification_graph_source = str(
            getattr(configs, "classification_graph_source", "blend")
        ).lower()
        self.sample_correlation_mode = getattr(configs, "sample_correlation_mode", "abs")
        self._forward_count = 0

        if self.use_causal_module2 and self.causal_learning_target == "temporal_sem":
            common_temporal_kwargs = dict(
                n_nodes=configs.channel,
                lag_order=int(getattr(configs, "temporal_lag_order", 3)),
                init_logit=float(getattr(configs, "temporal_causal_init_logit", -4.0)),
                input_norm=getattr(configs, "temporal_sem_input_norm", "time_zscore"),
                lambda_pred=float(getattr(configs, "lambda_temporal_pred", 1.0)),
                lambda_dag=float(getattr(configs, "lambda_causal_dag", 0.001)),
                lambda_sparse=float(
                    getattr(configs, "lambda_temporal_sparse", getattr(configs, "lambda_causal_l1", 0.0005))
                ),
                lambda_smooth=float(getattr(configs, "lambda_temporal_smooth", 0.0001)),
                dagma_warmup_epochs=int(getattr(configs, "temporal_dagma_warmup_epochs", 5)),
                dagma_barrier_epochs=int(getattr(configs, "temporal_dagma_barrier_epochs", 20)),
                reg_warmup_epochs=int(getattr(configs, "temporal_reg_warmup_epochs", 5)),
                decoder_activation=getattr(configs, "temporal_decoder_activation", "identity"),
            )
            if self.causal_graph_method == "attn_nts_notears":
                # A_lag 是跨时间主因果图；A0 只表示同时间片残余依赖，不默认传给模块3分类。
                self.causal_learner = AttentionGuidedTemporalCausalLearner(
                    **common_temporal_kwargs,
                    input_dim=1,
                    attention_heads=int(getattr(configs, "temporal_attention_heads", 2)),
                    attention_head_dim=int(getattr(configs, "temporal_attention_head_dim", 8)),
                    attention_dropout=float(getattr(configs, "temporal_attention_dropout", 0.0)),
                    classification_graph_scale=float(
                        getattr(configs, "temporal_attention_graph_scale", 1.0)
                    ),
                    prediction_loss_mode=getattr(configs, "temporal_prediction_loss_mode", "bold_alff"),
                    pred_huber_delta=float(getattr(configs, "temporal_pred_huber_delta", 1.0)),
                    lambda_pred_delta=float(getattr(configs, "lambda_temporal_pred_delta", 0.2)),
                    lambda_pred_lowfreq=float(getattr(configs, "lambda_temporal_pred_lowfreq", 0.2)),
                    lambda_pred_corr=float(getattr(configs, "lambda_temporal_pred_corr", 0.05)),
                    lowfreq_kernel_size=int(getattr(configs, "temporal_lowfreq_kernel_size", 9)),
                    a0_sparse_ratio=float(getattr(configs, "temporal_a0_sparse_ratio", 0.2)),
                )
            else:
                self.causal_learner = TemporalSEMCausalLearner(
                    **common_temporal_kwargs,
                    lambda_group_sparse=float(getattr(configs, "lambda_temporal_group_sparse", 0.0)),
                    lambda_lag_hierarchy=float(getattr(configs, "lambda_temporal_lag_hierarchy", 0.0)),
                    prediction_loss_mode=getattr(configs, "temporal_prediction_loss_mode", "bold_alff"),
                    pred_huber_delta=float(getattr(configs, "temporal_pred_huber_delta", 1.0)),
                    lambda_pred_delta=float(getattr(configs, "lambda_temporal_pred_delta", 0.2)),
                    lambda_pred_lowfreq=float(getattr(configs, "lambda_temporal_pred_lowfreq", 0.2)),
                    lambda_pred_corr=float(getattr(configs, "lambda_temporal_pred_corr", 0.05)),
                    lowfreq_kernel_size=int(getattr(configs, "temporal_lowfreq_kernel_size", 9)),
                    a0_sparse_ratio=float(getattr(configs, "temporal_a0_sparse_ratio", 0.2)),
                    prediction_target_mode=getattr(
                        configs,
                        "temporal_prediction_target_mode",
                        "innovation",
                    ),
                    a0_scale=float(getattr(configs, "temporal_a0_scale", 0.03)),
                    candidate_parent_topk=int(getattr(configs, "temporal_candidate_parent_topk", 0)),
                    use_sample_graph_residual=bool(getattr(configs, "use_sample_graph_residual", 0)),
                    sample_graph_delta_scale=float(getattr(configs, "temporal_sample_graph_delta_scale", 0.02)),
                    sample_lag_graph_mode=getattr(configs, "temporal_sample_lag_graph_mode", "abs"),
                    sample_graph_rank=int(getattr(configs, "temporal_sample_graph_rank", 4)),
                    lambda_sample_l1=float(getattr(configs, "lambda_sample_graph_l1", 0.0)),
                    lambda_sample_deviation=float(getattr(configs, "lambda_sample_graph_deviation", 0.0)),
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
                    readout_mode=getattr(configs, "hgcn_readout_mode", "mean_std"),
                    use_radius_head=bool(getattr(configs, "hgcn_use_radius_head", 0)),
                    radius_min_ratio=float(getattr(configs, "hgcn_radius_min_ratio", 0.25)),
                    causal_attention_heads=int(getattr(configs, "hgcn_causal_attention_heads", 4)),
                    causal_attention_graph_weight=float(getattr(configs, "hgcn_causal_attention_graph_weight", 0.5)),
                    network_gate_strength=float(getattr(configs, "hgcn_network_gate_strength", 0.35)),
                    use_graph_degree_encoding=bool(getattr(configs, "hgcn_use_graph_degree_encoding", 0)),
                    graph_degree_encoding_weight=float(getattr(configs, "hgcn_graph_degree_encoding_weight", 0.1)),
                    causal_subnetwork_count=int(getattr(configs, "hgcn_causal_subnetwork_count", 4)),
                    causal_subnetwork_topk=int(getattr(configs, "hgcn_causal_subnetwork_topk", 12)),
                    causal_subnetwork_weight=float(getattr(configs, "hgcn_causal_subnetwork_weight", 0.5)),
                    einstein_readout_weight=float(getattr(configs, "hgcn_einstein_readout_weight", 0.0)),
                    use_causal_role_readout=self.use_causal_role_readout,
                    causal_role_temperature=float(getattr(configs, "causal_role_temperature", 1.0)),
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
        if self.use_hgcn_module3 and self.use_multi_hop_causal_encoding:
            self.causal_reachability_projections = nn.ModuleList(
                [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(self.causal_reachability_hops)]
            )
            self.causal_reachability_gate_logits = nn.Parameter(
                torch.zeros(self.causal_reachability_hops)
            )
        else:
            self.causal_reachability_projections = nn.ModuleList()
            self.register_parameter("causal_reachability_gate_logits", None)
        self.fc_residual_norm = nn.LayerNorm(self.hgcn_hidden_dim) if self.hgcn_hidden_dim is not None else None
        if self.use_hgcn_module3 and self.use_hgcn_radial_calibration:
            self.hgcn_radial_head = nn.Sequential(
                nn.LayerNorm(self.hgcn_hidden_dim),
                nn.Linear(self.hgcn_hidden_dim, max(self.hgcn_hidden_dim // 2, 1)),
                nn.GELU(),
                nn.Linear(max(self.hgcn_hidden_dim // 2, 1), 1),
            )
            nn.init.zeros_(self.hgcn_radial_head[-1].weight)
            nn.init.zeros_(self.hgcn_radial_head[-1].bias)
        else:
            self.hgcn_radial_head = None

        if self.use_hpec_module4:
            # 模块 4 使用模块 3 的 z_global 与类别 prototype 计算 HPEC energy。
            hpec_manifold = self.hgcn_module3.manifold
            hpec_kwargs = dict(
                num_classes=configs.classes,
                embedding_dim=self.hgcn_hidden_dim,
                manifold=hpec_manifold,
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
                energy_ce_margin=self.hpec_energy_ce_margin,
                busemann_temperature=float(getattr(configs, "hpec_busemann_temperature", 1.0)),
                busemann_point_radius=float(getattr(configs, "hpec_busemann_point_radius", 0.0)),
                busemann_radius_gate_weight=float(getattr(configs, "hpec_busemann_radius_gate_weight", 0.0)),
                busemann_radius_gate_center=float(getattr(configs, "hpec_busemann_radius_gate_center", 0.3)),
                busemann_class_bias_weight=float(getattr(configs, "hpec_busemann_class_bias_weight", 0.0)),
                data_init=bool(getattr(configs, "hpec_data_init", 0)),
                use_sinkhorn_ema=self.hpec_use_sinkhorn_ema,
                prototype_update_mode=self.hpec_prototype_update_mode,
                reliable_confidence_threshold=self.hpec_reliable_confidence_threshold,
                reliable_min_samples=self.hpec_reliable_min_samples,
                sinkhorn_epsilon=self.hpec_sinkhorn_epsilon,
                sinkhorn_iters=self.hpec_sinkhorn_iters,
                ema_alpha=self.hpec_ema_alpha,
                ema_anchor_weight=self.hpec_ema_anchor_weight,
                intra_class_max_cos=float(getattr(configs, "hpec_intra_class_max_cos", 0.35)),
                prototype_min_radius_ratio=float(getattr(configs, "hpec_prototype_min_radius_ratio", 0.6)),
                prototype_max_radius_ratio=float(getattr(configs, "hpec_prototype_max_radius_ratio", 1.4)),
                prototype_parameterization=getattr(configs, "hpec_prototype_parameterization", "poincare_point"),
                eps=float(getattr(configs, "hpec_eps", 1e-7)),
            )
            hpec_seed = int(getattr(configs, "seed", 2024))
            self.hpec_module4 = HPECPrototypeEnergy(**hpec_kwargs, seed=hpec_seed)
            if self.use_causal_role_readout and self.hpec_causal_role_energy_weight > 0:
                self.hpec_role_modules = nn.ModuleList(
                    [
                        HPECPrototypeEnergy(**hpec_kwargs, seed=hpec_seed + role_idx + 1)
                        for role_idx in range(4)
                    ]
                )
                self.hpec_causal_role_gate_logits = nn.Parameter(
                    torch.zeros(int(configs.classes), 4)
                )
            else:
                self.hpec_role_modules = nn.ModuleList()
                self.register_parameter("hpec_causal_role_gate_logits", None)
            fusion_dim = self.hgcn_hidden_dim + int(configs.classes) * 3
            self.hpec_feature_classifier = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, self.hgcn_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.0))),
                nn.Linear(self.hgcn_hidden_dim, self.out_dim),
            )
            evidence_dim = int(configs.classes) * 4 + 3
            self.hpec_evidence_calibrator = nn.Sequential(
                nn.LayerNorm(evidence_dim),
                nn.Linear(evidence_dim, max(8, self.hgcn_hidden_dim // 2)),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.0))),
                nn.Linear(max(8, self.hgcn_hidden_dim // 2), self.out_dim),
            )
        else:
            self.hpec_module4 = None
            self.hpec_role_modules = nn.ModuleList()
            self.register_parameter("hpec_causal_role_gate_logits", None)
            self.hpec_feature_classifier = None
            self.hpec_evidence_calibrator = None
        if self.use_final_logit_calibration:
            identity_scale_raw = torch.log(torch.expm1(torch.ones(self.out_dim)))
            self.final_logit_scale = nn.Parameter(identity_scale_raw)
            self.final_logit_bias = nn.Parameter(torch.zeros(self.out_dim))
        else:
            self.register_parameter("final_logit_scale", None)
            self.register_parameter("final_logit_bias", None)
        if self.use_hyperbolic_residual_bias:
            self.hyperbolic_residual_bias = nn.Parameter(torch.zeros(self.out_dim))
        else:
            self.register_parameter("hyperbolic_residual_bias", None)

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

        # FC 生物标志分支：把样本相关矩阵编码为 embedding，在 gcn_fallback readout 层与图特征融合。
        # FC 是 MDD 最经典的边级判别生物标志，而 1 层 GCN 的 mean 池化会把它丢掉，这里显式补回。
        # 默认 network 模式：把 116 ROI 按 MDD 文献网络聚成 8×8 网络 FC（~72 维），低维强抗过拟合；
        # upper_tri = 6670 维全边（高判别但易过拟合）；both = 两者拼接。默认关闭，不影响旧入口/其它数据集。
        self.use_fc_readout_branch = bool(getattr(configs, "use_fc_readout_branch", 0))
        self.fc_readout_mode = str(getattr(configs, "fc_readout_mode", "network")).lower()
        if self.fc_readout_mode not in ("network", "upper_tri", "both"):
            raise ValueError(
                f"Unsupported fc_readout_mode={self.fc_readout_mode!r}. Use 'network', 'upper_tri' or 'both'."
            )
        self.fc_readout_fisher_z = bool(getattr(configs, "fc_readout_fisher_z", 1))
        self.fc_readout_edge_dropout = float(getattr(configs, "fc_readout_edge_dropout", 0.0))
        self.fc_readout_embed_dim = int(getattr(configs, "fc_readout_embed_dim", 64))
        fc_n_nodes = int(getattr(configs, "channel", 0))
        fc_input_dim = 0
        self.fc_readout_net_count = 0
        if self.use_fc_readout_branch and fc_n_nodes > 1:
            if self.fc_readout_mode in ("upper_tri", "both"):
                triu_idx = torch.triu_indices(fc_n_nodes, fc_n_nodes, offset=1)
                self.register_buffer("fc_triu_idx", triu_idx, persistent=False)
                fc_input_dim += int(triu_idx.shape[1])
            if self.fc_readout_mode in ("network", "both"):
                # 独立构建网络级掩码（模块3关闭时其 fc_network_masks 为空，不能依赖）。
                groups, _ = _aal116_network_masks()
                if fc_n_nodes == 116:
                    net_masks = torch.zeros(len(groups), 116, dtype=torch.float32)
                    for group_idx, (_, indices) in enumerate(groups):
                        net_masks[group_idx, indices] = 1.0
                    net_masks = net_masks / net_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
                else:
                    # 非 116 通道无文献网络先验，退化为单一全脑网络。
                    net_masks = torch.ones(1, fc_n_nodes, dtype=torch.float32) / float(fc_n_nodes)
                self.register_buffer("fc_readout_network_masks", net_masks, persistent=False)
                self.fc_readout_net_count = int(net_masks.shape[0])
                # 网络×网络平均 FC (G*G) + 每网络对全脑平均连接强度 (G)。
                fc_input_dim += self.fc_readout_net_count * self.fc_readout_net_count + self.fc_readout_net_count
        if self.use_fc_readout_branch and fc_input_dim > 0:
            fc_hidden = int(getattr(configs, "fc_readout_hidden_dim", 128))
            fc_dropout = float(getattr(configs, "fc_readout_dropout", 0.5))
            self.fc_readout_branch = nn.Sequential(
                nn.LayerNorm(fc_input_dim),
                nn.Linear(fc_input_dim, fc_hidden),
                nn.GELU(),
                nn.Dropout(fc_dropout),
                nn.Linear(fc_hidden, self.fc_readout_embed_dim),
            )
            self._fc_external_feature_dim = self.fc_readout_embed_dim
        else:
            self.use_fc_readout_branch = False
            self.fc_readout_branch = None
            self._fc_external_feature_dim = 0

        # 模块 3 FC 注入：把已证明的 FC 生物标志 embedding 投影后加到 z_tangent。
        # 治"模块3看不到FC"这一掉点主因——让 hgcn_classifier/HPEC 直接吃到 FC 边信号。
        # 默认权重 0（不影响现有 gcn_fallback 入口），仅在开启 3/4 的对照实验里启用。
        self.hgcn_fc_inject_weight = float(getattr(configs, "hgcn_fc_inject_weight", 0.0))
        self.hgcn_fc_anchor_norm_target = float(getattr(configs, "hgcn_fc_anchor_norm_target", 0.5))
        self.hgcn_fc_anchor_gate_init = float(getattr(configs, "hgcn_fc_anchor_gate_init", -1.5))
        if (
            self.use_hgcn_module3
            and self.use_fc_readout_branch
            and self._fc_external_feature_dim > 0
            and self.hgcn_fc_inject_weight > 0
            and self.hgcn_hidden_dim is not None
        ):
            self.hgcn_fc_inject = nn.Sequential(
                nn.LayerNorm(self._fc_external_feature_dim),
                nn.Linear(self._fc_external_feature_dim, self.hgcn_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.0))),
                nn.Linear(self.hgcn_hidden_dim, self.hgcn_hidden_dim),
            )
            self.hgcn_fc_anchor_norm = nn.LayerNorm(self.hgcn_hidden_dim)
            self.hgcn_fc_anchor_gate = nn.Sequential(
                nn.LayerNorm(self.hgcn_hidden_dim * 2),
                nn.Linear(self.hgcn_hidden_dim * 2, self.hgcn_hidden_dim),
                nn.GELU(),
                nn.Linear(self.hgcn_hidden_dim, self.hgcn_hidden_dim),
            )
            nn.init.zeros_(self.hgcn_fc_anchor_gate[-1].weight)
            nn.init.constant_(self.hgcn_fc_anchor_gate[-1].bias, self.hgcn_fc_anchor_gate_init)
        else:
            self.hgcn_fc_inject = None
            self.hgcn_fc_anchor_norm = None
            self.hgcn_fc_anchor_gate = None
        if (
            self.use_hgcn_module3
            and self.use_fc_readout_branch
            and self._fc_external_feature_dim > 0
            and self.hgcn_hidden_dim is not None
            and self.module34_film_weight > 0
        ):
            self.module34_film = nn.Sequential(
                nn.LayerNorm(self._fc_external_feature_dim),
                nn.Linear(self._fc_external_feature_dim, self.hgcn_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.0))),
                nn.Linear(self.hgcn_hidden_dim, self.hgcn_hidden_dim * 2),
            )
            nn.init.zeros_(self.module34_film[-1].weight)
            nn.init.zeros_(self.module34_film[-1].bias)
        else:
            self.module34_film = None

        if not self.use_hgcn_module3 or self.keep_gcn_fallback_with_hyperbolic:
            fallback_kwargs = dict(
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
                use_graph_stats=bool(getattr(configs, "gcn_fallback_use_graph_stats", 0)),
                graph_stats_mode=getattr(configs, "gcn_fallback_graph_stats_mode", "basic"),
                graph_stats_input=getattr(configs, "gcn_fallback_graph_stats_input", "normalized"),
                readout_mode=getattr(configs, "gcn_fallback_readout_mode", "mean_std"),
                input_residual_weight=float(getattr(configs, "gcn_fallback_input_residual_weight", 0.0)),
                external_feature_dim=self._fc_external_feature_dim,
                edge_readout_topk=int(getattr(configs, "gcn_fallback_edge_readout_topk", 0)),
                directional_propagation=bool(
                    getattr(configs, "gcn_fallback_directional_propagation", 1)
                ),
            )
            if self.use_hgcn_module3 and self.keep_gcn_fallback_with_hyperbolic:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(getattr(configs, "seed", 2024)) + 173)
                    self.gcn_fallback = ModuleGCNFallback(**fallback_kwargs)
            else:
                self.gcn_fallback = ModuleGCNFallback(**fallback_kwargs)
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
        self.latest_site_adversarial_loss = None
        self.latest_site_modulation_reg_loss = None
        self.latest_hgcn_aux_logits = None
        self.latest_prediction_logits = None
        self.latest_hpec_raw_margin = None
        self.latest_module34_branch_logits = None
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
        freeze_causal = (
            self.training
            and self.freeze_causal_after_epoch >= 0
            and self.current_epoch >= self.freeze_causal_after_epoch
        )
        if freeze_causal:
            with torch.no_grad():
                causal_output = self.causal_learner(causal_input)
                _, aux_parts = self.causal_learner.compute_losses(causal_output)
            aux_loss = causal_output.a_shared.sum() * 0.0
            self.latest_aux_losses["causal_module2_frozen"] = torch.ones(
                (),
                device=causal_output.a_shared.device,
                dtype=causal_output.a_shared.dtype,
            )
        else:
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
        stability_loss = None if freeze_causal else self._causal_stability_loss(causal_output)
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
                self._record_classification_adjacency(sample_adjacency)
                return sample_adjacency, True
            if self.classification_graph_source not in (
                "blend",
                "learned",
                "causal",
                "residual_blend",
                "topk_blend",
                "gated_fc",
                "causal_masked_fc",
                "causal_soft_masked_fc",
                "gated_fc_centered",
                "gated_fc_signed",
            ):
                raise ValueError(
                    f"Unsupported classification_graph_source={self.classification_graph_source!r}. "
                    "Use 'blend', 'learned', 'residual_blend', 'topk_blend', 'gated_fc', 'gated_fc_signed' or 'sample_correlation'."
                )
            # 下游主图必须来自稀疏跨时间 A_lag，而不是加入稠密样本残差后的
            # A_effective。后者几乎处处为正，用 >0 构造 mask 会把因果图退化成完整 FC。
            adjacency = causal_output.a_shared
            if self.classification_graph_source in ("learned", "causal"):
                causal_gate = adjacency
                sample_graph = getattr(causal_output, "a_effective", None)
                if sample_graph is not None and sample_graph.ndim == 3:
                    gate = self._normalize_learned_gate_unit(causal_gate)
                    if gate.ndim == 2:
                        gate = gate.unsqueeze(0).expand_as(sample_graph)
                    # 样本 lag 相关只在共享因果候选边上通过，兼顾个体差异和方向性。
                    adjacency = sample_graph * gate
                if self.detach_module2_graph_for_classification:
                    adjacency = adjacency.detach()
                    causal_gate = causal_gate.detach()
                self._record_classification_adjacency(adjacency, causal_gate=causal_gate)
                return adjacency, False
            learned_adjacency = adjacency
            if (
                correlation_matrix is not None
                and (
                    self.module2_sample_correlation_blend > 0.0
                    or self.classification_graph_source in (
                        "residual_blend",
                        "topk_blend",
                        "gated_fc",
                        "causal_masked_fc",
                        "causal_soft_masked_fc",
                        "gated_fc_centered",
                        "gated_fc_signed",
                    )
                )
            ):
                # 模块2启用时保留样本 FC 作为结构先验，再由学习到的因果图提供有向修正。
                sample_adjacency = self._prepare_sample_correlation_adjacency(
                    correlation_matrix,
                    device=adjacency.device,
                    dtype=adjacency.dtype,
                )
                if learned_adjacency.ndim == 2 and sample_adjacency.ndim == 3:
                    learned_adjacency = learned_adjacency.unsqueeze(0).expand_as(sample_adjacency)
                if self.detach_module2_graph_for_classification:
                    learned_adjacency = learned_adjacency.detach()
                if self.classification_graph_source == "residual_blend":
                    residual = self._scale_learned_graph_to_sample(
                        learned_adjacency,
                        sample_adjacency,
                    )
                    alpha = max(self.module2_graph_residual_alpha, 0.0)
                    adjacency = sample_adjacency + alpha * residual
                elif self.classification_graph_source == "gated_fc_signed":
                    signed_graph = getattr(causal_output, "signed_sample_graph", None)
                    if signed_graph is None:
                        signed_graph = learned_adjacency
                    elif self.detach_module2_graph_for_classification:
                        signed_graph = signed_graph.detach()
                    if signed_graph.ndim == 2 and sample_adjacency.ndim == 3:
                        signed_graph = signed_graph.unsqueeze(0).expand_as(sample_adjacency)
                    gate = self._normalize_signed_learned_gate(signed_graph)
                    alpha = min(max(self.module2_graph_residual_alpha, 0.0), 1.0)
                    adjacency = sample_adjacency * (1.0 + alpha * gate)
                    adjacency = adjacency.clamp_min(0.0)
                elif self.classification_graph_source == "gated_fc":
                    gate = self._normalize_learned_gate(learned_adjacency)
                    alpha = min(max(self.module2_graph_residual_alpha, 0.0), 1.0)
                    adjacency = sample_adjacency * ((1.0 - alpha) + alpha * gate)
                elif self.classification_graph_source == "causal_masked_fc":
                    mask = (learned_adjacency > 0).to(sample_adjacency.dtype)
                    adjacency = sample_adjacency * mask
                elif self.classification_graph_source == "causal_soft_masked_fc":
                    gate = self._normalize_learned_gate_unit(learned_adjacency)
                    floor = min(max(self.module2_graph_residual_alpha, 0.0), 1.0)
                    adjacency = sample_adjacency * (floor + (1.0 - floor) * gate)
                elif self.classification_graph_source == "gated_fc_centered":
                    gate = self._normalize_learned_gate(learned_adjacency)
                    centered_gate = torch.tanh(gate - 1.0)
                    alpha = min(max(self.module2_graph_residual_alpha, 0.0), 1.0)
                    adjacency = sample_adjacency * (1.0 + alpha * centered_gate)
                    adjacency = adjacency.clamp_min(0.0)
                else:
                    blend = min(max(self.module2_sample_correlation_blend, 0.0), 1.0)
                    adjacency = (1.0 - blend) * learned_adjacency + blend * sample_adjacency
                self.latest_sample_correlation_adjacency = sample_adjacency
            self._record_classification_adjacency(adjacency, causal_gate=learned_adjacency)
            return adjacency, False
        if correlation_matrix is None:
            raise RuntimeError(
                f"{module_name} requires sample correlation adjacency when module 2 is disabled. "
                "Provide correlation_matrix or enable use_causal_module2."
            )
        self.latest_sample_correlation_adjacency = correlation_matrix
        self._record_classification_adjacency(correlation_matrix)
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

    def _scale_learned_graph_to_sample(self, learned_adjacency, sample_adjacency):
        learned = learned_adjacency.clamp_min(0.0)
        sample = sample_adjacency.clamp_min(0.0)
        n_nodes = learned.shape[-1]
        off_diag = 1.0 - torch.eye(n_nodes, device=learned.device, dtype=learned.dtype)
        if learned.ndim == 3:
            off_diag = off_diag.unsqueeze(0)
        learned_mass = (learned * off_diag).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        sample_mass = (sample * off_diag).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return learned * (sample_mass / learned_mass)

    def _normalize_learned_gate(self, learned_adjacency):
        learned = learned_adjacency.clamp_min(0.0)
        n_nodes = learned.shape[-1]
        off_diag = 1.0 - torch.eye(n_nodes, device=learned.device, dtype=learned.dtype)
        if learned.ndim == 3:
            off_diag = off_diag.unsqueeze(0)
        mean_value = (learned * off_diag).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return learned / mean_value

    def _normalize_learned_gate_unit(self, learned_adjacency):
        """按每个 child 的最强 parent 将有向边缩放到 [0, 1]。"""
        learned = torch.nan_to_num(
            learned_adjacency.clamp_min(0.0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        scale = learned.amax(dim=-2, keepdim=True).clamp_min(1e-8)
        return (learned / scale).clamp(0.0, 1.0)

    def _record_classification_adjacency(self, adjacency, causal_gate=None):
        self.latest_classification_adjacency = adjacency
        graph = torch.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
        n_nodes = graph.shape[-1]
        off_diag = 1.0 - torch.eye(n_nodes, device=graph.device, dtype=graph.dtype)
        if graph.ndim == 3:
            off_diag = off_diag.unsqueeze(0)
        graph = graph * off_diag
        mass = graph.abs().mean().clamp_min(1e-8)
        asymmetry = (graph - graph.transpose(-1, -2)).abs().mean() / mass
        diagnostics = {
            "classification_graph_mass": mass.detach(),
            "classification_graph_asymmetry_ratio": asymmetry.detach(),
        }
        if causal_gate is not None:
            gate = torch.nan_to_num(causal_gate, nan=0.0, posinf=0.0, neginf=0.0)
            diagnostics["classification_graph_causal_support_density"] = (
                (gate.abs() > 1e-8).to(gate.dtype).mean().detach()
            )
        self.latest_aux_losses.update(diagnostics)

    def _normalize_signed_learned_gate(self, signed_adjacency):
        signed = torch.nan_to_num(signed_adjacency, nan=0.0, posinf=0.0, neginf=0.0)
        n_nodes = signed.shape[-1]
        off_diag = 1.0 - torch.eye(n_nodes, device=signed.device, dtype=signed.dtype)
        if signed.ndim == 3:
            off_diag = off_diag.unsqueeze(0)
        signed = signed * off_diag
        scale = signed.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return torch.tanh(signed / scale)

    def _project_tangent_to_module3_ball(self, module3_output, z_tangent):
        backclip_radius = float(
            getattr(
                self.hgcn_module3.backclip,
                "radius",
                getattr(self.hgcn_module3.backclip, "max_radius", 1.0),
            )
        )
        if backclip_radius > 0:
            norm = z_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            z_tangent = z_tangent * torch.clamp(backclip_radius / norm, max=1.0)
        z_global = self.hgcn_module3.manifold.expmap0(z_tangent, dim=-1, project=True)
        z_global = self.hgcn_module3.manifold.projx(z_global, dim=-1)
        z_tangent = self.hgcn_module3.manifold.logmap0(z_global, dim=-1)
        module3_output.z_global = z_global
        module3_output.z_tangent = z_tangent
        return module3_output

    def _apply_module34_film(self, module3_output, correlation_matrix):
        if (
            self.module34_film is None
            or correlation_matrix is None
            or self.module34_film_weight <= 0
        ):
            return module3_output
        fc_embed = self._fc_readout_features(
            correlation_matrix,
            module3_output.z_tangent.shape[0],
            module3_output.z_tangent.device,
            module3_output.z_tangent.dtype,
        )
        if fc_embed is None:
            return module3_output
        film = self.module34_film(fc_embed)
        scale_raw, shift = film.chunk(2, dim=-1)
        max_scale = max(float(self.module34_film_max_scale), 0.0)
        scale = max_scale * torch.tanh(scale_raw)
        if self.module34_film_shift_norm > 0:
            shift_norm = shift.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            shift = shift * torch.clamp(self.module34_film_shift_norm / shift_norm, max=1.0)
        weight = max(float(self.module34_film_weight), 0.0)
        z_tangent = module3_output.z_tangent * (1.0 + weight * scale) + weight * shift
        self._project_tangent_to_module3_ball(module3_output, z_tangent)
        self.latest_aux_losses.update(
            {
                "module34_film_weight": torch.as_tensor(
                    weight,
                    device=z_tangent.device,
                    dtype=z_tangent.dtype,
                ),
                "module34_film_scale_abs_mean": scale.abs().mean().detach(),
                "module34_film_shift_norm_mean": shift.norm(dim=-1).mean().detach(),
            }
        )
        return module3_output

    def _apply_multi_hop_causal_encoding(self, cycle_features, adjacency):
        """用 A_cls 的前向因果可达性增强模块 3 输入，不改写因果图本身。"""

        if not self.use_multi_hop_causal_encoding or not self.causal_reachability_projections:
            return cycle_features
        graph = torch.nan_to_num(adjacency.to(device=cycle_features.device, dtype=cycle_features.dtype)).abs()
        if graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(cycle_features.shape[0], -1, -1)
        n_nodes = graph.shape[-1]
        graph = graph * (1.0 - torch.eye(n_nodes, device=graph.device, dtype=graph.dtype).unsqueeze(0))
        transition = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        power = transition
        hop_gate = torch.softmax(self.causal_reachability_gate_logits, dim=0)
        encoded = torch.zeros_like(cycle_features)
        for hop_idx, projection in enumerate(self.causal_reachability_projections):
            parent_message = projection(cycle_features)
            # A[parent, child]；转置后由 child 聚合其 causal parent。
            hop_feature = torch.bmm(power.transpose(1, 2), parent_message)
            encoded = encoded + hop_gate[hop_idx] * hop_feature
            self.latest_aux_losses[f"causal_reachability_hop_{hop_idx + 1}_gate"] = hop_gate[hop_idx]
            self.latest_aux_losses[f"causal_reachability_hop_{hop_idx + 1}_norm"] = hop_feature.norm(dim=-1).mean()
            power = torch.bmm(power, transition)
        residual = self.causal_reachability_scale * encoded
        self.latest_aux_losses["causal_reachability_residual_norm"] = residual.norm(dim=-1).mean()
        return cycle_features + residual

    def _run_hgcn_module3(self, cycle_features, causal_output, correlation_matrix=None, cache_as_primary=True):
        if not self.use_hgcn_module3:
            return None
        if cycle_features is None:
            raise RuntimeError("Module 3 requires node feature.")

        adjacency, is_sample_correlation = self._resolve_graph_adjacency(
            causal_output,
            correlation_matrix=correlation_matrix,
            module_name="Module 3",
        )

        encoded_features = self._apply_multi_hop_causal_encoding(cycle_features, adjacency)
        # 模块 3 使用 Cycle feature C 与图结构；模块 2 开启时图来自可微 A_learned，
        # 模块 2 关闭时图来自样本相关矩阵，退化为模块 3 自身的 HGCN 设计。
        module3_output = self.hgcn_module3(
            encoded_features,
            adjacency,
            is_sample_correlation=is_sample_correlation,
            sample_correlation=correlation_matrix,
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
            self._project_tangent_to_module3_ball(module3_output, z_tangent)
            self.latest_aux_losses["fc_residual_norm_mean"] = fc_residual.norm(dim=-1).mean()
            self.latest_aux_losses["fc_residual_update_norm_mean"] = residual_update.norm(dim=-1).mean()
        # 注入网络级 FC 生物标志 embedding（复用已证明的 fc_readout_branch），治"模块3看不到FC"主因。
        if self.hgcn_fc_inject is not None:
            fc_embed = self._fc_readout_features(
                correlation_matrix,
                module3_output.z_tangent.shape[0],
                module3_output.z_tangent.device,
                module3_output.z_tangent.dtype,
            )
            if fc_embed is not None:
                fc_inject = self.hgcn_fc_inject(fc_embed)
                if self.hgcn_fc_anchor_norm is not None:
                    fc_inject = self.hgcn_fc_anchor_norm(fc_inject)
                if self.hgcn_fc_anchor_norm_target > 0:
                    anchor_norm = fc_inject.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    fc_inject = fc_inject * (self.hgcn_fc_anchor_norm_target / anchor_norm)
                if self.hgcn_fc_anchor_gate is not None:
                    gate_input = torch.cat([module3_output.z_tangent, fc_inject], dim=-1)
                    fc_anchor_gate = torch.sigmoid(self.hgcn_fc_anchor_gate(gate_input))
                else:
                    fc_anchor_gate = torch.ones_like(fc_inject)
                fc_anchor_update = self.hgcn_fc_inject_weight * fc_anchor_gate * fc_inject
                z_tangent = module3_output.z_tangent + fc_anchor_update
                self._project_tangent_to_module3_ball(module3_output, z_tangent)
                self.latest_aux_losses["hgcn_fc_inject_norm_mean"] = fc_inject.norm(dim=-1).mean()
                self.latest_aux_losses["hgcn_fc_anchor_gate_mean"] = fc_anchor_gate.mean()
                self.latest_aux_losses["hgcn_fc_anchor_update_norm_mean"] = fc_anchor_update.norm(dim=-1).mean()
        self._apply_module34_film(module3_output, correlation_matrix)
        self._apply_hgcn_radial_calibration(module3_output)
        if cache_as_primary:
            self.latest_module3_output = module3_output
        z_radius = torch.linalg.norm(module3_output.z_global, dim=-1)
        z_tangent_norm = torch.linalg.norm(module3_output.z_tangent, dim=-1)
        node_attention = module3_output.node_attention
        if node_attention is not None and node_attention.numel() > 0:
            node_entropy = -(
                node_attention * node_attention.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            node_peak = node_attention.max(dim=-1).values.mean()
        else:
            node_entropy = z_radius.sum() * 0.0
            node_peak = z_radius.sum() * 0.0
        network_attention = module3_output.network_attention
        if network_attention is not None and network_attention.numel() > 0:
            if network_attention.ndim == 1:
                net_attention = network_attention.unsqueeze(0)
            else:
                net_attention = network_attention
            net_entropy = -(
                net_attention * net_attention.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            net_peak = net_attention.max(dim=-1).values.mean()
        else:
            net_entropy = z_radius.sum() * 0.0
            net_peak = z_radius.sum() * 0.0
        self.latest_aux_losses.update(
            {
                "z_radius_mean": z_radius.mean(),
                "z_radius_max": z_radius.max(),
                "z_tangent_norm_mean": z_tangent_norm.mean(),
                "module3_node_attention_entropy": node_entropy,
                "module3_node_attention_peak": node_peak,
                "module3_network_attention_entropy": net_entropy,
                "module3_network_attention_peak": net_peak,
            }
        )
        return module3_output

    def _apply_hgcn_radial_calibration(self, module3_output):
        """保持 HGCN 学到的方向，用可学习半径把双曲中心点维持在有效环带内。"""

        if (
            module3_output is None
            or self.hgcn_radial_head is None
            or self.hgcn_hidden_dim is None
        ):
            return module3_output
        z_tangent = module3_output.z_tangent
        norm = z_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = z_tangent / norm
        min_radius = max(float(self.hgcn_radial_min), 1e-6)
        max_radius = max(float(self.hgcn_radial_max), min_radius + 1e-6)
        raw_radius = torch.sigmoid(self.hgcn_radial_head(z_tangent))
        target_norm = min_radius + (max_radius - min_radius) * raw_radius
        calibrated_tangent = direction * target_norm
        z_global = self.hgcn_module3.manifold.expmap0(calibrated_tangent, dim=-1, project=True)
        z_global = self.hgcn_module3.manifold.projx(z_global, dim=-1)
        calibrated_tangent = self.hgcn_module3.manifold.logmap0(z_global, dim=-1)
        module3_output.z_global = z_global
        module3_output.z_tangent = calibrated_tangent
        self.latest_aux_losses.update(
            {
                "hgcn_radial_calibration_enabled": torch.ones(
                    (),
                    device=z_tangent.device,
                    dtype=z_tangent.dtype,
                ),
                "hgcn_radial_target_norm_mean": target_norm.mean(),
                "hgcn_radial_target_norm_min": target_norm.min(),
                "hgcn_radial_target_norm_max": target_norm.max(),
            }
        )
        return module3_output

    def _fc_readout_features(self, correlation_matrix, batch_size, device, dtype):
        """从样本相关矩阵提取 FC 描述子并编码为 embedding。

        network 模式：116 ROI 按 MDD 文献网络聚成 8×8 网络 FC + 每网络全脑强度（~72 维，
        低维抗过拟合）。upper_tri 模式：6670 维全边。both：两者拼接。
        返回 None 时，ModuleGCNFallback 会按 external_feature_dim 自动补零，
        因此分支开启但某 batch 缺相关矩阵也不会破坏 classifier 输入维度。
        """

        if not self.use_fc_readout_branch or self.fc_readout_branch is None:
            return None
        if correlation_matrix is None:
            return None
        corr = torch.nan_to_num(
            correlation_matrix.to(device=device, dtype=dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if corr.ndim == 2:
            corr = corr.unsqueeze(0).expand(batch_size, -1, -1)
        # 去对角（自相关=1 不含连接信息），再 Fisher-z 稳定数值。
        n_nodes = corr.shape[-1]
        off_diag = 1.0 - torch.eye(n_nodes, device=device, dtype=dtype)
        corr = corr * off_diag
        if self.fc_readout_fisher_z:
            corr = torch.atanh(corr.clamp(min=-0.999, max=0.999))
        parts = []
        if self.fc_readout_mode in ("upper_tri", "both"):
            row, col = self.fc_triu_idx[0], self.fc_triu_idx[1]
            edges = corr[:, row, col]
            if self.training and self.fc_readout_edge_dropout > 0:
                keep_prob = max(1.0 - self.fc_readout_edge_dropout, 1e-6)
                mask = torch.rand_like(edges).lt(keep_prob)
                edges = edges * mask / keep_prob
            parts.append(edges)
        if self.fc_readout_mode in ("network", "both"):
            masks = self.fc_readout_network_masks.to(device=device, dtype=dtype)  # [G, N]
            # 网络×网络平均 FC：[B, G, G]
            network_fc = torch.einsum("gn,bnm,hm->bgh", masks, corr, masks)
            parts.append(network_fc.reshape(network_fc.shape[0], -1))
            # 每网络对全脑的平均连接强度：[B, G]
            node_strength = corr.mean(dim=-1)  # [B, N]
            parts.append(torch.einsum("gn,bn->bg", masks, node_strength))
        features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return self.fc_readout_branch(features)

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
        # FC 生物标志 embedding 在 readout 层与图特征融合；分支关闭时为 None。
        fc_external = self._fc_readout_features(
            correlation_matrix,
            node_features.shape[0],
            node_features.device,
            node_features.dtype,
        )
        # 模块 3/4 关闭时退化为普通 GCN，仍复用模块 2 或样本相关矩阵提供的图结构。
        gcn_output = self.gcn_fallback(
            node_features,
            adjacency,
            is_sample_correlation=is_sample_correlation,
            external_features=fc_external,
        )
        self.latest_gcn_fallback_output = gcn_output
        return gcn_output

    def _run_hpec_module4(self, module3_output, cache_as_primary=True):
        if not self.use_hpec_module4:
            return None
        if module3_output is None:
            raise RuntimeError("Module 4 requires module 3 z_global output.")

        # HPEC 默认使用 Poincare Ball 中的 z_global，而 z_tangent 继续作为调试/t-SNE 表示。
        hpec_input = self._calibrate_hpec_input_radius(module3_output.z_global)
        hpec_input = self._augment_hpec_input_tangent(hpec_input)
        module4_output = self.hpec_module4(hpec_input)
        role_points = getattr(module3_output, "causal_role_poincare", None)
        if (
            role_points is not None
            and role_points.ndim == 3
            and len(self.hpec_role_modules) == role_points.shape[1]
            and self.hpec_causal_role_energy_weight > 0
        ):
            role_outputs = []
            for role_idx, role_module in enumerate(self.hpec_role_modules):
                role_input = self._calibrate_hpec_input_radius(role_points[:, role_idx, :])
                role_input = self._augment_hpec_input_tangent(role_input)
                role_outputs.append(role_module(role_input))
            role_gate = torch.softmax(self.hpec_causal_role_gate_logits, dim=-1)
            role_energy_stack = torch.stack(
                [output.energy_matrix for output in role_outputs], dim=1
            )
            role_proto_energy_stack = torch.stack(
                [output.energy_per_proto for output in role_outputs], dim=1
            )
            role_similarity_stack = torch.stack(
                [output.prototype_similarity for output in role_outputs], dim=1
            )
            role_distance_stack = torch.stack(
                [output.prototype_distance_logits for output in role_outputs], dim=1
            )
            class_role_weight = role_gate.transpose(0, 1).unsqueeze(0)
            role_energy = (role_energy_stack * class_role_weight).sum(dim=1)
            proto_class_role_weight = class_role_weight.unsqueeze(-1)
            role_energy_per_proto = (
                role_proto_energy_stack * proto_class_role_weight
            ).sum(dim=1)
            role_similarity = (
                role_similarity_stack * proto_class_role_weight
            ).sum(dim=1)
            role_distance_logits = (
                role_distance_stack * class_role_weight
            ).sum(dim=1)
            role_weight = self.hpec_causal_role_energy_weight
            combined_energy = (
                (1.0 - role_weight) * module4_output.energy_matrix
                + role_weight * role_energy
            )
            combined_energy_per_proto = (
                (1.0 - role_weight) * module4_output.energy_per_proto
                + role_weight * role_energy_per_proto
            )
            combined_similarity = (
                (1.0 - role_weight) * module4_output.prototype_similarity
                + role_weight * role_similarity
            )
            combined_distance_logits = (
                (1.0 - role_weight) * module4_output.prototype_distance_logits
                + role_weight * role_distance_logits
            )
            module4_output = replace(
                module4_output,
                energy_per_proto=combined_energy_per_proto,
                prototype_similarity=combined_similarity,
                prototype_distance_logits=combined_distance_logits,
                energy_matrix=combined_energy,
                prediction=torch.argmin(combined_energy, dim=-1),
                probability=torch.softmax(-combined_energy, dim=-1),
                prototype_assignment=torch.argmin(
                    combined_energy_per_proto.reshape(combined_energy_per_proto.shape[0], -1),
                    dim=-1,
                ),
            )
            module4_output.causal_role_outputs = role_outputs
            module4_output.causal_role_energy = role_energy
            self.latest_aux_losses.update(
                {
                    "hpec_causal_role_energy_weight": torch.as_tensor(
                        role_weight,
                        device=combined_energy.device,
                        dtype=combined_energy.dtype,
                    ),
                    "hpec_causal_role_energy_std": torch.stack(
                        [output.energy_matrix for output in role_outputs], dim=1
                    ).std(dim=1, unbiased=False).mean().detach(),
                    "hpec_causal_role_gate_entropy": (
                        -(role_gate.clamp_min(1e-8) * role_gate.clamp_min(1e-8).log())
                        .sum(dim=-1)
                        .mean()
                    ).detach(),
                    "hpec_causal_role_gate_peak": role_gate.max(dim=-1).values.mean().detach(),
                }
            )
        if cache_as_primary:
            self.latest_module4_output = module4_output
        return module4_output

    def _augment_hpec_input_tangent(self, z_global):
        """训练期在切空间加入轻量噪声，防止 HPEC 原型边界贴死训练样本。"""

        if (not self.training) or self.hpec_input_tangent_noise_std <= 0:
            return z_global
        manifold = self.hpec_module4.manifold
        z_global = manifold.projx(z_global, dim=-1)
        z_tangent = manifold.logmap0(z_global, dim=-1)
        noise = torch.randn_like(z_tangent) * self.hpec_input_tangent_noise_std
        noisy_tangent = z_tangent + noise
        noisy = manifold.expmap0(noisy_tangent, dim=-1, project=True)
        noisy = manifold.projx(noisy, dim=-1)
        self.latest_aux_losses.update(
            {
                "hpec_input_tangent_noise_std": torch.as_tensor(
                    self.hpec_input_tangent_noise_std,
                    device=z_tangent.device,
                    dtype=z_tangent.dtype,
                ),
                "hpec_input_tangent_noise_norm": noise.norm(dim=-1).mean().detach(),
            }
        )
        return noisy

    def _calibrate_hpec_input_radius(self, z_global):
        """只校准 HPEC 输入半径，不改变样本方向。"""
        if self.hpec_input_radius_min <= 0 and self.hpec_input_radius_max <= 0:
            return z_global
        manifold = self.hpec_module4.manifold
        z_global = manifold.projx(z_global, dim=-1)
        z_tangent = manifold.logmap0(z_global, dim=-1)
        radius = torch.linalg.norm(z_tangent, dim=-1, keepdim=True)
        direction = torch.where(radius > 1e-8, z_tangent / radius.clamp_min(1e-8), z_tangent)
        target_radius = radius
        if self.hpec_input_radius_min > 0:
            target_radius = torch.maximum(
                target_radius,
                torch.as_tensor(self.hpec_input_radius_min, device=z_tangent.device, dtype=z_tangent.dtype),
            )
        if self.hpec_input_radius_max > 0:
            target_radius = torch.minimum(
                target_radius,
                torch.as_tensor(self.hpec_input_radius_max, device=z_tangent.device, dtype=z_tangent.dtype),
            )
        calibrated = manifold.expmap0(direction * target_radius, dim=-1, project=True)
        calibrated = manifold.projx(calibrated, dim=-1)
        self.latest_aux_losses.update(
            {
                "hpec_input_radius_raw_mean": torch.linalg.norm(z_global, dim=-1).mean().detach(),
                "hpec_input_radius_calibrated_mean": torch.linalg.norm(calibrated, dim=-1).mean().detach(),
            }
        )
        return calibrated

    def _network_hpec_energy(self, module3_output):
        if (
            module3_output is None
            or self.hpec_network_energy_weight <= 0
            or not hasattr(module3_output, "network_poincare")
            or module3_output.network_poincare.ndim != 3
            or module3_output.network_poincare.shape[1] <= 1
        ):
            return None
        points = module3_output.network_poincare
        batch_size, network_count, dim = points.shape
        flat_output = self.hpec_module4(points.reshape(batch_size * network_count, dim))
        network_energy = flat_output.energy_matrix.reshape(batch_size, network_count, -1)
        attention = module3_output.network_attention
        if attention.ndim == 1:
            attention = attention.unsqueeze(0).expand(batch_size, -1)
        attention = attention.to(device=network_energy.device, dtype=network_energy.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.hpec_network_energy_mode == "class_softmin":
            temperature = max(float(self.hpec_network_energy_temperature), 1e-6)
            selector_energy = network_energy
            if self.hpec_network_energy_normalize:
                # 每个样本、每个类别内部只比较不同子网络的相对能量，
                # 避免全局能量尺度过大时 softmin 退化为平均池化。
                energy_center = network_energy.mean(dim=1, keepdim=True)
                energy_scale = network_energy.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                selector_energy = (network_energy - energy_center) / energy_scale
            sharpness = max(float(self.hpec_network_selector_sharpness), 1e-6)
            selector_logits = -sharpness * selector_energy / temperature
            prior_weight = max(float(self.hpec_network_energy_prior_weight), 0.0)
            if prior_weight > 0:
                selector_logits = selector_logits + prior_weight * attention.clamp_min(1e-8).log().unsqueeze(-1)
            selector = torch.softmax(selector_logits, dim=1)
            pooled_energy = (selector * network_energy).sum(dim=1)
            selector_entropy = -(
                selector * selector.clamp_min(1e-8).log()
            ).sum(dim=1).mean()
            selector_peak = selector.max(dim=1).values.mean()
            self.latest_aux_losses.update(
                {
                    "hpec_network_selector_entropy": selector_entropy,
                    "hpec_network_selector_peak": selector_peak,
                    "hpec_network_energy_temperature": torch.as_tensor(
                        temperature,
                        device=network_energy.device,
                        dtype=network_energy.dtype,
                    ),
                    "hpec_network_energy_class_softmin": torch.ones(
                        (),
                        device=network_energy.device,
                        dtype=network_energy.dtype,
                    ),
                    "hpec_network_energy_normalized": torch.as_tensor(
                        float(self.hpec_network_energy_normalize),
                        device=network_energy.device,
                        dtype=network_energy.dtype,
                    ),
                    "hpec_network_selector_sharpness": torch.as_tensor(
                        sharpness,
                        device=network_energy.device,
                        dtype=network_energy.dtype,
                    ),
                }
            )
        else:
            pooled_energy = torch.einsum("bg,bgc->bc", attention, network_energy)
        self.latest_aux_losses.update(
            {
                "hpec_network_energy_mean": pooled_energy.mean(),
                "hpec_network_energy_std": pooled_energy.std(unbiased=False),
                "hpec_network_attention_max": attention.max(dim=-1)[0].mean(),
            }
        )
        return pooled_energy

    def _compose_prediction_logits(self, module4_output, module3_output):
        """组合模块 4 energy logits 与模块 3 切空间 logits，作为最终分类输出。"""

        if module4_output is None:
            return None
        energy_matrix = module4_output.energy_matrix
        network_energy = self._network_hpec_energy(module3_output)
        if network_energy is not None and network_energy.shape == energy_matrix.shape:
            weight = max(float(self.hpec_network_energy_weight), 0.0)
            energy_matrix = (1.0 - weight) * energy_matrix + weight * network_energy
            self.latest_aux_losses["hpec_network_energy_weight"] = torch.as_tensor(
                weight,
                device=energy_matrix.device,
                dtype=energy_matrix.dtype,
            )
        hpec_logits = -energy_matrix
        prototype_logits = None
        if hasattr(module4_output, "prototype_similarity"):
            temperature = max(self.hpec_logit_temperature, 1e-6)
            prototype_logits = torch.logsumexp(
                module4_output.prototype_similarity / temperature,
                dim=-1,
            ) * temperature
            prototype_logits = prototype_logits - prototype_logits.mean(dim=-1, keepdim=True)
        energy_centered_for_diag = hpec_logits - hpec_logits.mean(dim=-1, keepdim=True)
        if hpec_logits.ndim == 2 and hpec_logits.shape[-1] == 2:
            energy_margin_for_diag = energy_centered_for_diag[:, 1] - energy_centered_for_diag[:, 0]
        else:
            energy_margin_for_diag = (
                energy_centered_for_diag.max(dim=-1).values
                - energy_centered_for_diag.min(dim=-1).values
            )
        self.latest_aux_losses.update(
            {
                "hpec_energy_margin_mean": energy_margin_for_diag.abs().mean().detach(),
                "hpec_energy_margin_signed_mean": energy_margin_for_diag.mean().detach(),
                "hpec_energy_logit_abs_mean": energy_centered_for_diag.abs().mean().detach(),
            }
        )
        if prototype_logits is not None:
            if prototype_logits.ndim == 2 and prototype_logits.shape[-1] == 2:
                proto_margin_for_diag = prototype_logits[:, 1] - prototype_logits[:, 0]
            else:
                proto_margin_for_diag = (
                    prototype_logits.max(dim=-1).values
                    - prototype_logits.min(dim=-1).values
                )
            self.latest_aux_losses.update(
                {
                    "hpec_prototype_logit_margin_mean": proto_margin_for_diag.abs().mean().detach(),
                    "hpec_prototype_logit_signed_margin_mean": proto_margin_for_diag.mean().detach(),
                    "hpec_prototype_logit_abs_mean": prototype_logits.abs().mean().detach(),
                }
            )
        distance_logits = getattr(module4_output, "prototype_distance_logits", None)
        if distance_logits is not None:
            distance_margin = (
                distance_logits[:, 1] - distance_logits[:, 0]
                if distance_logits.ndim == 2 and distance_logits.shape[-1] == 2
                else distance_logits.max(dim=-1).values - distance_logits.min(dim=-1).values
            )
            self.latest_aux_losses.update(
                {
                    "hpec_distance_logit_margin_mean": distance_margin.abs().mean().detach(),
                    "hpec_distance_logit_signed_margin_mean": distance_margin.mean().detach(),
                    "hpec_distance_logit_abs_mean": distance_logits.abs().mean().detach(),
                }
            )
        if self.hpec_classification_mode == "distance_prototype" and distance_logits is not None:
            self.latest_aux_losses["hpec_distance_prototype_primary"] = torch.ones(
                (),
                device=distance_logits.device,
                dtype=distance_logits.dtype,
            )
            distance_evidence = -distance_logits
            self.latest_prediction_logits = distance_evidence
            return distance_evidence
        if self.hpec_classification_mode == "prototype_primary" and prototype_logits is not None:
            energy_centered = hpec_logits - hpec_logits.mean(dim=-1, keepdim=True)
            energy_scale = energy_centered.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
            blend = min(max(float(self.hpec_prototype_energy_blend), 0.0), 1.0)
            if self.hpec_prototype_logit_mode == "margin_preserving":
                proto_scale_value = max(float(self.hpec_prototype_logit_scale), 1e-6)
                prototype_logits = prototype_logits * proto_scale_value
                if blend > 0:
                    prototype_logits = (
                        (1.0 - blend) * prototype_logits
                        + blend * (energy_centered / energy_scale)
                    )
            else:
                proto_scale = prototype_logits.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
                prototype_logits = (
                    (1.0 - blend) * (prototype_logits / proto_scale)
                    + blend * (energy_centered / energy_scale)
                )
                proto_scale_value = float(proto_scale.detach().mean().item())
            self.latest_aux_losses["hpec_prototype_primary"] = torch.ones(
                (),
                device=prototype_logits.device,
                dtype=prototype_logits.dtype,
            )
            self.latest_aux_losses["hpec_prototype_energy_blend"] = torch.as_tensor(
                blend,
                device=prototype_logits.device,
                dtype=prototype_logits.dtype,
            )
            proto_margin = prototype_logits.max(dim=-1).values - prototype_logits.min(dim=-1).values
            self.latest_aux_losses["hpec_prototype_logit_margin_mean"] = proto_margin.mean().detach()
            self.latest_aux_losses["hpec_prototype_logit_abs_mean"] = prototype_logits.abs().mean().detach()
            self.latest_aux_losses["hpec_prototype_logit_mode_margin_preserving"] = torch.as_tensor(
                1.0 if self.hpec_prototype_logit_mode == "margin_preserving" else 0.0,
                device=prototype_logits.device,
                dtype=prototype_logits.dtype,
            )
            self.latest_aux_losses["hpec_prototype_logit_scale_value"] = torch.as_tensor(
                proto_scale_value,
                device=prototype_logits.device,
                dtype=prototype_logits.dtype,
            )
            self.latest_prediction_logits = prototype_logits
            return prototype_logits
        energy_mode = str(getattr(self.hpec_module4, "energy_mode", "")).lower()
        skip_duplicate_busemann_evidence = (
            self.hpec_avoid_busemann_double_count
            and energy_mode == "busemann"
            and self.hpec_classification_mode == "energy_primary"
        )
        if (
            self.hpec_evidence_weight > 0
            and hasattr(module4_output, "prototype_similarity")
            and not skip_duplicate_busemann_evidence
        ):
            evidence_logits = prototype_logits
            hpec_logits = hpec_logits + self.hpec_evidence_weight * evidence_logits
        if skip_duplicate_busemann_evidence:
            self.latest_aux_losses["hpec_skip_duplicate_busemann_evidence"] = torch.ones(
                (),
                device=hpec_logits.device,
                dtype=hpec_logits.dtype,
            )
            if self.hpec_evidence_weight > 0 and distance_logits is not None:
                # distance_logits 在 HPEC 层中实际保存的是中心化后的 geodesic distance energy；
                # 取负号后才是“距离越近、类别分数越高”的独立证据。
                distance_evidence = -distance_logits
                hpec_logits = hpec_logits + self.hpec_evidence_weight * distance_evidence
                self.latest_aux_losses["hpec_geodesic_secondary_evidence"] = torch.ones(
                    (),
                    device=hpec_logits.device,
                    dtype=hpec_logits.dtype,
                )
        if self.hpec_classification_mode == "energy_prototype_residual" and prototype_logits is not None:
            # Energy 负责主边界，prototype 只作为零均值残差补充类原型方向。
            # 这样模块4原型参与最终判断，但不会像 prototype_primary 那样整体接管概率校准。
            proto_centered = prototype_logits - prototype_logits.mean(dim=-1, keepdim=True)
            proto_scale = proto_centered.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
            proto_residual = proto_centered / proto_scale
            residual_weight = max(float(self.hpec_prototype_residual_weight), 0.0)
            logits = hpec_logits + residual_weight * proto_residual
            self.latest_aux_losses["hpec_energy_prototype_residual"] = torch.ones(
                (),
                device=logits.device,
                dtype=logits.dtype,
            )
            self.latest_aux_losses["hpec_prototype_residual_weight"] = torch.as_tensor(
                residual_weight,
                device=logits.device,
                dtype=logits.dtype,
            )
            self.latest_aux_losses["hpec_prototype_residual_abs_mean"] = proto_residual.abs().mean().detach()
            self.latest_prediction_logits = logits
            return logits
        logits = hpec_logits
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
            if (
                self.hpec_classification_mode == "tangent_prototype"
                and prototype_logits is not None
                and hgcn_logits.shape == prototype_logits.shape
            ):
                hgcn_centered = hgcn_logits - hgcn_logits.mean(dim=-1, keepdim=True)
                proto_centered = prototype_logits - prototype_logits.mean(dim=-1, keepdim=True)
                hgcn_scale = hgcn_centered.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
                proto_scale = proto_centered.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
                logits = (hgcn_centered / hgcn_scale) + (proto_centered / proto_scale)
                self.latest_aux_losses["hpec_tangent_prototype"] = torch.ones(
                    (),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                self.latest_prediction_logits = logits
                return logits
            if (
                self.hpec_classification_mode == "feature_fusion"
                and self.hpec_feature_classifier is not None
                and hasattr(module4_output, "prototype_similarity")
            ):
                similarity = module4_output.prototype_similarity
                sim_max = similarity.max(dim=-1).values
                sim_mean = similarity.mean(dim=-1)
                fusion_features = torch.cat(
                    [
                        module3_output.z_tangent,
                        hpec_logits,
                        sim_max,
                        sim_mean,
                    ],
                    dim=-1,
                )
                logits = self.hpec_feature_classifier(fusion_features)
                self.latest_aux_losses["hpec_feature_fusion"] = torch.ones(
                    (),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                self.latest_prediction_logits = logits
                return logits
            if (
                self.hpec_classification_mode == "energy_calibrated"
                and self.hpec_evidence_calibrator is not None
                and hasattr(module4_output, "prototype_similarity")
                and module3_output is not None
            ):
                similarity = module4_output.prototype_similarity
                sim_max = similarity.max(dim=-1).values
                sim_mean = similarity.mean(dim=-1)
                energy_centered = hpec_logits - hpec_logits.mean(dim=-1, keepdim=True)
                energy_margin = (
                    energy_centered[:, 1:2] - energy_centered[:, 0:1]
                    if energy_centered.shape[-1] == 2
                    else energy_centered.max(dim=-1, keepdim=True).values
                )
                sim_centered = sim_max - sim_max.mean(dim=-1, keepdim=True)
                sim_margin = (
                    sim_centered[:, 1:2] - sim_centered[:, 0:1]
                    if sim_centered.shape[-1] == 2
                    else sim_centered.max(dim=-1, keepdim=True).values
                )
                z_radius = torch.linalg.norm(module3_output.z_global, dim=-1, keepdim=True)
                evidence_features = torch.cat(
                    [
                        hpec_logits,
                        module4_output.energy_matrix,
                        sim_max,
                        sim_mean,
                        energy_margin,
                        sim_margin,
                        z_radius,
                    ],
                    dim=-1,
                )
                logits = self.hpec_evidence_calibrator(evidence_features)
                self.latest_aux_losses["hpec_energy_calibrated"] = torch.ones(
                    (),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                self.latest_aux_losses["hpec_energy_margin_mean"] = energy_margin.abs().mean().detach()
                self.latest_aux_losses["hpec_similarity_margin_mean"] = sim_margin.abs().mean().detach()
                self.latest_prediction_logits = logits
                return logits
            if self.hpec_classification_mode == "tangent_primary" and hgcn_logits.shape == hpec_logits.shape:
                logits = hgcn_logits
                self.latest_aux_losses["hpec_tangent_primary"] = torch.ones(
                    (),
                    device=hgcn_logits.device,
                    dtype=hgcn_logits.dtype,
                )
                self.latest_prediction_logits = logits
                return logits
            if hgcn_logits.shape == hpec_logits.shape and self.hpec_hgcn_logit_blend > 0:
                # 最终分类以 HPEC energy 为主；模块3线性头只提供小幅校准，避免绕开模块4。
                centered_hgcn = hgcn_logits - hgcn_logits.mean(dim=-1, keepdim=True)
                scale = centered_hgcn.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                gate = torch.sigmoid(self.hpec_logit_gate) * self.hpec_hgcn_logit_blend
                logits = hpec_logits + gate * (centered_hgcn / scale)
                self.latest_aux_losses["hpec_logit_gate"] = gate.detach()
        self.latest_prediction_logits = logits
        return logits

    def update_reliable_prototypes_after_step(self, labels):
        """训练 batch 更新后，独立更新或缓存 HPEC prototype 样本。"""

        if (
            not self.training
            or self.hpec_module4 is None
            or self.hpec_prototype_update_mode != "reliable_tp_ema"
            or self.latest_module3_output is None
            or self.latest_module4_output is None
            or self.latest_prediction_logits is None
        ):
            return {}
        should_update = self.current_epoch >= self.hpec_ema_start_epoch and (
            self.hpec_ema_update_epochs < 0
            or self.current_epoch < self.hpec_ema_start_epoch + self.hpec_ema_update_epochs
        )
        if not should_update:
            return {}
        stats = self.hpec_module4.update_prototypes_with_reliable_tp_ema(
            self.latest_module3_output.z_global,
            labels,
            self.latest_prediction_logits,
            energy_per_proto=self.latest_module4_output.energy_per_proto,
        )
        if stats:
            self.latest_aux_losses.update(stats)
        return stats

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

    def _apply_hyperbolic_logit_residual(self):
        """将欧氏局部结构 evidence 与双曲原型 evidence 在 logit 空间融合。

        函数名和配置名沿用 residual，是为了兼容已有实验脚本；论文叙事中应理解为
        dual-view evidence fusion。这样既不把欧氏 FC embedding 直接注入双曲切空间，
        也不把 HPEC 写成主从式附属分支。训练、验证和推理都复用此融合，
        避免 compute_primary_loss 重算 logits 时覆盖 forward 里的融合结果。
        """
        module3_output = self.latest_module3_output
        if self.latest_prediction_logits is not None:
            self.latest_module34_branch_logits = self.latest_prediction_logits
        gcn_output = self.latest_gcn_fallback_output
        if (
            module3_output is None
            or gcn_output is None
            or self.hyperbolic_logit_residual_weight <= 0
            or self.latest_prediction_logits is None
        ):
            return self.latest_prediction_logits

        base_logits = gcn_output.logits
        hyper_logits = self.latest_prediction_logits
        if (
            self.hyperbolic_residual_source == "tangent"
            and module3_output is not None
            and self.hgcn_classifier is not None
        ):
            hyper_logits = self.hgcn_classifier(module3_output.z_tangent)
        if base_logits.ndim == 2 and base_logits.shape[-1] == 1:
            if hyper_logits.ndim == 2 and hyper_logits.shape[-1] == 2:
                hyper_logits = hyper_logits[:, 1:2] - hyper_logits[:, 0:1]
        elif (
            base_logits.ndim == 2
            and base_logits.shape[-1] == 2
            and hyper_logits.ndim == 2
            and hyper_logits.shape[-1] == 1
        ):
            hyper_logits = torch.cat([torch.zeros_like(hyper_logits), hyper_logits], dim=-1)
        if hyper_logits.shape != base_logits.shape:
            return self.latest_prediction_logits
        if (
            self.hpec_residual_calibration in (
                "batch_margin",
                "tanh_margin",
                "running_batch_margin",
                "hybrid_batch_running_margin",
                "train_class_margin",
            )
            and hyper_logits.ndim == 2
            and hyper_logits.shape[-1] == 2
        ):
            # HPEC/prototype 的原始 margin 往往很小，直接作为残差时容易被 FC 基底淹没。
            # running_batch_margin 在测试期复用训练期 EMA 统计，避免测试 batch 自身参与校准。
            margin = hyper_logits[:, 1:2] - hyper_logits[:, 0:1]
            self.latest_hpec_raw_margin = margin
            batch_mean = margin.mean(dim=0, keepdim=True)
            batch_centered = margin - batch_mean
            batch_std = batch_centered.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            if self.hpec_residual_calibration == "tanh_margin":
                margin_scale = torch.as_tensor(
                    max(float(self.hyperbolic_residual_temperature), 1e-6),
                    device=margin.device,
                    dtype=margin.dtype,
                ).view(1, 1)
                calibrated_margin = (
                    torch.tanh(margin / margin_scale)
                    * max(float(self.hpec_residual_calibration_scale), 0.0)
                )
            elif self.hpec_residual_calibration == "train_class_margin":
                class_ready = torch.all(self.hpec_residual_class_margin_ready[:2] > 0.5)
                if class_ready:
                    class_means = self.hpec_residual_class_margin_mean[:2].to(
                        device=margin.device,
                        dtype=margin.dtype,
                    )
                    center = class_means.mean().view(1, 1)
                    class_half_gap = 0.5 * (class_means[1] - class_means[0]).abs()
                    running_scale = self.hpec_residual_running_margin_std.to(
                        device=margin.device,
                        dtype=margin.dtype,
                    ).view(1, 1)
                    margin_scale = torch.maximum(
                        running_scale,
                        class_half_gap.view(1, 1),
                    ).clamp_min(1e-4)
                else:
                    center = batch_mean
                    margin_scale = batch_std
                calibrated_margin = (
                    (margin - center)
                    / margin_scale
                    * max(float(self.hpec_residual_calibration_scale), 0.0)
                )
            elif self.hpec_residual_calibration in ("running_batch_margin", "hybrid_batch_running_margin"):
                if self.training:
                    with torch.no_grad():
                        momentum = min(max(float(self.hpec_residual_calibration_momentum), 0.0), 1.0)
                        running_mean = self.hpec_residual_running_margin_mean.to(
                            device=margin.device,
                            dtype=margin.dtype,
                        )
                        running_std = self.hpec_residual_running_margin_std.to(
                            device=margin.device,
                            dtype=margin.dtype,
                        )
                        if self.hpec_residual_running_margin_ready.item() < 0.5:
                            updated_mean = batch_mean.detach().reshape_as(running_mean)
                            updated_std = batch_std.detach().reshape_as(running_std)
                            self.hpec_residual_running_margin_ready.fill_(1.0)
                        else:
                            updated_mean = (
                                (1.0 - momentum) * running_mean
                                + momentum * batch_mean.detach().reshape_as(running_mean)
                            )
                            updated_std = (
                                (1.0 - momentum) * running_std
                                + momentum * batch_std.detach().reshape_as(running_std)
                            )
                        self.hpec_residual_running_margin_mean.copy_(updated_mean)
                        self.hpec_residual_running_margin_std.copy_(updated_std.clamp_min(1e-6))
                    center = batch_mean
                    margin_scale = batch_std
                elif self.hpec_residual_running_margin_ready.item() > 0.5:
                    running_center = self.hpec_residual_running_margin_mean.to(
                        device=margin.device,
                        dtype=margin.dtype,
                    ).view(1, 1)
                    running_scale = self.hpec_residual_running_margin_std.to(
                        device=margin.device,
                        dtype=margin.dtype,
                    ).view(1, 1).clamp_min(1e-6)
                    if self.hpec_residual_calibration == "hybrid_batch_running_margin":
                        batch_weight = min(max(float(self.hpec_residual_calibration_batch_weight), 0.0), 1.0)
                        center = (1.0 - batch_weight) * running_center + batch_weight * batch_mean
                        margin_scale = (1.0 - batch_weight) * running_scale + batch_weight * batch_std
                    else:
                        center = running_center
                        margin_scale = running_scale
                else:
                    center = batch_mean
                    margin_scale = batch_std
                margin_centered = margin - center
                calibrated_margin = (
                    margin_centered
                    / margin_scale
                    * max(float(self.hpec_residual_calibration_scale), 0.0)
                )
            else:
                margin_centered = batch_centered
                margin_scale = batch_std
                calibrated_margin = (
                    margin_centered
                    / margin_scale
                    * max(float(self.hpec_residual_calibration_scale), 0.0)
                )
            hyper_logits = torch.cat([-0.5 * calibrated_margin, 0.5 * calibrated_margin], dim=-1)
            self.latest_aux_losses.update(
                {
                    "hpec_residual_calibration_batch_margin": torch.ones(
                        (),
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ),
                    "hpec_residual_calibration_running_batch_margin": torch.as_tensor(
                        1.0 if self.hpec_residual_calibration == "running_batch_margin" else 0.0,
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ),
                    "hpec_residual_calibration_hybrid_batch_running_margin": torch.as_tensor(
                        1.0 if self.hpec_residual_calibration == "hybrid_batch_running_margin" else 0.0,
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ),
                    "hpec_residual_calibration_tanh_margin": torch.as_tensor(
                        1.0 if self.hpec_residual_calibration == "tanh_margin" else 0.0,
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ),
                    "hpec_residual_calibration_train_class_margin": torch.as_tensor(
                        1.0 if self.hpec_residual_calibration == "train_class_margin" else 0.0,
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ),
                    "hpec_residual_calibration_batch_weight": torch.as_tensor(
                        min(max(float(self.hpec_residual_calibration_batch_weight), 0.0), 1.0),
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ),
                    "hpec_residual_calibrated_margin_abs_mean": calibrated_margin.abs().mean().detach(),
                    "hpec_residual_raw_margin_abs_mean": margin.abs().mean().detach(),
                    "hpec_residual_raw_margin_std": batch_std.squeeze().detach(),
                    "hpec_residual_running_margin_mean": self.hpec_residual_running_margin_mean.to(
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ).squeeze().detach(),
                    "hpec_residual_running_margin_std": self.hpec_residual_running_margin_std.to(
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ).squeeze().detach(),
                    "hpec_residual_class_margin_gap": (
                        self.hpec_residual_class_margin_mean[1]
                        - self.hpec_residual_class_margin_mean[0]
                    ).abs().to(device=hyper_logits.device, dtype=hyper_logits.dtype).detach(),
                }
            )
        if self.hyperbolic_residual_bias is not None:
            bias = self.hyperbolic_residual_bias.to(device=hyper_logits.device, dtype=hyper_logits.dtype)
            view_shape = (1,) * (hyper_logits.ndim - 1) + (bias.numel(),)
            if hyper_logits.shape[-1] == bias.numel():
                hyper_logits = hyper_logits + bias.view(view_shape)
                self.latest_aux_losses["hyperbolic_residual_bias_mean"] = bias.mean().detach()
                self.latest_aux_losses["hyperbolic_residual_bias_abs_mean"] = bias.abs().mean().detach()

        if self.hyperbolic_residual_fusion_mode in ("dual_consensus", "dual_margin_consensus"):
            # 双视角共识：欧氏局部结构分支和双曲原型分支平等进入最终 logits。
            # 在该模式下，hyperbolic_logit_residual_weight 表示双曲视角最低占比，
            # 避免完整模型训练时模块3/4被 fallback 分支完全掩盖。
            hyper_centered = hyper_logits - hyper_logits.mean(dim=-1, keepdim=True)
            hyper_centered = hyper_logits - hyper_logits.mean(dim=-1, keepdim=True)
            base_centered = base_logits - base_logits.mean(dim=-1, keepdim=True)
            binary_margin_fusion = (
                base_centered.ndim == 2
                and hyper_centered.ndim == 2
                and base_centered.shape[-1] == 2
                and hyper_centered.shape[-1] == 2
            )
            margin_only_consensus = (
                self.hyperbolic_residual_fusion_mode == "dual_margin_consensus"
                and binary_margin_fusion
            )
            if margin_only_consensus:
                hyper_margin = hyper_centered[:, 1:2] - hyper_centered[:, 0:1]
                base_margin = base_centered[:, 1:2] - base_centered[:, 0:1]
                if self.hyperbolic_residual_norm == "sample":
                    hyper_scale = hyper_margin.detach().abs().mean(dim=0, keepdim=True).clamp_min(1e-6)
                    base_scale = base_margin.detach().abs().mean(dim=0, keepdim=True).clamp_min(1e-6)
                    hyper_margin = torch.tanh(hyper_margin / hyper_scale)
                    base_margin = torch.tanh(base_margin / base_scale)
                    self.latest_aux_losses["hyperbolic_dual_consensus_batch_margin_norm"] = torch.ones(
                        (),
                        device=base_logits.device,
                        dtype=base_logits.dtype,
                    )
                    self.latest_aux_losses["hyperbolic_dual_consensus_hyper_margin_std"] = hyper_scale.squeeze().detach()
                    self.latest_aux_losses["hyperbolic_dual_consensus_base_margin_std"] = base_scale.squeeze().detach()
                elif self.hyperbolic_residual_norm == "temperature":
                    temperature = max(float(self.hyperbolic_residual_temperature), 1e-6)
                    hyper_margin = hyper_margin / temperature
                min_hyper = min(max(float(self.hyperbolic_logit_residual_weight), 0.0), 1.0)
                if self.use_hyperbolic_residual_gate:
                    base_for_gate = torch.cat([-0.5 * base_margin, 0.5 * base_margin], dim=-1)
                    hyper_for_gate = torch.cat([-0.5 * hyper_margin, 0.5 * hyper_margin], dim=-1)
                    confidence_gate = self._hyperbolic_residual_gate(base_for_gate, hyper_for_gate)
                    hyper_weight = min_hyper + (1.0 - min_hyper) * confidence_gate
                else:
                    hyper_weight = torch.full_like(base_margin, min_hyper)
                fused_margin = (1.0 - hyper_weight) * base_margin + hyper_weight * hyper_margin
                y_hat = torch.cat([-0.5 * fused_margin, 0.5 * fused_margin], dim=-1)
                y_hat = y_hat + base_logits.mean(dim=-1, keepdim=True)
                self.latest_prediction_logits = y_hat
                self.latest_module34_branch_logits = torch.cat(
                    [-0.5 * hyper_margin, 0.5 * hyper_margin],
                    dim=-1,
                )
                self.latest_graph_path = "gcn_fallback_plus_hgcn_hpec_dual_margin_consensus"
                self.latest_aux_losses.update(
                    {
                        "hyperbolic_dual_margin_consensus": torch.ones(
                            (),
                            device=y_hat.device,
                            dtype=y_hat.dtype,
                        ),
                        "hyperbolic_consensus_weight_mean": hyper_weight.mean().detach(),
                        "hyperbolic_consensus_weight_min": hyper_weight.min().detach(),
                        "hyperbolic_consensus_weight_max": hyper_weight.max().detach(),
                        "hyperbolic_logit_residual_weight": torch.as_tensor(
                            min_hyper,
                            device=y_hat.device,
                            dtype=y_hat.dtype,
                        ),
                        "hyperbolic_update_logit_abs_mean": hyper_margin.abs().mean().detach(),
                    }
                )
                self.latest_aux_losses["final_positive_prob_mean"] = F.softmax(y_hat, dim=-1)[:, 1].mean().detach()
                return y_hat
            if self.hyperbolic_residual_norm == "sample" and binary_margin_fusion:
                # 二分类时逐样本 std 会把每个样本都压成固定幅度。
                # 用 batch margin 尺度归一化，保留样本间正负类证据强弱。
                hyper_margin = hyper_centered[:, 1:2] - hyper_centered[:, 0:1]
                base_margin = base_centered[:, 1:2] - base_centered[:, 0:1]
                hyper_scale = hyper_margin.detach().abs().mean(dim=0, keepdim=True).clamp_min(1e-6)
                base_scale = base_margin.detach().abs().mean(dim=0, keepdim=True).clamp_min(1e-6)
                hyper_margin = torch.tanh(hyper_margin / hyper_scale)
                base_margin = torch.tanh(base_margin / base_scale)
                hyper_for_fusion = torch.cat([-0.5 * hyper_margin, 0.5 * hyper_margin], dim=-1)
                base_for_fusion = torch.cat([-0.5 * base_margin, 0.5 * base_margin], dim=-1)
                self.latest_aux_losses["hyperbolic_dual_consensus_batch_margin_norm"] = torch.ones(
                    (),
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )
                self.latest_aux_losses["hyperbolic_dual_consensus_hyper_margin_std"] = hyper_scale.squeeze().detach()
                self.latest_aux_losses["hyperbolic_dual_consensus_base_margin_std"] = base_scale.squeeze().detach()
            elif self.hyperbolic_residual_norm == "sample":
                hyper_scale = hyper_centered.detach().std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                base_scale = base_centered.detach().std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                hyper_for_fusion = hyper_centered / hyper_scale
                base_for_fusion = base_centered / base_scale
            elif self.hyperbolic_residual_norm == "temperature":
                temperature = max(float(self.hyperbolic_residual_temperature), 1e-6)
                hyper_for_fusion = hyper_centered / temperature
                base_for_fusion = base_centered
            else:
                hyper_for_fusion = hyper_centered
                base_for_fusion = base_centered
            # 模块 3/4 的独立分支指标与辅助 CE 使用同一份双曲证据。
            # 否则最终融合用的是标定后 margin，branch 却仍评估未标定 HPEC logits，
            # 会误判模块 3/4 是否真正具备分类能力。
            self.latest_module34_branch_logits = hyper_for_fusion
            min_hyper = min(max(float(self.hyperbolic_logit_residual_weight), 0.0), 1.0)
            if self.use_hyperbolic_residual_gate:
                confidence_gate = self._hyperbolic_residual_gate(base_for_fusion, hyper_for_fusion)
                hyper_weight = min_hyper + (1.0 - min_hyper) * confidence_gate
            else:
                hyper_weight = torch.full(
                    base_logits[..., :1].shape,
                    min_hyper,
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )
            y_hat = (1.0 - hyper_weight) * base_for_fusion + hyper_weight * hyper_for_fusion
            if binary_margin_fusion:
                # 只融合判别 margin，保留欧氏分支学到的类别先验/校准偏置。
                # 否则 centered logits 会让测试概率整体严重偏向同一类。
                y_hat = y_hat + base_logits.mean(dim=-1, keepdim=True)
            self.latest_prediction_logits = y_hat
            self.latest_graph_path = "gcn_fallback_plus_hgcn_hpec_dual_consensus"
            self.latest_aux_losses.update(
                {
                    "hyperbolic_dual_consensus": torch.ones(
                        (),
                        device=y_hat.device,
                        dtype=y_hat.dtype,
                    ),
                    "hyperbolic_consensus_weight_mean": hyper_weight.mean().detach(),
                    "hyperbolic_consensus_weight_min": hyper_weight.min().detach(),
                    "hyperbolic_consensus_weight_max": hyper_weight.max().detach(),
                    "hyperbolic_logit_residual_weight": torch.as_tensor(
                        min_hyper,
                        device=y_hat.device,
                        dtype=y_hat.dtype,
                    ),
                    "hyperbolic_update_logit_abs_mean": hyper_for_fusion.abs().mean().detach(),
                }
            )
            if y_hat.ndim == 2 and y_hat.shape[-1] == 2:
                self.latest_aux_losses["final_positive_prob_mean"] = F.softmax(y_hat, dim=-1)[:, 1].mean().detach()
            return y_hat

        if self.hyperbolic_residual_fusion_mode in ("logit_blend", "binary_margin"):
            if hyper_logits.ndim == 2 and hyper_logits.shape[-1] == 1:
                hyper_centered = hyper_logits - hyper_logits.mean().detach()
            elif self.hyperbolic_residual_fusion_mode == "binary_margin" and hyper_logits.shape[-1] == 2:
                # 二分类时只取“正类相对负类”的 HPEC margin，避免同时推高 class0 并压低正类概率。
                hyper_centered = hyper_logits[:, 1:2] - hyper_logits[:, 0:1]
            else:
                hyper_centered = hyper_logits - hyper_logits.mean(dim=-1, keepdim=True)
            if self.hyperbolic_residual_norm == "sample":
                scale = hyper_centered.detach().std().clamp_min(1e-6)
                hyper_update = hyper_centered / scale
            elif self.hyperbolic_residual_norm == "temperature":
                temperature = max(float(self.hyperbolic_residual_temperature), 1e-6)
                hyper_update = hyper_centered / temperature
            else:
                hyper_update = hyper_centered
            residual_gate = self._hyperbolic_residual_gate(base_logits, hyper_logits)
            if self.hyperbolic_residual_fusion_mode == "binary_margin" and base_logits.ndim == 2 and base_logits.shape[-1] == 2:
                class_update = torch.cat([torch.zeros_like(hyper_update), hyper_update], dim=-1)
                y_hat = base_logits + self.hyperbolic_logit_residual_weight * residual_gate * class_update
            elif hyper_update.ndim == 2 and hyper_update.shape[-1] == 2 and base_logits.shape[-1] == 2:
                y_hat = base_logits + self.hyperbolic_logit_residual_weight * residual_gate * hyper_update
            else:
                y_hat = base_logits + self.hyperbolic_logit_residual_weight * residual_gate * hyper_update
            self.latest_prediction_logits = y_hat
            self.latest_aux_losses["hyperbolic_logit_residual_weight"] = torch.as_tensor(
                self.hyperbolic_logit_residual_weight,
                device=y_hat.device,
                dtype=y_hat.dtype,
            )
            self.latest_aux_losses["hyperbolic_residual_logit_blend"] = torch.ones(
                (),
                device=y_hat.device,
                dtype=y_hat.dtype,
            )
            self.latest_aux_losses["hyperbolic_residual_binary_margin"] = torch.as_tensor(
                1.0 if self.hyperbolic_residual_fusion_mode == "binary_margin" else 0.0,
                device=y_hat.device,
                dtype=y_hat.dtype,
            )
            self.latest_aux_losses["hyperbolic_residual_gate_mean"] = residual_gate.mean().detach()
            self.latest_aux_losses["hyperbolic_residual_gate_min"] = residual_gate.min().detach()
            self.latest_aux_losses["hyperbolic_residual_gate_max"] = residual_gate.max().detach()
            self.latest_aux_losses["hyperbolic_residual_norm_mean"] = hyper_update.abs().mean().detach()
            self.latest_aux_losses["hyperbolic_update_logit_mean"] = hyper_update.mean().detach()
            self.latest_aux_losses["hyperbolic_update_logit_abs_mean"] = hyper_update.abs().mean().detach()
            if y_hat.ndim == 2 and y_hat.shape[-1] == 1:
                self.latest_aux_losses["final_positive_prob_mean"] = torch.sigmoid(y_hat).mean().detach()
            elif y_hat.ndim == 2 and y_hat.shape[-1] == 2:
                self.latest_aux_losses["final_positive_prob_mean"] = F.softmax(y_hat, dim=-1)[:, 1].mean().detach()
            return y_hat

        if hyper_logits.ndim == 2 and hyper_logits.shape[-1] == 1:
            residual = torch.tanh(hyper_logits)
        else:
            centered = hyper_logits - hyper_logits.mean(dim=-1, keepdim=True)
            if self.hyperbolic_residual_norm == "sample":
                scale = centered.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
                residual_input = centered / scale
            elif self.hyperbolic_residual_norm == "temperature":
                temperature = max(float(self.hyperbolic_residual_temperature), 1e-6)
                residual_input = centered / temperature
            else:
                residual_input = centered
            residual = torch.tanh(residual_input)

        residual_gate = self._hyperbolic_residual_gate(base_logits, hyper_logits)
        residual_scale = torch.ones_like(residual_gate)
        if self.hyperbolic_residual_margin_gain > 0 and hyper_logits.ndim == 2 and hyper_logits.shape[-1] > 1:
            hyper_prob = F.softmax(hyper_logits, dim=-1)
            top2 = torch.topk(hyper_prob, k=min(2, hyper_prob.shape[-1]), dim=-1).values
            margin = top2[..., :1] if top2.shape[-1] == 1 else (top2[..., :1] - top2[..., 1:2]).clamp_min(0.0)
            max_scale = max(float(self.hyperbolic_residual_margin_max_scale), 1.0)
            residual_scale = (1.0 + self.hyperbolic_residual_margin_gain * margin).clamp(max=max_scale)
            self.latest_aux_losses["hyperbolic_residual_margin_scale_mean"] = residual_scale.mean().detach()
            self.latest_aux_losses["hyperbolic_residual_margin_scale_max"] = residual_scale.max().detach()
        y_hat = base_logits + self.hyperbolic_logit_residual_weight * residual_gate * residual_scale * residual
        self.latest_prediction_logits = y_hat
        self.latest_aux_losses["hyperbolic_logit_residual_weight"] = torch.as_tensor(
            self.hyperbolic_logit_residual_weight,
            device=y_hat.device,
            dtype=y_hat.dtype,
        )
        self.latest_aux_losses["hyperbolic_residual_gate_mean"] = residual_gate.mean().detach()
        self.latest_aux_losses["hyperbolic_residual_gate_min"] = residual_gate.min().detach()
        self.latest_aux_losses["hyperbolic_residual_gate_max"] = residual_gate.max().detach()
        self.latest_aux_losses["hyperbolic_residual_norm_mean"] = residual.abs().mean().detach()
        if y_hat.ndim == 2 and y_hat.shape[-1] == 1:
            self.latest_aux_losses["final_positive_prob_mean"] = torch.sigmoid(y_hat).mean().detach()
        elif y_hat.ndim == 2 and y_hat.shape[-1] == 2:
            self.latest_aux_losses["final_positive_prob_mean"] = F.softmax(y_hat, dim=-1)[:, 1].mean().detach()
        return y_hat

    def _hyperbolic_residual_gate(self, base_logits, hyper_logits):
        """根据模块4证据强弱生成逐样本残差门控。"""
        gate_shape = base_logits[..., :1].shape if base_logits.ndim > 1 else base_logits.reshape(-1, 1).shape
        if not self.use_hyperbolic_residual_gate:
            return torch.ones(gate_shape, device=base_logits.device, dtype=base_logits.dtype)

        base_two = self._logits_for_distillation(base_logits)
        hyper_two = self._logits_for_distillation(hyper_logits)
        if base_two is None or hyper_two is None or base_two.shape != hyper_two.shape or hyper_two.shape[-1] < 2:
            return torch.ones(gate_shape, device=base_logits.device, dtype=base_logits.dtype)

        hyper_prob = F.softmax(hyper_two, dim=-1)
        top2 = torch.topk(hyper_prob, k=min(2, hyper_prob.shape[-1]), dim=-1).values
        margin = top2[..., :1] if top2.shape[-1] == 1 else (top2[..., :1] - top2[..., 1:2]).clamp_min(0.0)
        agreement = (torch.argmax(base_two, dim=-1) == torch.argmax(hyper_two, dim=-1)).to(base_logits.dtype).unsqueeze(-1)

        if self.hyperbolic_residual_gate_mode == "agreement":
            # 模块4/双曲分支与 FC 基底方向一致时才放大残差；不一致时保留最低贡献。
            # 这样模块4仍参与决策，但不会在证据相反时强行拉坏 raw 0.5 边界。
            signed_agreement = 2.0 * agreement - 1.0
            raw_gate = torch.sigmoid(
                self.hyperbolic_residual_gate_gain * (signed_agreement + margin - 0.35)
                + self.hyperbolic_residual_gate_bias
            )
        elif self.hyperbolic_residual_gate_mode == "consensus":
            base_prob = F.softmax(base_two, dim=-1)
            base_top2 = torch.topk(base_prob, k=min(2, base_prob.shape[-1]), dim=-1).values
            base_margin = (
                base_top2[..., :1]
                if base_top2.shape[-1] == 1
                else (base_top2[..., :1] - base_top2[..., 1:2]).clamp_min(0.0)
            )
            consensus_margin = torch.minimum(base_margin, margin)
            # 双视角共识门控：只有欧氏局部图分支和双曲原型分支方向一致、且双方都有一定
            # margin 时，才提高模块3/4占比；相反时保留最低贡献，避免弱证据硬拉最终边界。
            raw_gate = torch.sigmoid(
                self.hyperbolic_residual_gate_gain * (agreement * consensus_margin - 0.20)
                + self.hyperbolic_residual_gate_bias
            )
            self.latest_aux_losses["hyperbolic_residual_base_margin_mean"] = base_margin.mean().detach()
            self.latest_aux_losses["hyperbolic_residual_consensus_margin_mean"] = (
                consensus_margin.mean().detach()
            )
        else:
            raw_gate = torch.sigmoid(
                self.hyperbolic_residual_gate_gain * (margin + 0.25 * agreement - 0.5)
                + self.hyperbolic_residual_gate_bias
            )
        gate = self.hyperbolic_residual_gate_min + (
            self.hyperbolic_residual_gate_max - self.hyperbolic_residual_gate_min
        ) * raw_gate
        self.latest_aux_losses["hyperbolic_residual_margin_mean"] = margin.mean().detach()
        self.latest_aux_losses["hyperbolic_residual_agreement"] = agreement.mean().detach()
        self.latest_aux_losses["hyperbolic_residual_gate_mode_agreement"] = torch.as_tensor(
            1.0 if self.hyperbolic_residual_gate_mode == "agreement" else 0.0,
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        self.latest_aux_losses["hyperbolic_residual_gate_mode_consensus"] = torch.as_tensor(
            1.0 if self.hyperbolic_residual_gate_mode == "consensus" else 0.0,
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        return gate

    def _logits_for_distillation(self, logits):
        if logits is None:
            return None
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if logits.ndim == 2 and logits.shape[-1] == 1:
            return torch.cat([torch.zeros_like(logits), logits], dim=-1)
        return logits

    def _apply_final_logit_calibration(self, logits):
        """训练内学习最终 logit 的尺度与偏置，缓解不同 fold 的概率校准漂移。"""
        if logits is None or self.final_logit_scale is None or self.final_logit_bias is None:
            return logits
        scale = F.softplus(self.final_logit_scale).view(1, -1).clamp_min(1e-4)
        bias = self.final_logit_bias.view(1, -1)
        if logits.ndim == 2 and logits.shape[-1] == scale.shape[-1]:
            calibrated = logits * scale + bias
        elif logits.ndim == 2 and logits.shape[-1] == 1 and scale.shape[-1] == 1:
            calibrated = logits * scale + bias
        else:
            return logits
        self.latest_aux_losses["final_logit_scale_mean"] = scale.mean().detach()
        self.latest_aux_losses["final_logit_bias_mean"] = bias.mean().detach()
        self.latest_prediction_logits = calibrated
        return calibrated

    def _hpec_teacher_distill_loss(self):
        if (
            not self.training
            or self.hpec_teacher_distill_weight <= 0
            or self.latest_prediction_logits is None
            or self.latest_gcn_fallback_output is None
        ):
            return None
        evidence_logits = self._logits_for_distillation(self.latest_prediction_logits)
        reference_logits = self._logits_for_distillation(self.latest_gcn_fallback_output.logits)
        if evidence_logits is None or reference_logits is None or evidence_logits.shape != reference_logits.shape:
            return None
        if self.hpec_teacher_detach:
            reference_logits = reference_logits.detach()
        temperature = max(float(self.hpec_teacher_distill_temperature), 1e-6)
        if self.hpec_teacher_distill_mode == "centered_kl":
            # Calibrate relative class structure only; remove per-sample common logit bias.
            evidence_for_loss = evidence_logits - evidence_logits.mean(dim=-1, keepdim=True)
            reference_for_loss = reference_logits - reference_logits.mean(dim=-1, keepdim=True)
            evidence_log_prob = F.log_softmax(evidence_for_loss / temperature, dim=-1)
            reference_prob = F.softmax(reference_for_loss / temperature, dim=-1)
            distill_loss = F.kl_div(evidence_log_prob, reference_prob, reduction="batchmean") * (temperature ** 2)
        elif self.hpec_teacher_distill_mode == "margin_mse":
            # Binary uses class margin calibration; multi-class falls back to centered-logit MSE.
            if evidence_logits.shape[-1] == 2:
                evidence_margin = evidence_logits[:, 1] - evidence_logits[:, 0]
                reference_margin = reference_logits[:, 1] - reference_logits[:, 0]
                distill_loss = F.smooth_l1_loss(
                    evidence_margin / temperature,
                    reference_margin / temperature,
                ) * (temperature ** 2)
            else:
                evidence_centered = evidence_logits - evidence_logits.mean(dim=-1, keepdim=True)
                reference_centered = reference_logits - reference_logits.mean(dim=-1, keepdim=True)
                distill_loss = F.smooth_l1_loss(
                    evidence_centered / temperature,
                    reference_centered / temperature,
                ) * (temperature ** 2)
            reference_prob = F.softmax(reference_logits / temperature, dim=-1)
        else:
            evidence_log_prob = F.log_softmax(evidence_logits / temperature, dim=-1)
            reference_prob = F.softmax(reference_logits / temperature, dim=-1)
            distill_loss = F.kl_div(evidence_log_prob, reference_prob, reduction="batchmean") * (temperature ** 2)
        weighted_loss = self.hpec_teacher_distill_weight * distill_loss
        reference_entropy = -(reference_prob * reference_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.latest_aux_losses.update(
            {
                "hpec_teacher_distill_loss": distill_loss,
                "hpec_teacher_distill_weighted_loss": weighted_loss,
                "hpec_teacher_entropy": reference_entropy,
                "hpec_teacher_distill_mode_centered_kl": torch.as_tensor(
                    1.0 if self.hpec_teacher_distill_mode == "centered_kl" else 0.0,
                    device=evidence_logits.device,
                    dtype=evidence_logits.dtype,
                ),
                "hpec_teacher_distill_mode_margin_mse": torch.as_tensor(
                    1.0 if self.hpec_teacher_distill_mode == "margin_mse" else 0.0,
                    device=evidence_logits.device,
                    dtype=evidence_logits.dtype,
                ),
            }
        )
        return weighted_loss

    def _prototype_similarity_ce_loss(self, module4_output, labels):
        if (
            module4_output is None
            or self.hpec_prototype_ce_loss_weight <= 0
            or not hasattr(module4_output, "prototype_similarity")
        ):
            return None
        similarity = module4_output.prototype_similarity
        temperature = max(float(self.hpec_logit_temperature), 1e-6)
        prototype_logits = torch.logsumexp(similarity / temperature, dim=-1) * temperature
        prototype_logits = prototype_logits - prototype_logits.mean(dim=-1, keepdim=True)
        label_index = labels.long().reshape(-1).to(device=prototype_logits.device)
        if prototype_logits.shape[0] != label_index.numel():
            return None
        prototype_ce_loss = self._classification_ce_loss(prototype_logits, label_index)
        weighted_loss = self.hpec_prototype_ce_loss_weight * prototype_ce_loss
        self.latest_aux_losses.update(
            {
                "hpec_prototype_ce_loss": prototype_ce_loss,
                "hpec_prototype_ce_weighted_loss": weighted_loss,
            }
        )
        return weighted_loss

    def _module34_supervised_contrastive_loss(self, labels):
        if (
            self.module34_supcon_loss_weight <= 0
            or self.latest_module3_output is None
            or self.latest_module3_output.z_tangent is None
        ):
            return None
        z = self.latest_module3_output.z_tangent
        label_index = labels.long().reshape(-1).to(device=z.device)
        if z.shape[0] != label_index.numel() or z.shape[0] < 2:
            return None

        z_norm = F.normalize(z, p=2, dim=-1)
        temperature = max(float(self.module34_supcon_temperature), 1e-6)
        logits = torch.matmul(z_norm, z_norm.T) / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
        positive_mask = (label_index[:, None] == label_index[None, :]) & (~eye)
        valid_anchor = positive_mask.any(dim=1)
        if not torch.any(valid_anchor):
            zero = z.sum() * 0.0
            self.latest_aux_losses.update(
                {
                    "module34_supcon_loss": zero,
                    "module34_supcon_weighted_loss": zero,
                    "module34_supcon_positive_pairs": zero,
                }
            )
            return zero

        exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
        log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8).log()
        positive_log_prob = (positive_mask.to(z.dtype) * log_prob).sum(dim=1)
        positive_count = positive_mask.sum(dim=1).clamp_min(1).to(z.dtype)
        loss = -(positive_log_prob / positive_count)[valid_anchor].mean()
        weighted_loss = self.module34_supcon_loss_weight * loss
        self.latest_aux_losses.update(
            {
                "module34_supcon_loss": loss,
                "module34_supcon_weighted_loss": weighted_loss,
                "module34_supcon_positive_pairs": positive_mask.sum().to(z.dtype),
            }
        )
        return weighted_loss

    def _module34_center_separation_loss(self, labels):
        if (
            self.module34_center_loss_weight <= 0
            or self.latest_module3_output is None
            or self.latest_module3_output.z_tangent is None
        ):
            return None
        z = self.latest_module3_output.z_tangent
        label_index = labels.long().reshape(-1).to(device=z.device)
        if z.shape[0] != label_index.numel() or z.shape[0] < 2:
            return None

        classes = torch.unique(label_index)
        if classes.numel() < 2:
            zero = z.sum() * 0.0
            self.latest_aux_losses.update(
                {
                    "module34_center_loss": zero,
                    "module34_center_weighted_loss": zero,
                    "module34_center_intra_loss": zero,
                    "module34_center_inter_loss": zero,
                }
            )
            return zero

        z_norm = F.normalize(z, p=2, dim=-1)
        centers = []
        intra_losses = []
        for class_id in classes:
            mask = label_index == class_id
            class_z = z_norm[mask]
            center = F.normalize(class_z.mean(dim=0), p=2, dim=-1)
            centers.append(center)
            intra_losses.append(1.0 - (class_z * center.unsqueeze(0)).sum(dim=-1).mean())
        centers = torch.stack(centers, dim=0)
        intra_loss = torch.stack(intra_losses).mean()
        center_cos = centers @ centers.T
        offdiag_mask = ~torch.eye(centers.shape[0], device=z.device, dtype=torch.bool)
        inter_cos = center_cos[offdiag_mask]
        margin = torch.as_tensor(
            self.module34_center_margin,
            device=z.device,
            dtype=z.dtype,
        )
        inter_loss = F.relu(inter_cos - margin).mean() if inter_cos.numel() else z.sum() * 0.0
        loss = (
            self.module34_center_intra_weight * intra_loss
            + self.module34_center_inter_weight * inter_loss
        )
        weighted_loss = self.module34_center_loss_weight * loss
        self.latest_aux_losses.update(
            {
                "module34_center_loss": loss,
                "module34_center_weighted_loss": weighted_loss,
                "module34_center_intra_loss": intra_loss,
                "module34_center_inter_loss": inter_loss,
                "module34_center_cos_mean": inter_cos.mean() if inter_cos.numel() else z.sum() * 0.0,
            }
        )
        return weighted_loss

    def compute_supervised_aux_loss(self, labels):
        losses = [
            self._module34_supervised_contrastive_loss(labels),
            self._module34_center_separation_loss(labels),
            self._gcn_fallback_branch_ce_loss(labels, "graph"),
            self._gcn_fallback_branch_ce_loss(labels, "fc"),
        ]
        losses = [loss for loss in losses if loss is not None]
        if not losses:
            return None
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total

    def _gcn_fallback_branch_ce_loss(self, labels, branch):
        """可选辅助监督：让 GCN 图分支或 FC 分支具备独立分类能力。

        默认权重为 0，只用于诊断；开启后不替代主分类损失。
        """

        gcn_output = self.latest_gcn_fallback_output
        if gcn_output is None:
            return None
        branch_name = str(branch).lower()
        if branch_name == "graph":
            weight = max(float(self.gcn_graph_branch_ce_loss_weight), 0.0)
            logits = getattr(gcn_output, "graph_only_logits", None)
            metric_prefix = "gcn_graph_branch"
        elif branch_name == "fc":
            weight = max(float(self.gcn_fc_branch_ce_loss_weight), 0.0)
            logits = getattr(gcn_output, "fc_only_logits", None)
            metric_prefix = "gcn_fc_branch"
        else:
            return None
        if weight <= 0 or logits is None:
            return None
        label_index = labels.long().reshape(-1)
        branch_loss = self._classification_ce_loss(logits, label_index)
        weighted_loss = weight * branch_loss
        self.latest_aux_losses.update(
            {
                f"{metric_prefix}_ce_loss": branch_loss,
                f"{metric_prefix}_ce_weighted_loss": weighted_loss,
            }
        )
        return weighted_loss

    def _module34_branch_ce_loss(self, labels):
        weight = max(float(self.module34_branch_ce_loss_weight), 0.0)
        if weight > 0 and self.module34_branch_ce_decay_epochs > 0:
            progress = min(
                max((self.current_epoch + 1) / float(self.module34_branch_ce_decay_epochs), 0.0),
                1.0,
            )
            min_ratio = min(max(float(self.module34_branch_ce_min_ratio), 0.0), 1.0)
            weight = weight * (1.0 - (1.0 - min_ratio) * progress)
        logits = self.latest_module34_branch_logits
        if weight <= 0 or logits is None:
            return None
        label_index = labels.long().reshape(-1)
        branch_loss = self._classification_ce_loss(logits, label_index)
        weighted_loss = weight * branch_loss
        self.latest_aux_losses.update(
            {
                "module34_branch_ce_loss": branch_loss,
                "module34_branch_ce_weighted_loss": weighted_loss,
                "module34_branch_ce_effective_weight": torch.as_tensor(
                    weight,
                    device=branch_loss.device,
                    dtype=branch_loss.dtype,
                ),
            }
        )
        return weighted_loss

    def _update_hpec_class_margin_statistics(self, labels):
        """仅用训练标签累计两类 HPEC margin 中心，推理阶段只读取缓冲区。"""

        if (
            not self.training
            or self.hpec_residual_calibration != "train_class_margin"
            or self.latest_hpec_raw_margin is None
        ):
            return
        margin = self.latest_hpec_raw_margin.detach().reshape(-1)
        label_index = labels.long().reshape(-1).to(device=margin.device)
        if margin.numel() != label_index.numel():
            return
        momentum = min(max(float(self.hpec_residual_calibration_momentum), 0.0), 1.0)
        with torch.no_grad():
            for class_idx in range(min(2, self.hpec_residual_class_margin_mean.numel())):
                class_margin = margin[label_index == class_idx]
                if class_margin.numel() == 0:
                    continue
                batch_class_mean = class_margin.mean().to(
                    device=self.hpec_residual_class_margin_mean.device,
                    dtype=self.hpec_residual_class_margin_mean.dtype,
                )
                if self.hpec_residual_class_margin_ready[class_idx] < 0.5:
                    updated = batch_class_mean
                    self.hpec_residual_class_margin_ready[class_idx] = 1.0
                else:
                    updated = (
                        (1.0 - momentum) * self.hpec_residual_class_margin_mean[class_idx]
                        + momentum * batch_class_mean
                    )
                self.hpec_residual_class_margin_mean[class_idx] = updated

            batch_std = margin.std(unbiased=False).clamp_min(1e-4).to(
                device=self.hpec_residual_running_margin_std.device,
                dtype=self.hpec_residual_running_margin_std.dtype,
            )
            if self.hpec_residual_running_margin_ready.item() < 0.5:
                self.hpec_residual_running_margin_std.copy_(batch_std.reshape_as(self.hpec_residual_running_margin_std))
                self.hpec_residual_running_margin_ready.fill_(1.0)
            else:
                updated_std = (
                    (1.0 - momentum) * self.hpec_residual_running_margin_std
                    + momentum * batch_std
                )
                self.hpec_residual_running_margin_std.copy_(updated_std.clamp_min(1e-4))

    def compute_primary_loss(self, labels):
        if self.latest_module4_output is None:
            return None
        if self.training and self.latest_module3_output is not None:
            # 首个训练 batch 上按真实标签做一次 prototype warm-start；推理/测试阶段不触碰标签。
            self.hpec_module4.maybe_initialize_from_batch(
                self.latest_module3_output.z_global,
                labels,
            )
            should_update_hpec_ema = (
                self.current_epoch >= self.hpec_ema_start_epoch
                and (
                    self.hpec_ema_update_epochs < 0
                    or self.current_epoch < self.hpec_ema_start_epoch + self.hpec_ema_update_epochs
                )
            )
            if should_update_hpec_ema and self.hpec_prototype_update_mode == "sinkhorn_ema":
                sinkhorn_stats = self.hpec_module4.update_prototypes_with_sinkhorn_ema(
                    self.latest_module3_output.z_global,
                    labels,
                )
                if sinkhorn_stats:
                    self.latest_aux_losses.update(sinkhorn_stats)
                hpec_input = self._calibrate_hpec_input_radius(self.latest_module3_output.z_global)
                hpec_input = self._augment_hpec_input_tangent(hpec_input)
                self.latest_module4_output = self.hpec_module4(hpec_input)
                self._compose_prediction_logits(self.latest_module4_output, self.latest_module3_output)
                # 重算后必须重新叠加 FC 基底残差融合，否则会覆盖 forward 的融合、让模块4 反噬性能。
                self._apply_hyperbolic_logit_residual()
        loss_energy_matrix = self.latest_module4_output.energy_matrix
        network_loss_energy = self._network_hpec_energy(self.latest_module3_output)
        if network_loss_energy is not None and network_loss_energy.shape == loss_energy_matrix.shape:
            weight = max(float(self.hpec_network_energy_weight), 0.0)
            loss_energy_matrix = (1.0 - weight) * loss_energy_matrix + weight * network_loss_energy
        hpec_energy_loss = self.hpec_module4.loss(
            loss_energy_matrix,
            labels,
        )
        if hasattr(self.hpec_module4, "_current_prototypes"):
            prototype_points = self.hpec_module4._current_prototypes(dtype=loss_energy_matrix.dtype)
        else:
            prototype_points = self.hpec_module4.manifold.projx(
                self.hpec_module4.prototypes,
                dim=-1,
            )
        prototype_tangent = self.hpec_module4.manifold.logmap0(
            prototype_points,
            dim=-1,
        )
        prototype_radius = torch.linalg.norm(prototype_points, dim=-1)
        prototype_tangent_norm = torch.linalg.norm(prototype_tangent, dim=-1)
        prototype_direction = F.normalize(prototype_tangent.reshape(-1, prototype_tangent.shape[-1]), p=2, dim=-1)
        prototype_cos = prototype_direction @ prototype_direction.T
        prototype_eye = torch.eye(
            prototype_cos.shape[0],
            device=prototype_cos.device,
            dtype=torch.bool,
        )
        prototype_offdiag = prototype_cos.abs().masked_select(~prototype_eye)
        same_class_cos_values = []
        if prototype_tangent.shape[1] > 1:
            for class_idx in range(prototype_tangent.shape[0]):
                class_direction = F.normalize(prototype_tangent[class_idx], p=2, dim=-1)
                class_cos = class_direction @ class_direction.T
                class_eye = torch.eye(
                    class_cos.shape[0],
                    device=class_cos.device,
                    dtype=torch.bool,
                )
                same_class_cos_values.append(class_cos.abs().masked_select(~class_eye).max())
        same_class_cos_max = (
            torch.stack(same_class_cos_values).max()
            if same_class_cos_values
            else prototype_cos.sum() * 0.0
        )
        self.latest_aux_losses.update(
            {
                "prototype_cos_abs_mean": prototype_offdiag.mean() if prototype_offdiag.numel() else prototype_cos.sum() * 0.0,
                "prototype_cos_abs_max": prototype_offdiag.max() if prototype_offdiag.numel() else prototype_cos.sum() * 0.0,
                "prototype_same_class_cos_max": same_class_cos_max,
                "prototype_radius_mean": prototype_radius.mean(),
                "prototype_radius_min": prototype_radius.min(),
                "prototype_radius_max": prototype_radius.max(),
                "prototype_tangent_norm_mean": prototype_tangent_norm.mean(),
                "prototype_tangent_norm_min": prototype_tangent_norm.min(),
                "prototype_tangent_norm_max": prototype_tangent_norm.max(),
                "hpec_prototype_parameterization_tangent_direction": torch.as_tensor(
                    1.0
                    if getattr(self.hpec_module4, "prototype_parameterization", "poincare_point")
                    == "tangent_direction"
                    else 0.0,
                    device=prototype_tangent.device,
                    dtype=prototype_tangent.dtype,
                ),
            }
        )
        if hasattr(self.hpec_module4, "busemann_class_bias"):
            class_bias = self.hpec_module4.busemann_class_bias
            self.latest_aux_losses.update(
                {
                    "hpec_busemann_class_bias_abs_mean": class_bias.abs().mean(),
                    "hpec_busemann_class_bias_gap": (
                        class_bias.max() - class_bias.min()
                        if class_bias.numel() > 1
                        else class_bias.sum() * 0.0
                    ),
                    "hpec_busemann_class_bias_weight": torch.as_tensor(
                        float(getattr(self.hpec_module4, "busemann_class_bias_weight", 0.0)),
                        device=class_bias.device,
                        dtype=class_bias.dtype,
                    ),
                }
            )
        prototype_separation_loss = prototype_cos.sum() * 0.0
        prototype_separation_weighted_loss = prototype_separation_loss
        if self.hpec_prototype_separation_loss_weight > 0 and prototype_offdiag.numel():
            prototype_separation_loss = F.relu(
                prototype_offdiag - self.hpec_prototype_separation_max_cos
            ).pow(2).mean()
            prototype_separation_weighted_loss = (
                self.hpec_prototype_separation_loss_weight * prototype_separation_loss
            )
            self.latest_aux_losses.update(
                {
                    "hpec_prototype_separation_loss": prototype_separation_loss,
                    "hpec_prototype_separation_weighted_loss": prototype_separation_weighted_loss,
                }
            )
        label_index = labels.long().reshape(-1)
        self._update_hpec_class_margin_statistics(label_index)
        if self.latest_prediction_logits is not None:
            final_ce_loss = self._classification_ce_loss(
                self.latest_prediction_logits,
                label_index,
            )
            hpec_weighted_loss = self.hpec_energy_loss_weight * hpec_energy_loss
            radius_loss = hpec_energy_loss * 0.0
            radius_weighted_loss = radius_loss
            prototype_radius_floor_loss = hpec_energy_loss * 0.0
            prototype_radius_floor_weighted_loss = prototype_radius_floor_loss
            if self.hpec_z_radius_loss_weight > 0 and self.latest_module3_output is not None:
                z_radius = torch.linalg.norm(self.latest_module3_output.z_global, dim=-1)
                if self.hpec_z_min_radius > 0:
                    min_radius = torch.as_tensor(
                        self.hpec_z_min_radius,
                        device=z_radius.device,
                        dtype=z_radius.dtype,
                    )
                    radius_loss = F.relu(min_radius - z_radius).pow(2).mean()
                else:
                    target_radius = torch.as_tensor(
                        self.hpec_z_radius_target,
                        device=z_radius.device,
                        dtype=z_radius.dtype,
                    )
                    radius_loss = (z_radius - target_radius).pow(2).mean()
                radius_weighted_loss = self.hpec_z_radius_loss_weight * radius_loss
            if self.hpec_prototype_min_radius_loss_weight > 0:
                proto_min_radius = torch.as_tensor(
                    self.hpec_prototype_min_radius,
                    device=prototype_radius.device,
                    dtype=prototype_radius.dtype,
                )
                prototype_radius_floor_loss = F.relu(proto_min_radius - prototype_radius).pow(2).mean()
                prototype_radius_floor_weighted_loss = (
                    self.hpec_prototype_min_radius_loss_weight * prototype_radius_floor_loss
                )
            teacher_distill_weighted_loss = self._hpec_teacher_distill_loss()
            prototype_ce_weighted_loss = self._prototype_similarity_ce_loss(
                self.latest_module4_output,
                labels,
            )
            self.latest_primary_loss = (
                final_ce_loss
                + hpec_weighted_loss
                + radius_weighted_loss
                + prototype_radius_floor_weighted_loss
                + prototype_separation_weighted_loss
            )
            prior_alignment_weighted_loss = self._class_prior_alignment_loss(
                self.latest_prediction_logits,
                label_index,
            )
            if prior_alignment_weighted_loss is not None:
                self.latest_primary_loss = self.latest_primary_loss + prior_alignment_weighted_loss
            if teacher_distill_weighted_loss is not None:
                self.latest_primary_loss = self.latest_primary_loss + teacher_distill_weighted_loss
            if prototype_ce_weighted_loss is not None:
                self.latest_primary_loss = self.latest_primary_loss + prototype_ce_weighted_loss
            branch_ce_weighted_loss = self._module34_branch_ce_loss(labels)
            if branch_ce_weighted_loss is not None:
                self.latest_primary_loss = self.latest_primary_loss + branch_ce_weighted_loss
            self.latest_aux_losses.update(
                {
                    "hpec_final_ce_loss": final_ce_loss,
                    "hpec_energy_loss": hpec_energy_loss,
                    "hpec_energy_weighted_loss": hpec_weighted_loss,
                    "hpec_z_radius_loss": radius_loss,
                    "hpec_z_radius_weighted_loss": radius_weighted_loss,
                    "hpec_prototype_radius_floor_loss": prototype_radius_floor_loss,
                    "hpec_prototype_radius_floor_weighted_loss": prototype_radius_floor_weighted_loss,
                }
            )
        else:
            self.latest_primary_loss = hpec_energy_loss
        return self.latest_primary_loss

    def _class_prior_alignment_loss(self, logits, labels):
        """训练期校准最终概率的类别比例，缓解 raw 0.5 下长期偏向负类。

        这里使用当前 batch 的真实标签比例作为弱约束，只作用于训练损失；
        测试阶段仍然只看模型输出概率，不读取测试标签或调整阈值。
        """

        weight = max(float(self.class_prior_alignment_weight), 0.0)
        if weight <= 0.0 or logits is None:
            return None
        labels = labels.long().reshape(-1)
        if labels.numel() <= 1:
            return None
        if logits.ndim != 2 or logits.shape[-1] <= 1:
            prob = torch.sigmoid(logits.reshape(-1))
            target_prior = labels.to(device=prob.device, dtype=prob.dtype).mean()
            prior_loss = (prob.mean() - target_prior).pow(2)
        else:
            prob_mean = torch.softmax(logits, dim=-1).mean(dim=0)
            target_prior = torch.bincount(
                labels.to(device=logits.device),
                minlength=logits.shape[-1],
            ).to(device=logits.device, dtype=logits.dtype)
            target_prior = target_prior / target_prior.sum().clamp_min(1.0)
            prior_loss = F.mse_loss(prob_mean, target_prior)
        weighted_loss = weight * prior_loss
        self.latest_aux_losses.update(
            {
                "class_prior_alignment_loss": prior_loss,
                "class_prior_alignment_weighted_loss": weighted_loss,
            }
        )
        return weighted_loss

    def _classification_ce_loss(self, logits, labels):
        """按配置计算分类 CE；MDD 小样本默认可用 batch 内类别均衡。"""

        labels = labels.long().reshape(-1)
        if logits.ndim != 2 or logits.shape[-1] <= 1:
            binary_logits = logits.reshape(-1)
            binary_labels = labels.to(device=binary_logits.device, dtype=binary_logits.dtype)
            if self.class_loss_weighting in (
                "batch_balanced",
                "balanced",
                "inverse_freq",
                "sqrt_batch_balanced",
                "sqrt_balanced",
            ):
                positive_count = binary_labels.sum().clamp_min(1.0)
                negative_count = (binary_labels.numel() - binary_labels.sum()).clamp_min(1.0)
                pos_weight = negative_count / positive_count
                if self.class_loss_weighting in ("sqrt_batch_balanced", "sqrt_balanced"):
                    pos_weight = pos_weight.sqrt()
                return F.binary_cross_entropy_with_logits(
                    binary_logits,
                    binary_labels,
                    pos_weight=pos_weight.to(device=binary_logits.device, dtype=binary_logits.dtype),
                )
            return F.binary_cross_entropy_with_logits(binary_logits, binary_labels)
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
        if self.class_loss_weighting in ("logit_adjusted", "sqrt_logit_adjusted"):
            counts = torch.bincount(labels, minlength=logits.shape[-1]).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            if torch.count_nonzero(counts).item() <= 1:
                return F.cross_entropy(logits, labels, label_smoothing=self.class_label_smoothing)
            prior = counts / counts.sum().clamp_min(1.0)
            if self.class_loss_weighting == "sqrt_logit_adjusted":
                prior = prior.sqrt()
                prior = prior / prior.sum().clamp_min(1e-8)
            adjusted_logits = logits + self.class_logit_adjust_tau * prior.clamp_min(1e-8).log().view(1, -1)
            return F.cross_entropy(
                adjusted_logits,
                labels,
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
                a_lag_raw_mean = (
                    causal_output.a_lag_raw.mean(dim=0)
                    if getattr(causal_output, "a_lag_raw", None) is not None
                    else None
                )
                candidate_mask_mean = None
                candidate_masked_lag_mean = None
                if getattr(causal_output, "candidate_lag_mask", None) is not None:
                    candidate_mask_mean = causal_output.candidate_lag_mask.mean(dim=0)
                    candidate_masked_lag_mean = a_lag_mean * candidate_mask_mean
                graph_style = causal_output.dag_metadata.get("graph_style", "temporal_nts_notears")
                if graph_style == "attention_guided_temporal_nts_notears":
                    temporal_prefix = "Attention-guided Temporal NTS-NOTEARS"
                else:
                    temporal_prefix = "Temporal NTS-NOTEARS"
                items.extend(
                    [
                        causal_output.target,
                        causal_output.x_hat,
                        pred_error,
                        causal_output.a0,
                        *([a_lag_raw_mean] if a_lag_raw_mean is not None else []),
                        a_lag_mean,
                        *(
                            [candidate_mask_mean, candidate_masked_lag_mean]
                            if candidate_mask_mean is not None
                            else []
                        ),
                        causal_output.a_shared,
                        a_effective,
                        adjacency_binary,
                        adjacency_direction_delta,
                    ]
                )
                titles.extend(
                    [
                        f"{temporal_prefix} target",
                        f"{temporal_prefix} X_hat",
                        f"{temporal_prefix} prediction error",
                        f"{temporal_prefix} A0 同时间片残余依赖图",
                        *(
                            [f"{temporal_prefix} A_lag_raw_mean before candidate top-k"]
                            if a_lag_raw_mean is not None
                            else []
                        ),
                        f"{temporal_prefix} A_lag_mean 跨时间主因果图",
                        *(
                            [
                                f"{temporal_prefix} candidate parent top-k mask",
                                f"{temporal_prefix} A_lag_mean after candidate top-k mask",
                            ]
                            if candidate_mask_mean is not None
                            else []
                        ),
                        f"{temporal_prefix} A_shared 跨时间主因果图",
                        "A_effective shared+delta",
                        "A_effective_binary",
                        "A_effective - A_effective.T",
                    ]
                )
                for lag_idx in range(causal_output.a_lag.shape[0]):
                    items.append(causal_output.a_lag[lag_idx])
                    titles.append(f"{temporal_prefix} A_lag[{lag_idx}]")
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
                titles.append("A_cls 最终分类图")
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
        if (
            module3_output is not None
            and gcn_output is not None
            and self.hyperbolic_logit_residual_weight > 0
        ):
            blended = self._apply_hyperbolic_logit_residual()
            if blended is not None:
                y_hat = blended
        y_hat = self._apply_final_logit_calibration(y_hat)
        # 独立 prototype EMA 使用与指标一致的最终标准视图 logits 评估 TP 可靠性。
        self.latest_prediction_logits = y_hat
        if self.out_dim == 1 and module4_output is None:
            y_hat = torch.sigmoid(y_hat)
        return y_hat
