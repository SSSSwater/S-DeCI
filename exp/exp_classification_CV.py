import json
import os
import time
import warnings
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping,evaluate,adjust_learning_rate
from data_provider.data_factory_CV import data_provider
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from scipy.special import softmax  
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

class WeightedBCELoss(nn.Module):
    def __init__(self, positive_weight=1.0, eps=1e-7):
        super().__init__()
        self.positive_weight = float(positive_weight)
        self.eps = float(eps)

    def forward(self, prediction, target):
        prediction = prediction.clamp(min=self.eps, max=1.0 - self.eps)
        target = target.to(prediction.dtype)
        loss = (
            -self.positive_weight * target * torch.log(prediction)
            -(1.0 - target) * torch.log(1.0 - prediction)
        )
        return loss.mean()

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self._ema_shadow = None

    def _build_model(self):
        self.initial_model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.initial_model = nn.DataParallel(self.initial_model, device_ids=self.args.device_ids)
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)

        total = sum([param.nelement() for param in self.model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))

        return self.model, self.initial_model
    def reset_model(self):
        self.model.load_state_dict(self.initial_model.state_dict())
        self._ema_shadow = None
        if self. args.print_process: print("Model has been reset to initial weights.")

    def _get_data(self):
        train_loaders, val_loaders= data_provider(self.args)
        return train_loaders, val_loaders
    def _select_optimizer(self):
        causal_lr = getattr(self.args, "causal_learning_rate", None)
        weight_decay = float(getattr(self.args, "weight_decay", 0.0) or 0.0)
        if causal_lr is None or causal_lr <= 0:
            model_optim = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=weight_decay,
            )
            return model_optim

        base_params = []
        causal_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "causal_learner" in name:
                causal_params.append(param)
            else:
                base_params.append(param)

        param_groups = []
        if base_params:
            param_groups.append(
                {"params": base_params, "lr": self.args.learning_rate, "weight_decay": weight_decay}
            )
        if causal_params:
            param_groups.append(
                {"params": causal_params, "lr": causal_lr, "weight_decay": weight_decay}
            )

        model_optim = optim.Adam(param_groups, lr=self.args.learning_rate, weight_decay=weight_decay)
        return model_optim

    def _init_model_ema(self):
        if not bool(getattr(self.args, "use_model_ema", 0)):
            self._ema_shadow = None
            return
        self._ema_shadow = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _update_model_ema(self):
        if self._ema_shadow is None:
            return
        decay = float(getattr(self.args, "model_ema_decay", 0.995))
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._ema_shadow:
                    self._ema_shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def _apply_model_ema(self):
        if self._ema_shadow is None:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._ema_shadow:
                    param.copy_(self._ema_shadow[name].to(device=param.device, dtype=param.dtype))
    def _select_criterion(self):
        loss_name = str(self.args.loss).lower()
        if self.args.classes == 2:
            if loss_name in ("bce", "binary_ce"):
                return nn.BCELoss()
            if loss_name in ("weighted_bce", "wbce"):
                return WeightedBCELoss(
                    positive_weight=float(getattr(self.args, "binary_positive_weight", 1.0))
                )
            return nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        return criterion
    def _model_for_aux_loss(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model
    def _get_model_aux_loss(self):
        model = self._model_for_aux_loss()
        get_aux_loss = getattr(model, "get_aux_loss", None)
        if not callable(get_aux_loss):
            return None
        return get_aux_loss()
    def _get_model_aux_losses(self):
        model = self._model_for_aux_loss()
        get_aux_losses = getattr(model, "get_aux_losses", None)
        if not callable(get_aux_losses):
            return {}
        return get_aux_losses()
    def _get_model_primary_loss(self, labels):
        model = self._model_for_aux_loss()
        compute_primary_loss = getattr(model, "compute_primary_loss", None)
        if not callable(compute_primary_loss):
            return None
        return compute_primary_loss(labels)
    def _set_model_train_epoch(self, epoch):
        model = self._model_for_aux_loss()
        set_train_epoch = getattr(model, "set_train_epoch", None)
        if callable(set_train_epoch):
            set_train_epoch(epoch)
    def _get_model_cached_prediction_probability(self):
        model = self._model_for_aux_loss()
        get_prediction = getattr(model, "get_latest_prediction", None)
        get_probability = getattr(model, "get_latest_probabilities", None)
        if not callable(get_prediction) or not callable(get_probability):
            return None, None
        prediction = get_prediction()
        probability = get_probability()
        if prediction is None or probability is None:
            return None, None
        return prediction, probability
    def _prediction_probability_for_metrics(self, y_hat, metric_label):
        cached_pred, cached_prob = self._get_model_cached_prediction_probability()
        if cached_pred is not None and cached_prob is not None:
            if self.args.classes == 2:
                if cached_prob.ndim > 1 and cached_prob.shape[-1] > 1:
                    pos_prob = cached_prob[:, 1]
                else:
                    pos_prob = cached_prob.reshape(-1)
                prob = pos_prob.detach().cpu().numpy()
                pred = (pos_prob > 0.5).to(metric_label.dtype).detach().cpu().numpy()
            else:
                prob = cached_prob.detach().cpu().numpy()
                pred = cached_pred.detach().cpu().numpy()
            target = metric_label.detach().cpu().numpy()
            return pred, prob, target

        if self.args.classes != 2:
            prob = torch.nn.functional.softmax(y_hat, dim=1)
            pred = torch.argmax(prob, dim=1).cpu().numpy()
            target = metric_label.cpu().numpy()
        else:
            prob = y_hat.squeeze(-1)
            pred = (prob > 0.5).to(metric_label.dtype).cpu().numpy()
            target = metric_label.cpu().numpy()
        return pred, prob.detach().cpu().numpy(), target
    def _prediction_probability_with_threshold(self, y_hat, metric_label, threshold=None):
        pred, prob, target = self._prediction_probability_for_metrics(y_hat, metric_label)
        if self.args.classes == 2 and threshold is not None:
            pred = (np.asarray(prob).reshape(-1) >= float(threshold)).astype(np.asarray(target).dtype)
        return pred, prob, target
    def _best_binary_threshold(self, targets, probs):
        if self.args.classes != 2:
            return None
        targets = np.asarray(targets).astype(int)
        probs = np.asarray(probs).reshape(-1)
        if len(np.unique(targets)) < 2:
            return None
        thresholds = np.linspace(0.05, 0.95, 91)
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in thresholds:
            pred = (probs >= threshold).astype(int)
            score = f1_score(targets, pred, average="macro", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        return best_threshold, best_f1
    def _prediction_diagnostics(self, targets, preds, probs):
        targets = np.asarray(targets).astype(int).reshape(-1)
        preds = np.asarray(preds).astype(int).reshape(-1)
        class_count = int(getattr(self.args, "classes", 2))
        target_counts = np.bincount(targets, minlength=class_count)
        pred_counts = np.bincount(preds, minlength=class_count)
        diagnostics = {
            "target_counts": target_counts.tolist(),
            "pred_counts": pred_counts.tolist(),
        }
        if class_count == 2 and probs is not None:
            prob_array = np.asarray(probs).reshape(-1)
            diagnostics.update(
                {
                    "prob_min": float(np.min(prob_array)),
                    "prob_mean": float(np.mean(prob_array)),
                    "prob_max": float(np.max(prob_array)),
                }
            )
        return diagnostics
    def _format_prediction_diagnostics(self, diagnostics):
        if not diagnostics:
            return ""
        text = (
            f"target_counts={diagnostics.get('target_counts')} | "
            f"pred_counts={diagnostics.get('pred_counts')}"
        )
        if "prob_min" in diagnostics:
            text += (
                f" | prob_pos[min/mean/max]="
                f"{diagnostics['prob_min']:.4f}/"
                f"{diagnostics['prob_mean']:.4f}/"
                f"{diagnostics['prob_max']:.4f}"
            )
        return text
    def _metrics_output_dir(self, check_path):
        output_dir = getattr(self.args, "causal_vis_dir", None) or check_path
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        output_dir = os.path.join(output_dir, "metrics")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _metrics_base_name(self, check_path):
        fold_name = os.path.basename(os.path.normpath(check_path))
        graph_path = self._current_graph_path_name()
        return f"{fold_name}_{graph_path}"

    def _metric_dict(self, prefix, values):
        names = ("accuracy", "precision", "recall", "macro_f1", "roc_auc")
        return {f"{prefix}_{name}": float(value) for name, value in zip(names, values)}

    def _json_safe(self, value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return value

    def _save_epoch_metrics(self, check_path, records):
        if not records:
            return
        output_dir = self._metrics_output_dir(check_path)
        base_name = self._metrics_base_name(check_path)
        csv_path = os.path.join(output_dir, f"{base_name}_epoch_metrics.csv")
        json_path = os.path.join(output_dir, f"{base_name}_epoch_metrics.json")
        pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(records), f, ensure_ascii=False, indent=2)
        if self.args.print_process:
            print(f"Epoch metrics saved to: {csv_path}")

    def _save_final_metrics(self, check_path, records):
        if not records:
            return
        output_dir = self._metrics_output_dir(check_path)
        base_name = self._metrics_base_name(check_path)
        csv_path = os.path.join(output_dir, f"{base_name}_final_metrics.csv")
        json_path = os.path.join(output_dir, f"{base_name}_final_metrics.json")
        flat_records = []
        for record in records:
            flat = {}
            for key, value in record.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        flat[f"{key}_{nested_key}"] = self._json_safe(nested_value)
                else:
                    flat[key] = self._json_safe(value)
            flat_records.append(flat)
        pd.DataFrame(flat_records).to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(records), f, ensure_ascii=False, indent=2)
        if self.args.print_process:
            print(f"Final metrics saved to: {csv_path}")
    def _unpack_batch(self, batch):
        site_label = None
        if len(batch) == 4:
            x_enc, label, correlation_matrix, site_label = batch
        elif len(batch) == 3:
            x_enc, label, third = batch
            if getattr(third, "ndim", 0) == 3:
                correlation_matrix = third
            else:
                correlation_matrix = None
                site_label = third
        elif len(batch) == 2:
            x_enc, label = batch
            correlation_matrix = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} fields.")
        return x_enc, label, correlation_matrix, site_label

    def _to_device(self, tensor):
        if tensor is None:
            return None
        return tensor.to(self.device, non_blocking=bool(getattr(self.args, "pin_memory", 0)))

    def _model_forward(self, x_enc, correlation_matrix=None, site_label=None):
        kwargs = {}
        if correlation_matrix is not None:
            kwargs["correlation_matrix"] = correlation_matrix
        if site_label is not None:
            kwargs["site_label"] = site_label
        return self.model(x_enc, **kwargs)
    def _get_model_site_adversarial_loss(self, site_label):
        model = self._model_for_aux_loss()
        compute_loss = getattr(model, "compute_site_adversarial_loss", None)
        if not callable(compute_loss):
            return None
        return compute_loss(site_label)
    def _format_metric_line(self, fold_epoch, train_loss, val_loss, train_metric, val_metric, loss_parts):
        names = ("acc", "precision", "recall", "macro_f1", "roc_auc")
        train_text = " | ".join(
            f"train_{name}: {value:.4f}" for name, value in zip(names, train_metric)
        )
        val_text = " | ".join(
            f"val_{name}: {value:.4f}" for name, value in zip(names, val_metric)
        )
        loss_items = [
            ("total_loss", train_loss),
            ("cls_loss", loss_parts.get("cls_loss", float("nan"))),
            ("hpec_loss", loss_parts.get("hpec_loss", 0.0)),
            ("module1_denoise", loss_parts.get("module1_denoise_loss", 0.0)),
            ("aux_loss", loss_parts.get("causal_aux_loss", 0.0)),
            ("causal_stability", loss_parts.get("causal_stability_loss", 0.0)),
            ("hgcn_radius", loss_parts.get("hgcn_radius_loss", 0.0)),
            ("hgcn_cls_aux", loss_parts.get("hgcn_cls_aux_loss", 0.0)),
            ("hgcn_view_cons", loss_parts.get("hgcn_view_consistency_loss", 0.0)),
            ("hgcn_supcon", loss_parts.get("hgcn_supcon_loss", 0.0)),
            ("proto_aux_loss", loss_parts.get("hpec_prototype_aux_loss", 0.0)),
            ("hpec_ce_aux", loss_parts.get("hpec_ce_aux_loss", 0.0)),
            ("hpec_final_ce", loss_parts.get("hpec_final_ce_loss", 0.0)),
            ("hpec_energy", loss_parts.get("hpec_energy_loss", 0.0)),
            ("hpec_radius", loss_parts.get("hpec_radius_loss", 0.0)),
            ("hpec_diversity", loss_parts.get("hpec_diversity_loss", 0.0)),
            ("hpec_hsic", loss_parts.get("hpec_hsic_loss", 0.0)),
            ("hpec_intra_orth", loss_parts.get("hpec_intra_orthogonal_loss", 0.0)),
            ("hpec_inter_margin", loss_parts.get("hpec_inter_margin_loss", 0.0)),
            ("hpec_class_margin", loss_parts.get("hpec_class_center_margin_loss", 0.0)),
            ("hpec_anchor", loss_parts.get("hpec_anchor_loss", 0.0)),
            ("hpec_gate", loss_parts.get("hpec_logit_gate", 0.0)),
            ("site_adv", loss_parts.get("site_adversarial_loss", 0.0)),
            ("site_film", loss_parts.get("site_modulation_reg_loss", 0.0)),
            ("hpec_mle_loss", loss_parts.get("hpec_mle_loss", 0.0)),
            ("hpec_pcl_loss", loss_parts.get("hpec_pcl_loss", 0.0)),
            ("hpec_pal_loss", loss_parts.get("hpec_pal_loss", 0.0)),
            ("recon_loss", loss_parts.get("causal_recon_loss", 0.0)),
            ("temporal_pred", loss_parts.get("temporal_pred_loss", 0.0)),
            ("dag_loss", loss_parts.get("causal_dag_loss", 0.0)),
            ("l1_loss", loss_parts.get("causal_l1_loss", 0.0)),
            ("temporal_sparse", loss_parts.get("temporal_sparse_loss", 0.0)),
            ("temporal_smooth", loss_parts.get("temporal_smooth_loss", 0.0)),
            ("temporal_cf", loss_parts.get("temporal_counterfactual_loss", 0.0)),
            ("sample_l1", loss_parts.get("sample_graph_l1_loss", 0.0)),
            ("sample_dev", loss_parts.get("sample_graph_deviation_loss", 0.0)),
            ("reg_scale", loss_parts.get("temporal_reg_scale", 0.0)),
            ("val_loss", val_loss),
        ]
        diagnostic_items = [
            ("dag_raw", loss_parts.get("causal_meta_dagma_spectral_radius", loss_parts.get("causal_meta_analytic_spectral_radius", 0.0))),
            ("graph_mass", loss_parts.get("causal_meta_shared_adjacency_mass_mean", 0.0)),
            ("direction", loss_parts.get("causal_meta_shared_adjacency_directionality_ratio", 0.0)),
            ("a0_mass", loss_parts.get("causal_meta_a0_adjacency_mass_mean", 0.0)),
            ("alag_mass", loss_parts.get("causal_meta_alag_mean_adjacency_mass_mean", 0.0)),
            ("alag_direction", loss_parts.get("causal_meta_alag_mean_adjacency_directionality_ratio", 0.0)),
            ("cf_effect", loss_parts.get("temporal_counterfactual_effect_mean", 0.0)),
            ("cf_edge", loss_parts.get("temporal_counterfactual_edge_mean", 0.0)),
            ("dagma_stage", loss_parts.get("causal_meta_dagma_stage_id", 0.0)),
            ("dagma_scale", loss_parts.get("causal_meta_dagma_effective_scale", 0.0)),
            ("delta_abs", loss_parts.get("causal_meta_sample_delta_abs_mean", 0.0)),
            ("z_radius", loss_parts.get("z_radius_mean", 0.0)),
            ("z_radius_max", loss_parts.get("z_radius_max", 0.0)),
            ("z_tangent", loss_parts.get("z_tangent_norm_mean", 0.0)),
            ("fc_residual", loss_parts.get("fc_residual_norm_mean", 0.0)),
            ("proto_cos_mean", loss_parts.get("prototype_cos_abs_mean", 0.0)),
            ("proto_cos_max", loss_parts.get("prototype_cos_abs_max", 0.0)),
            ("proto_same_cls_max", loss_parts.get("prototype_same_class_cos_max", 0.0)),
            ("sinkhorn_entropy", loss_parts.get("hpec_sinkhorn_assignment_entropy", 0.0)),
            ("sinkhorn_min", loss_parts.get("hpec_sinkhorn_usage_min", 0.0)),
            ("sinkhorn_max", loss_parts.get("hpec_sinkhorn_usage_max", 0.0)),
        ]
        def _fmt_metric(name, value):
            if name in ("dag_loss", "dag_raw") and abs(float(value)) < 1e-3:
                return f"{name}: {value:.3e}"
            return f"{name}: {value:.4f}"

        loss_text = " | ".join(_fmt_metric(name, value) for name, value in loss_items)
        diagnostic_text = " | ".join(_fmt_metric(name, value) for name, value in diagnostic_items)
        separator = "=" * 72
        section_separator = "-" * 72
        return (
            f"\n{separator}\n"
            f"{fold_epoch}\n"
            f"{section_separator}\n"
            f"[Loss] {loss_text}\n"
            f"{section_separator}\n"
            f"[Graph Diagnostics] {diagnostic_text}\n"
            f"{section_separator}\n"
            f"[Train Metrics] {train_text}\n"
            f"{section_separator}\n"
            f"[Validation Metrics] {val_text}\n"
            f"{separator}"
        )
    def _module_path_summary(self):
        if self.args.model != "S-DeCI":
            return None
        use_deci = int(getattr(self.args, "use_deci_module1", 1))
        use_causal = int(getattr(self.args, "use_causal_module2", 1))
        use_hgcn = int(getattr(self.args, "use_hgcn_module3", 0))
        use_hpec = int(getattr(self.args, "use_hpec_module4", 0))
        use_hyper = int(bool(use_hgcn or use_hpec))
        if use_hpec:
            path = "hgcn_hpec"
        elif use_hgcn:
            path = "hgcn_only"
        else:
            path = "gcn_fallback"
        module1_mode = str(getattr(self.args, "module1_feature_mode", "") or "").lower()
        if not use_deci:
            feature = "raw_projected_feature"
        elif module1_mode == "alff":
            feature = "alff_falff_physiological_feature"
        elif module1_mode == "deci":
            feature = "deci_cycle_feature"
        else:
            feature = f"{module1_mode or 'module1'}_feature"
        graph_source = str(getattr(self.args, "classification_graph_source", "") or "").lower()
        if graph_source in ("sample_correlation", "fc"):
            adjacency = "sample_correlation"
        elif graph_source in ("learned", "causal"):
            adjacency = "causal_graph"
        elif use_causal:
            adjacency = "causal_graph_blended_with_sample_correlation"
        else:
            adjacency = "sample_correlation"
        return (
            f"S-DeCI modules: use_deci_module1={use_deci}, "
            f"use_causal_module2={use_causal}, use_hgcn_module3={use_hgcn}, "
            f"use_hpec_module4={use_hpec}, use_hyperbolic_modules34={use_hyper}, "
            f"classification_path={path}, node_feature={feature}, adjacency={adjacency}"
        )
    def _current_graph_path_name(self):
        model = self._model_for_aux_loss()
        return getattr(
            model,
            "latest_graph_path",
            "hgcn_hpec"
            if int(getattr(self.args, "use_hpec_module4", 0))
            else ("hgcn_only" if int(getattr(self.args, "use_hgcn_module3", 0)) else "gcn_fallback"),
        )
    def _select_visualization_batch(self, loader):
        selected_samples = {}
        target_unique_count = max(int(getattr(self.args, "classes", 2)), 1)
        max_samples = max(int(getattr(self.args, "batch_size", 1)), target_unique_count)

        for batch in loader:
            x_enc, labels, correlation_matrix, site_label = self._unpack_batch(batch)
            for idx, label in enumerate(labels):
                label_value = int(label.item())
                if label_value in selected_samples:
                    continue
                sample = [x_enc[idx], label]
                if correlation_matrix is not None:
                    sample.append(correlation_matrix[idx])
                selected_samples[label_value] = sample
                if len(selected_samples) >= target_unique_count:
                    break
            if len(selected_samples) >= target_unique_count:
                break

        if not selected_samples:
            return None

        samples = [selected_samples[key] for key in sorted(selected_samples)]
        for batch in loader:
            x_enc, labels, correlation_matrix, site_label = self._unpack_batch(batch)
            for idx, label in enumerate(labels):
                if len(samples) >= max_samples:
                    break
                sample = [x_enc[idx], label]
                if correlation_matrix is not None:
                    sample.append(correlation_matrix[idx])
                samples.append(sample)
            if len(samples) >= max_samples:
                break

        data = torch.stack([sample[0] for sample in samples])
        labels = torch.stack([sample[1] for sample in samples])
        if len(samples[0]) == 3:
            correlations = torch.stack([sample[2] for sample in samples])
            return data, labels, correlations
        return data, labels

    def _save_causal_visualization_for_loader(self, loader, check_path, split_name):
        model = self._model_for_aux_loss()
        visualize = getattr(model, "visualize_causal_intermediates", None)
        if not callable(visualize):
            return

        batch = self._select_visualization_batch(loader)
        if batch is None:
            return
        x_enc, labels, correlation_matrix, site_label = self._unpack_batch(batch)
        x_enc = x_enc.to(self.device)
        if correlation_matrix is not None:
            correlation_matrix = correlation_matrix.to(self.device)
        if site_label is not None:
            site_label = site_label.to(self.device)

        with torch.no_grad():
            y_hat = self._model_forward(
                x_enc,
                correlation_matrix=correlation_matrix,
                site_label=site_label,
            )
            cached_pred, _ = self._get_model_cached_prediction_probability()
            if cached_pred is not None:
                predictions = cached_pred.to(labels.dtype).cpu()
            elif self.args.classes != 2:
                predictions = torch.argmax(torch.nn.functional.softmax(y_hat, dim=1), dim=1)
            else:
                predictions = (y_hat.squeeze(-1) > 0.5).to(labels.dtype)

        output_dir = getattr(self.args, "causal_vis_dir", None) or check_path
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(output_dir, exist_ok=True)

        fold_name = os.path.basename(os.path.normpath(check_path))
        graph_path = self._current_graph_path_name()
        save_path = os.path.join(output_dir, f"{fold_name}_{graph_path}_{split_name}_causal_intermediates.png")
        visualize(save_path=save_path, labels=labels, predictions=predictions)

        if self.args.print_process:
            print(f"{split_name} visualization labels: {labels.detach().cpu().tolist()}")
            print(f"{split_name} causal visualization saved to: {save_path}")

    def _save_causal_visualization_after_training(self, train_loader, val_loader, check_path):
        if not bool(getattr(self.args, "visualize_causal", 0)):
            return

        was_training = self.model.training
        self.model.eval()
        self._save_causal_visualization_for_loader(train_loader, check_path, "train")
        self._save_causal_visualization_for_loader(val_loader, check_path, "test")

        if was_training:
            self.model.train()
    def _save_graph_diagnostics_after_training(self, check_path):
        model = self._model_for_aux_loss()
        causal_output = getattr(model, "latest_causal_output", None)
        if causal_output is None:
            return

        output_dir = getattr(self.args, "causal_vis_dir", None) or check_path
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(output_dir, exist_ok=True)
        fold_name = os.path.basename(os.path.normpath(check_path))

        graph_pack = {}
        if hasattr(causal_output, "a0"):
            graph_pack["A0"] = causal_output.a0.detach().cpu().numpy()
            graph_pack["A_lag"] = causal_output.a_lag.detach().cpu().numpy()
        graph_pack["A_shared"] = causal_output.a_shared.detach().cpu().numpy()
        graph_pack["A_effective"] = causal_output.a_effective.detach().cpu().numpy()
        if causal_output.a_delta is not None:
            graph_pack["A_delta"] = causal_output.a_delta.detach().cpu().numpy()

        graph_path = os.path.join(output_dir, f"{fold_name}_graph_diagnostics.npz")
        np.savez(graph_path, **graph_pack)

        base_graph = graph_pack.get("A0", graph_pack["A_shared"])
        if base_graph.ndim == 3:
            base_graph = base_graph.mean(axis=0)
        k = min(20, base_graph.size)
        flat = np.abs(base_graph).reshape(-1)
        top_indices = np.argsort(flat)[-k:][::-1]
        rows, cols = np.unravel_index(top_indices, base_graph.shape)
        edge_path = os.path.join(output_dir, f"{fold_name}_top_edges.csv")
        pd.DataFrame(
            {
                "parent": rows,
                "child": cols,
                "weight": base_graph[rows, cols],
                "abs_weight": np.abs(base_graph[rows, cols]),
            }
        ).to_csv(edge_path, index=False)

        if self.args.print_process:
            print(f"Graph diagnostics saved to: {graph_path}")
            print(f"Top-k temporal edges saved to: {edge_path}")
    def _latest_embedding_after_forward(self, y_hat):
        model = self._model_for_aux_loss()
        module3_output = getattr(model, "latest_module3_output", None)
        if module3_output is not None:
            return module3_output.z_tangent.detach().cpu().numpy()
        gcn_output = getattr(model, "latest_gcn_fallback_output", None)
        if gcn_output is not None:
            return gcn_output.readout.detach().cpu().numpy()
        return y_hat.detach().cpu().reshape(y_hat.shape[0], -1).numpy()

    def _collect_embeddings_for_tsne(self, loader):
        features = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                x_enc, label, correlation_matrix, site_label = self._unpack_batch(batch)
                x_enc = x_enc.to(self.device)
                if correlation_matrix is not None:
                    correlation_matrix = correlation_matrix.to(self.device)
                if site_label is not None:
                    site_label = site_label.to(self.device)
                y_hat = self._model_forward(
                    x_enc,
                    correlation_matrix=correlation_matrix,
                    site_label=site_label,
                )
                features.append(self._latest_embedding_after_forward(y_hat))
                labels.append(label.detach().cpu().numpy())
        if not features:
            return None, None
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    def _latest_hpec_prototypes_for_tsne(self):
        model = self._model_for_aux_loss()
        module4_output = getattr(model, "latest_module4_output", None)
        module3_output = getattr(model, "latest_module3_output", None)
        if module4_output is None or module3_output is None:
            return None
        prototypes = getattr(module4_output, "prototypes", None)
        if prototypes is None:
            return None
        manifold = getattr(getattr(model, "hgcn_module3", None), "manifold", None)
        if manifold is None:
            return None
        prototype_features = manifold.logmap0(prototypes, dim=-1).detach().cpu().numpy()
        if prototype_features.ndim == 2:
            prototype_labels = np.arange(prototype_features.shape[0], dtype=int)
            return prototype_features, prototype_labels
        if prototype_features.ndim == 3:
            class_count, prototypes_per_class, hidden_dim = prototype_features.shape
            prototype_labels = np.repeat(np.arange(class_count, dtype=int), prototypes_per_class)
            return prototype_features.reshape(class_count * prototypes_per_class, hidden_dim), prototype_labels
        return None

    def _save_tsne_after_training(self, train_loader, val_loader, check_path):
        if not bool(getattr(self.args, "visualize_causal", 0)):
            return

        was_training = self.model.training
        self.model.eval()
        train_features, train_labels = self._collect_embeddings_for_tsne(train_loader)
        test_features, test_labels = self._collect_embeddings_for_tsne(val_loader)
        prototype_pack = self._latest_hpec_prototypes_for_tsne()
        if was_training:
            self.model.train()
        if train_features is None or test_features is None:
            return

        feature_parts = [train_features, test_features]
        prototype_features = None
        prototype_labels = None
        if prototype_pack is not None:
            prototype_features, prototype_labels = prototype_pack
        if prototype_features is not None:
            feature_parts.append(prototype_features)
        features = np.concatenate(feature_parts, axis=0)
        labels = np.concatenate([train_labels, test_labels], axis=0).astype(int)
        splits = np.array(["train"] * len(train_labels) + ["test"] * len(test_labels))
        if len(features) < 3:
            return

        perplexity = min(30, max(2, (len(features) - 1) // 3))
        embedding_2d = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=int(getattr(self.args, "seed", 2024)),
        ).fit_transform(features)
        sample_count = len(train_labels) + len(test_labels)
        sample_embedding_2d = embedding_2d[:sample_count]
        prototype_embedding_2d = embedding_2d[sample_count:] if prototype_features is not None else None

        output_dir = getattr(self.args, "causal_vis_dir", None) or check_path
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(output_dir, exist_ok=True)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.get_cmap("tab10")
        markers = {"train": "o", "test": "^"}
        for split_name in ("train", "test"):
            split_mask = splits == split_name
            for label_value in sorted(np.unique(labels)):
                mask = split_mask & (labels == label_value)
                if not np.any(mask):
                    continue
                ax.scatter(
                    sample_embedding_2d[mask, 0],
                    sample_embedding_2d[mask, 1],
                    c=[cmap(label_value % 10)],
                    marker=markers[split_name],
                    label=f"{split_name} label={label_value}",
                    alpha=0.78,
                    edgecolors="k" if split_name == "test" else "none",
                    linewidths=0.5,
                    s=36,
                )
        if prototype_embedding_2d is not None:
            shown_prototype_labels = set()
            for prototype_idx, point in enumerate(prototype_embedding_2d):
                prototype_label = int(prototype_labels[prototype_idx])
                legend_label = None
                if prototype_label not in shown_prototype_labels:
                    legend_label = f"prototype label={prototype_label}"
                    shown_prototype_labels.add(prototype_label)
                ax.scatter(
                    point[0],
                    point[1],
                    c=[cmap(prototype_label % 10)],
                    marker="*",
                    label=legend_label,
                    alpha=1.0,
                    edgecolors="k",
                    linewidths=0.8,
                    s=180,
                )
        graph_path = self._current_graph_path_name()
        ax.set_title(f"t-SNE of final-epoch embeddings ({graph_path})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        fold_name = os.path.basename(os.path.normpath(check_path))
        save_path = os.path.join(output_dir, f"{fold_name}_{graph_path}_train_test_tsne.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        if self.args.print_process:
            print(f"train/test t-SNE visualization saved to: {save_path}")
    
    def val(self, val_loader, criterion):
        total_loss = []
        self.model.eval()
        self.latest_val_best_threshold = None
        preds=[]
        targets=[]
        probs=[]
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                x_enc, label, correlation_matrix, site_label = self._unpack_batch(batch)
                x_enc = self._to_device(x_enc)
                correlation_matrix = self._to_device(correlation_matrix)
                label = self._to_device(label)
                class_label = label.long()
                metric_label = label
                
                if self.args.classes!=2:
                    one_hot_label = torch.zeros(len(label), self.args.classes, device=self.device)
                    one_hot_label.scatter_(1, label.unsqueeze(1), 1)
                    label=one_hot_label
                    metric_label = torch.argmax(label, dim=1)
                else:
                    label = label.to(torch.float32).view(-1, 1)
                
                if site_label is not None:
                    site_label = self._to_device(site_label)
                y_hat = self._model_forward(
                    x_enc,
                    correlation_matrix=correlation_matrix,
                    site_label=site_label,
                )
                primary_loss = None
                if self.model.training:
                    primary_loss = self._get_model_primary_loss(class_label)
                loss = primary_loss if primary_loss is not None else criterion(y_hat, label)
                total_loss.append(loss.cpu().numpy())
                pred, prob, target = self._prediction_probability_for_metrics(y_hat, metric_label)
                probs.append(prob)
                preds.append(pred)
                targets.append(target)
            if len(preds)>0:
                preds = np.concatenate(preds, axis=0)    
                targets = np.concatenate(targets, axis=0)  
                probs = np.concatenate(probs, axis=0)  
            else:
                preds=preds[0]
                targets=targets[0]
                probs=probs[0]
        total_loss = np.average(total_loss)
        self.latest_val_best_threshold = self._best_binary_threshold(targets, probs)
        self.model.train()
        return total_loss,evaluate(targets, preds,self.args.classes,probs)

    def train(self, train_loader, val_loader,check_path):
        if not os.path.exists(check_path):
            os.makedirs(check_path)
        
        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=self. args.print_process)
        best_monitor_score = -float("inf")
        best_binary_threshold = 0.5
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        self._init_model_ema()
        epoch_metric_records = []
        
        for epoch in range(self.args.train_epochs):
            self._set_model_train_epoch(epoch)
            train_loss = []
            cls_losses = []
            aux_loss_parts = {
                "causal_recon_loss": [],
                "temporal_pred_loss": [],
                "causal_dag_loss": [],
                "causal_l1_loss": [],
                "temporal_sparse_loss": [],
                "temporal_smooth_loss": [],
                "temporal_counterfactual_loss": [],
                "temporal_counterfactual_weighted_loss": [],
                "temporal_counterfactual_effect_mean": [],
                "temporal_counterfactual_effect_max": [],
                "temporal_counterfactual_edge_mean": [],
                "sample_graph_l1_loss": [],
                "sample_graph_deviation_loss": [],
                "temporal_reg_scale": [],
                "module1_denoise_loss": [],
                "module1_denoise_weighted_loss": [],
                "causal_stability_loss": [],
                "causal_stability_weighted_loss": [],
                "hgcn_radius_loss": [],
                "hgcn_radius_weighted_loss": [],
                "hgcn_cls_aux_loss": [],
                "hgcn_cls_aux_weighted_loss": [],
                "hgcn_view_consistency_loss": [],
                "hgcn_view_consistency_weighted_loss": [],
                "hgcn_supcon_loss": [],
                "hgcn_supcon_weighted_loss": [],
                "causal_recon_weighted_loss": [],
                "causal_dag_weighted_loss": [],
                "causal_l1_weighted_loss": [],
                "temporal_sparse_weighted_loss": [],
                "temporal_smooth_weighted_loss": [],
                "sample_graph_l1_weighted_loss": [],
                "sample_graph_deviation_weighted_loss": [],
                "causal_aux_loss": [],
                "hpec_mle_loss": [],
                "hpec_pcl_loss": [],
                "hpec_pal_loss": [],
                "hpec_radius_loss": [],
                "hpec_radius_weighted_loss": [],
                "hpec_diversity_loss": [],
                "hpec_diversity_weighted_loss": [],
                "hpec_hsic_loss": [],
                "hpec_hsic_weighted_loss": [],
                "hpec_intra_orthogonal_loss": [],
                "hpec_intra_orthogonal_weighted_loss": [],
                "hpec_inter_margin_loss": [],
                "hpec_inter_margin_weighted_loss": [],
                "hpec_class_center_margin_loss": [],
                "hpec_class_center_margin_weighted_loss": [],
                "hpec_anchor_loss": [],
                "hpec_anchor_weighted_loss": [],
                "hpec_logit_gate": [],
                "prototype_cos_abs_mean": [],
                "prototype_cos_abs_max": [],
                "prototype_same_class_cos_max": [],
                "hpec_sinkhorn_assignment_entropy": [],
                "hpec_sinkhorn_usage_min": [],
                "hpec_sinkhorn_usage_max": [],
                "hpec_prototype_aux_loss": [],
                "hpec_ce_aux_loss": [],
                "hpec_ce_aux_weighted_loss": [],
                "hpec_final_ce_loss": [],
                "hpec_energy_loss": [],
                "hpec_energy_weighted_loss": [],
                "site_adversarial_loss": [],
                "site_modulation_reg_loss": [],
                "site_adversarial_weighted_loss": [],
                "z_radius_mean": [],
                "z_radius_max": [],
                "z_tangent_norm_mean": [],
                "fc_residual_norm_mean": [],
                "causal_meta_analytic_spectral_radius": [],
                "causal_meta_dagma_spectral_radius": [],
                "causal_meta_shared_adjacency_mass_mean": [],
                "causal_meta_shared_adjacency_directionality_ratio": [],
                "causal_meta_a0_adjacency_mass_mean": [],
                "causal_meta_a0_adjacency_directionality_ratio": [],
                "causal_meta_alag_mean_adjacency_mass_mean": [],
                "causal_meta_alag_mean_adjacency_directionality_ratio": [],
                "causal_meta_dagma_stage_id": [],
                "causal_meta_dagma_effective_scale": [],
                "causal_meta_sample_delta_abs_mean": [],
            }
            preds=[]
            targets=[]
            probs=[]
            self.model.train()
            epoch_time = time.time()
            for _, batch in enumerate(train_loader):
                x_enc, label, correlation_matrix, site_label = self._unpack_batch(batch)
                x_enc = self._to_device(x_enc)
                correlation_matrix = self._to_device(correlation_matrix)
                if site_label is not None:
                    site_label = self._to_device(site_label)
                label = self._to_device(label)
                class_label = label.long()
                metric_label = label
                
                if self.args.classes!=2:
                    one_hot_label = torch.zeros(len(label), self.args.classes, device=self.device)
                    one_hot_label.scatter_(1, label.unsqueeze(1), 1)
                    label=one_hot_label
                    metric_label = torch.argmax(label, dim=1)
                else:
                    label = label.to(torch.float32).view(-1, 1)
                
                
                model_optim.zero_grad()
                y_hat= self._model_forward(
                    x_enc,
                    correlation_matrix=correlation_matrix,
                    site_label=site_label,
                )
                primary_loss = None
                if self.model.training:
                    primary_loss = self._get_model_primary_loss(class_label)
                loss = primary_loss if primary_loss is not None else criterion(y_hat, label)
                aux_loss = self._get_model_aux_loss()
                aux_losses = self._get_model_aux_losses()
                prototype_aux_loss = aux_losses.get("hpec_prototype_aux_loss")
                site_adversarial_loss = self._get_model_site_adversarial_loss(site_label)
                # 模块 3 启用后，分类 loss 与模块 2 的因果图共享同一条计算图。
                # 合并成一次 backward，保证 Loss_cls 能经 HGCN 回传到 A_learned。
                total_loss = loss + aux_loss if aux_loss is not None else loss
                if prototype_aux_loss is not None:
                    total_loss = total_loss + prototype_aux_loss
                if site_adversarial_loss is not None:
                    total_loss = total_loss + site_adversarial_loss
                total_loss.backward()
                model_optim.step()
                self._update_model_ema()
                cls_losses.append(loss.detach().cpu().item())
                for key in aux_loss_parts:
                    value = aux_losses.get(key)
                    if value is not None:
                        aux_loss_parts[key].append(value.detach().cpu().item())
                total_batch_loss = loss.detach()
                if aux_loss is not None:
                    total_batch_loss = total_batch_loss + aux_loss.detach()
                if prototype_aux_loss is not None:
                    total_batch_loss = total_batch_loss + prototype_aux_loss.detach()
                if site_adversarial_loss is not None:
                    total_batch_loss = total_batch_loss + site_adversarial_loss.detach()
                train_loss.append(total_batch_loss.cpu().numpy())
                pred, prob, target = self._prediction_probability_for_metrics(y_hat, metric_label)
                
                probs.append(prob)
                preds.append(pred)
                targets.append(target)
            if len(preds)>0:
                preds = np.concatenate(preds, axis=0)    
                targets = np.concatenate(targets, axis=0)   
                probs = np.concatenate(probs, axis=0) 
            else:
                preds=preds[0]
                targets=targets[0]
                probs=probs[0]
            train_metric =evaluate(targets, preds,self.args.classes,probs)
            best_threshold_info = self._best_binary_threshold(targets, probs)
            train_loss = np.average(train_loss)
            avg_cls_loss = float(np.average(cls_losses)) if cls_losses else float("nan")
            loss_parts = {"cls_loss": avg_cls_loss}
            if bool(getattr(self.args, "use_hpec_module4", 0)):
                loss_parts["hpec_loss"] = avg_cls_loss
            for key, values in aux_loss_parts.items():
                if values:
                    loss_parts[key] = float(np.average(values))
            val_loss,val_metric = self.val(val_loader,criterion)
            val_threshold_info = getattr(self, "latest_val_best_threshold", None)
            epoch_record = {
                "epoch": int(epoch + 1),
                "total_epochs": int(self.args.train_epochs),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            epoch_record.update(self._metric_dict("train", train_metric))
            epoch_record.update(self._metric_dict("val", val_metric))
            if best_threshold_info is not None:
                epoch_record["train_best_threshold"] = float(best_threshold_info[0])
                epoch_record["train_best_macro_f1"] = float(best_threshold_info[1])
            if val_threshold_info is not None:
                epoch_record["val_best_threshold"] = float(val_threshold_info[0])
                epoch_record["val_best_macro_f1"] = float(val_threshold_info[1])
            for key, value in loss_parts.items():
                epoch_record[key] = float(value)
            epoch_metric_records.append(epoch_record)
            metric_name_to_index = {
                "accuracy": 0,
                "acc": 0,
                "precision": 1,
                "recall": 2,
                "macro_f1": 3,
                "f1": 3,
                "roc_auc": 4,
                "auc": 4,
            }
            monitor_name = str(getattr(self.args, "early_stop_metric", "accuracy")).lower()
            if monitor_name in ("best_macro_f1", "best_f1") and val_threshold_info is not None:
                monitor_score = float(val_threshold_info[1])
            else:
                monitor_index = metric_name_to_index.get(monitor_name, 0)
                monitor_score = float(val_metric[monitor_index])
            if monitor_score > best_monitor_score:
                best_monitor_score = monitor_score
                if val_threshold_info is not None:
                    best_binary_threshold = float(val_threshold_info[0])
                    threshold_path = os.path.join(check_path, "best_threshold.txt")
                    with open(threshold_path, "w", encoding="utf-8") as f:
                        f.write(f"{best_binary_threshold:.6f}\n")

            print_metric_every = int(getattr(self.args, "print_metric_every", 0) or 0)
            should_print_metric = bool(getattr(self.args, "print_process", 0)) or (
                print_metric_every > 0
                and ((epoch + 1) % print_metric_every == 0 or epoch == 0 or epoch + 1 == self.args.train_epochs)
            )
            if should_print_metric:
                threshold_suffix = ""
                if best_threshold_info is not None:
                    threshold_suffix = (
                        f" | train_best_threshold: {best_threshold_info[0]:.2f}"
                        f" | train_best_macro_f1: {best_threshold_info[1]:.4f}"
                    )
                if val_threshold_info is not None:
                    threshold_suffix += (
                        f" | val_best_threshold: {val_threshold_info[0]:.2f}"
                        f" | val_best_macro_f1: {val_threshold_info[1]:.4f}"
                    )
                print(
                    self._format_metric_line(
                        f"Epoch {epoch + 1}/{self.args.train_epochs}{threshold_suffix}",
                        train_loss,
                        val_loss,
                        train_metric,
                        val_metric,
                        loss_parts,
                    )
                )
            early_stopping(-monitor_score, self.model, check_path)
            if self. args.print_process:
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        self._apply_model_ema()
        self._save_causal_visualization_after_training(train_loader, val_loader, check_path)
        self._save_graph_diagnostics_after_training(check_path)
        self._save_tsne_after_training(train_loader, val_loader, check_path)
        self._save_epoch_metrics(check_path, epoch_metric_records)
        final_threshold = None
        if epoch_metric_records:
            final_threshold = epoch_metric_records[-1].get("val_best_threshold")
        return final_threshold
    def _load_best_threshold(self, check_path):
        threshold_path = os.path.join(check_path, "best_threshold.txt")
        if not os.path.exists(threshold_path):
            return None
        try:
            with open(threshold_path, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except ValueError:
            return None

    def val_with_threshold(self, val_loader, criterion, threshold=None):
        total_loss = []
        self.model.eval()
        preds=[]
        targets=[]
        probs=[]
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                x_enc, label, correlation_matrix, site_label = self._unpack_batch(batch)
                x_enc = self._to_device(x_enc)
                correlation_matrix = self._to_device(correlation_matrix)
                label = self._to_device(label)
                class_label = label.long()
                metric_label = label
                if self.args.classes!=2:
                    one_hot_label = torch.zeros(len(label), self.args.classes, device=self.device)
                    one_hot_label.scatter_(1, label.unsqueeze(1), 1)
                    label=one_hot_label
                    metric_label = torch.argmax(label, dim=1)
                else:
                    label = label.to(torch.float32).view(-1, 1)
                if site_label is not None:
                    site_label = self._to_device(site_label)
                y_hat = self._model_forward(
                    x_enc,
                    correlation_matrix=correlation_matrix,
                    site_label=site_label,
                )
                primary_loss = self._get_model_primary_loss(class_label)
                loss = primary_loss if primary_loss is not None else criterion(y_hat, label)
                total_loss.append(loss.cpu().numpy())
                pred, prob, target = self._prediction_probability_with_threshold(
                    y_hat,
                    metric_label,
                    threshold=threshold,
                )
                probs.append(prob)
                preds.append(pred)
                targets.append(target)
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        probs = np.concatenate(probs, axis=0)
        self._latest_eval_arrays = (preds, probs, targets)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss,evaluate(targets, preds,self.args.classes,probs)

    def kf_train(self, setting):
        train_loaders,val_loaders=self._get_data()
        val_metrics=[]
        final_metric_records = []
        max_folds = int(getattr(self.args, "max_folds", 0) or 0)
        fold_pairs = list(zip(train_loaders, val_loaders))
        if max_folds > 0:
            fold_pairs = fold_pairs[:max_folds]
        total_folds = len(fold_pairs)
        for fold, (train_loader, val_loader) in tqdm(enumerate(fold_pairs), total=total_folds, desc="Cross-validation", ncols=100):
            check_path = os.path.join(self.args.checkpoints, setting+'fold'+str(fold + 1))
            self.reset_model()
            module_summary = self._module_path_summary()
            if module_summary is not None:
                print(f"Fold {fold + 1}/{self.args.kfold} {module_summary}")
            if self. args.print_process: print(f"Fold {fold + 1}/{self.args.kfold} Start>>>>>>>>>>>>>>>>>>>\n")
            final_epoch_threshold = self.train(train_loader, val_loader, check_path)

            _, raw_test_metric = self.val_with_threshold(
                val_loader,
                self._select_criterion(),
                threshold=None,
            )
            raw_preds, raw_probs, raw_targets = self._latest_eval_arrays
            threshold_for_report = (
                final_epoch_threshold
                if bool(getattr(self.args, "use_best_threshold", 1))
                else None
            )
            _,test_metric = self.val_with_threshold(
                val_loader,
                self._select_criterion(),
                threshold=threshold_for_report,
            )
            final_preds, final_probs, final_targets = self._latest_eval_arrays
            raw_diagnostics = self._prediction_diagnostics(raw_targets, raw_preds, raw_probs)
            final_diagnostics = self._prediction_diagnostics(final_targets, final_preds, final_probs)
            print(
                f"Fold {fold + 1} final-epoch test raw-threshold: "
                f"accuracy={raw_test_metric[0]:.4f}, precision={raw_test_metric[1]:.4f}, "
                f"recall={raw_test_metric[2]:.4f}, macro_f1={raw_test_metric[3]:.4f}, "
                f"roc_auc={raw_test_metric[4]:.4f}"
            )
            print(f"Fold {fold + 1} final-epoch test raw diagnostics: {self._format_prediction_diagnostics(raw_diagnostics)}")
            if threshold_for_report is not None:
                print(f"Fold {fold + 1} final-epoch test uses final-epoch threshold: {threshold_for_report:.4f}")
                print(f"Fold {fold + 1} final-epoch test thresholded diagnostics: {self._format_prediction_diagnostics(final_diagnostics)}")
            val_metrics.append(raw_test_metric)
            fold_record = {
                "record_type": "fold",
                "fold": int(fold + 1),
                "total_folds": int(total_folds),
                "epoch_source": "final_epoch_current_model",
                "summary_metric_source": "raw_final_epoch_test",
                "threshold_source": "final_epoch_test_threshold_reference_only" if threshold_for_report is not None else "fixed_0.5",
                "best_threshold": None if threshold_for_report is None else float(threshold_for_report),
            }
            fold_record.update(self._metric_dict("raw", raw_test_metric))
            fold_record.update(self._metric_dict("raw_final_epoch_test", raw_test_metric))
            fold_record.update(self._metric_dict("final", raw_test_metric))
            fold_record.update(self._metric_dict("thresholded_final_epoch_test", test_metric))
            fold_record["raw_diagnostics"] = raw_diagnostics
            fold_record["final_diagnostics"] = final_diagnostics
            final_metric_records.append(fold_record)
            self._save_final_metrics(check_path, final_metric_records)
            
            ## Uncomment below code for save space on device
            if self. args.del_weight: 
                self.del_weight(check_path)
            if self. args.print_process: print(f"Fold {fold + 1}/{self.args.kfold} End<<<<<<<<<<<<<<<<<<<\n")
        avg_metric=[np.mean([val_metrics[i][j] for i in range(len(val_metrics))]) for j in range(5)]
        print(f'Final-epoch Test Avg (raw) accuracy: {avg_metric[0]:.4f}, precision: {avg_metric[1]:.4f}, recall: {avg_metric[2]:.4f}, macro_f1: {avg_metric[3]:.4f}, roc_auc: {avg_metric[4]:.4f}')
        if fold_pairs:
            summary_record = {
                "record_type": "summary",
                "fold": "avg",
                "total_folds": int(total_folds),
            }
            summary_record.update(self._metric_dict("final", avg_metric))
            summary_check_path = os.path.join(self.args.checkpoints, setting + "summary")
            final_metric_records.append(summary_record)
            self._save_final_metrics(summary_check_path, final_metric_records)
        return avg_metric
    
    def del_weight(self, path):
        if os.path.exists(os.path.join(os.path.join(path, 'checkpoint.pth'))):
            os.remove(os.path.join(os.path.join(path, 'checkpoint.pth')))
            if self. args.print_process: print('Model weights deleted....')

    def svm(self, train_loader, val_loader):
        X_train, y_train = map(lambda batches: np.concatenate(batches, axis=0), zip(*train_loader))
        X_val,   y_val   = map(lambda batches: np.concatenate(batches, axis=0), zip(*val_loader))

        # Compute the Pearson correlation matrix (FC) for each sample and stack into shape (B, N, N)
        FC_train = np.stack([np.corrcoef(x.T) for x in X_train], axis=0)
        FC_val   = np.stack([np.corrcoef(x.T) for x in X_val],   axis=0)
        FC_train = np.nan_to_num(FC_train, nan=0.0)
        FC_val   = np.nan_to_num(FC_val,   nan=0.0)

        # Determine the number of ROIs (N) and create a boolean mask for the upper triangle (including diagonal)
        N = FC_train.shape[-1]
        mask = np.triu(np.ones((N, N), dtype=bool), k=0)  # shape = (N, N)

        # Extract and flatten only the upper-triangle (including diagonal) elements: shape -> (B, N*(N+1)/2)
        X_train_feat = FC_train[:, mask]
        X_val_feat   = FC_val[:,   mask]

        # Initialize and train the SVM with probability outputs enabled
        svm = SVC(C=0.1, kernel='rbf', probability=True, random_state=self.args.seed)
        svm.fit(X_train_feat, y_train)

        # Predict class probabilities
        proba = svm.predict_proba(X_val_feat)
        if self.args.classes == 2:
            # For binary classification, take the probability of the positive class
            probs = proba[:, 1]
            preds = (probs > 0.5).astype(int)
        else:
            # For multiclass, apply softmax (optional calibration) and choose the class with highest probability
            probs = softmax(proba, axis=1)
            preds = np.argmax(probs, axis=1)

        # Evaluate predictions against ground truth
        val_metric = evaluate(y_val, preds, self.args.classes, probs)
        return None, val_metric

    def rf(self, train_loader, val_loader):
        # Concatenate all batches of time-series data (B, T, N) and labels
        X_train_ts, y_train = map(lambda batches: np.concatenate(batches, axis=0), zip(*train_loader))
        X_val_ts,   y_val   = map(lambda batches: np.concatenate(batches, axis=0), zip(*val_loader))

        # Compute the Pearson correlation matrix (FC) for each sample and stack into shape (B, N, N)
        FC_train = np.stack([np.corrcoef(x.T) for x in X_train_ts], axis=0)
        FC_val   = np.stack([np.corrcoef(x.T) for x in X_val_ts],   axis=0)
        FC_train = np.nan_to_num(FC_train, nan=0.0)
        FC_val   = np.nan_to_num(FC_val,   nan=0.0)

        # Determine the number of ROIs (N) and create a boolean mask for the upper triangle (including diagonal)
        N    = FC_train.shape[-1]
        mask = np.triu(np.ones((N, N), dtype=bool), k=0)  # shape = (N, N)

        # Extract and flatten only the upper-triangle (including diagonal) elements: shape -> (B, N*(N+1)/2)
        X_train_feat = FC_train[:, mask]
        X_val_feat   = FC_val[:,   mask]

        # Initialize and train the Random Forest classifier with specified hyperparameters
        rf_clf = RandomForestClassifier(
            n_estimators=12,  # number of trees
            max_depth=2,        # maximum tree depth
            random_state=self.args.seed,
            n_jobs=4               # number of parallel jobs
        )
        rf_clf.fit(X_train_feat, y_train)

        # Predict class probabilities
        proba = rf_clf.predict_proba(X_val_feat)
        if self.args.classes == 2:
            # For binary classification, take the probability of the positive class
            probs = proba[:, 1]
            preds = (probs > 0.5).astype(int)
        else:
            # For multiclass, apply softmax (optional calibration) and choose the class with highest probability
            probs = softmax(proba, axis=1)
            preds = np.argmax(probs, axis=1)

        # Evaluate predictions against ground truth
        val_metric = evaluate(y_val, preds, self.args.classes, probs)
        return None, val_metric


    
    def kf_ML(self, setting):
        train_loaders,val_loaders=self._get_data()
        val_metrics=[]
        for fold, (train_loader, val_loader) in tqdm(enumerate(zip(train_loaders, val_loaders)), total=self.args.kfold, desc="Cross-validation", ncols=100):
            _,val_metric = self.svm(train_loader, val_loader) if (self.args.Method == 'SVM' or self.args.Method=='svm') else self.rf(train_loader, val_loader)
            val_metrics.append(val_metric)
        avg_metric=[np.mean([val_metrics[i][j] for i in range(self.args.kfold)]) for j in range(5)]
        print(f'Test Avg accuracy: {avg_metric[0]:.4f}, precision: {avg_metric[1]:.4f}, recall: {avg_metric[2]:.4f}, macro_f1: {avg_metric[3]:.4f}, roc_auc: {avg_metric[4]:.4f}')
        return avg_metric
