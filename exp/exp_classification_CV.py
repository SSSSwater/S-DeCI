import json
import hashlib
import os
import re
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

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None

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

class WeightedMSELoss(nn.Module):
    def __init__(self, positive_weight=1.0):
        super().__init__()
        self.positive_weight = float(positive_weight)

    def forward(self, prediction, target):
        target = target.to(prediction.dtype)
        weight = torch.where(
            target > 0.5,
            torch.as_tensor(self.positive_weight, device=prediction.device, dtype=prediction.dtype),
            torch.ones((), device=prediction.device, dtype=prediction.dtype),
        )
        return (weight * (prediction - target).pow(2)).mean()

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
        hpec_proto_lr_scale = float(getattr(self.args, "hpec_prototype_lr_scale", 1.0) or 1.0)
        use_hpec_proto_group = hpec_proto_lr_scale > 0 and abs(hpec_proto_lr_scale - 1.0) > 1e-8
        if (causal_lr is None or causal_lr <= 0) and not use_hpec_proto_group:
            model_optim = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=weight_decay,
            )
            return model_optim

        base_params = []
        causal_params = []
        hpec_proto_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "causal_learner" in name:
                causal_params.append(param)
            elif (
                use_hpec_proto_group
                and "hpec_module4" in name
                and (
                    "prototypes" in name
                    or "prototype_tangent_direction" in name
                    or "busemann_class_bias" in name
                )
            ):
                hpec_proto_params.append(param)
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
        if hpec_proto_params:
            param_groups.append(
                {
                    "params": hpec_proto_params,
                    "lr": self.args.learning_rate * hpec_proto_lr_scale,
                    "weight_decay": 0.0,
                }
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
            if loss_name in ("weighted_mse", "wmse"):
                return WeightedMSELoss(
                    positive_weight=float(getattr(self.args, "binary_positive_weight", 1.0))
                )
            return nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        return criterion
    def _supervised_loss(self, y_hat, binary_label, class_label, criterion):
        """兼容二分类单概率输出和二分类双 logits 输出。"""
        if self.args.classes == 2 and y_hat.ndim >= 2 and y_hat.shape[-1] > 1:
            return nn.functional.cross_entropy(y_hat, class_label.view(-1).long())
        return criterion(y_hat, binary_label)
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
    def _get_model_supervised_aux_loss(self, labels):
        model = self._model_for_aux_loss()
        compute_aux_loss = getattr(model, "compute_supervised_aux_loss", None)
        if not callable(compute_aux_loss):
            return None
        return compute_aux_loss(labels)
    def _set_model_train_epoch(self, epoch):
        model = self._model_for_aux_loss()
        set_train_epoch = getattr(model, "set_train_epoch", None)
        if callable(set_train_epoch):
            set_train_epoch(epoch)
    def _update_reliable_prototypes_after_step(self, labels):
        model = self._model_for_aux_loss()
        update = getattr(model, "update_reliable_prototypes_after_step", None)
        if not callable(update):
            return {}
        return update(labels)
    def _finalize_epoch_prototype_update(self):
        model = self._model_for_aux_loss()
        finalize = getattr(model, "finalize_epoch_prototype_update", None)
        if not callable(finalize):
            return {}
        return finalize()
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
            if y_hat.ndim >= 2 and y_hat.shape[-1] > 1:
                prob = torch.nn.functional.softmax(y_hat, dim=1)[:, 1]
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

    def _prediction_probability_from_logits(self, logits, metric_label):
        """把任意分支 logits 转成统一的 pred/prob/target，便于比较各模块贡献。"""
        if logits is None:
            return None
        if self.args.classes != 2:
            prob_tensor = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(prob_tensor, dim=1).detach().cpu().numpy()
            prob = prob_tensor.detach().cpu().numpy()
            target = metric_label.detach().cpu().numpy()
            return pred, prob, target
        if logits.ndim >= 2 and logits.shape[-1] > 1:
            prob_tensor = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        else:
            prob_tensor = torch.sigmoid(logits.reshape(-1))
        pred = (prob_tensor > 0.5).to(metric_label.dtype).detach().cpu().numpy()
        prob = prob_tensor.detach().cpu().numpy()
        target = metric_label.detach().cpu().numpy()
        return pred, prob, target

    def _branch_logits_after_forward(self):
        model = self._model_for_aux_loss()
        branches = {}
        gcn_output = getattr(model, "latest_gcn_fallback_output", None)
        if gcn_output is not None and getattr(gcn_output, "logits", None) is not None:
            branches["gcn_fallback"] = gcn_output.logits
            graph_only_logits = getattr(gcn_output, "graph_only_logits", None)
            if graph_only_logits is not None:
                branches["gcn_graph_only"] = graph_only_logits
            fc_only_logits = getattr(gcn_output, "fc_only_logits", None)
            if fc_only_logits is not None:
                branches["gcn_fc_only"] = fc_only_logits
        module34_logits = getattr(model, "latest_module34_branch_logits", None)
        if module34_logits is not None:
            branches["module34_branch"] = module34_logits
        module4_output = getattr(model, "latest_module4_output", None)
        if module4_output is not None and getattr(module4_output, "energy_matrix", None) is not None:
            branches["hpec_energy"] = -module4_output.energy_matrix
        final_logits = getattr(model, "latest_prediction_logits", None)
        if final_logits is not None:
            branches["final_cached"] = final_logits
        return branches

    def _collect_branch_batch_metrics(self, metric_label):
        batch_metrics = {}
        for branch_name, logits in self._branch_logits_after_forward().items():
            converted = self._prediction_probability_from_logits(logits, metric_label)
            if converted is None:
                continue
            batch_metrics[branch_name] = converted
        return batch_metrics

    def _append_branch_batch_metrics(self, storage, batch_metrics):
        for branch_name, (pred, prob, target) in batch_metrics.items():
            branch_storage = storage.setdefault(branch_name, {"preds": [], "probs": [], "targets": []})
            branch_storage["preds"].append(pred)
            branch_storage["probs"].append(prob)
            branch_storage["targets"].append(target)

    def _finalize_branch_metrics(self, storage):
        metrics = {}
        for branch_name, values in storage.items():
            if not values["preds"]:
                continue
            preds = np.concatenate(values["preds"], axis=0)
            probs = np.concatenate(values["probs"], axis=0)
            targets = np.concatenate(values["targets"], axis=0)
            metrics[branch_name] = evaluate(targets, preds, self.args.classes, probs)
        return metrics

    def _branch_metric_record(self, prefix, branch_metrics):
        record = {}
        names = ("accuracy", "precision", "recall", "macro_f1", "roc_auc")
        for branch_name, metric in branch_metrics.items():
            safe_branch_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(branch_name)).strip("_")
            for name, value in zip(names, metric):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    record[f"{prefix}_{safe_branch_name}_{name}"] = value
        return record

    def _format_branch_metric_line(self, split_name, branch_metrics):
        if not branch_metrics:
            return ""
        pieces = []
        for branch_name in sorted(branch_metrics):
            metric = branch_metrics[branch_name]
            pieces.append(
                f"{branch_name}: acc={metric[0]:.4f}, f1={metric[3]:.4f}, auc={metric[4]:.4f}"
            )
        return f"[{split_name} Branch Metrics] " + " | ".join(pieces)
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
        # setting 中包含 kfold_5，必须优先匹配末尾的 runfoldN，避免五折图片都误标 fold_5。
        match = re.search(r"runfold\d+", fold_name, flags=re.IGNORECASE)
        if match is None:
            candidates = list(re.finditer(r"(?<!k)fold[_-]?\d+", fold_name, flags=re.IGNORECASE))
            match = candidates[-1] if candidates else None
        short_fold = match.group(0) if match else "fold"
        digest = hashlib.md5(f"{fold_name}_{graph_path}".encode("utf-8")).hexdigest()[:8]
        return f"{short_fold}_{graph_path}_{digest}"

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

    def _tensorboard_enabled(self):
        if not bool(int(getattr(self.args, "use_tensorboard", 1) or 0)):
            return False
        if bool(int(getattr(self.args, "tensorboard_disable_smoke_runs", 1) or 0)):
            run_name = str(getattr(self.args, "tensorboard_run_name", "") or "")
            if "smoke" in run_name.lower():
                return False
        return True

    def _safe_tensorboard_name(self, name):
        text = str(name or "run")
        text = re.sub(r"[^\w.\-+=]+", "_", text, flags=re.UNICODE)
        return text.strip("_") or "run"

    def _safe_tensorboard_path(self, name):
        parts = re.split(r"[\\/]+", str(name or "run"))
        safe_parts = [self._safe_tensorboard_name(part) for part in parts if str(part).strip()]
        return os.path.join(*safe_parts) if safe_parts else "run"

    def _tensorboard_root(self):
        root = getattr(self.args, "tensorboard_dir", "outputs/tensorboard") or "outputs/tensorboard"
        if not os.path.isabs(root):
            root = os.path.join(os.getcwd(), root)
        return root

    def _tensorboard_run_name(self, setting):
        configured = getattr(self.args, "tensorboard_run_name", None)
        return self._safe_tensorboard_path(configured or setting)

    def _tensorboard_config_label(self):
        items = [
            ("data", getattr(self.args, "data", None)),
            ("prot", getattr(self.args, "protocol", None)),
            ("seq", getattr(self.args, "seq_len", None)),
            ("bs", getattr(self.args, "batch_size", None)),
            ("lr", getattr(self.args, "learning_rate", None)),
            ("dp", getattr(self.args, "dropout", None)),
            ("dm", getattr(self.args, "d_model", None)),
            ("m1", getattr(self.args, "use_deci_module1", None)),
            ("m2", getattr(self.args, "use_causal_module2", None)),
            ("m3", getattr(self.args, "use_hgcn_module3", None)),
            ("m4", getattr(self.args, "use_hpec_module4", None)),
            ("lag", getattr(self.args, "temporal_lag_order", None)),
            ("graph", getattr(self.args, "causal_graph_method", None)),
            ("blend", getattr(self.args, "module2_sample_correlation_blend", None)),
        ]
        parts = []
        for key, value in items:
            if value is None:
                continue
            if isinstance(value, float):
                value = f"{value:g}"
            parts.append(f"{key}_{value}")
        return self._safe_tensorboard_name("_".join(parts) or "config")

    def _tensorboard_hparams(self):
        candidates = {
            "learning_rate": getattr(self.args, "learning_rate", None),
            "dropout": getattr(self.args, "dropout", None),
            "batch_size": getattr(self.args, "batch_size", None),
            "d_model": getattr(self.args, "d_model", None),
            "temporal_lag_order": getattr(self.args, "temporal_lag_order", None),
            "lambda_temporal_pred": getattr(self.args, "lambda_temporal_pred", None),
            "lambda_temporal_sparse": getattr(self.args, "lambda_temporal_sparse", None),
            "lambda_temporal_smooth": getattr(self.args, "lambda_temporal_smooth", None),
            "lambda_causal_dag": getattr(self.args, "lambda_causal_dag", None),
            "module2_sample_correlation_blend": getattr(self.args, "module2_sample_correlation_blend", None),
            "module2_graph_residual_alpha": getattr(self.args, "module2_graph_residual_alpha", None),
            "mac_min_radius": getattr(self.args, "mac_min_radius", None),
            "mac_max_radius": getattr(self.args, "mac_max_radius", None),
            "hbr_loss_weight": getattr(self.args, "hbr_loss_weight", None),
            "hpec_energy_loss_weight": getattr(self.args, "hpec_energy_loss_weight", None),
            "hpec_logit_temperature": getattr(self.args, "hpec_logit_temperature", None),
            "hpec_prototypes_per_class": getattr(self.args, "hpec_prototypes_per_class", None),
        }
        hparams = {}
        for key, value in candidates.items():
            if value is None:
                continue
            if isinstance(value, (bool, int, float, str)):
                hparams[key] = value
        return hparams

    def _tensorboard_hparam_step(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(value):
            return 0
        scale = 1000000.0 if abs(value) < 1.0 else 1000.0
        return int(round(value * scale))

    def _html_entity_encode_non_ascii(self, text):
        # Keep TensorBoard event text ASCII-only, while the browser renders Chinese normally.
        return "".join(char if ord(char) < 128 else f"&#x{ord(char):X};" for char in str(text))

    def _tensorboard_notes_markdown_zh(self, run_part):
        config_label = self._tensorboard_config_label()
        return f"""# TensorBoard 图表说明

当前 run: `{run_part}`

配置摘要: `{config_label}`

## 图表分类

- `FoldTrend/<配置摘要>/Loss/*`: 每个 epoch 的损失趋势。每个 fold 是一个独立 run，因此同一张图里可以叠加查看 `fold_1` 到 `fold_5`。
- `FoldTrend/<配置摘要>/Metrics/*`: 每个 epoch 的训练集和验证集指标趋势，用于观察是否过拟合或欠拟合。
- `FoldTrend/<配置摘要>/Threshold/*`: 二分类阈值诊断，只用于观察训练过程中的最佳阈值变化。
- `FoldTrend/<配置摘要>/Diagnostics/*`: 模块诊断量，用于看因果图、双曲表示和原型是否在训练中发生有效变化。
- `Final/raw/<配置摘要>/*`: 每个 fold 最后一个 epoch 的测试集指标，横轴 step 是 fold 编号。
- `Summary/avg/<配置摘要>/*`: 所有 fold 的最终平均指标，表示当前配置的整体结果。
- `HparamEffect/<超参数>/<指标>`: 超参数取值对最终指标的影响。横轴 step 是超参数值的数值编码，小于 1 的值按 `value * 1e6` 显示，大于等于 1 的值按 `value * 1e3` 显示。
- `CompareTrend/*`: 当前配置 5 个 fold 逐 epoch 均值趋势，适合跨超参数取值直接叠加比较。
- `CompareFinal/*`: 当前配置 5 个 fold 的最终平均指标，适合快速看不同配置最终结果。
- `Custom Scalars`: TensorBoard 自定义分类面板，把常用趋势按 Overview、Loss、Module2、Module34、Final 分组。
- `HParams`: TensorBoard 自带的超参数表格视图，记录本次配置和最终指标。

## 常用指标

- `accuracy`: 准确率。
- `precision`: 精确率。
- `recall`: 召回率。
- `macro_f1`: 宏平均 F1，更适合类别不均衡时观察整体分类质量。
- `auc`: ROC AUC，用于观察预测概率的排序能力。

## 主要损失

- `Loss/train_total`: 训练总损失，包含分类损失和启用模块的辅助损失。
- `Loss/val_total`: 验证集主损失。
- `Loss/cls`: 分类主损失。
- `Loss/module2_temporal_pred`: 模块 2 时间序列预测式因果学习损失。
- `Loss/module2_dag`: 模块 2 中 `A0` 的 DAG 约束损失。
- `Loss/module2_sparse`: 模块 2 因果图稀疏约束。
- `Loss/module2_smooth`: 模块 2 lag 间平滑约束。
- `Loss/module4_hpec_final_ce`: 模块 4 HPEC 最终分类 CE 损失，模块 4 关闭时为 0。
- `Loss/module4_hpec_energy_weighted`: 模块 4 HPEC 能量损失，模块 4 关闭时为 0。

## 诊断量

- `Diagnostics/graph_mass`: 当前用于分类的图平均边强度。
- `Diagnostics/directionality`: 图的方向性/非对称性比例，越大表示有向差异越明显。
- `Diagnostics/a_lag_mass`: 模块 2 跨时间 lag 因果图的平均边强度。
- `Diagnostics/a0_mass`: 同一时间片残余依赖图 `A0` 的平均边强度。
- `Diagnostics/z_radius`: 双曲表示半径，模块 3/4 关闭时通常为 0。
- `Diagnostics/z_tangent_norm`: 双曲切空间表示范数，模块 3/4 关闭时通常为 0。
- `Diagnostics/prototype_cos_mean`: 原型余弦相似度平均值，越高说明原型越接近。
- `Diagnostics/prototype_cos_max`: 原型最大余弦相似度，用于观察是否有原型挤在一起。

## 读图建议

- 先看 `Metrics/val_macro_f1` 和 `Metrics/val_auc`，确认泛化趋势。
- 再对比 `Loss/train_total` 与 `Loss/val_total`，如果训练下降但验证上升，通常是过拟合。
- 看模块 2 时重点观察 `module2_temporal_pred`、`a_lag_mass` 和 `directionality`。
- 做 sweep 时优先看 `CompareTrend/Metrics/val_macro_f1`、`CompareTrend/Metrics/val_auc` 和 `HparamEffect/<被扫超参数>/macro_f1`。
"""

    def _tensorboard_notes_markdown(self, run_part):
        config_label = self._tensorboard_config_label()
        return f"""# TensorBoard Chart Guide

Current run: `{run_part}`

Config label: `{config_label}`

## Chart Groups

- `FoldTrend/<config>/Loss/*`: epoch-level loss curves. Each fold is a separate run, so `fold_1` to `fold_5` can be overlaid in one scalar chart.
- `FoldTrend/<config>/Metrics/*`: epoch-level train and validation metrics. Use these charts to inspect overfitting or underfitting.
- `FoldTrend/<config>/Threshold/*`: binary-classification threshold diagnostics.
- `FoldTrend/<config>/Diagnostics/*`: module diagnostics, such as causal graph mass, graph directionality, hyperbolic radius, and prototype similarity.
- `Final/raw/<config>/*`: final-epoch test metrics for each fold. The x-axis step is the fold index.
- `Summary/avg/<config>/*`: averaged final metrics across all folds for the current configuration.
- `HparamEffect/<hparam>/<metric>`: effect of a hyperparameter value on final metrics. For numeric values below 1, the x-axis step is `value * 1e6`; for values greater than or equal to 1, the x-axis step is `value * 1e3`.
- `CompareTrend/*`: mean epoch-level trend across folds for the current config. This is the main view for comparing different hyperparameter values.
- `CompareFinal/*`: mean final metrics across folds for the current config.
- `Custom Scalars`: grouped TensorBoard dashboards for Overview, Loss, Module2, Module34, and Final.
- `HParams`: TensorBoard hparams plugin view with the recorded config and final metrics.

## Metrics

- `accuracy`: classification accuracy.
- `precision`: precision.
- `recall`: recall.
- `macro_f1`: macro-averaged F1, useful when classes are imbalanced.
- `auc`: ROC AUC, measuring ranking quality of predicted probabilities.

## Losses

- `Loss/train_total`: total training loss, including classification loss and enabled auxiliary losses.
- `Loss/val_total`: validation primary loss.
- `Loss/cls`: primary classification loss.
- `Loss/module2_temporal_pred`: Module 2 temporal predictive causal learning loss.
- `Loss/module2_dag`: DAG constraint loss for Module 2 `A0`.
- `Loss/module2_sparse`: sparsity regularization for Module 2 causal graph.
- `Loss/module2_smooth`: lag smoothness regularization for Module 2.
- `Loss/module4_hpec_final_ce`: final CE loss from Module 4 HPEC. It is 0 when Module 4 is disabled.
- `Loss/module4_hpec_energy_weighted`: weighted HPEC energy loss. It is 0 when Module 4 is disabled.

## Diagnostics

- `Diagnostics/graph_mass`: average edge strength of the graph used for classification.
- `Diagnostics/directionality`: graph asymmetry ratio. Larger values indicate stronger directed differences.
- `Diagnostics/a_lag_mass`: average edge strength of the cross-time lag causal graph from Module 2.
- `Diagnostics/a0_mass`: average edge strength of the same-time residual dependency graph `A0`.
- `Diagnostics/z_radius`: hyperbolic representation radius. It is usually 0 when Module 3/4 are disabled.
- `Diagnostics/z_tangent_norm`: tangent-space norm of hyperbolic features. It is usually 0 when Module 3/4 are disabled.
- `Diagnostics/prototype_cos_mean`: mean cosine similarity between prototypes. Higher values mean prototypes are closer.
- `Diagnostics/prototype_cos_max`: maximum cosine similarity between prototypes, useful for spotting collapsed prototypes.

## Reading Tips

- Start with `Metrics/val_macro_f1` and `Metrics/val_auc` to inspect generalization.
- Compare `Loss/train_total` and `Loss/val_total`; falling train loss with rising val loss usually indicates overfitting.
- For Module 2, focus on `module2_temporal_pred`, `a_lag_mass`, and `directionality`.
- For sweeps, first inspect `CompareTrend/Metrics/val_macro_f1`, `CompareTrend/Metrics/val_auc`, and `HparamEffect/<swept_hparam>/macro_f1`.
"""

    def _write_tensorboard_notes(self, writer, run_part):
        if writer is None:
            return
        chinese_notes = self._html_entity_encode_non_ascii(self._tensorboard_notes_markdown_zh(run_part))
        writer.add_text("Notes/chart_guide_cn", chinese_notes, 0)
        writer.add_text("Notes/chart_guide_zh", chinese_notes, 0)
        writer.add_text("Notes/chart_guide", self._tensorboard_notes_markdown(run_part), 0)
        writer.flush()

    def _write_tensorboard_custom_scalars(self, writer):
        if writer is None:
            return
        layout = {
            "Overview": {
                "validation_quality": [
                    "Multiline",
                    [
                        "CompareTrend/Metrics/val_macro_f1",
                        "CompareTrend/Metrics/val_auc",
                        "CompareTrend/Metrics/val_accuracy",
                    ],
                ],
                "accuracy_gap": [
                    "Multiline",
                    [
                        "CompareTrend/Metrics/train_accuracy",
                        "CompareTrend/Metrics/val_accuracy",
                    ],
                ],
                "branch_accuracy": [
                    "Multiline",
                    [
                        "CompareTrend/Branches/val_branch_gcn_fallback_accuracy",
                        "CompareTrend/Branches/val_branch_module34_branch_accuracy",
                        "CompareTrend/Branches/val_branch_hpec_energy_accuracy",
                        "CompareTrend/Branches/val_branch_final_cached_accuracy",
                    ],
                ],
            },
            "Loss": {
                "main_losses": [
                    "Multiline",
                    [
                        "CompareTrend/Loss/train_total",
                        "CompareTrend/Loss/val_total",
                        "CompareTrend/Loss/cls",
                        "CompareTrend/Loss/module2_temporal_pred",
                    ],
                ],
            },
            "Module2": {
                "causal_graph": [
                    "Multiline",
                    [
                        "CompareTrend/Module2/a_lag_mass",
                        "CompareTrend/Module2/a0_mass",
                        "CompareTrend/Module2/a0_to_alag_mass_ratio",
                        "CompareTrend/Module2/directionality",
                        "CompareTrend/Module2/graph_mass",
                        "CompareTrend/Module2/classification_graph_asymmetry",
                        "CompareTrend/Module2/causal_support_density",
                    ],
                ],
                "regularization": [
                    "Multiline",
                    [
                        "CompareTrend/Module2/dag_loss",
                        "CompareTrend/Module2/sparse_loss",
                        "CompareTrend/Module2/smooth_loss",
                        "CompareTrend/Module2/pred_std_ratio",
                        "CompareTrend/Module2/pred_corr_value",
                    ],
                ],
            },
            "Module34": {
                "hyperbolic_and_prototype": [
                    "Multiline",
                    [
                        "CompareTrend/Module34/z_radius",
                        "CompareTrend/Module34/z_tangent_norm",
                        "CompareTrend/Module34/lp_mac_radius",
                        "CompareTrend/Module34/lp_hbr_loss",
                        "CompareTrend/Module34/prototype_cos_mean",
                        "CompareTrend/Module34/prototype_cos_max",
                        "CompareTrend/Module34/prototype_same_class_cos_max",
                        "CompareTrend/Module34/prototype_radius_mean",
                        "CompareTrend/Module34/prototype_radius_min",
                        "CompareTrend/Module34/prototype_radius_max",
                    ],
                ],
                "branch_auc": [
                    "Multiline",
                    [
                        "CompareTrend/Branches/val_branch_module34_branch_auc",
                        "CompareTrend/Branches/val_branch_hpec_energy_auc",
                        "CompareTrend/Branches/val_branch_final_cached_auc",
                    ],
                ],
            },
            "Complementary": {
                "mask_and_invariance": [
                    "Multiline",
                    [
                        "CompareTrend/Complementary/mask_ratio",
                        "CompareTrend/Complementary/poincare_distance",
                        "CompareTrend/Complementary/view_loss",
                        "CompareTrend/Complementary/instance_infonce_loss",
                        "CompareTrend/Complementary/masked_ce_loss",
                    ],
                ],
                "prototype_updates": [
                    "Multiline",
                    [
                        "CompareTrend/PrototypeUpdate/reliable_tp_ratio",
                        "CompareTrend/PrototypeUpdate/view_consistency_mean",
                        "CompareTrend/PrototypeUpdate/assignment_entropy",
                        "CompareTrend/PrototypeUpdate/ema_displacement_mean",
                    ],
                ],
            },
            "CausalReachability": {
                "encoding": [
                    "Multiline",
                    [
                        "CompareTrend/CausalReachability/residual_norm",
                        "CompareTrend/CausalReachability/hop_1_gate",
                        "CompareTrend/CausalReachability/hop_2_gate",
                        "CompareTrend/CausalReachability/hop_3_gate",
                    ],
                ],
            },
            "Final": {
                "final_mean_metrics": [
                    "Multiline",
                    [
                        "CompareFinal/accuracy",
                        "CompareFinal/precision",
                        "CompareFinal/recall",
                        "CompareFinal/macro_f1",
                        "CompareFinal/auc",
                    ],
                ],
                "training_time": [
                    "Multiline",
                    [
                        "CompareFinal/train_seconds",
                    ],
                ],
            },
            "Timing": {
                "epoch_time": [
                    "Multiline",
                    [
                        "CompareTrend/Timing/epoch_seconds",
                    ],
                ],
            },
        }
        writer.add_custom_scalars(layout)
        writer.flush()

    def _create_tensorboard_writer(self, setting, run_part):
        if not self._tensorboard_enabled():
            return None
        if SummaryWriter is None:
            warnings.warn(
                "TensorBoard is enabled but tensorboard is not installed. "
                "Install it with `pip install tensorboard` or set use_tensorboard=0.",
                RuntimeWarning,
            )
            return None
        log_dir = os.path.join(
            self._tensorboard_root(),
            self._tensorboard_run_name(setting),
            self._safe_tensorboard_name(run_part),
        )
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        self._write_tensorboard_notes(writer, run_part)
        if str(run_part) == "summary":
            self._write_tensorboard_custom_scalars(writer)
        return writer

    def _tensorboard_scalar_items(self, epoch_record):
        mapping = {
            "train_loss": "Loss/train_total",
            "val_loss": "Loss/val_total",
            "cls_loss": "Loss/cls",
            "temporal_pred_loss": "Loss/module2_temporal_pred",
            "temporal_pred_base_loss": "Loss/module2_pred_base",
            "temporal_pred_delta_loss": "Loss/module2_pred_delta",
            "temporal_pred_lowfreq_loss": "Loss/module2_pred_lowfreq",
            "temporal_pred_corr_loss": "Loss/module2_pred_corr",
            "temporal_pred_std_ratio": "Diagnostics/module2_pred_std_ratio",
            "temporal_pred_corr_value": "Diagnostics/module2_pred_corr_value",
            "causal_dag_loss": "Loss/module2_dag",
            "temporal_sparse_loss": "Loss/module2_sparse",
            "temporal_parameter_sparse_loss": "Loss/module2_parameter_sparse",
            "temporal_smooth_loss": "Loss/module2_smooth",
            "temporal_group_sparse_loss": "Loss/module2_group_sparse",
            "temporal_lag_hierarchy_loss": "Loss/module2_lag_hierarchy",
            "hpec_final_ce_loss": "Loss/module4_hpec_final_ce",
            "hpec_energy_weighted_loss": "Loss/module4_hpec_energy_weighted",
            "complementary_view_loss": "Complementary/view_loss",
            "complementary_view_weighted_loss": "Complementary/view_loss_weighted",
            "complementary_view_distance": "Complementary/poincare_distance",
            "complementary_instance_infonce_loss": "Complementary/instance_infonce_loss",
            "complementary_instance_infonce_weighted_loss": "Complementary/instance_infonce_weighted",
            "complementary_masked_ce_loss": "Complementary/masked_ce_loss",
            "complementary_masked_ce_weighted_loss": "Complementary/masked_ce_weighted",
            "complementary_mask_ratio": "Complementary/mask_ratio",
            "complementary_topology_salience_entropy": "Complementary/topology_salience_entropy",
            "complementary_semantic_salience_entropy": "Complementary/semantic_salience_entropy",
            "hpec_reliable_tp_ratio": "PrototypeUpdate/reliable_tp_ratio",
            "hpec_reliable_confidence_mean": "PrototypeUpdate/reliable_confidence_mean",
            "hpec_reliable_view_consistency_mean": "PrototypeUpdate/view_consistency_mean",
            "hpec_reliable_assignment_entropy": "PrototypeUpdate/assignment_entropy",
            "hpec_reliable_updated_prototype_count": "PrototypeUpdate/updated_count",
            "hpec_reliable_unupdated_prototype_count": "PrototypeUpdate/unupdated_count",
            "hpec_reliable_ema_displacement_mean": "PrototypeUpdate/ema_displacement_mean",
            "causal_reachability_residual_norm": "CausalReachability/residual_norm",
            "hpec_teacher_distill_loss": "Loss/module4_teacher_distill",
            "hpec_teacher_distill_weighted_loss": "Loss/module4_teacher_distill_weighted",
            "hpec_teacher_distill_mode_centered_kl": "Diagnostics/hpec_teacher_distill_centered_kl",
            "hpec_teacher_distill_mode_margin_mse": "Diagnostics/hpec_teacher_distill_margin_mse",
            "hpec_prototype_ce_loss": "Loss/module4_prototype_ce",
            "hpec_prototype_ce_weighted_loss": "Loss/module4_prototype_ce_weighted",
            "hpec_z_radius_loss": "Loss/module4_z_radius",
            "hpec_z_radius_weighted_loss": "Loss/module4_z_radius_weighted",
            "hpec_prototype_radius_floor_loss": "Loss/module4_prototype_radius_floor",
            "hpec_prototype_radius_floor_weighted_loss": "Loss/module4_prototype_radius_floor_weighted",
            "hpec_prototype_separation_loss": "Loss/module4_prototype_separation",
            "hpec_prototype_separation_weighted_loss": "Loss/module4_prototype_separation_weighted",
            "class_prior_alignment_loss": "Loss/class_prior_alignment",
            "class_prior_alignment_weighted_loss": "Loss/class_prior_alignment_weighted",
            "module34_supcon_loss": "Loss/module34_supcon",
            "module34_supcon_weighted_loss": "Loss/module34_supcon_weighted",
            "module34_center_loss": "Loss/module34_center",
            "module34_center_weighted_loss": "Loss/module34_center_weighted",
            "module34_center_intra_loss": "Loss/module34_center_intra",
            "module34_center_inter_loss": "Loss/module34_center_inter",
            "module34_branch_ce_loss": "Loss/module34_branch_ce",
            "module34_branch_ce_weighted_loss": "Loss/module34_branch_ce_weighted",
            "lp_hbr_loss": "Loss/module34_lp_hbr",
            "lp_hbr_weighted_loss": "Loss/module34_lp_hbr_weighted",
            "epoch_seconds": "Timing/epoch_seconds",
            "train_accuracy": "Metrics/train_accuracy",
            "train_precision": "Metrics/train_precision",
            "train_recall": "Metrics/train_recall",
            "train_macro_f1": "Metrics/train_macro_f1",
            "train_roc_auc": "Metrics/train_auc",
            "val_accuracy": "Metrics/val_accuracy",
            "val_precision": "Metrics/val_precision",
            "val_recall": "Metrics/val_recall",
            "val_macro_f1": "Metrics/val_macro_f1",
            "val_roc_auc": "Metrics/val_auc",
            "train_best_threshold": "Threshold/train_best_threshold",
            "val_best_threshold": "Threshold/val_best_threshold",
            "causal_meta_shared_adjacency_mass_mean": "Diagnostics/graph_mass",
            "causal_meta_shared_adjacency_directionality_ratio": "Diagnostics/directionality",
            "causal_meta_alag_mean_adjacency_mass_mean": "Diagnostics/a_lag_mass",
            "causal_meta_a0_adjacency_mass_mean": "Diagnostics/a0_mass",
            "causal_meta_a0_to_alag_mass_ratio": "Diagnostics/a0_to_alag_mass_ratio",
            "classification_graph_mass": "Diagnostics/classification_graph_mass",
            "classification_graph_asymmetry_ratio": "Diagnostics/classification_graph_asymmetry",
            "classification_graph_causal_support_density": "Diagnostics/classification_graph_causal_support_density",
            "causal_module2_frozen": "Diagnostics/causal_module2_frozen",
            "z_radius_mean": "Diagnostics/z_radius",
            "z_tangent_norm_mean": "Diagnostics/z_tangent_norm",
            "module3_node_attention_entropy": "Diagnostics/module3_node_attention_entropy",
            "module3_node_attention_peak": "Diagnostics/module3_node_attention_peak",
            "module3_network_attention_entropy": "Diagnostics/module3_network_attention_entropy",
            "module3_network_attention_peak": "Diagnostics/module3_network_attention_peak",
            "hgcn_fc_anchor_gate_mean": "Diagnostics/hgcn_fc_anchor_gate",
            "hgcn_fc_anchor_update_norm_mean": "Diagnostics/hgcn_fc_anchor_update_norm",
            "module34_film_weight": "Diagnostics/module34_film_weight",
            "module34_film_scale_abs_mean": "Diagnostics/module34_film_scale_abs_mean",
            "module34_film_shift_norm_mean": "Diagnostics/module34_film_shift_norm_mean",
            "hyperbolic_residual_gate_mean": "Diagnostics/hyperbolic_residual_gate",
            "hyperbolic_residual_gate_min": "Diagnostics/hyperbolic_residual_gate_min",
            "hyperbolic_residual_gate_max": "Diagnostics/hyperbolic_residual_gate_max",
            "hyperbolic_residual_gate_mode_agreement": "Diagnostics/hyperbolic_residual_gate_mode_agreement",
            "hyperbolic_residual_gate_mode_consensus": "Diagnostics/hyperbolic_residual_gate_mode_consensus",
            "hyperbolic_residual_base_margin_mean": "Diagnostics/hyperbolic_residual_base_margin",
            "hyperbolic_residual_consensus_margin_mean": "Diagnostics/hyperbolic_residual_consensus_margin",
            "hyperbolic_residual_margin_mean": "Diagnostics/hyperbolic_residual_margin",
            "hyperbolic_residual_agreement": "Diagnostics/hyperbolic_residual_agreement",
            "hyperbolic_residual_norm_mean": "Diagnostics/hyperbolic_residual_norm",
            "hyperbolic_residual_margin_scale_mean": "Diagnostics/hyperbolic_residual_margin_scale",
            "hyperbolic_residual_margin_scale_max": "Diagnostics/hyperbolic_residual_margin_scale_max",
            "hyperbolic_residual_bias_mean": "Diagnostics/hyperbolic_residual_bias_mean",
            "hyperbolic_residual_bias_abs_mean": "Diagnostics/hyperbolic_residual_bias_abs_mean",
            "hyperbolic_residual_logit_blend": "Diagnostics/hyperbolic_residual_logit_blend",
            "hyperbolic_residual_binary_margin": "Diagnostics/hyperbolic_residual_binary_margin",
            "hyperbolic_dual_consensus": "Diagnostics/hyperbolic_dual_consensus",
            "hyperbolic_consensus_weight_mean": "Diagnostics/hyperbolic_consensus_weight_mean",
            "hyperbolic_consensus_weight_min": "Diagnostics/hyperbolic_consensus_weight_min",
            "hyperbolic_consensus_weight_max": "Diagnostics/hyperbolic_consensus_weight_max",
            "hyperbolic_update_logit_mean": "Diagnostics/hyperbolic_update_logit_mean",
            "hyperbolic_update_logit_abs_mean": "Diagnostics/hyperbolic_update_logit_abs_mean",
            "hpec_residual_calibration_batch_margin": "Diagnostics/hpec_residual_calibration_batch_margin",
            "hpec_residual_calibration_running_batch_margin": "Diagnostics/hpec_residual_calibration_running_batch_margin",
            "hpec_residual_calibration_hybrid_batch_running_margin": "Diagnostics/hpec_residual_calibration_hybrid_batch_running_margin",
            "hpec_residual_calibration_batch_weight": "Diagnostics/hpec_residual_calibration_batch_weight",
            "hpec_residual_calibrated_margin_abs_mean": "Diagnostics/hpec_residual_calibrated_margin_abs_mean",
            "hpec_residual_raw_margin_abs_mean": "Diagnostics/hpec_residual_raw_margin_abs_mean",
            "hpec_residual_raw_margin_std": "Diagnostics/hpec_residual_raw_margin_std",
            "hpec_residual_running_margin_mean": "Diagnostics/hpec_residual_running_margin_mean",
            "hpec_residual_running_margin_std": "Diagnostics/hpec_residual_running_margin_std",
            "hpec_residual_class_margin_gap": "Diagnostics/hpec_residual_class_margin_gap",
            "final_positive_prob_mean": "Diagnostics/final_positive_prob_mean",
            "hpec_input_radius_raw_mean": "Diagnostics/hpec_input_radius_raw",
            "hpec_input_radius_calibrated_mean": "Diagnostics/hpec_input_radius_calibrated",
            "hpec_input_tangent_noise_std": "Diagnostics/hpec_input_tangent_noise_std",
            "hpec_input_tangent_noise_norm": "Diagnostics/hpec_input_tangent_noise_norm",
            "final_logit_scale_mean": "Diagnostics/final_logit_scale",
            "final_logit_bias_mean": "Diagnostics/final_logit_bias",
            "hgcn_radial_calibration_enabled": "Diagnostics/hgcn_radial_calibration_enabled",
            "hgcn_radial_target_norm_mean": "Diagnostics/hgcn_radial_target_norm_mean",
            "lp_lorentz_constraint_error": "Diagnostics/lp_lorentz_constraint",
            "lp_in_aggregation_norm": "Diagnostics/lp_in_aggregation_norm",
            "lp_out_aggregation_norm": "Diagnostics/lp_out_aggregation_norm",
            "lp_alpha_out": "Diagnostics/lp_alpha_out",
            "lp_centroid_message_weight": "Diagnostics/lp_centroid_message_weight",
            "lp_in_attention_temperature": "Diagnostics/lp_in_attention_temperature",
            "lp_out_attention_temperature": "Diagnostics/lp_out_attention_temperature",
            "lp_mac_radius_mean": "Diagnostics/lp_mac_radius",
            "lp_mac_radius_max": "Diagnostics/lp_mac_radius_max",
            "lp_mac_low_clip_ratio": "Diagnostics/lp_mac_low_clip_ratio",
            "lp_mac_high_clip_ratio": "Diagnostics/lp_mac_high_clip_ratio",
            "lp_stats_update_gate_mean": "Diagnostics/lp_stats_update_gate",
            "prototype_cos_abs_mean": "Diagnostics/prototype_cos_mean",
            "prototype_cos_abs_max": "Diagnostics/prototype_cos_max",
            "prototype_same_class_cos_max": "Diagnostics/prototype_same_class_cos_max",
            "prototype_radius_mean": "Diagnostics/prototype_radius_mean",
            "prototype_radius_min": "Diagnostics/prototype_radius_min",
            "prototype_radius_max": "Diagnostics/prototype_radius_max",
            "prototype_tangent_norm_mean": "Diagnostics/prototype_tangent_norm_mean",
            "prototype_tangent_norm_min": "Diagnostics/prototype_tangent_norm_min",
            "prototype_tangent_norm_max": "Diagnostics/prototype_tangent_norm_max",
            "hpec_causal_role_energy_weight": "Diagnostics/hpec_causal_role_energy_weight",
            "hpec_causal_role_energy_std": "Diagnostics/hpec_causal_role_energy_std",
            "hpec_causal_role_gate_entropy": "Diagnostics/hpec_causal_role_gate_entropy",
            "hpec_causal_role_gate_peak": "Diagnostics/hpec_causal_role_gate_peak",
            "hpec_epoch_sample_count": "Diagnostics/hpec_epoch_sample_count",
            "hpec_epoch_reliability_mean": "Diagnostics/hpec_epoch_reliability_mean",
            "hpec_epoch_occupancy_min": "Diagnostics/hpec_epoch_occupancy_min",
            "hpec_epoch_occupancy_max": "Diagnostics/hpec_epoch_occupancy_max",
            "hpec_epoch_occupancy_entropy": "Diagnostics/hpec_epoch_occupancy_entropy",
            "hpec_epoch_updated_prototype_count": "Diagnostics/hpec_epoch_updated_prototype_count",
            "hpec_epoch_prototype_movement_mean": "Diagnostics/hpec_epoch_prototype_movement_mean",
            "hpec_epoch_prototype_movement_max": "Diagnostics/hpec_epoch_prototype_movement_max",
            "hpec_busemann_class_bias_abs_mean": "Diagnostics/hpec_busemann_class_bias_abs_mean",
            "hpec_busemann_class_bias_gap": "Diagnostics/hpec_busemann_class_bias_gap",
            "hpec_busemann_class_bias_weight": "Diagnostics/hpec_busemann_class_bias_weight",
            "hpec_residual_calibration_tanh_margin": "Diagnostics/hpec_residual_calibration_tanh_margin",
            "hpec_residual_calibration_train_class_margin": "Diagnostics/hpec_residual_calibration_train_class_margin",
            "hpec_prototype_logit_margin_mean": "Diagnostics/hpec_prototype_logit_margin_mean",
            "hpec_prototype_logit_signed_margin_mean": "Diagnostics/hpec_prototype_logit_signed_margin_mean",
            "hpec_prototype_logit_abs_mean": "Diagnostics/hpec_prototype_logit_abs_mean",
            "hpec_prototype_logit_mode_margin_preserving": "Diagnostics/hpec_prototype_logit_mode_margin_preserving",
            "hpec_prototype_logit_scale_value": "Diagnostics/hpec_prototype_logit_scale_value",
            "hpec_energy_prototype_residual": "Diagnostics/hpec_energy_prototype_residual",
            "hpec_prototype_residual_weight": "Diagnostics/hpec_prototype_residual_weight",
            "hpec_prototype_residual_abs_mean": "Diagnostics/hpec_prototype_residual_abs_mean",
            "hpec_energy_margin_mean": "Diagnostics/hpec_energy_margin_mean",
            "hpec_energy_margin_signed_mean": "Diagnostics/hpec_energy_margin_signed_mean",
            "hpec_energy_logit_abs_mean": "Diagnostics/hpec_energy_logit_abs_mean",
            "hpec_network_energy_mean": "Diagnostics/hpec_network_energy_mean",
            "hpec_network_energy_std": "Diagnostics/hpec_network_energy_std",
            "hpec_network_attention_max": "Diagnostics/hpec_network_attention_max",
            "hpec_network_selector_entropy": "Diagnostics/hpec_network_selector_entropy",
            "hpec_network_selector_peak": "Diagnostics/hpec_network_selector_peak",
            "hpec_network_energy_temperature": "Diagnostics/hpec_network_energy_temperature",
            "hpec_network_energy_normalized": "Diagnostics/hpec_network_energy_normalized",
            "hpec_network_selector_sharpness": "Diagnostics/hpec_network_selector_sharpness",
            "hpec_similarity_margin_mean": "Diagnostics/hpec_similarity_margin_mean",
        }
        zero_when_missing = {
            "cls_loss",
            "temporal_pred_loss",
            "temporal_pred_base_loss",
            "temporal_pred_delta_loss",
            "temporal_pred_lowfreq_loss",
            "temporal_pred_corr_loss",
            "temporal_pred_std_ratio",
            "temporal_pred_corr_value",
            "causal_dag_loss",
            "temporal_sparse_loss",
            "temporal_parameter_sparse_loss",
            "temporal_smooth_loss",
            "temporal_group_sparse_loss",
            "temporal_lag_hierarchy_loss",
            "hpec_final_ce_loss",
            "hpec_energy_weighted_loss",
            "hpec_teacher_distill_loss",
            "hpec_teacher_distill_weighted_loss",
            "hpec_prototype_ce_loss",
            "hpec_prototype_ce_weighted_loss",
            "hpec_z_radius_loss",
            "hpec_z_radius_weighted_loss",
            "hpec_prototype_radius_floor_loss",
            "hpec_prototype_radius_floor_weighted_loss",
            "hpec_prototype_separation_loss",
            "hpec_prototype_separation_weighted_loss",
            "module34_supcon_loss",
            "module34_supcon_weighted_loss",
            "module34_center_loss",
            "module34_center_weighted_loss",
            "module34_center_intra_loss",
            "module34_center_inter_loss",
            "module34_branch_ce_loss",
            "module34_branch_ce_weighted_loss",
            "module34_center_cos_mean",
            "lp_hbr_loss",
            "lp_hbr_weighted_loss",
            "epoch_seconds",
            "causal_meta_shared_adjacency_mass_mean",
            "causal_meta_shared_adjacency_directionality_ratio",
            "causal_meta_alag_mean_adjacency_mass_mean",
            "causal_meta_a0_adjacency_mass_mean",
            "causal_meta_a0_to_alag_mass_ratio",
            "classification_graph_mass",
            "classification_graph_asymmetry_ratio",
            "classification_graph_causal_support_density",
            "causal_module2_frozen",
            "z_radius_mean",
            "z_tangent_norm_mean",
            "hgcn_fc_anchor_gate_mean",
            "hgcn_fc_anchor_update_norm_mean",
            "module34_film_weight",
            "module34_film_scale_abs_mean",
            "module34_film_shift_norm_mean",
            "hyperbolic_residual_gate_mean",
            "hyperbolic_residual_gate_min",
            "hyperbolic_residual_gate_max",
            "hyperbolic_residual_gate_mode_agreement",
            "hyperbolic_residual_margin_mean",
            "hyperbolic_residual_agreement",
            "hyperbolic_residual_norm_mean",
            "hyperbolic_residual_margin_scale_mean",
            "hyperbolic_residual_margin_scale_max",
            "hyperbolic_residual_bias_mean",
            "hyperbolic_residual_bias_abs_mean",
            "hyperbolic_residual_logit_blend",
            "hyperbolic_residual_binary_margin",
            "hyperbolic_dual_consensus",
            "hyperbolic_consensus_weight_mean",
            "hyperbolic_consensus_weight_min",
            "hyperbolic_consensus_weight_max",
            "hyperbolic_update_logit_mean",
            "hyperbolic_update_logit_abs_mean",
            "hpec_residual_calibration_batch_margin",
            "hpec_residual_calibrated_margin_abs_mean",
            "hpec_residual_raw_margin_abs_mean",
            "hpec_residual_raw_margin_std",
            "final_positive_prob_mean",
            "hpec_input_radius_raw_mean",
            "hpec_input_radius_calibrated_mean",
            "final_logit_scale_mean",
            "final_logit_bias_mean",
            "hgcn_radial_calibration_enabled",
            "hgcn_radial_target_norm_mean",
            "lp_lorentz_constraint_error",
            "lp_in_aggregation_norm",
            "lp_out_aggregation_norm",
            "lp_alpha_out",
            "lp_centroid_message_weight",
            "lp_in_attention_temperature",
            "lp_out_attention_temperature",
            "lp_mac_radius_mean",
            "lp_mac_radius_max",
            "lp_mac_low_clip_ratio",
            "lp_mac_high_clip_ratio",
            "lp_stats_update_gate_mean",
            "prototype_cos_abs_mean",
            "prototype_cos_abs_max",
            "prototype_same_class_cos_max",
            "prototype_radius_mean",
            "prototype_radius_min",
            "prototype_radius_max",
            "prototype_tangent_norm_mean",
            "prototype_tangent_norm_min",
            "prototype_tangent_norm_max",
            "hpec_prototype_logit_margin_mean",
            "hpec_prototype_logit_abs_mean",
            "hpec_prototype_logit_mode_margin_preserving",
            "hpec_prototype_logit_scale_value",
            "hpec_energy_margin_mean",
            "hpec_network_energy_mean",
            "hpec_network_energy_std",
            "hpec_network_attention_max",
            "hpec_network_selector_entropy",
            "hpec_network_selector_peak",
            "hpec_network_energy_temperature",
            "hpec_similarity_margin_mean",
        }
        for key, tag in mapping.items():
            value = epoch_record.get(key)
            if value is None:
                if key not in zero_when_missing:
                    continue
                value = 0.0
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                yield tag, value
        for key, value in epoch_record.items():
            if key.startswith("causal_reachability_hop_"):
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(scalar):
                    suffix = key.removeprefix("causal_reachability_")
                    yield f"CausalReachability/{suffix}", scalar
                continue
            if not (
                key.startswith("train_branch_")
                or key.startswith("val_branch_")
                or key.startswith("raw_branch_")
            ):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                yield f"Branches/{key}", value

    def _write_tensorboard_epoch(self, writer, epoch_record):
        if writer is None:
            return
        step = int(epoch_record.get("epoch", 1))
        config_label = self._tensorboard_config_label()
        for tag, value in self._tensorboard_scalar_items(epoch_record):
            writer.add_scalar(f"FoldTrend/{config_label}/{tag}", value, step)
        writer.flush()

    def _write_tensorboard_final_metrics(self, writer, metrics, step, prefix="Final/raw"):
        if writer is None or metrics is None:
            return
        config_label = self._tensorboard_config_label()
        names = ("accuracy", "precision", "recall", "macro_f1", "auc", "train_seconds")
        for name, value in zip(names, metrics):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                writer.add_scalar(f"{prefix}/{config_label}/{name}", value, int(step))
        writer.flush()

    def _tensorboard_compare_scalar_items(self, epoch_record):
        mapping = {
            "train_loss": "CompareTrend/Loss/train_total",
            "val_loss": "CompareTrend/Loss/val_total",
            "cls_loss": "CompareTrend/Loss/cls",
            "temporal_pred_loss": "CompareTrend/Loss/module2_temporal_pred",
            "temporal_pred_base_loss": "CompareTrend/Module2/pred_base_loss",
            "temporal_pred_delta_loss": "CompareTrend/Module2/pred_delta_loss",
            "temporal_pred_lowfreq_loss": "CompareTrend/Module2/pred_lowfreq_loss",
            "temporal_pred_corr_loss": "CompareTrend/Module2/pred_corr_loss",
            "temporal_pred_std_ratio": "CompareTrend/Module2/pred_std_ratio",
            "temporal_pred_corr_value": "CompareTrend/Module2/pred_corr_value",
            "causal_dag_loss": "CompareTrend/Module2/dag_loss",
            "temporal_sparse_loss": "CompareTrend/Module2/sparse_loss",
            "temporal_parameter_sparse_loss": "CompareTrend/Module2/parameter_sparse_loss",
            "temporal_smooth_loss": "CompareTrend/Module2/smooth_loss",
            "temporal_group_sparse_loss": "CompareTrend/Module2/group_sparse_loss",
            "temporal_lag_hierarchy_loss": "CompareTrend/Module2/lag_hierarchy_loss",
            "complementary_view_loss": "CompareTrend/Complementary/view_loss",
            "complementary_view_weighted_loss": "CompareTrend/Complementary/view_loss_weighted",
            "complementary_view_distance": "CompareTrend/Complementary/poincare_distance",
            "complementary_instance_infonce_loss": "CompareTrend/Complementary/instance_infonce_loss",
            "complementary_instance_infonce_weighted_loss": "CompareTrend/Complementary/instance_infonce_weighted",
            "complementary_masked_ce_loss": "CompareTrend/Complementary/masked_ce_loss",
            "complementary_masked_ce_weighted_loss": "CompareTrend/Complementary/masked_ce_weighted",
            "complementary_mask_ratio": "CompareTrend/Complementary/mask_ratio",
            "complementary_topology_salience_entropy": "CompareTrend/Complementary/topology_salience_entropy",
            "complementary_semantic_salience_entropy": "CompareTrend/Complementary/semantic_salience_entropy",
            "hpec_reliable_tp_ratio": "CompareTrend/PrototypeUpdate/reliable_tp_ratio",
            "hpec_reliable_confidence_mean": "CompareTrend/PrototypeUpdate/reliable_confidence_mean",
            "hpec_reliable_view_consistency_mean": "CompareTrend/PrototypeUpdate/view_consistency_mean",
            "hpec_reliable_assignment_entropy": "CompareTrend/PrototypeUpdate/assignment_entropy",
            "hpec_reliable_updated_prototype_count": "CompareTrend/PrototypeUpdate/updated_count",
            "hpec_reliable_unupdated_prototype_count": "CompareTrend/PrototypeUpdate/unupdated_count",
            "hpec_reliable_ema_displacement_mean": "CompareTrend/PrototypeUpdate/ema_displacement_mean",
            "causal_reachability_residual_norm": "CompareTrend/CausalReachability/residual_norm",
            "hpec_teacher_distill_loss": "CompareTrend/Module34/teacher_distill",
            "hpec_teacher_distill_weighted_loss": "CompareTrend/Module34/teacher_distill_weighted",
            "hpec_teacher_distill_mode_centered_kl": "CompareTrend/Module34/teacher_distill_centered_kl",
            "hpec_teacher_distill_mode_margin_mse": "CompareTrend/Module34/teacher_distill_margin_mse",
            "hpec_prototype_ce_loss": "CompareTrend/Module34/prototype_ce",
            "hpec_prototype_ce_weighted_loss": "CompareTrend/Module34/prototype_ce_weighted",
            "hpec_z_radius_loss": "CompareTrend/Module34/z_radius_loss",
            "hpec_z_radius_weighted_loss": "CompareTrend/Module34/z_radius_weighted_loss",
            "hpec_prototype_radius_floor_loss": "CompareTrend/Module34/prototype_radius_floor_loss",
            "hpec_prototype_radius_floor_weighted_loss": "CompareTrend/Module34/prototype_radius_floor_weighted_loss",
            "hpec_prototype_separation_loss": "CompareTrend/Module34/prototype_separation_loss",
            "hpec_prototype_separation_weighted_loss": "CompareTrend/Module34/prototype_separation_weighted_loss",
            "module34_supcon_loss": "CompareTrend/Module34/supcon_loss",
            "module34_supcon_weighted_loss": "CompareTrend/Module34/supcon_weighted_loss",
            "module34_center_loss": "CompareTrend/Module34/center_loss",
            "module34_center_weighted_loss": "CompareTrend/Module34/center_weighted_loss",
            "module34_center_intra_loss": "CompareTrend/Module34/center_intra_loss",
            "module34_center_inter_loss": "CompareTrend/Module34/center_inter_loss",
            "module34_branch_ce_loss": "CompareTrend/Module34/branch_ce",
            "module34_branch_ce_weighted_loss": "CompareTrend/Module34/branch_ce_weighted",
            "module34_center_cos_mean": "CompareTrend/Module34/center_cos_mean",
            "lp_hbr_loss": "CompareTrend/Module34/lp_hbr_loss",
            "lp_hbr_weighted_loss": "CompareTrend/Module34/lp_hbr_weighted_loss",
            "epoch_seconds": "CompareTrend/Timing/epoch_seconds",
            "train_accuracy": "CompareTrend/Metrics/train_accuracy",
            "train_precision": "CompareTrend/Metrics/train_precision",
            "train_recall": "CompareTrend/Metrics/train_recall",
            "train_macro_f1": "CompareTrend/Metrics/train_macro_f1",
            "train_roc_auc": "CompareTrend/Metrics/train_auc",
            "val_accuracy": "CompareTrend/Metrics/val_accuracy",
            "val_precision": "CompareTrend/Metrics/val_precision",
            "val_recall": "CompareTrend/Metrics/val_recall",
            "val_macro_f1": "CompareTrend/Metrics/val_macro_f1",
            "val_roc_auc": "CompareTrend/Metrics/val_auc",
            "causal_meta_shared_adjacency_mass_mean": "CompareTrend/Module2/graph_mass",
            "causal_meta_shared_adjacency_directionality_ratio": "CompareTrend/Module2/directionality",
            "causal_meta_alag_mean_adjacency_mass_mean": "CompareTrend/Module2/a_lag_mass",
            "causal_meta_a0_adjacency_mass_mean": "CompareTrend/Module2/a0_mass",
            "causal_meta_a0_to_alag_mass_ratio": "CompareTrend/Module2/a0_to_alag_mass_ratio",
            "classification_graph_mass": "CompareTrend/Module2/classification_graph_mass",
            "classification_graph_asymmetry_ratio": "CompareTrend/Module2/classification_graph_asymmetry",
            "classification_graph_causal_support_density": "CompareTrend/Module2/causal_support_density",
            "causal_module2_frozen": "CompareTrend/Module2/frozen",
            "z_radius_mean": "CompareTrend/Module34/z_radius",
            "z_tangent_norm_mean": "CompareTrend/Module34/z_tangent_norm",
            "module3_node_attention_entropy": "CompareTrend/Module34/node_attention_entropy",
            "module3_node_attention_peak": "CompareTrend/Module34/node_attention_peak",
            "module3_network_attention_entropy": "CompareTrend/Module34/network_attention_entropy",
            "module3_network_attention_peak": "CompareTrend/Module34/network_attention_peak",
            "hgcn_fc_anchor_gate_mean": "CompareTrend/Module34/fc_anchor_gate",
            "hgcn_fc_anchor_update_norm_mean": "CompareTrend/Module34/fc_anchor_update_norm",
            "hpec_residual_calibration_batch_margin": "CompareTrend/Module34/residual_calibration_batch_margin",
            "hpec_residual_calibration_hybrid_batch_running_margin": "CompareTrend/Module34/residual_calibration_hybrid_batch_running_margin",
            "hpec_residual_calibration_batch_weight": "CompareTrend/Module34/residual_calibration_batch_weight",
            "hpec_residual_calibrated_margin_abs_mean": "CompareTrend/Module34/residual_calibrated_margin_abs",
            "hpec_residual_raw_margin_abs_mean": "CompareTrend/Module34/residual_raw_margin_abs",
            "hpec_residual_raw_margin_std": "CompareTrend/Module34/residual_raw_margin_std",
            "hyperbolic_dual_consensus": "CompareTrend/Module34/dual_consensus",
            "hyperbolic_consensus_weight_mean": "CompareTrend/Module34/consensus_weight_mean",
            "hyperbolic_consensus_weight_min": "CompareTrend/Module34/consensus_weight_min",
            "hyperbolic_consensus_weight_max": "CompareTrend/Module34/consensus_weight_max",
            "hyperbolic_update_logit_abs_mean": "CompareTrend/Module34/hyperbolic_update_logit_abs",
            "hgcn_radial_calibration_enabled": "CompareTrend/Module34/hgcn_radial_calibration_enabled",
            "hgcn_radial_target_norm_mean": "CompareTrend/Module34/hgcn_radial_target_norm_mean",
            "lp_lorentz_constraint_error": "CompareTrend/Module34/lp_lorentz_constraint",
            "lp_in_aggregation_norm": "CompareTrend/Module34/lp_in_aggregation_norm",
            "lp_out_aggregation_norm": "CompareTrend/Module34/lp_out_aggregation_norm",
            "lp_alpha_out": "CompareTrend/Module34/lp_alpha_out",
            "lp_centroid_message_weight": "CompareTrend/Module34/lp_centroid_message_weight",
            "lp_in_attention_temperature": "CompareTrend/Module34/lp_in_attention_temperature",
            "lp_out_attention_temperature": "CompareTrend/Module34/lp_out_attention_temperature",
            "lp_mac_radius_mean": "CompareTrend/Module34/lp_mac_radius",
            "lp_mac_radius_max": "CompareTrend/Module34/lp_mac_radius_max",
            "lp_mac_low_clip_ratio": "CompareTrend/Module34/lp_mac_low_clip_ratio",
            "lp_mac_high_clip_ratio": "CompareTrend/Module34/lp_mac_high_clip_ratio",
            "lp_stats_update_gate_mean": "CompareTrend/Module34/lp_stats_update_gate",
            "prototype_cos_abs_mean": "CompareTrend/Module34/prototype_cos_mean",
            "prototype_cos_abs_max": "CompareTrend/Module34/prototype_cos_max",
            "prototype_same_class_cos_max": "CompareTrend/Module34/prototype_same_class_cos_max",
            "prototype_radius_mean": "CompareTrend/Module34/prototype_radius_mean",
            "prototype_radius_min": "CompareTrend/Module34/prototype_radius_min",
            "prototype_radius_max": "CompareTrend/Module34/prototype_radius_max",
            "prototype_tangent_norm_mean": "CompareTrend/Module34/prototype_tangent_norm_mean",
            "prototype_tangent_norm_min": "CompareTrend/Module34/prototype_tangent_norm_min",
            "prototype_tangent_norm_max": "CompareTrend/Module34/prototype_tangent_norm_max",
            "hpec_prototype_logit_margin_mean": "CompareTrend/Module34/prototype_logit_margin",
            "hpec_prototype_logit_signed_margin_mean": "CompareTrend/Module34/prototype_logit_signed_margin",
            "hpec_prototype_logit_abs_mean": "CompareTrend/Module34/prototype_logit_abs",
            "hpec_prototype_logit_mode_margin_preserving": "CompareTrend/Module34/prototype_logit_margin_preserving",
            "hpec_prototype_logit_scale_value": "CompareTrend/Module34/prototype_logit_scale",
            "hpec_energy_prototype_residual": "CompareTrend/Module34/energy_prototype_residual",
            "hpec_prototype_residual_weight": "CompareTrend/Module34/prototype_residual_weight",
            "hpec_prototype_residual_abs_mean": "CompareTrend/Module34/prototype_residual_abs",
            "hpec_energy_margin_mean": "CompareTrend/Module34/energy_margin",
            "hpec_energy_margin_signed_mean": "CompareTrend/Module34/energy_signed_margin",
            "hpec_energy_logit_abs_mean": "CompareTrend/Module34/energy_logit_abs",
            "hpec_residual_calibration_tanh_margin": "CompareTrend/Module34/residual_calibration_tanh_margin",
            "hpec_similarity_margin_mean": "CompareTrend/Module34/similarity_margin",
        }
        for key, tag in mapping.items():
            value = epoch_record.get(key)
            if value is None:
                value = 0.0
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                yield tag, value
        for key, value in epoch_record.items():
            if key.startswith("causal_reachability_hop_"):
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(scalar):
                    suffix = key.removeprefix("causal_reachability_")
                    yield f"CompareTrend/CausalReachability/{suffix}", scalar
                continue
            if not (
                key.startswith("train_branch_")
                or key.startswith("val_branch_")
                or key.startswith("raw_branch_")
            ):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                yield f"CompareTrend/Branches/{key}", value

    def _mean_epoch_records(self, fold_epoch_records):
        epoch_buckets = {}
        for records in fold_epoch_records:
            for record in records:
                epoch = int(record.get("epoch", 0) or 0)
                if epoch <= 0:
                    continue
                epoch_buckets.setdefault(epoch, []).append(record)
        mean_records = []
        for epoch in sorted(epoch_buckets):
            records = epoch_buckets[epoch]
            keys = set().union(*(record.keys() for record in records))
            mean_record = {"epoch": epoch}
            for key in keys:
                values = []
                for record in records:
                    if key not in record:
                        continue
                    try:
                        value = float(record[key])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        values.append(value)
                if values:
                    mean_record[key] = float(np.mean(values))
            mean_records.append(mean_record)
        return mean_records

    def _write_tensorboard_compare_trends(self, writer, fold_epoch_records):
        if writer is None:
            return
        mean_records = self._mean_epoch_records(fold_epoch_records)
        for record in mean_records:
            step = int(record.get("epoch", 0) or 0)
            if step <= 0:
                continue
            for tag, value in self._tensorboard_compare_scalar_items(record):
                writer.add_scalar(tag, value, step)
        writer.flush()

    def _write_tensorboard_compare_final(self, writer, metrics):
        if writer is None or metrics is None:
            return
        names = ("accuracy", "precision", "recall", "macro_f1", "auc", "train_seconds")
        for name, value in zip(names, metrics):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                writer.add_scalar(f"CompareFinal/{name}", value, 1)
        writer.flush()

    def _write_tensorboard_hparam_effects(self, writer, metrics):
        if writer is None or metrics is None:
            return
        names = ("accuracy", "precision", "recall", "macro_f1", "auc", "train_seconds")
        metric_values = {}
        for name, value in zip(names, metrics):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                metric_values[name] = value
        hparams = self._tensorboard_hparams()
        for hparam_name, hparam_value in hparams.items():
            if not isinstance(hparam_value, (int, float)):
                continue
            step = self._tensorboard_hparam_step(hparam_value)
            for metric_name, metric_value in metric_values.items():
                writer.add_scalar(f"HparamEffect/{hparam_name}/{metric_name}", metric_value, step)
        if hparams and metric_values:
            try:
                writer.add_hparams(
                    hparams,
                    {f"hparam/{name}": value for name, value in metric_values.items()},
                    run_name="hparams",
                )
            except (TypeError, ValueError):
                pass
        writer.flush()

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
            ("aux_loss", loss_parts.get("causal_aux_loss", 0.0)),
            ("hpec_final_ce", loss_parts.get("hpec_final_ce_loss", 0.0)),
            ("hpec_energy_w", loss_parts.get("hpec_energy_weighted_loss", 0.0)),
            ("temporal_pred", loss_parts.get("temporal_pred_loss", 0.0)),
            ("dag_loss", loss_parts.get("causal_dag_loss", 0.0)),
            ("temporal_sparse", loss_parts.get("temporal_sparse_loss", 0.0)),
            ("temporal_smooth", loss_parts.get("temporal_smooth_loss", 0.0)),
            ("val_loss", val_loss),
        ]
        optional_loss_items = [
            ("module1_denoise", loss_parts.get("module1_denoise_loss", 0.0)),
            ("teacher_distill", loss_parts.get("hpec_teacher_distill_loss", 0.0)),
            ("teacher_distill_w", loss_parts.get("hpec_teacher_distill_weighted_loss", 0.0)),
            ("proto_ce", loss_parts.get("hpec_prototype_ce_loss", 0.0)),
            ("proto_ce_w", loss_parts.get("hpec_prototype_ce_weighted_loss", 0.0)),
            ("z_radius_w", loss_parts.get("hpec_z_radius_weighted_loss", 0.0)),
            ("proto_sep_w", loss_parts.get("hpec_prototype_separation_weighted_loss", 0.0)),
            ("prior_align_w", loss_parts.get("class_prior_alignment_weighted_loss", 0.0)),
            ("m34_supcon_w", loss_parts.get("module34_supcon_weighted_loss", 0.0)),
            ("m34_center_w", loss_parts.get("module34_center_weighted_loss", 0.0)),
            ("m34_branch_ce_w", loss_parts.get("module34_branch_ce_weighted_loss", 0.0)),
            ("lp_hbr_w", loss_parts.get("lp_hbr_weighted_loss", 0.0)),
            ("site_adv", loss_parts.get("site_adversarial_loss", 0.0)),
            ("pred_base", loss_parts.get("temporal_pred_base_loss", 0.0)),
            ("pred_delta", loss_parts.get("temporal_pred_delta_loss", 0.0)),
            ("pred_lowfreq", loss_parts.get("temporal_pred_lowfreq_loss", 0.0)),
            ("pred_corr", loss_parts.get("temporal_pred_corr_loss", 0.0)),
            ("param_sparse", loss_parts.get("temporal_parameter_sparse_loss", 0.0)),
            ("sample_l1", loss_parts.get("sample_graph_l1_loss", 0.0)),
            ("sample_dev", loss_parts.get("sample_graph_deviation_loss", 0.0)),
        ]
        loss_items.extend(
            (name, value)
            for name, value in optional_loss_items
            if value is not None and abs(float(value)) > 1e-12
        )
        diagnostic_items = [
            ("dag_raw", loss_parts.get("causal_meta_dagma_spectral_radius", loss_parts.get("causal_meta_analytic_spectral_radius", 0.0))),
            ("graph_mass", loss_parts.get("causal_meta_shared_adjacency_mass_mean", 0.0)),
            ("direction", loss_parts.get("causal_meta_shared_adjacency_directionality_ratio", 0.0)),
            ("alag_mass", loss_parts.get("causal_meta_alag_mean_adjacency_mass_mean", 0.0)),
            ("alag_direction", loss_parts.get("causal_meta_alag_mean_adjacency_directionality_ratio", 0.0)),
            ("a0_mass", loss_parts.get("causal_meta_a0_adjacency_mass_mean", 0.0)),
            ("a0_direction", loss_parts.get("causal_meta_a0_adjacency_directionality_ratio", 0.0)),
            ("a0/alag", loss_parts.get("causal_meta_a0_to_alag_mass_ratio", 0.0)),
            ("pred_std_ratio", loss_parts.get("temporal_pred_std_ratio", 0.0)),
            ("pred_corr_value", loss_parts.get("temporal_pred_corr_value", 0.0)),
            ("A_cls_asym", loss_parts.get("classification_graph_asymmetry_ratio", 0.0)),
            ("causal_support", loss_parts.get("classification_graph_causal_support_density", 0.0)),
            ("attn_entropy", loss_parts.get("causal_meta_attention_entropy", 0.0)),
            ("gate_mass", loss_parts.get("causal_meta_gate_mass", 0.0)),
            ("graph_delta", loss_parts.get("causal_meta_sample_delta_abs_mean", 0.0)),
            ("z_radius", loss_parts.get("z_radius_mean", 0.0)),
            ("z_tangent", loss_parts.get("z_tangent_norm_mean", 0.0)),
            ("fc_residual", loss_parts.get("fc_residual_norm_mean", 0.0)),
            ("fc_anchor_gate", loss_parts.get("hgcn_fc_anchor_gate_mean", 0.0)),
            ("fc_anchor_update", loss_parts.get("hgcn_fc_anchor_update_norm_mean", 0.0)),
            ("film_scale", loss_parts.get("module34_film_scale_abs_mean", 0.0)),
            ("film_shift", loss_parts.get("module34_film_shift_norm_mean", 0.0)),
            ("proto_cos_mean", loss_parts.get("prototype_cos_abs_mean", 0.0)),
            ("proto_cos_max", loss_parts.get("prototype_cos_abs_max", 0.0)),
            ("proto_same_max", loss_parts.get("prototype_same_class_cos_max", 0.0)),
            ("proto_radius", loss_parts.get("prototype_radius_mean", 0.0)),
            ("proto_logit_margin", loss_parts.get("hpec_prototype_logit_margin_mean", 0.0)),
            ("proto_logit_abs", loss_parts.get("hpec_prototype_logit_abs_mean", 0.0)),
            ("energy_margin", loss_parts.get("hpec_energy_margin_mean", 0.0)),
            ("sim_margin", loss_parts.get("hpec_similarity_margin_mean", 0.0)),
            ("hpec_cal_margin", loss_parts.get("hpec_residual_calibrated_margin_abs_mean", 0.0)),
            ("pos_prob", loss_parts.get("final_positive_prob_mean", 0.0)),
        ]
        optional_diagnostic_items = [
            ("lp_constraint", loss_parts.get("lp_lorentz_constraint_error", 0.0)),
            ("lp_in_norm", loss_parts.get("lp_in_aggregation_norm", 0.0)),
            ("lp_out_norm", loss_parts.get("lp_out_aggregation_norm", 0.0)),
            ("lp_alpha_out", loss_parts.get("lp_alpha_out", 0.0)),
            ("lp_centroid", loss_parts.get("lp_centroid_message_weight", 0.0)),
            ("lp_mac_radius", loss_parts.get("lp_mac_radius_mean", 0.0)),
            ("lp_mac_high", loss_parts.get("lp_mac_high_clip_ratio", 0.0)),
            ("lp_mac_low", loss_parts.get("lp_mac_low_clip_ratio", 0.0)),
            ("lp_stats_gate", loss_parts.get("lp_stats_update_gate_mean", 0.0)),
            ("lp_hbr", loss_parts.get("lp_hbr_loss", 0.0)),
            ("hpec_net_std", loss_parts.get("hpec_network_energy_std", 0.0)),
            ("hpec_net_sel_peak", loss_parts.get("hpec_network_selector_peak", 0.0)),
            ("hpec_net_sel_ent", loss_parts.get("hpec_network_selector_entropy", 0.0)),
        ]
        diagnostic_items.extend(
            (name, value)
            for name, value in optional_diagnostic_items
            if value is not None and abs(float(value)) > 1e-12
        )
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
        keep_fallback = int(getattr(self.args, "keep_gcn_fallback_with_hyperbolic", 0))
        residual_weight = float(getattr(self.args, "hyperbolic_logit_residual_weight", 0.0) or 0.0)
        teacher_weight = float(getattr(self.args, "hpec_teacher_distill_weight", 0.0) or 0.0)
        arch_label = "hgcn_hpec"
        if use_hpec and keep_fallback and teacher_weight > 0 and residual_weight <= 0:
            path = f"{arch_label}_with_gcn_teacher"
        elif keep_fallback and use_hgcn and residual_weight > 0:
            fusion_mode = str(getattr(self.args, "hyperbolic_residual_fusion_mode", "residual"))
            path = f"gcn_fallback_plus_{arch_label}_{fusion_mode}"
        elif keep_fallback and use_hgcn:
            path = f"gcn_fallback_with_{arch_label}_available"
        elif use_hpec:
            path = arch_label
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
        fusion_mode = str(getattr(self.args, "hyperbolic_residual_fusion_mode", "residual"))
        fallback_hyper_path = f"gcn_fallback_plus_hgcn_hpec_{fusion_mode}"
        return getattr(
            model,
            "latest_graph_path",
            (
                fallback_hyper_path
                if int(getattr(self.args, "keep_gcn_fallback_with_hyperbolic", 0))
                and int(getattr(self.args, "use_hgcn_module3", 0))
                and float(getattr(self.args, "hyperbolic_logit_residual_weight", 0.0) or 0.0) > 0
                else (
                    "hgcn_hpec"
                    if int(getattr(self.args, "use_hpec_module4", 0))
                    else ("hgcn_only" if int(getattr(self.args, "use_hgcn_module3", 0)) else "gcn_fallback")
                )
            ),
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

        base_name = self._metrics_base_name(check_path)
        save_path = os.path.join(output_dir, f"{base_name}_{split_name}_causal_intermediates.png")
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

    def _collect_labelwise_adjacency(self, loader):
        model = self._model_for_aux_loss()
        sums = {}
        counts = {}
        with torch.no_grad():
            for batch in loader:
                x_enc, label, correlation_matrix, site_label = self._unpack_batch(batch)
                x_enc = x_enc.to(self.device)
                if correlation_matrix is not None:
                    correlation_matrix = correlation_matrix.to(self.device)
                if site_label is not None:
                    site_label = site_label.to(self.device)
                self._model_forward(
                    x_enc,
                    correlation_matrix=correlation_matrix,
                    site_label=site_label,
                )
                adjacency = getattr(model, "latest_classification_adjacency", None)
                if adjacency is None:
                    causal_output = getattr(model, "latest_causal_output", None)
                    adjacency = getattr(causal_output, "a_effective", None)
                if adjacency is None:
                    continue
                adjacency = adjacency.detach().cpu()
                if adjacency.ndim == 2:
                    adjacency = adjacency.unsqueeze(0).expand(label.shape[0], -1, -1)
                labels = label.detach().cpu().view(-1).long()
                for class_id in labels.unique(sorted=True).tolist():
                    mask = labels == int(class_id)
                    if not torch.any(mask):
                        continue
                    class_sum = adjacency[mask].sum(dim=0)
                    sums[class_id] = sums.get(class_id, torch.zeros_like(class_sum)) + class_sum
                    counts[class_id] = counts.get(class_id, 0) + int(mask.sum().item())
        means = {
            class_id: (sums[class_id] / max(counts.get(class_id, 0), 1)).numpy()
            for class_id in sums
        }
        return means, counts

    def _save_labelwise_graph_diagnostics_after_training(self, train_loader, val_loader, check_path):
        if not bool(getattr(self.args, "visualize_causal", 0)):
            return
        output_dir = getattr(self.args, "causal_vis_dir", None) or check_path
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(output_dir, exist_ok=True)

        was_training = self.model.training
        self.model.eval()
        try:
            split_data = {
                "train": self._collect_labelwise_adjacency(train_loader),
                "test": self._collect_labelwise_adjacency(val_loader),
            }
        finally:
            if was_training:
                self.model.train()

        fold_name = self._metrics_base_name(check_path)
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        for split_name, (means, counts) in split_data.items():
            if len(means) < 2:
                continue
            class_ids = sorted(means.keys())[:2]
            graph0 = means[class_ids[0]]
            graph1 = means[class_ids[1]]
            diff = graph1 - graph0
            npz_path = os.path.join(output_dir, f"{fold_name}_{split_name}_labelwise_adjacency.npz")
            np.savez(
                npz_path,
                label0=graph0,
                label1=graph1,
                diff_label1_minus_label0=diff,
                label_ids=np.array(class_ids, dtype=int),
                counts=np.array([counts.get(cid, 0) for cid in class_ids], dtype=int),
            )

            k = min(30, diff.size)
            flat = np.abs(diff).reshape(-1)
            top_indices = np.argsort(flat)[-k:][::-1]
            rows, cols = np.unravel_index(top_indices, diff.shape)
            csv_path = os.path.join(output_dir, f"{fold_name}_{split_name}_labelwise_diff_top_edges.csv")
            pd.DataFrame(
                {
                    "parent": rows,
                    "child": cols,
                    "label1_minus_label0": diff[rows, cols],
                    "abs_diff": np.abs(diff[rows, cols]),
                    "label0_mean": graph0[rows, cols],
                    "label1_mean": graph1[rows, cols],
                }
            ).to_csv(csv_path, index=False, encoding="utf-8-sig")

            if plt is not None:
                vmax = max(
                    float(np.max(np.abs(graph0))) if graph0.size else 0.0,
                    float(np.max(np.abs(graph1))) if graph1.size else 0.0,
                    1e-8,
                )
                diff_vmax = max(float(np.max(np.abs(diff))) if diff.size else 0.0, 1e-8)
                fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
                im0 = axes[0].imshow(graph0, cmap="viridis", vmin=-vmax, vmax=vmax)
                axes[0].set_title(f"label {class_ids[0]} mean (n={counts.get(class_ids[0], 0)})")
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                im1 = axes[1].imshow(graph1, cmap="viridis", vmin=-vmax, vmax=vmax)
                axes[1].set_title(f"label {class_ids[1]} mean (n={counts.get(class_ids[1], 0)})")
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                im2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax)
                axes[2].set_title(f"diff: label {class_ids[1]} - label {class_ids[0]}")
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                for ax in axes:
                    ax.set_xlabel("child ROI")
                    ax.set_ylabel("parent ROI")
                png_path = os.path.join(output_dir, f"{fold_name}_{split_name}_labelwise_adjacency_diff.png")
                fig.savefig(png_path, dpi=160)
                plt.close(fig)

            if self.args.print_process:
                print(f"{split_name} labelwise adjacency diagnostics saved to: {npz_path}")
                print(f"{split_name} labelwise top diff edges saved to: {csv_path}")

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
            graph_pack["A_lag_mean"] = causal_output.a_lag.mean(dim=0).detach().cpu().numpy()
            if getattr(causal_output, "a_lag_raw", None) is not None:
                graph_pack["A_lag_raw"] = causal_output.a_lag_raw.detach().cpu().numpy()
                graph_pack["A_lag_raw_mean"] = causal_output.a_lag_raw.mean(dim=0).detach().cpu().numpy()
            if getattr(causal_output, "candidate_lag_mask", None) is not None:
                mask_mean = causal_output.candidate_lag_mask.mean(dim=0)
                graph_pack["candidate_mask_mean"] = mask_mean.detach().cpu().numpy()
                graph_pack["A_lag_candidate_masked_mean"] = (
                    causal_output.a_lag.mean(dim=0) * mask_mean
                ).detach().cpu().numpy()
        graph_pack["A_shared"] = causal_output.a_shared.detach().cpu().numpy()
        graph_pack["A_effective"] = causal_output.a_effective.detach().cpu().numpy()
        classification_adjacency = getattr(model, "latest_classification_adjacency", None)
        if classification_adjacency is not None:
            graph_pack["A_cls"] = classification_adjacency.detach().cpu().numpy()
        if causal_output.a_delta is not None:
            graph_pack["A_delta"] = causal_output.a_delta.detach().cpu().numpy()

        graph_path = os.path.join(output_dir, f"{fold_name}_graph_diagnostics.npz")
        np.savez(graph_path, **graph_pack)

        # Top edges 使用下游主图而不是 A0；A0 只是同时间片残余依赖和 DAG 约束载体。
        base_graph = graph_pack.get("A_lag_mean", graph_pack["A_shared"])
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
        complementary_features = []
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
                model = self._model_for_aux_loss()
                complementary = getattr(model, "latest_complementary_module3_output", None)
                if complementary is not None:
                    complementary_features.append(complementary.z_tangent.detach().cpu().numpy())
        if not features:
            return None, None, None
        complementary = (
            np.concatenate(complementary_features, axis=0)
            if complementary_features
            else None
        )
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), complementary

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
        model = self._model_for_aux_loss()
        self.model.eval()
        train_features, train_labels, train_complementary = self._collect_embeddings_for_tsne(train_loader)
        test_features, test_labels, test_complementary = self._collect_embeddings_for_tsne(val_loader)
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

        fold_name = self._metrics_base_name(check_path)
        save_path = os.path.join(output_dir, f"{fold_name}_train_test_tsne.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        if self.args.print_process:
            print(f"train/test t-SNE visualization saved to: {save_path}")

        if train_complementary is None or test_complementary is None:
            return
        paired_features = np.concatenate(
            [train_features, test_features, train_complementary, test_complementary], axis=0
        )
        paired_labels = np.concatenate(
            [train_labels, test_labels, train_labels, test_labels], axis=0
        ).astype(int)
        paired_split = np.array(
            ["train-standard"] * len(train_labels)
            + ["test-standard"] * len(test_labels)
            + ["train-complementary"] * len(train_labels)
            + ["test-complementary"] * len(test_labels)
        )
        perplexity = min(30, max(2, (len(paired_features) - 1) // 3))
        paired_2d = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=int(getattr(self.args, "seed", 2024)),
        ).fit_transform(paired_features)
        fig, ax = plt.subplots(figsize=(7, 6))
        marker_map = {
            "train-standard": "o",
            "test-standard": "^",
            "train-complementary": "x",
            "test-complementary": "s",
        }
        for split_name, marker in marker_map.items():
            split_mask = paired_split == split_name
            for label_value in sorted(np.unique(paired_labels)):
                point_mask = split_mask & (paired_labels == label_value)
                if np.any(point_mask):
                    ax.scatter(
                        paired_2d[point_mask, 0],
                        paired_2d[point_mask, 1],
                        c=[cmap(label_value % 10)],
                        marker=marker,
                        label=f"{split_name} label={label_value}",
                        alpha=0.72,
                        s=34,
                    )
        ax.set_title("t-SNE of standard and complementary embeddings")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        paired_path = os.path.join(output_dir, f"{fold_name}_standard_complementary_tsne.png")
        fig.savefig(paired_path, bbox_inches="tight")
        plt.close(fig)
    
    def val(self, val_loader, criterion):
        total_loss = []
        self.model.eval()
        self.latest_val_best_threshold = None
        preds=[]
        targets=[]
        probs=[]
        branch_storage = {}
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
                loss = primary_loss if primary_loss is not None else self._supervised_loss(
                    y_hat,
                    label,
                    class_label,
                    criterion,
                )
                total_loss.append(loss.cpu().numpy())
                pred, prob, target = self._prediction_probability_for_metrics(y_hat, metric_label)
                probs.append(prob)
                preds.append(pred)
                targets.append(target)
                self._append_branch_batch_metrics(
                    branch_storage,
                    self._collect_branch_batch_metrics(metric_label),
                )
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
        self.latest_val_branch_metrics = self._finalize_branch_metrics(branch_storage)
        self.model.train()
        return total_loss,evaluate(targets, preds,self.args.classes,probs)

    def train(self, train_loader, val_loader, check_path, tensorboard_writer=None):
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
                "temporal_pred_base_loss": [],
                "temporal_pred_delta_loss": [],
                "temporal_pred_lowfreq_loss": [],
                "temporal_pred_corr_loss": [],
                "temporal_pred_std_ratio": [],
                "temporal_pred_corr_value": [],
                "causal_dag_loss": [],
                "causal_l1_loss": [],
                "temporal_sparse_loss": [],
                "temporal_parameter_sparse_loss": [],
                "temporal_smooth_loss": [],
                "temporal_group_sparse_loss": [],
                "temporal_lag_hierarchy_loss": [],
                "complementary_view_loss": [],
                "complementary_view_weighted_loss": [],
                "complementary_view_distance": [],
                "complementary_instance_infonce_loss": [],
                "complementary_instance_infonce_weighted_loss": [],
                "complementary_masked_ce_loss": [],
                "complementary_masked_ce_weighted_loss": [],
                "complementary_mask_ratio": [],
                "complementary_topology_salience_entropy": [],
                "complementary_semantic_salience_entropy": [],
                "hpec_reliable_tp_ratio": [],
                "hpec_reliable_confidence_mean": [],
                "hpec_reliable_view_consistency_mean": [],
                "hpec_reliable_assignment_entropy": [],
                "hpec_reliable_updated_prototype_count": [],
                "hpec_reliable_unupdated_prototype_count": [],
                "hpec_reliable_ema_displacement_mean": [],
                "causal_reachability_residual_norm": [],
                "sample_graph_l1_loss": [],
                "sample_graph_deviation_loss": [],
                "temporal_reg_scale": [],
                "module1_denoise_loss": [],
                "module1_denoise_weighted_loss": [],
                "causal_stability_loss": [],
                "causal_stability_weighted_loss": [],
                "causal_recon_weighted_loss": [],
                "causal_dag_weighted_loss": [],
                "causal_l1_weighted_loss": [],
                "temporal_sparse_weighted_loss": [],
                "temporal_smooth_weighted_loss": [],
                "sample_graph_l1_weighted_loss": [],
                "sample_graph_deviation_weighted_loss": [],
                "causal_aux_loss": [],
                "hpec_logit_gate": [],
                "prototype_cos_abs_mean": [],
                "prototype_cos_abs_max": [],
                "prototype_same_class_cos_max": [],
                "prototype_radius_mean": [],
                "prototype_radius_min": [],
                "prototype_radius_max": [],
                "prototype_tangent_norm_mean": [],
                "prototype_tangent_norm_min": [],
                "prototype_tangent_norm_max": [],
                "hpec_causal_role_energy_weight": [],
                "hpec_causal_role_energy_std": [],
                "hpec_causal_role_gate_entropy": [],
                "hpec_causal_role_gate_peak": [],
                "hpec_prototype_logit_margin_mean": [],
                "hpec_prototype_logit_abs_mean": [],
                "hpec_prototype_logit_mode_margin_preserving": [],
                "hpec_prototype_logit_scale_value": [],
                "hpec_energy_margin_mean": [],
                "hpec_similarity_margin_mean": [],
                "hpec_final_ce_loss": [],
                "hpec_energy_loss": [],
                "hpec_energy_weighted_loss": [],
                "hpec_teacher_distill_loss": [],
                "hpec_teacher_distill_weighted_loss": [],
                "hpec_teacher_entropy": [],
                "hpec_z_radius_loss": [],
                "hpec_z_radius_weighted_loss": [],
                "hpec_prototype_separation_loss": [],
                "hpec_prototype_separation_weighted_loss": [],
                "class_prior_alignment_loss": [],
                "class_prior_alignment_weighted_loss": [],
                "module34_supcon_loss": [],
                "module34_supcon_weighted_loss": [],
                "module34_supcon_positive_pairs": [],
                "module34_center_loss": [],
                "module34_center_weighted_loss": [],
                "module34_center_intra_loss": [],
                "module34_center_inter_loss": [],
                "module34_center_cos_mean": [],
                "module34_branch_ce_loss": [],
                "module34_branch_ce_weighted_loss": [],
                "lp_lorentz_constraint_error": [],
                "lp_in_aggregation_norm": [],
                "lp_out_aggregation_norm": [],
                "lp_alpha_out": [],
                "lp_centroid_message_weight": [],
                "lp_in_attention_temperature": [],
                "lp_out_attention_temperature": [],
                "lp_mac_radius_mean": [],
                "lp_mac_radius_max": [],
                "lp_mac_low_clip_ratio": [],
                "lp_mac_high_clip_ratio": [],
                "lp_stats_update_gate_mean": [],
                "lp_hbr_loss": [],
                "lp_hbr_weighted_loss": [],
                "site_adversarial_loss": [],
                "site_modulation_reg_loss": [],
                "site_adversarial_weighted_loss": [],
                "z_radius_mean": [],
                "z_radius_max": [],
                "z_tangent_norm_mean": [],
                "fc_residual_norm_mean": [],
                "hgcn_fc_inject_norm_mean": [],
                "hgcn_fc_anchor_gate_mean": [],
                "hgcn_fc_anchor_update_norm_mean": [],
                "module34_film_weight": [],
                "module34_film_scale_abs_mean": [],
                "module34_film_shift_norm_mean": [],
                "hyperbolic_residual_margin_scale_mean": [],
                "hyperbolic_residual_margin_scale_max": [],
                "hyperbolic_residual_bias_mean": [],
                "hyperbolic_residual_bias_abs_mean": [],
                "hyperbolic_residual_logit_blend": [],
                "hyperbolic_update_logit_mean": [],
                "hyperbolic_update_logit_abs_mean": [],
                "hpec_residual_calibration_batch_margin": [],
                "hpec_residual_calibrated_margin_abs_mean": [],
                "hpec_residual_raw_margin_abs_mean": [],
                "hpec_residual_raw_margin_std": [],
                "final_positive_prob_mean": [],
                "hpec_energy_margin_mean": [],
                "hpec_energy_margin_signed_mean": [],
                "hpec_energy_logit_abs_mean": [],
                "hpec_network_energy_mean": [],
                "hpec_network_energy_std": [],
                "hpec_network_attention_max": [],
                "hpec_network_selector_entropy": [],
                "hpec_network_selector_peak": [],
                "hpec_network_energy_temperature": [],
                "hpec_network_energy_class_softmin": [],
                "hpec_network_energy_weight": [],
                "hpec_network_energy_normalized": [],
                "hpec_network_selector_sharpness": [],
                "hpec_prototype_logit_margin_mean": [],
                "hpec_prototype_logit_signed_margin_mean": [],
                "hpec_prototype_logit_abs_mean": [],
                "causal_meta_analytic_spectral_radius": [],
                "causal_meta_dagma_spectral_radius": [],
                "causal_meta_shared_adjacency_mass_mean": [],
                "causal_meta_shared_adjacency_directionality_ratio": [],
                "causal_meta_a0_adjacency_mass_mean": [],
                "causal_meta_a0_adjacency_directionality_ratio": [],
                "causal_meta_a0_to_alag_mass_ratio": [],
                "causal_meta_alag_mean_adjacency_mass_mean": [],
                "causal_meta_alag_mean_adjacency_directionality_ratio": [],
                "causal_meta_attention_entropy": [],
                "causal_meta_gate_mass": [],
                "causal_meta_dagma_stage_id": [],
                "causal_meta_dagma_effective_scale": [],
                "causal_meta_sample_delta_abs_mean": [],
                "classification_graph_mass": [],
                "classification_graph_asymmetry_ratio": [],
                "classification_graph_causal_support_density": [],
            }
            preds=[]
            targets=[]
            probs=[]
            train_branch_storage = {}
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
                loss = primary_loss if primary_loss is not None else self._supervised_loss(
                    y_hat,
                    label,
                    class_label,
                    criterion,
                )
                aux_loss = self._get_model_aux_loss()
                supervised_aux_loss = self._get_model_supervised_aux_loss(class_label)
                aux_losses = self._get_model_aux_losses()
                site_adversarial_loss = self._get_model_site_adversarial_loss(site_label)
                # 模块 3 启用后，分类 loss 与模块 2 的因果图共享同一条计算图。
                # 合并成一次 backward，保证 Loss_cls 能经 HGCN 回传到 A_learned。
                total_loss = loss + aux_loss if aux_loss is not None else loss
                if supervised_aux_loss is not None:
                    total_loss = total_loss + supervised_aux_loss
                if site_adversarial_loss is not None:
                    total_loss = total_loss + site_adversarial_loss
                total_loss.backward()
                model_optim.step()
                self._update_reliable_prototypes_after_step(class_label)
                self._update_model_ema()
                # prototype 独立 EMA 在 step 后更新，重新读取以写入同一 batch 的诊断。
                aux_losses = self._get_model_aux_losses()
                # 多阶因果编码的 hop 数由配置决定，动态收集每个 hop 的 gate 和范数，
                # 使 TensorBoard 能展示实际参与融合的各阶权重。
                for key, value in aux_losses.items():
                    if key.startswith("causal_reachability_hop_"):
                        aux_loss_parts.setdefault(key, []).append(value.detach().cpu().item())
                cls_losses.append(loss.detach().cpu().item())
                for key in aux_loss_parts:
                    value = aux_losses.get(key)
                    if value is not None:
                        aux_loss_parts[key].append(value.detach().cpu().item())
                total_batch_loss = loss.detach()
                if aux_loss is not None:
                    total_batch_loss = total_batch_loss + aux_loss.detach()
                if supervised_aux_loss is not None:
                    total_batch_loss = total_batch_loss + supervised_aux_loss.detach()
                if site_adversarial_loss is not None:
                    total_batch_loss = total_batch_loss + site_adversarial_loss.detach()
                train_loss.append(total_batch_loss.cpu().numpy())
                pred, prob, target = self._prediction_probability_for_metrics(y_hat, metric_label)
                
                probs.append(prob)
                preds.append(pred)
                targets.append(target)
                self._append_branch_batch_metrics(
                    train_branch_storage,
                    self._collect_branch_batch_metrics(metric_label),
                )
            epoch_prototype_stats = self._finalize_epoch_prototype_update()
            for key, value in epoch_prototype_stats.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    aux_loss_parts.setdefault(key, []).append(value.detach().cpu().item())
            if len(preds)>0:
                preds = np.concatenate(preds, axis=0)    
                targets = np.concatenate(targets, axis=0)   
                probs = np.concatenate(probs, axis=0) 
            else:
                preds=preds[0]
                targets=targets[0]
                probs=probs[0]
            train_metric =evaluate(targets, preds,self.args.classes,probs)
            train_branch_metrics = self._finalize_branch_metrics(train_branch_storage)
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
            val_branch_metrics = getattr(self, "latest_val_branch_metrics", {})
            val_threshold_info = getattr(self, "latest_val_best_threshold", None)
            epoch_seconds = time.time() - epoch_time
            epoch_record = {
                "epoch": int(epoch + 1),
                "total_epochs": int(self.args.train_epochs),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "epoch_seconds": float(epoch_seconds),
            }
            epoch_record.update(self._metric_dict("train", train_metric))
            epoch_record.update(self._metric_dict("val", val_metric))
            epoch_record.update(self._branch_metric_record("train_branch", train_branch_metrics))
            epoch_record.update(self._branch_metric_record("val_branch", val_branch_metrics))
            if best_threshold_info is not None:
                epoch_record["train_best_threshold"] = float(best_threshold_info[0])
                epoch_record["train_best_macro_f1"] = float(best_threshold_info[1])
            if val_threshold_info is not None:
                epoch_record["val_best_threshold"] = float(val_threshold_info[0])
                epoch_record["val_best_macro_f1"] = float(val_threshold_info[1])
            for key, value in loss_parts.items():
                epoch_record[key] = float(value)
            epoch_metric_records.append(epoch_record)
            self._write_tensorboard_epoch(tensorboard_writer, epoch_record)
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
                train_branch_line = self._format_branch_metric_line("Train", train_branch_metrics)
                val_branch_line = self._format_branch_metric_line("Validation", val_branch_metrics)
                if train_branch_line:
                    print(train_branch_line)
                if val_branch_line:
                    print(val_branch_line)
            early_stopping(-monitor_score, self.model, check_path)
            if self. args.print_process:
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        self._apply_model_ema()
        self._save_causal_visualization_after_training(train_loader, val_loader, check_path)
        self._save_graph_diagnostics_after_training(check_path)
        self._save_labelwise_graph_diagnostics_after_training(train_loader, val_loader, check_path)
        self._save_tsne_after_training(train_loader, val_loader, check_path)
        self._save_epoch_metrics(check_path, epoch_metric_records)
        final_threshold = None
        if epoch_metric_records:
            final_threshold = epoch_metric_records[-1].get("val_best_threshold")
        return final_threshold, epoch_metric_records
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
        branch_storage = {}
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
                loss = primary_loss if primary_loss is not None else self._supervised_loss(
                    y_hat,
                    label,
                    class_label,
                    criterion,
                )
                total_loss.append(loss.cpu().numpy())
                pred, prob, target = self._prediction_probability_with_threshold(
                    y_hat,
                    metric_label,
                    threshold=threshold,
                )
                probs.append(prob)
                preds.append(pred)
                targets.append(target)
                self._append_branch_batch_metrics(
                    branch_storage,
                    self._collect_branch_batch_metrics(metric_label),
                )
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        probs = np.concatenate(probs, axis=0)
        self._latest_eval_arrays = (preds, probs, targets)
        self.latest_eval_branch_metrics = self._finalize_branch_metrics(branch_storage)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss,evaluate(targets, preds,self.args.classes,probs)

    def kf_train(self, setting):
        cv_start_time = time.time()
        train_loaders,val_loaders=self._get_data()
        val_metrics=[]
        fold_seconds_records = []
        final_metric_records = []
        fold_epoch_metric_records = []
        max_folds = int(getattr(self.args, "max_folds", 0) or 0)
        fold_pairs = list(zip(train_loaders, val_loaders))
        if max_folds > 0:
            fold_pairs = fold_pairs[:max_folds]
        total_folds = len(fold_pairs)
        summary_writer = self._create_tensorboard_writer(setting, "summary")
        try:
            for fold, (train_loader, val_loader) in tqdm(enumerate(fold_pairs), total=total_folds, desc="Cross-validation", ncols=100):
                fold_start_time = time.time()
                check_path = os.path.join(self.args.checkpoints, setting+'fold'+str(fold + 1))
                fold_writer = self._create_tensorboard_writer(setting, f"fold_{fold + 1}")
                try:
                    self.reset_model()
                    module_summary = self._module_path_summary()
                    if module_summary is not None:
                        print(f"Fold {fold + 1}/{self.args.kfold} {module_summary}")
                    if self. args.print_process: print(f"Fold {fold + 1}/{self.args.kfold} Start>>>>>>>>>>>>>>>>>>>\n")
                    final_epoch_threshold, fold_epoch_records = self.train(
                        train_loader,
                        val_loader,
                        check_path,
                        tensorboard_writer=fold_writer,
                    )
                    fold_epoch_metric_records.append(fold_epoch_records)

                    _, raw_test_metric = self.val_with_threshold(
                        val_loader,
                        self._select_criterion(),
                        threshold=None,
                    )
                    raw_preds, raw_probs, raw_targets = self._latest_eval_arrays
                    raw_branch_metrics = getattr(self, "latest_eval_branch_metrics", {})
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
                    thresholded_branch_metrics = getattr(self, "latest_eval_branch_metrics", {})
                    raw_diagnostics = self._prediction_diagnostics(raw_targets, raw_preds, raw_probs)
                    final_diagnostics = self._prediction_diagnostics(final_targets, final_preds, final_probs)
                    print(
                        f"Fold {fold + 1} final-epoch test raw-threshold: "
                        f"accuracy={raw_test_metric[0]:.4f}, precision={raw_test_metric[1]:.4f}, "
                        f"recall={raw_test_metric[2]:.4f}, macro_f1={raw_test_metric[3]:.4f}, "
                        f"roc_auc={raw_test_metric[4]:.4f}"
                    )
                    fold_seconds = time.time() - fold_start_time
                    fold_seconds_records.append(fold_seconds)
                    print(f"Fold {fold + 1} training wall time: {fold_seconds:.2f}s")
                    print(f"Fold {fold + 1} final-epoch test raw diagnostics: {self._format_prediction_diagnostics(raw_diagnostics)}")
                    raw_branch_line = self._format_branch_metric_line(
                        f"Fold {fold + 1} Raw Test",
                        raw_branch_metrics,
                    )
                    if raw_branch_line:
                        print(raw_branch_line)
                    if threshold_for_report is not None:
                        print(f"Fold {fold + 1} final-epoch test uses final-epoch threshold: {threshold_for_report:.4f}")
                        print(f"Fold {fold + 1} final-epoch test thresholded diagnostics: {self._format_prediction_diagnostics(final_diagnostics)}")
                    val_metrics.append(raw_test_metric)
                    fold_metric_with_time = list(raw_test_metric) + [float(fold_seconds)]
                    self._write_tensorboard_final_metrics(fold_writer, fold_metric_with_time, fold + 1)
                    self._write_tensorboard_final_metrics(summary_writer, fold_metric_with_time, fold + 1)
                    fold_record = {
                        "record_type": "fold",
                        "fold": int(fold + 1),
                        "total_folds": int(total_folds),
                        "epoch_source": "final_epoch_current_model",
                        "summary_metric_source": "raw_final_epoch_test",
                        "threshold_source": "final_epoch_test_threshold_reference_only" if threshold_for_report is not None else "fixed_0.5",
                        "best_threshold": None if threshold_for_report is None else float(threshold_for_report),
                        "train_seconds": float(fold_seconds),
                    }
                    fold_record.update(self._metric_dict("raw", raw_test_metric))
                    fold_record.update(self._metric_dict("raw_final_epoch_test", raw_test_metric))
                    fold_record.update(self._metric_dict("final", raw_test_metric))
                    fold_record.update(self._metric_dict("thresholded_final_epoch_test", test_metric))
                    fold_record.update(self._branch_metric_record("raw_branch", raw_branch_metrics))
                    fold_record.update(self._branch_metric_record("thresholded_branch", thresholded_branch_metrics))
                    fold_record["raw_diagnostics"] = raw_diagnostics
                    fold_record["final_diagnostics"] = final_diagnostics
                    final_metric_records.append(fold_record)
                    self._save_final_metrics(check_path, final_metric_records)

                    ## Uncomment below code for save space on device
                    if self. args.del_weight:
                        self.del_weight(check_path)
                    if self. args.print_process: print(f"Fold {fold + 1}/{self.args.kfold} End<<<<<<<<<<<<<<<<<<<\n")
                finally:
                    if fold_writer is not None:
                        fold_writer.close()
            cv_train_seconds = time.time() - cv_start_time
            avg_metric=[np.mean([val_metrics[i][j] for i in range(len(val_metrics))]) for j in range(5)]
            avg_metric.append(float(cv_train_seconds))
            avg_fold_seconds = float(np.mean(fold_seconds_records)) if fold_seconds_records else 0.0
            print(f'Final-epoch Test Avg (raw) accuracy: {avg_metric[0]:.4f}, precision: {avg_metric[1]:.4f}, recall: {avg_metric[2]:.4f}, macro_f1: {avg_metric[3]:.4f}, roc_auc: {avg_metric[4]:.4f}, train_seconds: {avg_metric[5]:.2f}, avg_fold_seconds: {avg_fold_seconds:.2f}')
            self._write_tensorboard_final_metrics(summary_writer, avg_metric, total_folds + 1, prefix="Summary/avg")
            self._write_tensorboard_compare_trends(summary_writer, fold_epoch_metric_records)
            self._write_tensorboard_compare_final(summary_writer, avg_metric)
            self._write_tensorboard_hparam_effects(summary_writer, avg_metric)
            if fold_pairs:
                summary_record = {
                    "record_type": "summary",
                    "fold": "avg",
                    "total_folds": int(total_folds),
                }
                summary_record.update(self._metric_dict("final", avg_metric))
                summary_record["train_seconds"] = float(cv_train_seconds)
                summary_record["avg_fold_seconds"] = float(avg_fold_seconds)
                summary_check_path = os.path.join(self.args.checkpoints, setting + "summary")
                final_metric_records.append(summary_record)
                self._save_final_metrics(summary_check_path, final_metric_records)
            return avg_metric
        finally:
            if summary_writer is not None:
                summary_writer.close()
    
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
