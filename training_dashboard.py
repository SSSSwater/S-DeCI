"""Local training workbench for this repository.

Run with: .venv\\Scripts\\python.exe training_dashboard.py
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import shlex
import subprocess
import sys
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web_dashboard"
OUTPUT_ROOT = ROOT / "outputs"
DASHBOARD_RUNS = OUTPUT_ROOT / "dashboard_runs"
TRAINING_RECORDS = OUTPUT_ROOT / "training_records"
ENTRY_SCRIPTS = {
    "abide": "test_abide_best_config.py",
    "mdd": "test_mdd_best_config.py",
    "matai": "test_matai_best_config.py",
    "taowu": "test_taowu_best_config.py",
}
DATASET_DEFAULTS = {
    "abide": {"label": "ABIDE", "seq_len": 180, "classes": 2, "protocol": "AAL116"},
    "mdd": {"label": "MDD", "seq_len": 230, "classes": 2, "protocol": "AAL116"},
    "matai": {"label": "Matai", "seq_len": 200, "classes": 2, "protocol": "AAL116"},
    "taowu": {"label": "Taowu", "seq_len": 239, "classes": 2, "protocol": "AAL116"},
}
PARAMETER_GROUPS = [
    {"title": "训练与优化", "fields": [
        {"key": "train_epochs", "label": "训练轮数", "kind": "number", "default": 50},
        {"key": "max_folds", "label": "最大 Fold", "kind": "number", "default": 5},
        {"key": "iterations", "label": "重复次数", "kind": "number", "default": 1},
        {"key": "batch_size", "label": "批大小", "kind": "number", "default": 32},
        {"key": "learning_rate", "label": "学习率", "kind": "number", "step": "any", "default": 0.001},
        {"key": "lradj", "label": "学习率策略", "kind": "select", "default": "cosine", "options": ["constant", "cosine", "type1", "type2", "type3", "type4"]},
        {"key": "weight_decay", "label": "Weight decay", "kind": "number", "step": "any", "default": 0.01},
        {"key": "dropout", "label": "全局 Dropout", "kind": "number", "step": "0.01", "default": 0.2},
        {"key": "d_model", "label": "隐藏维度", "kind": "number", "default": 32},
        {"key": "loss", "label": "分类损失", "kind": "select", "default": "bce", "options": ["bce", "mse", "weighted_bce", "weighted_mse", "ce"]},
        {"key": "patience", "label": "Early stop patience", "kind": "number", "default": 50},
        {"key": "seed", "label": "随机种子", "kind": "number", "default": 2024},
        {"key": "use_model_ema", "label": "启用权重 EMA", "kind": "checkbox", "default": 1},
        {"key": "use_norm", "label": "输入归一化", "kind": "checkbox", "default": 1},
    ]},
    {"title": "模块 1 特征", "fields": [
        {"key": "use_deci_module1", "label": "启用模块 1", "kind": "checkbox", "default": 1},
        {"key": "module1_feature_mode", "label": "特征模式", "kind": "select", "default": "alff", "options": ["alff", "deci", "raw"]},
        {"key": "module1_alff_time_weight", "label": "ALFF 时序权重", "kind": "number", "step": "0.01", "default": 0.2},
        {"key": "module1_random_crop", "label": "随机时间裁剪", "kind": "checkbox", "default": 1},
        {"key": "module1_temporal_dropout", "label": "时序 Dropout", "kind": "number", "step": "0.01", "default": 0.03},
        {"key": "module1_roi_dropout", "label": "ROI Dropout", "kind": "number", "step": "0.01", "default": 0.02},
        {"key": "module1_denoise_loss_weight", "label": "去噪损失权重", "kind": "number", "step": "any", "default": 0},
        {"key": "module1_temporal_stats_weight", "label": "时序统计权重", "kind": "number", "step": "any", "default": 0},
    ]},
    {"title": "模块 2 因果学习", "fields": [
        {"key": "use_causal_module2", "label": "启用因果模块", "kind": "checkbox", "default": 1},
        {"key": "causal_learning_target", "label": "学习目标", "kind": "select", "default": "temporal_sem", "options": ["temporal_sem", "static_feature"]},
        {"key": "causal_graph_method", "label": "因果图方法", "kind": "select", "default": "nts_notears", "options": ["nts_notears", "attn_nts_notears", "dagma_logdet", "dag_sampling"]},
        {"key": "temporal_lag_order", "label": "滞后阶数", "kind": "number", "default": 5},
        {"key": "temporal_candidate_parent_topk", "label": "候选父节点 Top-K", "kind": "number", "default": 4},
        {"key": "causal_learning_rate", "label": "因果图学习率", "kind": "number", "step": "any", "default": 0.0005},
        {"key": "lambda_causal_dag", "label": "DAG 约束权重", "kind": "number", "step": "any", "default": 0.0001},
        {"key": "lambda_causal_l1", "label": "因果 L1 权重", "kind": "number", "step": "any", "default": 0.00001},
        {"key": "lambda_causal_stability", "label": "图稳定性权重", "kind": "number", "step": "any", "default": 0},
        {"key": "lambda_temporal_pred", "label": "时序预测权重", "kind": "number", "step": "any", "default": 1},
        {"key": "lambda_temporal_sparse", "label": "时序稀疏权重", "kind": "number", "step": "any", "default": 0.0005},
        {"key": "lambda_temporal_smooth", "label": "时序平滑权重", "kind": "number", "step": "any", "default": 0.0001},
        {"key": "classification_graph_source", "label": "分类图来源", "kind": "select", "default": "causal_soft_masked_fc", "options": ["causal_soft_masked_fc", "blend", "gated_fc", "gated_fc_signed", "sample_correlation", "fc"]},
        {"key": "module2_sample_correlation_blend", "label": "样本 FC 混合比例", "kind": "number", "step": "0.01", "default": 0.75},
        {"key": "module2_graph_residual_alpha", "label": "图残差强度", "kind": "number", "step": "0.01", "default": 0.1},
        {"key": "causal_edge_dropout", "label": "因果边 Dropout", "kind": "number", "step": "0.01", "default": 0.25},
        {"key": "sample_correlation_mode", "label": "样本相关模式", "kind": "select", "default": "abs", "options": ["abs", "positive", "raw"]},
    ]},
    {"title": "模块 3 HGCN 与模块 4 HPEC", "fields": [
        {"key": "use_hyperbolic_modules34", "label": "启用模块 3/4", "kind": "checkbox", "default": 1},
        {"key": "use_hgcn_module3", "label": "启用 HGCN", "kind": "checkbox", "default": 1},
        {"key": "use_hpec_module4", "label": "启用 HPEC", "kind": "checkbox", "default": 1},
        {"key": "module34_arch", "label": "模块 3/4 架构", "kind": "select", "default": "hgcn_hpec", "options": ["hgcn_hpec", "lp_brain_hpec"]},
        {"key": "hgcn_hidden_dim", "label": "HGCN 隐藏维度", "kind": "number", "default": 64},
        {"key": "hgcn_layers", "label": "HGCN 层数", "kind": "number", "default": 1},
        {"key": "hgcn_dropout", "label": "HGCN Dropout", "kind": "number", "step": "0.01", "default": 0.6},
        {"key": "hgcn_residual_alpha", "label": "HGCN 残差比例", "kind": "number", "step": "0.01", "default": 0.35},
        {"key": "hgcn_readout_mode", "label": "HGCN Readout", "kind": "select", "default": "mean_std", "options": ["mean_std", "causal_weighted_mean_std", "graph_weighted_mean_std", "network_stats", "causal_attention"]},
        {"key": "hyperbolic_logit_residual_weight", "label": "双曲 Logit 权重", "kind": "number", "step": "0.01", "default": 0.5},
        {"key": "hpec_classification_mode", "label": "HPEC 分类策略", "kind": "select", "default": "energy_primary", "options": ["energy_primary", "prototype_primary", "energy_prototype_residual", "distance_prototype"]},
        {"key": "hpec_energy_mode", "label": "HPEC 能量模式", "kind": "select", "default": "busemann", "options": ["busemann", "cone"]},
        {"key": "hpec_prototypes_per_class", "label": "每类原型数", "kind": "number", "default": 2},
        {"key": "hpec_energy_loss_weight", "label": "能量损失权重", "kind": "number", "step": "any", "default": 0.2},
        {"key": "hpec_prototype_ce_loss_weight", "label": "原型 CE 权重", "kind": "number", "step": "any", "default": 0.05},
        {"key": "hpec_ema_alpha", "label": "原型 EMA Alpha", "kind": "number", "step": "0.001", "default": 0.995},
    ]},
    {"title": "FC 读出与诊断", "fields": [
        {"key": "use_fc_readout_branch", "label": "启用 FC 读出", "kind": "checkbox", "default": 1},
        {"key": "fc_readout_mode", "label": "FC 读出模式", "kind": "select", "default": "network", "options": ["network", "upper_tri", "both"]},
        {"key": "fc_readout_dropout", "label": "FC Dropout", "kind": "number", "step": "0.01", "default": 0.5},
        {"key": "fc_readout_edge_dropout", "label": "FC 边 Dropout", "kind": "number", "step": "0.01", "default": 0},
        {"key": "gcn_fallback_hidden_dim", "label": "Fallback 隐藏维度", "kind": "number", "default": 32},
        {"key": "gcn_fallback_dropout", "label": "Fallback Dropout", "kind": "number", "step": "0.01", "default": 0.5},
        {"key": "gcn_fallback_readout_mode", "label": "Fallback Readout", "kind": "select", "default": "mean", "options": ["mean", "attention", "mean_max"]},
        {"key": "visualize_causal", "label": "保存因果可视化", "kind": "checkbox", "default": 0},
        {"key": "print_metric_every", "label": "指标打印间隔", "kind": "number", "default": 10},
    ]},
]
TRAINING_FIELDS = {field["key"]: "--" + field["key"].replace("_", "-") for group in PARAMETER_GROUPS for field in group["fields"]}
DEFAULTS_CACHE: dict[str, dict] = {}
DEFAULTS_LOCK = threading.Lock()
METRIC_RE = re.compile(
    r"(?:Final-epoch Test Avg \(raw\)\s+)?accuracy:\s*(?P<accuracy>[-+0-9.eE]+),\s*"
    r"precision:\s*(?P<precision>[-+0-9.eE]+),\s*recall:\s*(?P<recall>[-+0-9.eE]+),\s*"
    r"macro_f1:\s*(?P<macro_f1>[-+0-9.eE]+),\s*(?:roc_auc|auc):\s*(?P<roc_auc>[-+0-9.eE]+)"
    r"(?:,\s*train_seconds:\s*(?P<train_seconds>[-+0-9.eE]+)s?)?",
    re.IGNORECASE,
)
SWEEP_HEADER_RE = re.compile(r"^\s*(?P<parameter>--[\w-]+)\s*\|\s*Status", re.MULTILINE)
SWEEP_ROW_RE = re.compile(
    r"^\s*(?P<value>[^|\r\n]+?)\s*\|\s*ok\s*\|\s*"
    r"(?P<accuracy>[-+0-9.]+)%\s*\|\s*(?P<precision>[-+0-9.]+)%\s*\|\s*"
    r"(?P<recall>[-+0-9.]+)%\s*\|\s*(?P<macro_f1>[-+0-9.]+)%\s*\|\s*"
    r"(?P<roc_auc>[-+0-9.]+)%\s*\|\s*(?P<train_seconds>[-+0-9.]+)",
    re.MULTILINE,
)
RUN_SUMMARY_RE = re.compile(
    r"\[run\]\s+data=(?P<data>[^,]+),.*?protocol=(?P<protocol>[^,]+),\s*"
    r"folds=(?P<kfold>\d+),\s*max_folds=(?P<max_folds>\d+),\s*"
    r"epochs=(?P<train_epochs>\d+),\s*batch_size=(?P<batch_size>\d+),\s*"
    r"lr=(?P<learning_rate>[-+0-9.eE]+)",
    re.IGNORECASE,
)
MODULE_SUMMARY_RE = re.compile(
    r"modules:\s*module1=(?P<use_deci_module1>\d+),\s*module2=(?P<use_causal_module2>\d+),\s*"
    r"module3\(HGCN\)=(?P<use_hgcn_module3>\d+),\s*module4\(HPEC\)=(?P<use_hpec_module4>\d+)",
    re.IGNORECASE,
)
RUNS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()


def read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16", errors="replace")
        return raw.decode("utf-8-sig", errors="replace")
    except OSError:
        return ""


def parse_metrics(text: str) -> dict | None:
    matches = list(METRIC_RE.finditer(text))
    if not matches:
        return None
    return {key: float(value) for key, value in matches[-1].groupdict().items() if value is not None}


def parse_experiment_records(text: str) -> list[tuple[str, dict]]:
    """Parse every summarized candidate in a sweep, or one final run otherwise."""
    header = SWEEP_HEADER_RE.search(text)
    sweep_rows = list(SWEEP_ROW_RE.finditer(text))
    if header and sweep_rows:
        parameter = header.group("parameter")
        records = []
        for match in sweep_rows:
            metrics = {
                key: float(value) / 100.0
                for key, value in match.groupdict().items()
                if key != "value"
            }
            metrics["train_seconds"] = float(match.group("train_seconds"))
            records.append((f"{parameter}={match.group('value').strip()}", metrics))
        return records
    metrics = parse_metrics(text)
    return [("", metrics)] if metrics else []


def entry_default_parameters(entry: str) -> dict:
    """Get the effective argparse defaults for a dataset wrapper without training."""
    with DEFAULTS_LOCK:
        cached = DEFAULTS_CACHE.get(entry)
        if cached is not None:
            return copy.deepcopy(cached)
        original_argv = sys.argv[:]
        try:
            common = importlib.import_module("test_mdd_best_config")
            if entry != "mdd":
                wrapper = importlib.import_module(f"test_{entry}_best_config")
                sys.argv = [str(ROOT / ENTRY_SCRIPTS[entry])]
                wrapper._prepend_defaults()
            else:
                sys.argv = [str(ROOT / ENTRY_SCRIPTS[entry])]
            defaults = vars(common.parse_args())
        finally:
            sys.argv = original_argv
        DEFAULTS_CACHE[entry] = defaults
        return copy.deepcopy(defaults)


def command_overrides(text: str, entry: str, label: str) -> dict:
    """Find the command matching a sweep candidate and return its CLI values."""
    script_name = ENTRY_SCRIPTS.get(entry)
    if not script_name:
        return {}
    command_matches = list(re.finditer(rf"^.*{re.escape(script_name)}.*$", text, re.MULTILINE))
    if label and "=" in label:
        flag, value = label.split("=", 1)
        expected = f"{flag} {value}"
        matching = [match for match in command_matches if expected in match.group(0)]
        if matching:
            command_matches = matching
    overrides = {}
    if command_matches:
        command_match = command_matches[0]
        next_start = next((match.start() for match in command_matches[1:] if match.start() > command_match.start()), len(text))
        segment = text[command_match.start():next_start]
        tokens = shlex.split(command_match.group(0).strip(), posix=False)
        try:
            script_index = next(index for index, token in enumerate(tokens) if token.endswith(script_name))
        except StopIteration:
            script_index = len(tokens)
    else:
        segment = text
        tokens = []
        script_index = len(tokens)
    index = script_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                overrides[key] = tokens[index + 1]
                index += 2
            else:
                overrides[key] = 1
                index += 1
        else:
            index += 1
    run_summary = RUN_SUMMARY_RE.search(segment)
    if run_summary:
        overrides.update({key: value.strip() for key, value in run_summary.groupdict().items()})
    module_summary = MODULE_SUMMARY_RE.search(segment)
    if module_summary:
        overrides.update(module_summary.groupdict())
        overrides["use_hyperbolic_modules34"] = str(int(bool(int(overrides["use_hgcn_module3"]) or int(overrides["use_hpec_module4"]))))
    return overrides


def dataset_from_name(name: str) -> str:
    lowered = name.lower()
    for key in ENTRY_SCRIPTS:
        if key in lowered:
            return key
    return "other"


def dataset_from_parameters(parameters: dict) -> str:
    data = str(parameters.get("data", "")).lower()
    normalized = data.replace("ā", "a")
    for entry in ENTRY_SCRIPTS:
        if entry in normalized:
            return entry
    return "other"


def manifest_experiments() -> list[dict]:
    if not TRAINING_RECORDS.exists():
        return []
    rows = []
    for path in sorted(TRAINING_RECORDS.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parameters = record.get("parameters") or {}
        if not isinstance(parameters, dict):
            continue
        dataset = dataset_from_parameters(parameters)
        metrics = record.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative = path.name
        rows.append({
            "id": f"manifest:{relative}",
            "name": path.stem,
            "dataset": dataset,
            "modified": record.get("finished_at") or record.get("started_at") or datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "metrics": metrics,
            "parameters": parameters,
            "status": record.get("status", "unknown"),
            "source": "manifest",
        })
    return rows


def log_files() -> list[Path]:
    # `outputs` also holds TensorBoard events and per-fold diagnostics.  Scan
    # only the directories that conventionally contain textual training logs.
    paths = list(OUTPUT_ROOT.glob("*.log"))
    for directory in (
        OUTPUT_ROOT / "logs",
        OUTPUT_ROOT / "sweep_logs",
        OUTPUT_ROOT / "model_compare_logs",
        DASHBOARD_RUNS,
    ):
        if directory.exists():
            paths.extend(directory.rglob("*.log"))
    return sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)


def experiments(limit: int = 250) -> list[dict]:
    rows = manifest_experiments()
    if len(rows) >= limit:
        return rows[:limit]
    for path in log_files():
        text = read_text(path)
        records = parse_experiment_records(text)
        if not records:
            continue
        stat = path.stat()
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative = path.name
        for index, (label, metrics) in enumerate(records, start=1):
            display_name = path.stem.replace(".out", "")
            if label:
                display_name = f"{display_name} [{label}]"
            rows.append({
                "id": f"{relative}#{index}",
                "name": display_name,
                "dataset": dataset_from_name(path.name),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "metrics": metrics,
                "parameters": {
                    **entry_default_parameters(dataset_from_name(path.name)),
                    **command_overrides(text, dataset_from_name(path.name), label),
                } if dataset_from_name(path.name) in ENTRY_SCRIPTS else {},
                "status": "completed",
                "source": "legacy-log",
            })
            if len(rows) >= limit:
                return rows
    return rows


def overview(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["dataset"], []).append(row)
    summary = []
    ordered_datasets = [*ENTRY_SCRIPTS, *(key for key in groups if key not in ENTRY_SCRIPTS)]
    for dataset in ordered_datasets:
        items = groups.get(dataset, [])
        best = max(items, key=lambda row: row["metrics"].get("macro_f1", -1)) if items else None
        summary.append({
            "dataset": DATASET_DEFAULTS.get(dataset, {}).get("label", dataset.upper()),
            "runs": len(items),
            "best": best,
            "parameters": best["parameters"] if best else entry_default_parameters(dataset) if dataset in ENTRY_SCRIPTS else {},
        })
    return summary


def sanitize_value(value: object) -> str:
    text = str(value).strip()
    if not text or len(text) > 100 or any(char in text for char in "\r\n\x00"):
        raise ValueError("Invalid parameter value")
    return text


def training_command(payload: dict) -> tuple[list[str], str]:
    entry = str(payload.get("entry", "")).lower()
    if entry not in ENTRY_SCRIPTS:
        raise ValueError("Unknown dataset entry")
    command = [sys.executable, str(ROOT / ENTRY_SCRIPTS[entry])]
    for key, flag in TRAINING_FIELDS.items():
        value = payload.get(key)
        if value not in (None, ""):
            command.extend((flag, sanitize_value(value)))
    extra_args = str(payload.get("extra_args", "")).strip()
    if extra_args:
        tokens = shlex.split(extra_args, posix=False)
        if any(len(token) > 100 or "\x00" in token for token in tokens):
            raise ValueError("Invalid extra CLI parameter")
        command.extend(tokens)
    run_name = f"{entry}_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command.extend(("--tensorboard-run-name", run_name))
    return command, run_name


def start_process(command: list[str], name: str, kind: str) -> dict:
    DASHBOARD_RUNS.mkdir(parents=True, exist_ok=True)
    run_id = f"{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    log_path = DASHBOARD_RUNS / f"{run_id}.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with RUNS_LOCK:
        RUNS[run_id] = {"id": run_id, "name": name, "kind": kind, "command": command, "process": process, "log": log_path}
    return {"id": run_id, "name": name, "command": command}


def running_jobs() -> list[dict]:
    with RUNS_LOCK:
        items = list(RUNS.values())
    jobs = []
    for item in items:
        code = item["process"].poll()
        jobs.append({
            "id": item["id"], "name": item["name"], "kind": item["kind"],
            "status": "running" if code is None else ("finished" if code == 0 else "failed"),
            "returncode": code,
            "command": item["command"],
            "tail": "\n".join(read_text(item["log"]).splitlines()[-18:]),
        })
    return jobs


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format, *args):
        return

    def guess_type(self, path):
        content_type = super().guess_type(path)
        if content_type in {"text/html", "text/css", "application/javascript"}:
            return f"{content_type}; charset=utf-8"
        return content_type

    def _json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/api/state":
            rows = experiments()
            self._json({"entries": DATASET_DEFAULTS, "parameterGroups": PARAMETER_GROUPS, "experiments": rows, "overview": overview(rows), "jobs": running_jobs()})
            return
        if request.path == "/api/log":
            run_id = parse_qs(request.query).get("id", [""])[0]
            with RUNS_LOCK:
                item = RUNS.get(run_id)
            if not item:
                self._json({"error": "Unknown run"}, HTTPStatus.NOT_FOUND)
            else:
                self._json({"text": read_text(item["log"])[-16000:]})
            return
        super().do_GET()

    def do_POST(self):
        try:
            payload = self._payload()
            if self.path == "/api/run":
                command, name = training_command(payload)
                self._json(start_process(command, name, "train"), HTTPStatus.CREATED)
                return
            if self.path == "/api/sweep":
                entry = str(payload.get("entry", "")).lower()
                if entry not in ENTRY_SCRIPTS:
                    raise ValueError("Unknown dataset entry")
                param = sanitize_value(payload.get("param", ""))
                values = [sanitize_value(value) for value in str(payload.get("values", "")).split(",") if value.strip()]
                if not values:
                    raise ValueError("Provide at least one sweep value")
                command = [sys.executable, str(ROOT / "sweep_hparam.py"), "--entry", entry, "--param", param, "--values", *values, "--continue-on-error"]
                self._json(start_process(command, f"{entry} sweep: {param}", "sweep"), HTTPStatus.CREATED)
                return
            self._json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    parser = argparse.ArgumentParser(description="Serve the local training workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not WEB_ROOT.exists():
        raise FileNotFoundError(f"Missing frontend directory: {WEB_ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Training workbench: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
