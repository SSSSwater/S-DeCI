import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ENTRY_SCRIPTS = {
    "mdd": "test_mdd_best_config.py",
    "abide": "test_abide_best_config.py",
    "matai": "test_matai_best_config.py",
    "taowu": "test_taowu_best_config.py",
}

METRIC_NAMES = ("accuracy", "precision", "recall", "macro_f1", "roc_auc", "train_seconds")
DISPLAY_NAMES = {
    "accuracy": "Acc",
    "precision": "Precision",
    "recall": "Recall",
    "macro_f1": "Macro-F1",
    "roc_auc": "AUC",
    "train_seconds": "Time(s)",
}

FINAL_AVG_RE = re.compile(
    r"Final-epoch Test Avg \(raw\)\s+"
    r"accuracy:\s*(?P<accuracy>[-+0-9.eE]+),\s*"
    r"precision:\s*(?P<precision>[-+0-9.eE]+),\s*"
    r"recall:\s*(?P<recall>[-+0-9.eE]+),\s*"
    r"macro_f1:\s*(?P<macro_f1>[-+0-9.eE]+),\s*"
    r"roc_auc:\s*(?P<roc_auc>[-+0-9.eE]+),\s*"
    r"train_seconds:\s*(?P<train_seconds>[-+0-9.eE]+)",
    re.IGNORECASE,
)
FORMAT_METRICS_RE = re.compile(
    r"(?:^|\b)"
    r"accuracy:\s*(?P<accuracy>[-+0-9.eE]+),\s*"
    r"precision:\s*(?P<precision>[-+0-9.eE]+),\s*"
    r"recall:\s*(?P<recall>[-+0-9.eE]+),\s*"
    r"macro_f1:\s*(?P<macro_f1>[-+0-9.eE]+),\s*"
    r"(?:roc_auc|auc):\s*(?P<roc_auc>[-+0-9.eE]+)"
    r"(?:,\s*train_seconds:\s*(?P<train_seconds>[-+0-9.eE]+)s?)?",
    re.IGNORECASE,
)


def _safe_name(text):
    keep = []
    for char in str(text):
        if char.isalnum() or char in ("-", "_", ".", "="):
            keep.append(char)
        else:
            keep.append("_")
    name = "".join(keep).strip("_")
    return name or "value"


def _script_arg_name(param_name):
    name = str(param_name).strip()
    if not name:
        raise ValueError("超参数名不能为空")
    if name.startswith("--"):
        return name
    name = name.lstrip("-")
    return "--" + name.replace("_", "-")


def _parse_values(raw_values):
    values = []
    for item in raw_values:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not values:
        raise ValueError("至少需要给出一个超参数取值")
    return values


def _run_with_live_output(command, cwd):
    completed_output = []
    stdout_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        safe_line = line.encode(stdout_encoding, errors="replace").decode(
            stdout_encoding,
            errors="replace",
        )
        print(safe_line, end="", flush=True)
        completed_output.append(line)
    return_code = process.wait()
    return return_code, "".join(completed_output)


def _parse_metrics_from_output(output):
    matches = list(FINAL_AVG_RE.finditer(output))
    if not matches:
        matches = list(FORMAT_METRICS_RE.finditer(output))
    if not matches:
        return None
    match = matches[-1]
    metrics = {}
    for name in METRIC_NAMES:
        value = match.groupdict().get(name)
        if value is None:
            continue
        metrics[name] = float(value)
    return metrics


def _format_metric_value(name, value):
    if value is None:
        return "-"
    if name == "train_seconds":
        return f"{float(value):.2f}"
    return f"{float(value) * 100.0:.2f}%"


def _print_summary_table(results, param_arg):
    if not results:
        return
    columns = [
        ("value", param_arg),
        ("status", "Status"),
        *[(name, DISPLAY_NAMES[name]) for name in METRIC_NAMES],
    ]
    rows = []
    for item in results:
        metrics = item.get("metrics") or {}
        row = {
            "value": str(item["value"]),
            "status": item["status"],
        }
        for name in METRIC_NAMES:
            row[name] = _format_metric_value(name, metrics.get(name))
        rows.append(row)

    widths = {}
    for key, header in columns:
        widths[key] = max(len(header), *(len(row[key]) for row in rows))

    print("", flush=True)
    print("=" * 104, flush=True)
    print("超参数对比总结（最终稳定指标，百分比保留两位）", flush=True)
    print("=" * 104, flush=True)
    header = " | ".join(header.ljust(widths[key]) for key, header in columns)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        print(" | ".join(row[key].ljust(widths[key]) for key, _ in columns), flush=True)

    successful = [item for item in results if item.get("metrics")]
    if successful:
        print("-" * len(header), flush=True)
        for metric_name in ("accuracy", "macro_f1", "roc_auc"):
            best = max(successful, key=lambda item: item["metrics"].get(metric_name, float("-inf")))
            print(
                f"Best {DISPLAY_NAMES[metric_name]}: {param_arg}={best['value']} "
                f"-> {_format_metric_value(metric_name, best['metrics'][metric_name])}",
                flush=True,
            )
    missing = [item for item in results if not item.get("metrics")]
    if missing:
        missing_values = ", ".join(str(item["value"]) for item in missing)
        print(f"未解析到指标的取值: {missing_values}", flush=True)
    print("=" * 104, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="按给定超参数取值自动运行多组完整训练，并写入 TensorBoard。"
    )
    parser.add_argument(
        "--entry",
        default="mdd",
        choices=sorted(ENTRY_SCRIPTS),
        help="使用哪个 best config 入口脚本，默认 mdd。",
    )
    parser.add_argument(
        "--script",
        default=None,
        help="自定义训练入口脚本；设置后会覆盖 --entry。",
    )
    parser.add_argument(
        "--param",
        required=True,
        help="要扫描的超参数名，可写 learning-rate 或 learning_rate。",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        required=True,
        help="超参数候选值，可用空格或逗号分隔，例如 0.001 0.003 0.01。",
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="TensorBoard run 分组名；默认自动生成。",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default="outputs/tensorboard",
        help="TensorBoard 日志根目录。",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="用于启动训练脚本的 Python 解释器，默认使用当前解释器。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不真正训练。",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个取值训练失败时继续跑后续取值。",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="传给训练入口的额外参数；如需使用，请放在 -- 后面。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    script = args.script or ENTRY_SCRIPTS[args.entry]
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = root / script_path
    if not script_path.exists():
        raise FileNotFoundError(f"训练入口脚本不存在: {script_path}")

    param_arg = _script_arg_name(args.param)
    param_key = param_arg.lstrip("-").replace("-", "_")
    values = _parse_values(args.values)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = args.sweep_name or f"sweep_{args.entry}_{param_key}_{timestamp}"

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    print("Hyperparameter sweep", flush=True)
    print(f"  script: {script_path}", flush=True)
    print(f"  param : {param_arg}", flush=True)
    print(f"  values: {', '.join(values)}", flush=True)
    print(f"  tensorboard group: {sweep_name}", flush=True)
    print("", flush=True)

    failures = []
    results = []
    for index, value in enumerate(values, start=1):
        value_name = _safe_name(value)
        run_name = f"{sweep_name}/{param_key}_{value_name}"
        command = [
            args.python,
            str(script_path),
            param_arg,
            str(value),
            "--use-tensorboard",
            "1",
            "--tensorboard-dir",
            args.tensorboard_dir,
            "--tensorboard-run-name",
            run_name,
            "--tensorboard-disable-smoke-runs",
            "1",
            *extra_args,
        ]
        print("=" * 80, flush=True)
        print(f"[{index}/{len(values)}] {param_arg}={value}", flush=True)
        print(" ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
        print("=" * 80, flush=True)
        if args.dry_run:
            results.append(
                {
                    "value": value,
                    "status": "dry-run",
                    "returncode": 0,
                    "metrics": None,
                    "run_name": run_name,
                }
            )
            continue
        return_code, output = _run_with_live_output(command, cwd=root)
        metrics = _parse_metrics_from_output(output)
        status = "ok" if return_code == 0 else "failed"
        results.append(
            {
                "value": value,
                "status": status,
                "returncode": return_code,
                "metrics": metrics,
                "run_name": run_name,
            }
        )
        if metrics is not None:
            compact = " | ".join(
                f"{DISPLAY_NAMES[name]}={_format_metric_value(name, metrics.get(name))}"
                for name in METRIC_NAMES
                if name in metrics
            )
            print(f"[当前结果] {param_arg}={value} -> {compact}", flush=True)
        else:
            print(f"[提示] {param_arg}={value} 未能从输出中解析最终指标。", flush=True)
        if return_code != 0:
            failures.append((value, return_code))
            print(f"[失败] {param_arg}={value}, returncode={return_code}", flush=True)
            if not args.continue_on_error:
                break

    print("", flush=True)
    _print_summary_table(results, param_arg)
    if failures:
        print("Sweep finished with failures:", flush=True)
        for value, code in failures:
            print(f"  {param_arg}={value}: returncode={code}", flush=True)
        raise SystemExit(1)

    print("Sweep finished successfully.", flush=True)
    print(f"TensorBoard: .\\.venv\\Scripts\\tensorboard.exe --logdir {args.tensorboard_dir}", flush=True)
    print(f"Run group : {sweep_name}", flush=True)
    print("Compare view: Scalars/Custom Scalars -> CompareTrend and CompareFinal", flush=True)


if __name__ == "__main__":
    main()
