import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from exp.exp_classification_CV import Exp_Main


METRIC_NAMES = ("accuracy", "precision", "recall", "macro_f1", "roc_auc")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an effect-oriented training check for the local Matai dataset."
    )
    parser.add_argument("--data", default="Mātai", help="Dataset key used by data_provider.")
    parser.add_argument("--data-path", default=None, help="Defaults to dataset/Mātai.")
    parser.add_argument("--data-type", default="TS", choices=("TS", "FC"))
    parser.add_argument("--model", default="S-DeCI")
    parser.add_argument("--protocol", default="AAL116")
    parser.add_argument("--channel", type=int, default=116)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--use-norm", type=int, default=1)
    parser.add_argument("--use-deci-module1", type=int, default=1, help="是否启用 S-DeCI 模块 1 DeCI/Cycle 分解。")
    parser.add_argument("--use-causal-module2", type=int, default=1)
    parser.add_argument("--causal-feature-source", default="sum", choices=("sum", "last"))
    parser.add_argument("--causal-graph-method", default="nts_notears", choices=("nts_notears", "dag_sampling"))
    parser.add_argument("--causal-init-logit", type=float, default=-2.0)
    parser.add_argument("--causal-learning-rate", type=float, default=1e-2)
    parser.add_argument("--causal-graph-hidden-dim", type=int, default=0)
    parser.add_argument("--dag-sampling-temperature", type=float, default=1.0)
    parser.add_argument("--dag-sampling-noise", type=float, default=0.0)
    parser.add_argument("--dag-sampling-sinkhorn-iters", type=int, default=20)
    parser.add_argument("--dag-sampling-hard", type=int, default=1)
    parser.add_argument("--causal-threshold", type=float, default=0.05)
    parser.add_argument("--detach-causal-input", type=int, default=1)
    parser.add_argument("--lambda-causal-recon", type=float, default=0.02)
    parser.add_argument("--lambda-causal-dag", type=float, default=0.001)
    parser.add_argument("--lambda-causal-l1", type=float, default=0.0001)
    parser.add_argument("--causal-analytic-margin", type=float, default=0.1)
    parser.add_argument("--causal-analytic-power-iters", type=int, default=5)
    parser.add_argument("--use-hyperbolic-modules34", type=int, default=1, help="是否联合启用模块 3 HGCN 与模块 4 HPEC。")
    parser.add_argument("--use-hgcn-module3", type=int, default=1)
    parser.add_argument("--hgcn-hidden-dim", type=int, default=128)
    parser.add_argument("--hgcn-layers", type=int, default=1)
    parser.add_argument("--hgcn-curvature", type=float, default=1.0)
    parser.add_argument("--hgcn-backclip-radius", type=float, default=1.0)
    parser.add_argument("--hgcn-dropout", type=float, default=0.0)
    parser.add_argument("--hgcn-add-self-loop", type=int, default=1)
    parser.add_argument("--hgcn-adjacency-normalization", default="row", choices=("row", "sym", "none"))
    parser.add_argument("--use-sample-correlation-when-module2-disabled", type=int, default=1)
    parser.add_argument("--sample-correlation-mode", default="abs", choices=("abs", "positive", "raw"))
    parser.add_argument("--use-hpec-module4", type=int, default=1)
    parser.add_argument("--gcn-fallback-hidden-dim", type=int, default=16)
    parser.add_argument("--gcn-fallback-layers", type=int, default=1)
    parser.add_argument("--gcn-fallback-dropout", type=float, default=0.0)
    parser.add_argument("--gcn-fallback-add-self-loop", type=int, default=1)
    parser.add_argument("--gcn-fallback-adjacency-normalization", default="row", choices=("row", "sym", "none"))
    parser.add_argument("--hpec-prototype-radius", type=float, default=0.3)
    parser.add_argument("--hpec-cone-k", type=float, default=0.1)
    parser.add_argument("--hpec-margin", type=float, default=1.0)
    parser.add_argument("--hpec-prototypes-per-class", type=int, default=1)
    parser.add_argument("--hpec-proto-temperature", type=float, default=0.2)
    parser.add_argument("--lambda-hpec-mle", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-pcl", type=float, default=0.0)
    parser.add_argument("--lambda-hpec-pal", type=float, default=0.0)
    parser.add_argument("--hpec-trainable-prototypes", type=int, default=0)
    parser.add_argument("--hpec-init-steps", type=int, default=500)
    parser.add_argument("--hpec-eps", type=float, default=1e-7)
    parser.add_argument("--visualize-causal", type=int, default=0)
    parser.add_argument("--causal-vis-dir", default=".tmp_training_tests/s_deci_causal_matai")
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=0,
        help="Deprecated; causal visualization is saved once after each fold training.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--checkpoints", default=".tmp_training_tests/matai")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--print-process", type=int, default=0)
    parser.add_argument("--print-metric-every", type=int, default=1)
    parser.add_argument("--print-data-info", type=int, default=0)
    parser.add_argument("--keep-weight", action="store_true", help="Keep generated checkpoint weights.")
    return parser.parse_args()


def normalize_s_deci_module_args(args):
    if args.model != "S-DeCI":
        return args
    args.use_deci_module1 = int(bool(args.use_deci_module1))
    args.use_causal_module2 = int(bool(args.use_causal_module2))
    args.use_hyperbolic_modules34 = int(bool(args.use_hyperbolic_modules34))
    args.use_hgcn_module3 = args.use_hyperbolic_modules34
    args.use_hpec_module4 = args.use_hyperbolic_modules34
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def repo_root():
    return Path(__file__).resolve().parent


def available_datasets(root):
    dataset_root = root / "dataset"
    if not dataset_root.exists():
        return []
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def resolve_matai_path(root, data_path):
    if data_path:
        path = Path(data_path)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path
        expected = path
    else:
        expected = root / "dataset" / "Mātai"
        if expected.exists():
            return expected
        fallback = root / "dataset" / "Matai"
        if fallback.exists():
            return fallback

    choices = ", ".join(available_datasets(root)) or "none"
    raise SystemExit(
        f"Matai dataset path not found: {expected}\n"
        f"Expected local Matai dataset under: {root / 'dataset' / 'Mātai'}\n"
        f"Available dataset directories: {choices}"
    )


def build_experiment_args(cli_args):
    root = repo_root()
    data_path = resolve_matai_path(root, cli_args.data_path)
    checkpoints = Path(cli_args.checkpoints)
    if not checkpoints.is_absolute():
        checkpoints = root / checkpoints
    checkpoints.mkdir(parents=True, exist_ok=True)

    use_gpu = bool(cli_args.use_gpu and torch.cuda.is_available())

    return SimpleNamespace(
        Method="DL",
        model=cli_args.model,
        data=cli_args.data,
        data_path=str(data_path),
        data_type=cli_args.data_type,
        protocol=cli_args.protocol,
        kfold=cli_args.kfold,
        checkpoints=str(checkpoints),
        del_weight=not cli_args.keep_weight,
        seq_len=cli_args.seq_len,
        channel=cli_args.channel,
        classes=cli_args.classes,
        d_model=cli_args.d_model,
        layer=cli_args.layer,
        n_head=8,
        dropout=cli_args.dropout,
        use_norm=cli_args.use_norm,
        use_deci_module1=cli_args.use_deci_module1,
        use_causal_module2=cli_args.use_causal_module2,
        causal_feature_source=cli_args.causal_feature_source,
        causal_graph_method=cli_args.causal_graph_method,
        causal_init_logit=cli_args.causal_init_logit,
        causal_learning_rate=cli_args.causal_learning_rate,
        causal_graph_hidden_dim=cli_args.causal_graph_hidden_dim,
        dag_sampling_temperature=cli_args.dag_sampling_temperature,
        dag_sampling_noise=cli_args.dag_sampling_noise,
        dag_sampling_sinkhorn_iters=cli_args.dag_sampling_sinkhorn_iters,
        dag_sampling_hard=cli_args.dag_sampling_hard,
        causal_threshold=cli_args.causal_threshold,
        detach_causal_input=cli_args.detach_causal_input,
        lambda_causal_recon=cli_args.lambda_causal_recon,
        lambda_causal_dag=cli_args.lambda_causal_dag,
        lambda_causal_l1=cli_args.lambda_causal_l1,
        causal_analytic_margin=cli_args.causal_analytic_margin,
        causal_analytic_power_iters=cli_args.causal_analytic_power_iters,
        use_hyperbolic_modules34=cli_args.use_hyperbolic_modules34,
        use_hgcn_module3=cli_args.use_hgcn_module3,
        hgcn_hidden_dim=cli_args.hgcn_hidden_dim,
        hgcn_layers=cli_args.hgcn_layers,
        hgcn_curvature=cli_args.hgcn_curvature,
        hgcn_backclip_radius=cli_args.hgcn_backclip_radius,
        hgcn_dropout=cli_args.hgcn_dropout,
        hgcn_add_self_loop=cli_args.hgcn_add_self_loop,
        hgcn_adjacency_normalization=cli_args.hgcn_adjacency_normalization,
        use_sample_correlation_when_module2_disabled=cli_args.use_sample_correlation_when_module2_disabled,
        sample_correlation_mode=cli_args.sample_correlation_mode,
        use_hpec_module4=cli_args.use_hpec_module4,
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
        lambda_hpec_mle=cli_args.lambda_hpec_mle,
        lambda_hpec_pcl=cli_args.lambda_hpec_pcl,
        lambda_hpec_pal=cli_args.lambda_hpec_pal,
        hpec_trainable_prototypes=cli_args.hpec_trainable_prototypes,
        hpec_init_steps=cli_args.hpec_init_steps,
        hpec_eps=cli_args.hpec_eps,
        visualize_causal=cli_args.visualize_causal,
        causal_vis_dir=str(root / cli_args.causal_vis_dir)
        if not Path(cli_args.causal_vis_dir).is_absolute()
        else cli_args.causal_vis_dir,
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
        train_epochs=cli_args.train_epochs,
        batch_size=cli_args.batch_size,
        patience=cli_args.patience,
        learning_rate=cli_args.learning_rate,
        loss=cli_args.loss,
        lradj="constant",
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
        f"lr_{args.learning_rate}_dp_{args.dropout}_dm_{args.d_model}_seq_{args.seq_len}"
    )


def format_metrics(metrics):
    return ", ".join(f"{name}: {float(value):.4f}" for name, value in zip(METRIC_NAMES, metrics))


def main():
    cli_args = normalize_s_deci_module_args(parse_args())
    seed_everything(cli_args.seed)
    args = build_experiment_args(cli_args)

    print("Matai small-sample training configuration:")
    print(
        f"  data={args.data}, data_path={args.data_path}, model={args.model}, "
        f"protocol={args.protocol}, channel={args.channel}, seq_len={args.seq_len}, "
        f"kfold={args.kfold}, epochs={args.train_epochs}, batch_size={args.batch_size}, "
        f"d_model={args.d_model}, device={'cuda' if args.use_gpu else 'cpu'}"
    )

    try:
        exp = Exp_Main(args)
        metrics = exp.kf_train(setting_name(args))
    except Exception as exc:
        raise SystemExit(f"Matai small-sample training failed during execution: {exc}") from exc

    print("Matai small-sample training completed.")
    print(format_metrics(metrics))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Matai small-sample training interrupted.")
