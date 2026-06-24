import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from layers.causal_graph_learner import (
    CausalGraphLearner,
    adjacency_directionality,
    normalized_dag_loss,
    threshold_adjacency,
)
from utils.tensor_visualization import visualize_tensors


@dataclass
class SyntheticCausalData:
    c: torch.Tensor
    adjacency: torch.Tensor
    structure: torch.Tensor
    topological_order: torch.Tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="只训练当前模块2因果图学习器，用模拟 DAG 数据观察因果图学习效果。"
    )

    data = parser.add_argument_group("模拟数据")
    data.add_argument("--batch-size", type=int, default=64, help="模拟样本数")
    data.add_argument("--n-nodes", type=int, default=32, help="节点/ROI 数量")
    data.add_argument("--feature-dim", type=int, default=16, help="每个节点的 Cycle 特征维度")
    data.add_argument("--noise-std", type=float, default=0.05, help="结构方程生成数据时的噪声强度")
    data.add_argument("--edge-probability", type=float, default=0.03, help="基础边之外额外随机边的概率")
    data.add_argument("--min-weight", type=float, default=0.55, help="真实边权重随机下界")
    data.add_argument("--max-weight", type=float, default=0.95, help="真实边权重随机上界")
    data.add_argument("--no-shuffle-nodes", action="store_true", help="不打乱观测节点编号，真实图会更接近上三角")

    model = parser.add_argument_group("当前模块2")
    model.add_argument(
        "--graph-methods",
        choices=["nts_notears", "dagma_logdet", "dag_sampling", "both"],
        default="both",
        help="训练哪种因果图结构；both 会依次训练 nts_notears、dagma_logdet 与 dag_sampling 方便比较",
    )
    model.add_argument(
        "--dag-method",
        choices=["notears", "analytic"],
        default="notears",
        help="DAG acyclicity 约束形式",
    )
    model.add_argument("--init-logit", type=float, default=-2.0, help="因果边初始化 logit")
    model.add_argument("--graph-hidden-dim", type=int, default=8, help="NTS-NOTEARS 第一层 hidden 维度")
    model.add_argument("--dag-sampling-temperature", type=float, default=1.0, help="dag_sampling 的 Sinkhorn 温度")
    model.add_argument("--dag-sampling-noise", type=float, default=0.0, help="dag_sampling 训练期 Gumbel 噪声")
    model.add_argument("--dag-sampling-sinkhorn-iters", type=int, default=20, help="dag_sampling Sinkhorn 迭代次数")
    model.add_argument("--dag-sampling-hard", type=int, default=1, help="是否使用 hard straight-through permutation")
    model.add_argument("--analytic-margin", type=float, default=0.1, help="analytic DAG 的安全 margin")
    model.add_argument("--analytic-power-iters", type=int, default=5, help="analytic DAG 的 power iteration 步数")

    loss = parser.add_argument_group("损失权重")
    loss.add_argument("--lambda-recon", type=float, default=1.0, help="重构损失权重")
    loss.add_argument("--lambda-dag", type=float, default=0.001, help="DAG 约束损失权重")
    loss.add_argument("--lambda-l1", type=float, default=0.0001, help="稀疏 L1 损失权重")

    train = parser.add_argument_group("训练")
    train.add_argument("--epochs", type=int, default=500, help="训练 epoch 数")
    train.add_argument("--learning-rate", type=float, default=0.01, help="Adam 学习率")
    train.add_argument("--seed", type=int, default=2024, help="随机种子")
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="训练设备")
    train.add_argument("--print-every", type=int, default=50, help="每隔多少 epoch 打印一次")

    eval_group = parser.add_argument_group("评估与输出")
    eval_group.add_argument("--threshold", type=float, default=0.05, help="生成 A_learned_binary 的阈值")
    eval_group.add_argument("--output-dir", default="outputs/module2_current_synthetic", help="输出目录")
    eval_group.add_argument("--print-matrices", action="store_true", help="打印完整矩阵，节点多时不建议开启")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境没有可用 CUDA。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def selected_graph_methods(args):
    if args.graph_methods == "both":
        return ["nts_notears", "dagma_logdet", "dag_sampling"]
    return [args.graph_methods]


def default_structure(n_nodes=8, edge_probability=0.08, seed=2024, shuffle_nodes=True):
    if n_nodes < 8:
        raise ValueError("--n-nodes 至少需要为 8。")

    generator = torch.Generator()
    generator.manual_seed(seed)
    latent_structure = torch.zeros(n_nodes, n_nodes)

    required_edges = [
        (0, 1),
        (1, 2),
        (0, 3),
        (3, 4),
        (1, 5),
        (3, 5),
        (2, 6),
        (5, 7),
    ]
    for parent, child in required_edges:
        latent_structure[parent, child] = 1.0

    for parent in range(n_nodes):
        for child in range(parent + 1, n_nodes):
            if latent_structure[parent, child] > 0:
                continue
            if torch.rand((), generator=generator).item() < edge_probability:
                latent_structure[parent, child] = 1.0

    if not shuffle_nodes:
        return latent_structure, torch.arange(n_nodes)

    latent_to_observed = torch.randperm(n_nodes, generator=generator)
    observed_structure = torch.zeros_like(latent_structure)
    for latent_parent in range(n_nodes):
        children = torch.where(latent_structure[latent_parent] > 0)[0]
        for latent_child in children:
            observed_parent = latent_to_observed[latent_parent].item()
            observed_child = latent_to_observed[latent_child].item()
            observed_structure[observed_parent, observed_child] = 1.0
    return observed_structure, latent_to_observed


def is_dag(adjacency):
    reachability = adjacency.clone().bool()
    for _ in range(adjacency.shape[0]):
        if torch.diagonal(reachability).any():
            return False
        reachability = torch.matmul(reachability.float(), adjacency.float()).bool()
    return not torch.diagonal(reachability).any().item()


def random_weighted_adjacency(structure, seed=2024, min_weight=0.55, max_weight=0.95):
    generator = torch.Generator()
    generator.manual_seed(seed + 17)
    weights = min_weight + (max_weight - min_weight) * torch.rand(
        structure.shape,
        generator=generator,
        dtype=structure.dtype,
    )
    return structure * weights


def generate_cycle_like_data(args):
    seed_everything(args.seed)
    structure, topological_order = default_structure(
        n_nodes=args.n_nodes,
        edge_probability=args.edge_probability,
        seed=args.seed,
        shuffle_nodes=not args.no_shuffle_nodes,
    )
    if not is_dag(structure):
        raise ValueError("生成的真实结构不是 DAG，请检查模拟数据逻辑。")

    adjacency = random_weighted_adjacency(
        structure=structure,
        seed=args.seed,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    c = torch.zeros(args.batch_size, args.n_nodes, args.feature_dim)
    base = torch.randn(args.batch_size, args.n_nodes, args.feature_dim)
    for node in topological_order.tolist():
        parents = torch.where(structure[:, node] > 0)[0]
        if len(parents) == 0:
            c[:, node, :] = base[:, node, :]
            continue
        parent_sum = torch.zeros(args.batch_size, args.feature_dim)
        for parent in parents:
            parent_sum = parent_sum + adjacency[parent, node] * c[:, parent, :]
        c[:, node, :] = parent_sum + args.noise_std * base[:, node, :]

    return SyntheticCausalData(
        c=c,
        adjacency=adjacency,
        structure=structure,
        topological_order=topological_order,
    )


def edge_metrics(pred_binary, target):
    pred = pred_binary.bool()
    truth = target.bool()
    true_positive = torch.logical_and(pred, truth).sum().item()
    false_positive = torch.logical_and(pred, torch.logical_not(truth)).sum().item()
    false_negative = torch.logical_and(torch.logical_not(pred), truth).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-12)
    recall = true_positive / (true_positive + false_negative + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    shd = false_positive + false_negative
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "shd": shd,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def topk_adjacency(adjacency, k):
    binary = torch.zeros_like(adjacency)
    off_diag_mask = ~torch.eye(adjacency.shape[0], device=adjacency.device, dtype=torch.bool)
    values = adjacency.masked_fill(~off_diag_mask, float("-inf")).flatten()
    k = min(int(k), int(off_diag_mask.sum().item()))
    if k <= 0:
        return binary
    indices = torch.topk(values, k=k).indices
    binary.flatten()[indices] = 1.0
    return binary


def weight_error_metrics(a_learned, a_true, a_structure_true):
    weight_diff = a_learned - a_true
    true_edge_mask = a_structure_true.bool()
    off_diag_mask = ~torch.eye(a_structure_true.shape[0], device=a_structure_true.device, dtype=torch.bool)
    non_edge_mask = (~true_edge_mask) & off_diag_mask
    return {
        "weight_mae_overall": weight_diff.abs().mean().item(),
        "weight_mae_true_edges": weight_diff[true_edge_mask].abs().mean().item()
        if true_edge_mask.any()
        else 0.0,
        "false_edge_mean_abs": a_learned[non_edge_mask].abs().mean().item(),
    }


def save_outputs(output_dir, graph_method, data, c_hat, a_learned, a_binary, a_topk):
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_diff = a_learned - data.adjacency
    structure_diff = a_binary - data.structure
    direction_delta = a_learned - a_learned.T

    np.save(output_dir / "C.npy", data.c.detach().cpu().numpy())
    np.save(output_dir / "C_hat.npy", c_hat.detach().cpu().numpy())
    np.save(output_dir / "A_true.npy", data.adjacency.detach().cpu().numpy())
    np.save(output_dir / "A_structure_true.npy", data.structure.detach().cpu().numpy())
    np.save(output_dir / "A_learned.npy", a_learned.detach().cpu().numpy())
    np.save(output_dir / "A_learned_binary.npy", a_binary.detach().cpu().numpy())
    np.save(output_dir / "A_learned_topk.npy", a_topk.detach().cpu().numpy())
    np.save(output_dir / "A_diff.npy", weight_diff.detach().cpu().numpy())
    np.save(output_dir / "A_structure_diff.npy", structure_diff.detach().cpu().numpy())
    np.save(output_dir / "A_direction_delta.npy", direction_delta.detach().cpu().numpy())

    heatmap_path = output_dir / "causal_matrix_comparison.png"
    visualize_tensors(
        data.c,
        c_hat,
        data.adjacency,
        data.structure,
        a_learned,
        a_binary,
        a_topk,
        weight_diff,
        direction_delta,
        titles=[
            "C simulated",
            "C_hat reconstruction",
            "A_true weighted",
            "A_true structure",
            f"A_learned ({graph_method})",
            f"A_binary threshold ({graph_method})",
            f"A_binary topk ({graph_method})",
            f"A_learned - A_true ({graph_method})",
            f"A_learned - A_learned.T ({graph_method})",
        ],
        cmap="viridis",
        save_path=heatmap_path,
        show=False,
    )
    return heatmap_path


def train_one_graph_method(args, data, graph_method, device):
    seed_everything(args.seed)
    run_name = f"{graph_method}_{args.dag_method}"
    output_dir = Path(args.output_dir) / run_name

    model = CausalGraphLearner(
        n_nodes=args.n_nodes,
        feature_dim=args.feature_dim,
        init_logit=args.init_logit,
        dag_method=args.dag_method,
        analytic_margin=args.analytic_margin,
        analytic_power_iters=args.analytic_power_iters,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_method=graph_method,
        dag_sampling_temperature=args.dag_sampling_temperature,
        dag_sampling_noise=args.dag_sampling_noise,
        dag_sampling_sinkhorn_iters=args.dag_sampling_sinkhorn_iters,
        dag_sampling_hard=bool(args.dag_sampling_hard),
    ).to(device)
    c = data.c.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"\n[{run_name}] 开始训练：device={device}, epochs={args.epochs}")
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model(c)
        recon = F.mse_loss(output.c_hat, c)
        dag_loss = normalized_dag_loss(output.dag_penalty, args.n_nodes)
        l1_loss = model.first_layer_l1_loss()
        loss = args.lambda_recon * recon + args.lambda_dag * dag_loss + args.lambda_l1 * l1_loss
        loss.backward()
        optimizer.step()

        if args.print_every > 0 and (epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs):
            direction = adjacency_directionality(output.adjacency)
            print(
                f"[{run_name}] epoch={epoch:04d} loss={loss.item():.6f} "
                f"recon={recon.item():.6f} dag={dag_loss.item():.6f} "
                f"l1={l1_loss.item():.6f} dir={direction['adjacency_directionality_ratio'].item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        output = model(c)
        a_learned = output.adjacency.detach().cpu()
        c_hat = output.c_hat.detach().cpu()
        a_binary = threshold_adjacency(a_learned, threshold=args.threshold)
        true_edge_count = int(data.structure.sum().item())
        a_topk = topk_adjacency(a_learned, true_edge_count)
        heatmap_path = save_outputs(output_dir, graph_method, data, c_hat, a_learned, a_binary, a_topk)

        recon = F.mse_loss(c_hat, data.c).item()
        dag_loss = normalized_dag_loss(output.dag_penalty.detach().cpu(), args.n_nodes).item()
        l1_loss = model.first_layer_l1_loss().detach().cpu().item()
        threshold_metrics = edge_metrics(a_binary, data.structure)
        topk_metrics = edge_metrics(a_topk, data.structure)
        direction = {k: v.detach().cpu().item() for k, v in adjacency_directionality(a_learned).items()}
        weight_metrics = weight_error_metrics(a_learned, data.adjacency, data.structure)

    summary = {
        "graph_method": graph_method,
        "dag_method": args.dag_method,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "reconstruction_loss": recon,
        "dag_loss_normalized": dag_loss,
        "l1_loss": l1_loss,
        "threshold": args.threshold,
        "threshold_metrics": threshold_metrics,
        "topk_metrics": topk_metrics,
        "true_edge_count": true_edge_count,
        "learned_edge_count_threshold": int(a_binary.sum().item()),
        "heatmap_path": str(heatmap_path),
    }
    summary.update(direction)
    summary.update(weight_metrics)
    summary.update(output.dag_metadata)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.print_matrices:
        print(f"[{run_name}] A_true weighted:\n{data.adjacency.numpy()}")
        print(f"[{run_name}] A_structure_true:\n{data.structure.numpy()}")
        print(f"[{run_name}] A_learned:\n{np.round(a_learned.numpy(), 4)}")
        print(f"[{run_name}] A_binary threshold:\n{a_binary.numpy()}")
        print(f"[{run_name}] A_binary topk:\n{a_topk.numpy()}")

    print(f"[{run_name}] 训练完成，summary 保存到: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = generate_cycle_like_data(args)
    print("模拟数据已生成")
    print(f"  C shape: {tuple(data.c.shape)}")
    print(f"  true edge count: {int(data.structure.sum().item())}")
    print(f"  topological order in observed node ids: {data.topological_order.numpy()}")

    summaries = {}
    for graph_method in selected_graph_methods(args):
        summaries[graph_method] = train_one_graph_method(args, data, graph_method, device)

    comparison_path = output_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n整体对比保存到: {comparison_path}")


if __name__ == "__main__":
    main()
