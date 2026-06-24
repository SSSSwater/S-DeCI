import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module_2_test.causal_graph_learner import (  # noqa: E402
    CausalGraphLearner,
    adjacency_difference,
    edge_metrics,
    random_edge_f1_baseline,
    threshold_adjacency,
)
from module_2_test.synthetic_data import generate_cycle_like_data  # noqa: E402
from utils.tensor_visualization import visualize_tensors  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train standalone module 2 causal graph learner.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-nodes", type=int, default=116)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--edge-probability", type=float, default=0.08)
    parser.add_argument("--min-weight", type=float, default=0.55)
    parser.add_argument("--max-weight", type=float, default=0.95)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-dag", type=float, default=0.001)
    parser.add_argument("--lambda-analytic-dag", type=float, default=None)
    parser.add_argument("--lambda-l1", type=float, default=0.0001)
    parser.add_argument("--dag-methods", choices=["notears", "analytic", "both"], default="both")
    parser.add_argument("--analytic-margin", type=float, default=0.1)
    parser.add_argument("--analytic-power-iters", type=int, default=5)
    parser.add_argument("--no-shuffle-nodes", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", default="module_2_test/outputs")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--print-matrices", action="store_true")
    return parser.parse_args()


def selected_methods(args):
    if args.dag_methods == "both":
        return ["notears", "analytic"]
    return [args.dag_methods]


def method_dag_weight(args, method):
    if method == "analytic" and args.lambda_analytic_dag is not None:
        return args.lambda_analytic_dag
    return args.lambda_dag


def reconstruction_loss(c_hat, c):
    return F.mse_loss(c_hat, c)


def normalized_dag_loss(dag_penalty, n_nodes):
    return dag_penalty / max(float(n_nodes), 1.0)


def normalized_l1_loss(adjacency):
    off_diag_count = adjacency.numel() - adjacency.shape[0]
    return adjacency.abs().sum() / max(float(off_diag_count), 1.0)


def weight_error_metrics(a_learned, a_true, a_structure_true):
    weight_diff = a_learned - a_true
    true_edge_mask = a_structure_true.bool()
    off_diag_mask = ~torch.eye(a_structure_true.shape[0], device=a_structure_true.device, dtype=torch.bool)
    non_edge_mask = (~true_edge_mask) & off_diag_mask
    overall_mae = weight_diff.abs().mean().item()
    true_edge_mae = weight_diff[true_edge_mask].abs().mean().item() if true_edge_mask.any() else 0.0
    false_edge_mean_abs = a_learned[non_edge_mask].abs().mean().item()
    return {
        "weight_mae_overall": overall_mae,
        "weight_mae_true_edges": true_edge_mae,
        "false_edge_mean_abs": false_edge_mean_abs,
    }


def save_outputs(output_dir, method, data, a_learned, a_binary):
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_diff = adjacency_difference(a_learned, data.adjacency)
    structure_diff = adjacency_difference(a_binary, data.structure)

    np.save(output_dir / "A_true.npy", data.adjacency.detach().cpu().numpy())
    np.save(output_dir / "A_structure_true.npy", data.structure.detach().cpu().numpy())
    np.save(output_dir / "A_learned.npy", a_learned.detach().cpu().numpy())
    np.save(output_dir / "A_learned_binary.npy", a_binary.detach().cpu().numpy())
    np.save(output_dir / "A_diff.npy", weight_diff.detach().cpu().numpy())
    np.save(output_dir / "A_structure_diff.npy", structure_diff.detach().cpu().numpy())

    heatmap_path = output_dir / "causal_matrix_comparison.png"
    visualize_tensors(
        data.adjacency,
        data.structure,
        a_learned,
        a_binary,
        weight_diff,
        structure_diff,
        titles=[
            "A_true weighted",
            "A_true structure",
            f"A_learned ({method})",
            f"A_learned_binary ({method})",
            f"A_learned - A_true ({method})",
            f"A_binary - A_true_structure ({method})",
        ],
        cmap="viridis",
        save_path=heatmap_path,
        show=False,
    )
    return heatmap_path, weight_diff, structure_diff


def train_one_method(args, data, method):
    torch.manual_seed(args.seed)
    model = CausalGraphLearner(
        n_nodes=args.n_nodes,
        feature_dim=args.feature_dim,
        dag_method=method,
        analytic_margin=args.analytic_margin,
        analytic_power_iters=args.analytic_power_iters,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lambda_dag = method_dag_weight(args, method)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model(data.c)
        recon = reconstruction_loss(output.c_hat, data.c)
        dag_loss = normalized_dag_loss(output.dag_penalty, args.n_nodes)
        l1_loss = normalized_l1_loss(output.adjacency)
        loss = args.lambda_recon * recon + lambda_dag * dag_loss + args.lambda_l1 * l1_loss
        loss.backward()
        optimizer.step()

        if args.print_every > 0 and (epoch == 1 or epoch % args.print_every == 0):
            print(
                f"method={method} epoch={epoch:04d} loss={loss.item():.6f} "
                f"recon={recon.item():.6f} dag={dag_loss.item():.6f} "
                f"l1={l1_loss.item():.6f}"
            )

    with torch.no_grad():
        output = model(data.c)
        a_learned = output.adjacency
        a_binary = threshold_adjacency(a_learned, threshold=args.threshold)
        heatmap_path, weight_diff, structure_diff = save_outputs(
            Path(args.output_dir) / method,
            method,
            data,
            a_learned,
            a_binary,
        )
        metrics = edge_metrics(a_binary, data.structure)
        baseline_f1 = random_edge_f1_baseline(data.structure, seed=args.seed)
        weight_metrics = weight_error_metrics(a_learned, data.adjacency, data.structure)
        recon = reconstruction_loss(output.c_hat, data.c)
        dag_loss = normalized_dag_loss(output.dag_penalty, args.n_nodes)
        l1_loss = normalized_l1_loss(output.adjacency)

    summary = {
        "method": method,
        "reconstruction_loss": recon.item(),
        "dag_penalty": output.dag_penalty.item(),
        "dag_loss_normalized": dag_loss.item(),
        "l1_loss_normalized": l1_loss.item(),
        "lambda_recon": args.lambda_recon,
        "lambda_dag": lambda_dag,
        "lambda_l1": args.lambda_l1,
        "edge_precision": metrics["edge_precision"],
        "edge_recall": metrics["edge_recall"],
        "edge_f1": metrics["edge_f1"],
        "random_baseline_f1": baseline_f1,
        "shd": metrics["shd"],
        "threshold": args.threshold,
        "heatmap_path": str(heatmap_path),
        "beats_random_baseline": metrics["edge_f1"] > baseline_f1,
    }
    summary.update(weight_metrics)
    summary.update(output.dag_metadata)

    if args.print_matrices:
        print(f"[{method}] A_true weighted:")
        print(data.adjacency.cpu().numpy())
        print(f"[{method}] A_true structure:")
        print(data.structure.cpu().numpy())
        print(f"[{method}] Topological order in observed node ids:")
        print(data.topological_order.cpu().numpy())
        print(f"[{method}] A_learned:")
        print(np.round(a_learned.cpu().numpy(), 3))
        print(f"[{method}] A_learned_binary:")
        print(a_binary.cpu().numpy())
        print(f"[{method}] A_learned - A_true:")
        print(np.round(weight_diff.cpu().numpy(), 3))
        print(f"[{method}] A_learned_binary - A_true_structure:")
        print(structure_diff.cpu().numpy())
    else:
        print(f"[{method}] Topological order in observed node ids:")
        print(data.topological_order.cpu().numpy())
    print(f"[{method}] Training summary:")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data = generate_cycle_like_data(
        batch_size=args.batch_size,
        n_nodes=args.n_nodes,
        feature_dim=args.feature_dim,
        edge_probability=args.edge_probability,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        shuffle_nodes=not args.no_shuffle_nodes,
        seed=args.seed,
    )

    summaries = {}
    for method in selected_methods(args):
        summaries[method] = train_one_method(args, data, method)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("Comparison summary:")
    print(json.dumps(summaries, indent=2))
    print(f"Comparison summary saved to: {comparison_path}")

    dag_success = any(summary["beats_random_baseline"] for summary in summaries.values())
    if not dag_success:
        raise SystemExit("No DAG method exceeded the random baseline F1.")


if __name__ == "__main__":
    main()
