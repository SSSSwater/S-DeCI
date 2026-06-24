import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from layers.temporal_sem_causal_learner import TemporalSEMCausalLearner
from utils.tensor_visualization import visualize_tensors


def parse_args():
    parser = argparse.ArgumentParser(description="快速测试 temporal SEM 因果学习器。")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--lag-order", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", default="outputs/temporal_sem_smoke")
    return parser.parse_args()


def make_synthetic_series(args):
    generator = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.batch_size, args.seq_len, args.n_nodes, generator=generator) * 0.1
    adjacency = torch.zeros(args.n_nodes, args.n_nodes)
    for parent, child in [(0, 1), (1, 2), (0, 3), (3, 4), (2, 5), (5, 6), (4, 7)]:
        if child < args.n_nodes:
            adjacency[parent, child] = 0.45 + 0.35 * torch.rand((), generator=generator)
    for t in range(1, args.seq_len):
        driven = torch.einsum("pc,bp->bc", adjacency, x[:, t - 1, :])
        x[:, t, :] = 0.55 * x[:, t, :] + driven
    return x, adjacency


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, a_true = make_synthetic_series(args)
    learner = TemporalSEMCausalLearner(
        n_nodes=args.n_nodes,
        lag_order=args.lag_order,
        lambda_pred=1.0,
        lambda_dag=0.001,
        lambda_sparse=0.0005,
        lambda_smooth=0.0001,
    )
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.learning_rate)

    first_loss = None
    last_loss = None
    for epoch in range(args.epochs):
        learner.set_epoch(epoch)
        optimizer.zero_grad()
        output = learner(x)
        loss, parts = learner.compute_losses(output)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = last_loss
        print(
            f"epoch={epoch + 1:03d} loss={last_loss:.6f} "
            f"pred={parts['temporal_pred_loss'].item():.6f} "
            f"dag={parts['causal_dag_loss'].item():.6f}"
        )

    with torch.no_grad():
        output = learner(x)
        pred_loss = F.mse_loss(output.x_hat, output.target).item()
        visualize_tensors(
            x,
            output.target,
            output.x_hat,
            a_true,
            output.a0,
            output.a_lag,
            output.a_shared,
            titles=[
                "Synthetic temporal series",
                "Temporal SEM target",
                "Temporal SEM X_hat",
                "A_true lag structure",
                "A0 learned",
                "A_lag learned",
                "A_shared A0+mean(A_lag)",
            ],
            save_path=output_dir / "temporal_sem_smoke.png",
            show=False,
        )
        torch.save(
            {
                "A_true": a_true,
                "A0": output.a0,
                "A_lag": output.a_lag,
                "A_shared": output.a_shared,
                "first_loss": first_loss,
                "last_loss": last_loss,
                "pred_loss": pred_loss,
            },
            output_dir / "temporal_sem_smoke.pt",
        )
    print(f"temporal_sem_smoke saved to: {output_dir}")


if __name__ == "__main__":
    main()
