from dataclasses import dataclass

import torch
import torch.nn as nn

from module_2_test.analytic_dag_constraint import AnalyticDAGConstraint


@dataclass
class CausalGraphOutput:
    c_hat: torch.Tensor
    adjacency: torch.Tensor
    dag_penalty: torch.Tensor
    dag_metadata: dict


def dag_penalty(adjacency):
    """NOTEARS-style differentiable acyclicity penalty."""
    n_nodes = adjacency.shape[0]
    expm = torch.matrix_exp(adjacency * adjacency)
    return torch.trace(expm) - n_nodes


def threshold_adjacency(adjacency, threshold=0.5):
    return (adjacency >= threshold).to(adjacency.dtype)


def adjacency_difference(left, right):
    return left.to(right.dtype) - right


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


def random_edge_f1_baseline(target, seed=2024):
    _ = seed
    off_diag = 1.0 - torch.eye(target.shape[0], device=target.device, dtype=target.dtype)
    edge_rate = target.sum() / off_diag.sum().clamp_min(1.0)
    return edge_rate.item()


class CausalGraphLearner(nn.Module):
    """Learn a shared causal adjacency matrix for Cycle-like node features.

    Edge direction convention: adjacency[parent, child].
    """

    def __init__(
        self,
        n_nodes,
        feature_dim=64,
        init_logit=-4.0,
        dag_method="notears",
        analytic_margin=0.1,
        analytic_power_iters=5,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.feature_dim = feature_dim
        self.dag_method = dag_method
        logits = torch.full((n_nodes, n_nodes), float(init_logit))
        logits = logits + 0.01 * torch.randn(n_nodes, n_nodes)
        self.adjacency_logits = nn.Parameter(logits)
        off_diag = 1.0 - torch.eye(n_nodes)
        self.register_buffer("off_diag_mask", off_diag)
        if dag_method == "notears":
            self.dag_constraint = None
        elif dag_method == "analytic":
            self.dag_constraint = AnalyticDAGConstraint(
                num_nodes=n_nodes,
                margin=analytic_margin,
                power_iters=analytic_power_iters,
            )
        else:
            raise ValueError(f"Unsupported dag_method={dag_method!r}. Use 'notears' or 'analytic'.")

    def effective_adjacency(self):
        return torch.sigmoid(self.adjacency_logits) * self.off_diag_mask

    def reconstruct(self, c, adjacency=None):
        if c.ndim != 3:
            raise ValueError(f"Expected C with shape [B, N, F], got {tuple(c.shape)}.")
        if c.shape[1] != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {c.shape[1]}.")
        if c.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {c.shape[2]}.")

        adjacency = self.effective_adjacency() if adjacency is None else adjacency
        return torch.einsum("ij,bif->bjf", adjacency, c)

    def forward(self, c):
        adjacency = self.effective_adjacency()
        c_hat = self.reconstruct(c, adjacency)
        if self.dag_method == "notears":
            penalty = dag_penalty(adjacency)
            metadata = {"dag_method": "notears"}
        else:
            penalty = self.dag_constraint(adjacency)
            metadata = {"dag_method": "analytic"}
            metadata.update(self.dag_constraint.metadata())
        return CausalGraphOutput(
            c_hat=c_hat,
            adjacency=adjacency,
            dag_penalty=penalty,
            dag_metadata=metadata,
        )
