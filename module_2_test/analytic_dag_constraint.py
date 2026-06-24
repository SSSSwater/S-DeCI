import torch
import torch.nn as nn


class AnalyticDAGConstraint(nn.Module):
    """Analytic DAG constraint using a scaled matrix inverse.

    The constraint follows the design in docs/新DAG因果方法.md:
    h(A) = trace((I - W_scaled)^-1) - N, W = A * A.
    """

    def __init__(self, num_nodes, margin=0.1, power_iters=5, eps=1e-5):
        super().__init__()
        self.num_nodes = num_nodes
        self.margin = margin
        self.power_iters = power_iters
        self.eps = eps
        self.register_buffer("identity", torch.eye(num_nodes))
        self.last_spectral_radius = None
        self.last_scale = None

    def estimate_spectral_radius(self, matrix):
        x = torch.ones((self.num_nodes, 1), device=matrix.device, dtype=matrix.dtype)
        for _ in range(self.power_iters):
            x = torch.matmul(matrix, x)
            x = x / (torch.norm(x, p=2) + self.eps)

        numerator = torch.matmul(torch.matmul(x.T, matrix), x)
        denominator = torch.matmul(x.T, x).clamp_min(self.eps)
        return (numerator / denominator).squeeze()

    def forward(self, adjacency):
        identity = self.identity.to(device=adjacency.device, dtype=adjacency.dtype)
        off_diag = 1.0 - identity
        masked_adjacency = adjacency * off_diag
        nonnegative_matrix = masked_adjacency * masked_adjacency

        spectral_radius = self.estimate_spectral_radius(nonnegative_matrix)
        scale = ((1.0 + self.margin) * spectral_radius).clamp_min(self.eps)
        scaled_matrix = nonnegative_matrix / scale

        inverse = torch.linalg.solve(identity - scaled_matrix, identity)
        dag_loss = torch.trace(inverse) - self.num_nodes

        self.last_spectral_radius = spectral_radius.detach()
        self.last_scale = scale.detach()
        return dag_loss

    def metadata(self):
        metadata = {}
        if self.last_spectral_radius is not None:
            metadata["analytic_spectral_radius"] = self.last_spectral_radius.item()
        if self.last_scale is not None:
            metadata["analytic_scale"] = self.last_scale.item()
        metadata["analytic_margin"] = self.margin
        metadata["analytic_power_iters"] = self.power_iters
        return metadata
