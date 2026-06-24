import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SyntheticCausalData:
    c: torch.Tensor
    adjacency: torch.Tensor
    structure: torch.Tensor
    parent_mask: torch.Tensor
    topological_order: torch.Tensor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_structure(n_nodes=8, edge_probability=0.08, seed=2024, shuffle_nodes=True):
    if n_nodes < 8:
        raise ValueError("default_structure requires n_nodes >= 8.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    latent_structure = torch.zeros(n_nodes, n_nodes)
    edges = [
        (0, 1),  # chain
        (1, 2),
        (0, 3),  # fork
        (3, 4),
        (1, 5),  # multi-parent target
        (3, 5),
        (2, 6),
        (5, 7),
    ]
    for parent, child in edges:
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
        for latent_child in torch.where(latent_structure[latent_parent] > 0)[0]:
            observed_parent = latent_to_observed[latent_parent].item()
            observed_child = latent_to_observed[latent_child].item()
            observed_structure[observed_parent, observed_child] = 1.0

    topological_order = latent_to_observed.clone()
    return observed_structure, topological_order


def random_weighted_adjacency(structure, seed=2024, min_weight=0.55, max_weight=0.95):
    generator = torch.Generator()
    generator.manual_seed(seed + 17)
    weights = min_weight + (max_weight - min_weight) * torch.rand(
        structure.shape,
        generator=generator,
        dtype=structure.dtype,
    )
    return structure * weights


def is_dag(adjacency):
    reachability = adjacency.clone().bool()
    for _ in range(adjacency.shape[0]):
        if torch.diagonal(reachability).any():
            return False
        reachability = torch.matmul(reachability.float(), adjacency.float()).bool()
    return not torch.diagonal(reachability).any().item()


def generate_cycle_like_data(
    batch_size=256,
    n_nodes=8,
    feature_dim=64,
    noise_std=0.05,
    edge_probability=0.08,
    min_weight=0.55,
    max_weight=0.95,
    shuffle_nodes=True,
    seed=2024,
):
    """Generate Cycle-like features from a random weighted structural equation DAG.

    Returns C plus weighted and binary ground-truth adjacency for evaluation.
    Ground truth is not used by the training loss.
    """
    seed_everything(seed)
    structure, topological_order = default_structure(
        n_nodes=n_nodes,
        edge_probability=edge_probability,
        seed=seed,
        shuffle_nodes=shuffle_nodes,
    )
    if not is_dag(structure):
        raise ValueError("Default structure must be a DAG.")

    adjacency = random_weighted_adjacency(
        structure=structure,
        seed=seed,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    c = torch.zeros(batch_size, n_nodes, feature_dim)
    base = torch.randn(batch_size, n_nodes, feature_dim)
    parent_mask = (structure.sum(dim=0) > 0).float().view(1, n_nodes, 1)

    for node in topological_order.tolist():
        parents = torch.where(structure[:, node] > 0)[0]
        if len(parents) == 0:
            c[:, node, :] = base[:, node, :]
            continue
        parent_sum = torch.zeros(batch_size, feature_dim)
        for parent in parents:
            parent_sum = parent_sum + adjacency[parent, node] * c[:, parent, :]
        c[:, node, :] = parent_sum + noise_std * base[:, node, :]

    return SyntheticCausalData(
        c=c,
        adjacency=adjacency,
        structure=structure,
        parent_mask=parent_mask,
        topological_order=topological_order,
    )
