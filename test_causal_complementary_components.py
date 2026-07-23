"""模块 3/4 因果显著性互补学习的张量级回归测试。"""

import unittest

import torch

from layers.hpec_energy_layer import HPECPrototypeEnergy
from layers.hyperbolic_gcn_layer import Module3HGCNReadout


class ReliablePrototypeUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.labels = torch.tensor([0, 1, 0, 1])
        self.logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])

    def _points(self, layer):
        tangent = torch.randn(4, 8) * 0.05
        return layer.manifold.expmap0(tangent, dim=-1, project=True)

    def test_reliable_tp_ema_updates_without_autograd(self):
        layer = HPECPrototypeEnergy(
            2,
            8,
            prototypes_per_class=2,
            prototype_update_mode="reliable_tp_ema",
            reliable_confidence_threshold=0.0,
            reliable_min_samples=1,
            trainable_prototypes=True,
        )
        points = self._points(layer)
        output = layer(points)
        before = layer._current_prototypes().detach().clone()
        stats = layer.update_prototypes_with_reliable_tp_ema(
            points,
            self.labels,
            self.logits,
            energy_per_proto=output.energy_per_proto,
        )
        after = layer._current_prototypes().detach()
        self.assertTrue(torch.isfinite(after).all())
        self.assertGreater(float((after - before).abs().sum()), 0.0)
        self.assertIn("hpec_reliable_tp_ratio", stats)
        self.assertFalse(layer.prototypes.requires_grad)

    def test_legacy_and_frozen_modes(self):
        for mode in ("sinkhorn_ema", "none"):
            layer = HPECPrototypeEnergy(
                2,
                8,
                prototypes_per_class=2,
                prototype_update_mode=mode,
                use_sinkhorn_ema=True,
                trainable_prototypes=True,
            )
            points = self._points(layer)
            before = layer._current_prototypes().detach().clone()
            if mode == "sinkhorn_ema":
                layer.update_prototypes_with_sinkhorn_ema(points, self.labels)
            after = layer._current_prototypes().detach()
            self.assertTrue(torch.isfinite(after).all())
            if mode == "none":
                self.assertTrue(torch.equal(before, after))

class CausalRoleReadoutTests(unittest.TestCase):
    def test_four_role_centers_stay_inside_poincare_ball(self):
        torch.manual_seed(11)
        batch_size, nodes, dim = 3, 10, 8
        features = torch.randn(batch_size, nodes, dim) * 0.05
        adjacency = torch.rand(batch_size, nodes, nodes)
        adjacency = adjacency * (1.0 - torch.eye(nodes).unsqueeze(0))
        layer = Module3HGCNReadout(
            input_dim=dim,
            hidden_dim=dim,
            use_causal_role_readout=True,
        )
        output = layer(features, adjacency)
        self.assertEqual(output.causal_role_poincare.shape, (batch_size, 4, dim))
        self.assertEqual(output.causal_role_weights.shape, (batch_size, 4, nodes))
        self.assertTrue(torch.isfinite(output.causal_role_poincare).all())
        self.assertTrue(torch.all(output.causal_role_poincare.norm(dim=-1) < 1.0))
        output.causal_role_tangent.square().mean().backward()
        self.assertIsNotNone(layer.layers[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
