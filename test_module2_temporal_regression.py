import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from exp.exp_classification_CV import Exp_Main
from layers.gcn_fallback_layer import ModuleGCNFallback
from layers.temporal_sem_causal_learner import TemporalSEMCausalLearner
from models.S_DeCI import Model as SDeCIModel


class TemporalModule2RegressionTest(unittest.TestCase):
    def test_innovation_prediction_is_dynamic_and_backward_works(self):
        torch.manual_seed(7)
        batch, steps, nodes = 4, 32, 6
        signal = torch.zeros(batch, steps, nodes)
        signal[:, 0] = torch.randn(batch, nodes)
        for time_idx in range(1, steps):
            signal[:, time_idx] = 0.75 * signal[:, time_idx - 1]
            signal[:, time_idx, 1] += 0.25 * signal[:, time_idx - 1, 0]
            signal[:, time_idx] += 0.15 * torch.randn(batch, nodes)

        learner = TemporalSEMCausalLearner(
            n_nodes=nodes,
            lag_order=3,
            candidate_parent_topk=2,
            decoder_activation="identity",
            prediction_target_mode="innovation",
            a0_scale=0.03,
            use_sample_graph_residual=True,
            sample_graph_delta_scale=0.15,
        )
        output = learner(signal)
        loss, parts = learner.compute_losses(output)

        self.assertEqual(output.x_hat.shape, (batch, steps - 3, nodes))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts["temporal_pred_std_ratio"]), 0.5)
        self.assertGreater(float(parts["temporal_pred_base_loss"].detach()), 0.0)
        self.assertLess(float(parts["causal_meta_a0_to_alag_mass_ratio"]), 5.0)

        loss.backward()
        self.assertIsNotNone(learner.lag_pos_raw.grad)
        self.assertGreater(float(learner.lag_pos_raw.grad.abs().sum()), 0.0)

    def test_soft_masked_fc_preserves_direction_and_gradient(self):
        model = SDeCIModel.__new__(SDeCIModel)
        nn.Module.__init__(model)
        model.classification_graph_source = "causal_soft_masked_fc"
        model.module2_sample_correlation_blend = 0.75
        model.module2_graph_residual_alpha = 0.10
        model.detach_module2_graph_for_classification = False
        model.sample_correlation_mode = "abs"
        model.latest_aux_losses = {}
        model.latest_sample_correlation_adjacency = None
        model.latest_classification_adjacency = None

        learned = torch.zeros(4, 4, requires_grad=True)
        with torch.no_grad():
            learned[0, 1] = 0.8
            learned[2, 1] = 0.4
            learned[1, 2] = 0.6
        dense_effective = torch.full((2, 4, 4), 0.01)
        causal_output = SimpleNamespace(
            a_shared=learned,
            a_effective=dense_effective,
            signed_sample_graph=None,
        )
        correlation = torch.full((2, 4, 4), 0.5)
        correlation.diagonal(dim1=-2, dim2=-1).fill_(1.0)

        adjacency, is_correlation = model._resolve_graph_adjacency(
            causal_output,
            correlation_matrix=correlation,
            module_name="test",
        )
        self.assertFalse(is_correlation)
        self.assertFalse(torch.allclose(adjacency, adjacency.transpose(-1, -2)))
        self.assertLess(
            float(model.latest_aux_losses["classification_graph_causal_support_density"]),
            0.5,
        )
        adjacency.sum().backward()
        self.assertIsNotNone(learned.grad)
        self.assertGreater(float(learned.grad.abs().sum()), 0.0)

        model.classification_graph_source = "learned"
        causal_output.a_effective = dense_effective.clone()
        causal_output.a_effective[1] *= 2.0
        strict_adjacency, _ = model._resolve_graph_adjacency(
            causal_output,
            correlation_matrix=None,
            module_name="strict-test",
        )
        self.assertEqual(strict_adjacency.shape, (2, 4, 4))
        self.assertFalse(torch.allclose(strict_adjacency[0], strict_adjacency[1]))
        self.assertFalse(
            torch.allclose(strict_adjacency, strict_adjacency.transpose(-1, -2))
        )

    def test_metrics_name_prefers_runfold_over_kfold(self):
        exp = Exp_Main.__new__(Exp_Main)
        exp._current_graph_path_name = lambda: "gcn_fallback"
        name = exp._metrics_base_name(
            "MDD_protocol_AAL116_kfold_5_model_S-DeCI_runfold3"
        )
        self.assertTrue(name.startswith("runfold3_gcn_fallback_"), name)

    def test_directional_gcn_and_mean_std_preserve_graph_differences(self):
        torch.manual_seed(11)
        model = ModuleGCNFallback(
            input_dim=3,
            hidden_dim=8,
            out_dim=2,
            num_layers=1,
            dropout=0.0,
            add_self_loop=True,
            readout_mode="mean_std",
            directional_propagation=True,
        )
        node_features = torch.randn(2, 5, 3)
        adjacency = torch.zeros(2, 5, 5)
        adjacency[0, 0, 1] = 1.0
        adjacency[0, 1, 2] = 0.7
        adjacency[1] = adjacency[0].T

        output = model(node_features, adjacency)
        self.assertEqual(output.readout.shape, (2, 16))
        self.assertFalse(torch.allclose(output.h_gcn[0], output.h_gcn[1]))
        self.assertFalse(torch.allclose(output.logits[0], output.logits[1]))


if __name__ == "__main__":
    unittest.main()
