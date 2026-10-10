import unittest
import torch

from lib.forecast_optimization import ForecastEMA, balanced_forecast_loss, validation_score, configure_forecast_training, chronological_sample_weights


class ForecastOptimizationTests(unittest.TestCase):
    def test_default_loss_matches_previous_training_objective(self):
        torch.manual_seed(1)
        pred = torch.randn(2, 3, 4, 1, requires_grad=True)
        labels, teacher = torch.randn_like(pred), torch.randn_like(pred)
        true = torch.rand_like(pred)*100
        true[0, 0] = 0
        diff = pred-labels
        expected = diff.abs().mean()+0.35*diff.square().mean()
        expected += 0.05*(diff[true > 0].abs()*156/true[true > 0].clamp_min(1)).mean()
        expected += 0.03*(pred-teacher).square().mean()
        actual, _ = balanced_forecast_loss(pred, labels, true, teacher, 156, 0.35, 0.05, 0.03)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, pred, retain_graph=True)[0],
                                   torch.autograd.grad(expected, pred)[0])

    def test_near_horizon_weight_increases_early_error_gradient_without_changing_loss_scale(self):
        pred = torch.ones(1, 3, 2, 1, requires_grad=True)
        zeros, real = torch.zeros_like(pred), torch.ones_like(pred)
        ordinary, _ = balanced_forecast_loss(pred, zeros, real, zeros, 1, 1, 0, 0)
        weighted, _ = balanced_forecast_loss(pred, zeros, real, zeros, 1, 1, 0, 0, 1.5)
        torch.testing.assert_close(weighted, ordinary)
        gradient = torch.autograd.grad(weighted, pred)[0]
        self.assertGreater(float(gradient[:, 0].mean()), float(gradient[:, -1].mean()))

    def test_zero_flow_and_zero_percentage_weight_are_finite(self):
        pred = torch.ones(2, 3, 4, 1, requires_grad=True)
        zeros = torch.zeros_like(pred)
        loss, _ = balanced_forecast_loss(pred, zeros, zeros, zeros, 156, 1, 0, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_ema_snapshot_reproduces_next_update_and_keeps_integer_buffers(self):
        module = torch.nn.BatchNorm1d(2)
        first = ForecastEMA(module, 0.9)
        with torch.no_grad():
            module.weight.fill_(3)
            module.num_batches_tracked.fill_(7)
        first.update(module)
        torch.testing.assert_close(first.shadow['weight'], torch.full((2,), 1.2))
        restored = ForecastEMA(module, 0.8)
        restored.load_state_dict(first.state_dict())
        with torch.no_grad():
            module.weight.fill_(4)
        first.update(module); restored.update(module)
        for key in first.shadow:
            torch.testing.assert_close(first.shadow[key], restored.shadow[key])
        self.assertEqual(int(restored.shadow['num_batches_tracked']), 7)
        self.assertEqual(restored.updates, 2)

    def test_balanced_selection_does_not_reward_one_metric_at_others_expense(self):
        reference = dict(rmse=30, mae=18, mape=11)
        balanced = dict(rmse=29.7, mae=17.82, mape=10.89)
        skewed = dict(rmse=30.3, mae=18.18, mape=9)
        self.assertLess(validation_score(balanced, reference), validation_score(skewed, reference))

    def test_joint_training_unfreezes_only_live_short_encoder_and_head(self):
        from types import SimpleNamespace
        from model.PDG2Seq import PDG2Seq
        from lib.long_history import context_batch
        args = SimpleNamespace(num_nodes=4, input_dim=1, output_dim=1, rnn_units=8,
            num_layers=1, use_day=True, use_week=True, use_context_graph_refine=True,
            use_periodic_context=True, steps_per_day=16, steps_per_week=7,
            lr_decay_step=1500, embed_dim=2, time_dim=4, cheb_k=1,
            lag=6, horizon=3, use_signal_decouple=True, use_dgq=True,
            use_long_short_learning=True, long_history_steps=32, long_patch_size=4,
            long_prior_width=8, long_prior_layers=1, long_forecast_width=16, long_dropout=0.0)
        model = PDG2Seq(args)
        for parameter in model.parameters():
            if parameter.ndim > 1:
                torch.nn.init.xavier_uniform_(parameter)
            else:
                torch.nn.init.zeros_(parameter)
        head, core = configure_forecast_training(model, True)
        self.assertTrue(head and core)
        self.assertFalse(any(value.requires_grad for value in model.long_history_encoder.parameters()))
        self.assertFalse(any(value.requires_grad for value in model.decoder.parameters()))
        source, target, periodic = context_batch(torch.randn(100, 4), torch.tensor([80]), 6, 3, 16, 64)
        prior = torch.randn(1, 4, 4, 8)
        output = model(source, target, long_prior=prior, periodic_features=periodic)
        output.square().mean().backward()
        self.assertTrue(any(value.grad is not None and float(value.grad.abs().sum()) > 0 for value in core))
        self.assertTrue(any(value.grad is not None and float(value.grad.abs().sum()) > 0 for value in head))
        self.assertFalse(any(value.grad is not None for value in model.long_history_encoder.parameters()))

    def test_rmse_term_increases_gradient_for_large_forecast_errors(self):
        pred = torch.tensor([1.0, 10.0]).reshape(1, 2, 1, 1).requires_grad_()
        labels = torch.zeros_like(pred)
        loss, _ = balanced_forecast_loss(pred, labels, torch.ones_like(pred), labels, 1, 0, 0, 0,
                                         rmse_weight=0.25)
        gradient = torch.autograd.grad(loss, pred)[0].flatten()
        self.assertGreater(float(gradient[1]), float(gradient[0]))

    def test_recency_sampling_covers_only_existing_training_origins(self):
        weights = chronological_sample_weights(100, 0.3, 2.0)
        torch.testing.assert_close(weights[:70], torch.ones(70))
        torch.testing.assert_close(weights[70:], torch.full((30,), 2.0))
        generator = torch.Generator().manual_seed(5)
        indices = torch.multinomial(weights, 10000, replacement=True, generator=generator)
        self.assertTrue(((indices >= 0) & (indices < 100)).all())
        self.assertGreater(float((indices >= 70).float().mean()), 0.4)


if __name__ == '__main__':
    unittest.main()
