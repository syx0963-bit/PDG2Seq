import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lib.long_history import context_batch, long_windows, split_origins, structured_mask
from model.LongShortTemporal import HorizonAdaptiveForecaster, LongHistoryMaskedEncoder
from model.PDG2Seq import PDG2Seq


class LongShortLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        torch.set_num_threads(1)

    def test_chronological_splits_have_no_shared_target_timestamps(self):
        (train, validation, test), normalization_end = split_origins(16992)
        self.assertLess(int(train[-1])+11, int(validation[0]))
        self.assertLess(int(validation[-1])+11, int(test[0]))
        self.assertLess(normalization_end, int(validation[0]))
        self.assertEqual((len(train), len(validation), len(test)), (10171, 3383, 3393))

    def test_past_context_does_not_change_when_future_traffic_changes(self):
        raw = torch.randn(160, 4)
        changed = raw.clone(); changed[80:] = 9999
        origins = torch.tensor([80])
        original = context_batch(raw, origins, 6, 3, 16, 64)
        modified = context_batch(changed, origins, 6, 3, 16, 64)
        for first, second in zip(original, modified):
            torch.testing.assert_close(first, second)
        history, available, start = long_windows(raw, origins, 32)
        future_changed, _, _ = long_windows(changed, origins, 32)
        torch.testing.assert_close(history, future_changed)
        self.assertTrue(available.all())
        self.assertEqual(int(start), 48)
        self.assertTrue((original[1][..., 0] == 0).all())

    def test_missing_history_is_padded_with_explicit_availability(self):
        raw = torch.ones(100, 4)
        history, available, _ = long_windows(raw, torch.tensor([4]), 32)
        self.assertEqual(int(available.sum()), 4)
        self.assertTrue((history[:, :-4] == 0).all())

    def test_masked_values_cannot_leak_into_reconstruction(self):
        model = LongHistoryMaskedEncoder(4, 32, 4, 8, 1, 0.0, 16).eval()
        history = torch.randn(2, 32, 4)
        available = torch.ones(2, 32, 1, dtype=torch.bool)
        mask = torch.zeros_like(history, dtype=torch.bool); mask[:, 8:20, 1] = True
        modified = history.clone(); modified[mask] = 9000
        first = model.reconstruct(history, torch.tensor([0, 16]), available, mask)
        second = model.reconstruct(modified, torch.tensor([0, 16]), available, mask)
        torch.testing.assert_close(first, second)
        loss = (first[mask]-history[mask]).square().mean()
        loss.backward()
        self.assertGreater(float(model.spatial_proj.weight.grad.abs().sum()), 0)

    def test_entirely_missing_long_history_produces_finite_zero_prior(self):
        model = LongHistoryMaskedEncoder(4, 32, 4, 8, 1, 0.0, 16).eval()
        prior = model(torch.zeros(1, 32, 4), torch.tensor([-32]),
                      torch.zeros(1, 32, 1, dtype=torch.bool))
        self.assertTrue(torch.isfinite(prior).all())
        self.assertTrue((prior == 0).all())

    def test_decomposition_reconstructs_observed_sequence(self):
        history = torch.randn(2, 12, 4)
        short, middle, trend = HorizonAdaptiveForecaster.decompose(history)
        torch.testing.assert_close(short+middle+trend, history.transpose(1, 2))

    def test_structured_masks_cover_whole_patches_and_only_available_history(self):
        history = torch.randn(2, 192, 20)
        available = torch.ones(2, 192, 1, dtype=torch.bool); available[:, :12] = False
        mask = structured_mask(history, available, 4, 32)
        self.assertFalse(mask[:, :12].any())
        patches = mask[:, 12:].reshape(2, -1, 4, 20)
        self.assertTrue((patches == patches[:, :, :1]).all())
        self.assertTrue(mask.any())

    def test_direct_prediction_uses_original_encoder_and_no_autoregressive_decoder(self):
        args = SimpleNamespace(num_nodes=4, input_dim=1, output_dim=1, rnn_units=8,
            num_layers=1, use_day=True, use_week=True, use_context_graph_refine=True,
            use_periodic_context=True, use_periodic_consistency=True,
            use_decoder_periodic_context=True, steps_per_day=16, steps_per_week=7,
            lr_decay_step=1500, embed_dim=2, time_dim=4, cheb_k=1,
            lag=6, horizon=3, use_signal_decouple=True, use_dgq=True,
            use_long_short_learning=True, long_history_steps=32, long_patch_size=4,
            long_prior_width=8, long_prior_layers=1, long_forecast_width=16, long_dropout=0.0)
        model = PDG2Seq(args)
        for value in model.parameters():
            if value.ndim > 1:
                torch.nn.init.xavier_uniform_(value)
            else:
                torch.nn.init.zeros_(value)
        model.eval()
        raw = torch.randn(160, 4)
        source, target, periodic = context_batch(raw, torch.tensor([80, 96]), 6, 3, 16, 64)
        history, available, start = long_windows(raw, torch.tensor([80, 96]), 32)
        with patch.object(model.decoder, 'forward', side_effect=AssertionError('Autoregressive decoder called')):
            prediction, gates = model(source, target, long_history=history,
                history_start=start, history_available=available, periodic_features=periodic, return_gates=True)
            prior = model.long_history_encoder(history, start, available)
            state = model.encode_short(source)
            cached = model(source, target, long_prior=prior, short_state=state, periodic_features=periodic)
        self.assertEqual(tuple(prediction.shape), (2, 3, 4, 1))
        torch.testing.assert_close(prediction, cached)
        torch.testing.assert_close(gates.sum(-1), torch.ones(2, 4, 3))
        changed_target = target.clone(); changed_target[..., 0] = 99999
        unchanged = model(source, changed_target, long_prior=prior, short_state=state, periodic_features=periodic)
        torch.testing.assert_close(prediction, unchanged)

    def test_recent_spatial_temporal_attention_learns_without_future_inputs(self):
        head = HorizonAdaptiveForecaster(4, 6, 3, 8, 8, 16, 0.0, attention_layers=2)
        torch.nn.init.normal_(head.output[-1].weight, std=0.05)
        history = torch.randn(2, 6, 4)
        core = torch.randn(2, 4, 8)
        prior = torch.randn(2, 4, 4, 8)
        periodic = torch.randn(2, 3, 4, 4)
        calendar = torch.zeros(2, 3, 2)
        output, weights = head(history, core, prior, periodic, True, calendar)
        output.square().mean().backward()
        self.assertGreater(float(head.recent_project.weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.short_spatial[0].attention.out_proj.weight.grad.abs().sum()), 0)
        self.assertEqual(tuple(output.shape), (2, 3, 4, 1))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 3))

    def test_daily_phase_memory_keeps_past_tokens_and_preserves_warmstart(self):
        encoder = LongHistoryMaskedEncoder(4, 32, 4, 8, 1, 0.0, 16).eval()
        history = torch.randn(2, 32, 4)
        start = torch.tensor([0, 16])
        tokens, valid = encoder.encode_tokens(history, start)
        prior = encoder(history, start, phase_memory=True)
        self.assertEqual(tuple(prior.shape), (2, 4, 8, 8))
        torch.testing.assert_close(prior[:, :, 4:], tokens[:, :, [4, 5, 0, 1]])
        old = HorizonAdaptiveForecaster(4, 6, 3, 8, 8, 16, 0.0).eval()
        new = HorizonAdaptiveForecaster(4, 6, 3, 8, 8, 16, 0.0, phase_memory=True).eval()
        new.load_state_dict(old.state_dict(), strict=False)
        torch.nn.init.normal_(old.output[-1].weight, std=0.05)
        new.output.load_state_dict(old.output.state_dict())
        recent, core, periodic = torch.randn(2, 6, 4), torch.randn(2, 4, 8), torch.randn(2, 3, 4, 4)
        expected = old(recent, core, prior[:, :, :4], periodic)
        actual = new(recent, core, prior, periodic)
        torch.testing.assert_close(expected, actual)
        actual.square().mean().backward()
        self.assertGreater(float(new.phase_project.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
