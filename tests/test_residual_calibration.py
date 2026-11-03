import unittest

import numpy as np
import torch

from tools.optimize_history_calibration import HistoryFeatures
from tools.train_residual_head import ResidualHead, matrix
from model.AuxiliaryTeacher import AuxiliaryTeacher
from tools.refine_auxiliary_results import augment


class ResidualCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(10)
        self.raw = self.rng.normal(size=(2500, 4, 3)).astype(np.float32)
        self.starts = np.arange(2100, 2105)
        p = self.rng.normal(size=(5, 12, 4, 1)).astype(np.float32)
        self.values = [p.copy(), p.copy(), p.copy(), p.copy(), np.ones_like(p),
                       p[:, 0].copy(), p[:, 0].copy(), p[:, 0].copy()]
        self.weights = np.ones((1, 12, 1, 1), np.float32)
        self.neighbors = np.array([[1, 2], [0, 2], [1, 3], [0, 2]])

    def test_features_ignore_unobserved_traffic(self):
        b = HistoryFeatures(self.raw, self.starts, self.values, self.weights,
                            self.weights * 0, self.neighbors)
        # First origin ends at 2111; all 12 horizons have that same cutoff.
        indices = np.arange(12 * 4)
        before = b.rows(indices, True)
        self.raw[2112:] += 10000
        self.values[2][...] = 99999
        np.testing.assert_array_equal(before, b.rows(indices, True))

    def test_multi_horizon_rows_stay_aligned(self):
        x, node, base, true = matrix(self.raw, self.starts, self.values,
                                   self.weights, self.weights * 0, self.neighbors)
        self.assertEqual(x.shape[0], 5 * 4)
        for t in range(5):
            for n in range(4):
                np.testing.assert_array_equal(base[t * 4 + n], np.maximum(self.values[0][t, :, n, 0], 0))
                np.testing.assert_array_equal(true[t * 4 + n], self.values[2][t, :, n, 0])
                self.assertEqual(node[t * 4 + n], n)

    def test_head_initially_preserves_main_prediction(self):
        head = ResidualHead(5, nodes=4)
        out = head(torch.randn(8, 5), torch.arange(8) % 4)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        (out - 1).square().mean().backward()
        self.assertTrue(torch.isfinite(head.net[-1].weight.grad).all())
        self.assertGreater(head.net[-1].weight.grad.abs().sum().item(), 0)

    def test_direct_teacher_shape_and_gradients(self):
        teacher = AuxiliaryTeacher(nodes=4, layers=1)
        x = torch.randn(2, 12, 4, 3)
        prediction = teacher(x, torch.tensor([0, 280]))
        self.assertEqual(tuple(prediction.shape), (2, 12, 4, 1))
        torch.testing.assert_close(prediction, x[:, -1, :, :1][:, None].expand(-1, 12, -1, -1))
        prediction.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in teacher.parameters() if p.grad is not None))

    def test_three_model_fusion_is_preserved_by_history_adapter(self):
        main, legacy, auxiliary = [self.rng.normal(size=(5, 12, 4, 1)).astype(np.float32) for _ in range(3)]
        values = list(self.values)
        values[0], values[1] = main, legacy
        mw = np.full((1, 12, 1, 1), .4, np.float32)
        lw = np.full_like(mw, .2)
        converted = augment(values, auxiliary, mw, lw)
        reconstructed = (mw + lw) * converted[0] + (1 - mw - lw) * converted[1]
        expected = mw * main + lw * legacy + (1 - mw - lw) * auxiliary
        np.testing.assert_allclose(reconstructed, expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
