import ast
import subprocess
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from lib.online_benchmark import OnlineBenchmark, SEPTEMBER_SETTINGS


class OnlineBenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        torch.set_num_threads(1)
        self.embedding = torch.randn(4, 3)
        self.pred = torch.rand(12, 3, 4, 1)*25+5
        self.true = torch.rand_like(self.pred)*25+5

    def test_immediate_adapter_matches_historical_source_including_warmup(self):
        source = subprocess.check_output(['git', 'show', 'innovation2-3-4:model/BasicTrainer.py'], text=True)
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
                    and node.name == 'GraphAwareOnlineAdapter')
        namespace = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<historical adapter>', 'exec'), namespace)
        key_map = dict(lr='online_adapt_lr', scale_lr='online_scale_lr', global_lr='online_global_adapt_lr',
            global_scale_lr='online_global_scale_lr', bias_decay='online_adapt_decay',
            error_decay='online_error_decay', sensitivity='online_drift_sensitivity',
            neighbor_expand='online_neighbor_expand', bias_clip='online_bias_clip',
            scale_clip='online_scale_clip', horizon_lr_start='online_horizon_lr_start',
            horizon_lr_end='online_horizon_lr_end', post_bias_shrink='online_val_bias_shrink')
        args = SimpleNamespace(device='cpu', horizon=3, num_nodes=4, online_graph_topk=8,
            online_overlap_memory=False, online_overlap_blend=0.0,
            **{key_map[key]:value for key,value in SEPTEMBER_SETTINGS.items()})
        original = namespace['GraphAwareOnlineAdapter'](args, SimpleNamespace(node_embeddings1=self.embedding))
        restored = OnlineBenchmark(self.embedding, 3)
        first = original.adapt_batch(self.pred[:8], self.true[:8])
        second = restored.adapt(self.pred[:8], self.true[:8])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        original.set_post_bias((self.true[:8]-first).mean(0))
        restored.fit_post_bias(self.true[:8]-second)
        torch.testing.assert_close(original.adapt_batch(self.pred[8:], self.true[8:]),
                                   restored.adapt(self.pred[8:], self.true[8:]), rtol=0, atol=0)

    def test_historical_mode_uses_previous_future_labels_without_copying_current_truth(self):
        altered = self.true.clone(); altered[0, -1] += 500
        first = OnlineBenchmark(self.embedding, 3).adapt(self.pred, self.true)
        second = OnlineBenchmark(self.embedding, 3).adapt(self.pred, altered)
        torch.testing.assert_close(first[0], self.pred[0], rtol=0, atol=0)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[1, -1], second[1, -1]))

    def test_causal_comparison_waits_until_labels_are_available(self):
        altered = self.true.clone(); altered[4:] += 500
        first = OnlineBenchmark(self.embedding, 3, delay_steps=3).adapt(self.pred, self.true)
        second = OnlineBenchmark(self.embedding, 3, delay_steps=3).adapt(self.pred, altered)
        torch.testing.assert_close(first[:3], self.pred[:3], rtol=0, atol=0)
        torch.testing.assert_close(first[:7], second[:7], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
