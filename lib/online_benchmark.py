"""Explicit historical future-label-assisted benchmark; never model inputs."""
from collections import deque

import torch
from torch.nn import functional as F


SEPTEMBER_SETTINGS = dict(lr=0.08, scale_lr=0.16, global_lr=0.02,
    global_scale_lr=0.0, bias_decay=0.97, error_decay=0.92, sensitivity=0.7,
    neighbor_expand=0.5, bias_clip=4.0, scale_clip=0.4, horizon_lr_start=0.35,
    horizon_lr_end=1.35, post_bias_shrink=-0.5)


class OnlineBenchmark:
    def __init__(self, node_embedding, horizon=12, settings=None, delay_steps=0):
        self.settings = dict(SEPTEMBER_SETTINGS, **(settings or {}))
        self.horizon, self.delay_steps = horizon, delay_steps
        self.uses_future_labels = delay_steps < horizon
        self.pending = deque()
        embedding = F.normalize(node_embedding.detach().float().cpu(), dim=-1, eps=1e-8)
        nodes = len(embedding)
        similarity = torch.relu(embedding@embedding.T)
        similarity.fill_diagonal_(0)
        values, indices = torch.topk(similarity, min(8, nodes-1), -1)
        graph = torch.zeros_like(similarity).scatter_(1, indices, values)
        graph = (graph+graph.T)/2+torch.eye(nodes)
        self.graph = graph/graph.sum(-1, keepdim=True).clamp_min(1e-8)
        self.bias = torch.zeros(1, horizon, nodes, 1)
        self.scale = torch.ones_like(self.bias)
        self.post_bias = torch.zeros_like(self.bias)
        self.error = torch.zeros(horizon, nodes, 1)
        self.gain = torch.linspace(self.settings['horizon_lr_start'], self.settings['horizon_lr_end'], horizon).view(horizon, 1, 1)
        self.updates = 0

    def update(self, pred, true):
        config = self.settings
        residual = true-pred
        error = residual.abs().mean(0)
        self.error = error if self.updates == 0 else config['error_decay']*self.error+(1-config['error_decay'])*error
        neighbor_error = torch.einsum('ij,hjo->hio', self.graph, self.error)
        drift = self.error+config['neighbor_expand']*(self.error-neighbor_error).abs()
        mask = (drift > drift.mean(1, keepdim=True)+config['sensitivity']*drift.std(1, keepdim=True).clamp_min(1e-6)).float()
        mask = (torch.maximum(mask, torch.einsum('ij,hjo->hio', self.graph, mask)) > 0).float()
        mean = residual.mean(0)
        self.bias = (config['bias_decay']*self.bias+config['lr']*self.gain*mean*mask).clamp(-config['bias_clip'], config['bias_clip'])
        self.bias = (self.bias+config['global_lr']*self.gain*mean).clamp(-config['bias_clip'], config['bias_clip'])
        relative = (residual/pred.abs().clamp_min(1)).mean(0)
        self.scale = (self.scale+config['scale_lr']*self.gain*relative*mask+
                      config['global_scale_lr']*self.gain*relative).clamp(1-config['scale_clip'], 1+config['scale_clip'])
        self.updates += 1

    def adapt(self, pred, true):
        output = []
        for index in range(len(pred)):
            if self.delay_steps and len(self.pending) >= self.delay_steps:
                self.update(*self.pending.popleft())
            value = (self.scale*pred[index:index+1]+self.bias+self.post_bias).clamp_min(0)
            output.append(value)
            if self.delay_steps:
                self.pending.append((value, true[index:index+1]))
            else:
                # Intentionally uses the entire just-issued future horizon.
                # Reproduces September; output above never contains copied truth.
                self.update(value, true[index:index+1])
        return torch.cat(output)

    def fit_post_bias(self, residual):
        self.post_bias = (self.settings['post_bias_shrink']*residual.mean(0)).clamp(
            -self.settings['bias_clip'], self.settings['bias_clip']).unsqueeze(0)

    def reset_pending(self):
        self.pending.clear()


def assisted_validation(pred, true, node_embedding):
    adapter = OnlineBenchmark(node_embedding)
    prefix = len(pred)*2//3
    warmup = adapter.adapt(pred[:prefix], true[:prefix])
    adapter.fit_post_bias(true[:prefix]-warmup)
    return adapter.adapt(pred[prefix:], true[prefix:]), true[prefix:]
