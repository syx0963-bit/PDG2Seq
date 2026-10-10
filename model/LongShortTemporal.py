"""Masked long-history priors and horizon-adaptive direct traffic prediction."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LongHistoryMaskedEncoder(nn.Module):
    def __init__(self, num_nodes, history_steps=2016, patch_size=12, width=48,
                 layers=2, dropout=0.1, steps_per_day=288):
        super().__init__()
        if history_steps % patch_size:
            raise ValueError('Long history must contain whole patches')
        self.history_steps, self.patch_size = history_steps, patch_size
        self.width, self.steps_per_day = width, steps_per_day
        self.patch_embed = nn.Linear(2*patch_size, width)
        self.calendar_embed = nn.Linear(4, width, bias=False)
        self.position = nn.Parameter(torch.randn(history_steps//patch_size, width)*0.02)
        self.node_embed = nn.Parameter(torch.randn(num_nodes, 16)*0.05)
        self.spatial_proj = nn.Linear(width, width)
        self.spatial_norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(width, 4, width*3, dropout,
                                            activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(width)
        self.reconstruction = nn.Linear(width, patch_size)
        self.register_buffer('physical_graph', torch.eye(num_nodes))

    def encode_tokens(self, history, start_times, available=None, masked=None):
        """All attention is within the supplied past, never forecast targets."""
        batch, length, nodes = history.shape
        if length != self.history_steps:
            raise ValueError('Unexpected long-history length')
        if available is None:
            available = torch.ones_like(history, dtype=torch.bool)
        else:
            available = available.expand_as(history).bool()
        if masked is None:
            masked = torch.zeros_like(available)
        visible = available & ~masked
        patches = (history*visible).transpose(1, 2).reshape(batch, nodes, -1, self.patch_size)
        visibility = visible.transpose(1, 2).reshape_as(patches).to(history.dtype)
        tokens = self.patch_embed(torch.cat((patches, visibility), -1))
        timestamps = start_times[:, None]+torch.arange(0, length, self.patch_size, device=history.device)[None]
        day = 2*math.pi*timestamps/self.steps_per_day
        week = day/7
        calendar = torch.stack((day.sin(), day.cos(), week.sin(), week.cos()), -1)
        tokens = tokens+self.calendar_embed(calendar).unsqueeze(1)+self.position[None, None]
        learned_graph = torch.softmax(self.node_embed@self.node_embed.T/4, -1)
        graph = 0.8*self.physical_graph+0.2*learned_graph
        neighbors = torch.einsum('ij,bjpc->bipc', graph.to(tokens.dtype), tokens)
        tokens = self.spatial_norm(tokens+0.2*self.spatial_proj(neighbors))
        valid_patch = available.transpose(1, 2).reshape(batch, nodes, -1, self.patch_size).any(-1)
        padding = ~valid_patch.reshape(batch*nodes, -1)
        # Keep attention finite for an entirely absent history; pooling below
        # still zeroes this representation.
        padding = padding.clone()
        padding[padding.all(-1), -1] = False
        encoded = self.temporal(tokens.reshape(batch*nodes, -1, self.width),
                                src_key_padding_mask=padding)
        encoded = self.output_norm(encoded).reshape(batch, nodes, -1, self.width)
        return encoded, valid_patch

    def reconstruct(self, history, start_times, available, masked):
        tokens, _ = self.encode_tokens(history, start_times, available, masked)
        return self.reconstruction(tokens).reshape(history.shape[0], history.shape[2], -1).transpose(1, 2)

    def forward(self, history, start_times, available=None, phase_memory=False):
        tokens, valid = self.encode_tokens(history, start_times, available)
        weights = valid.to(tokens.dtype).unsqueeze(-1)
        mean = (tokens*weights).sum(2)/weights.sum(2).clamp_min(1)
        day_count = min(tokens.shape[2], self.steps_per_day//self.patch_size)
        day_tokens, day_weights = tokens[:, :, -day_count:], weights[:, :, -day_count:]
        day_mean = (day_tokens*day_weights).sum(2)/day_weights.sum(2).clamp_min(1)
        variance = ((day_tokens-day_mean.unsqueeze(2)).square()*day_weights).sum(2)/day_weights.sum(2).clamp_min(1)
        last = tokens[:, :, -1]*weights[:, :, -1]
        prior = torch.stack((last, day_mean, mean, (variance+1e-6).sqrt()), 2)
        prior = prior*valid.any(2)[:, :, None, None]
        if phase_memory:
            # Retrieve the next two patches at the same phase on each past day.
            # Every index lies inside the supplied history, before prediction.
            indices = []
            for day in range(1, self.history_steps//self.steps_per_day+1):
                first = tokens.shape[2]-day*(self.steps_per_day//self.patch_size)
                indices.extend((first, min(first+1, tokens.shape[2]-1)))
            index = torch.tensor(indices, device=history.device, dtype=torch.long)
            aligned = tokens.index_select(2, index)*weights.index_select(2, index)
            prior = torch.cat((prior, aligned), 2)
        return prior


class ShortSpatialAttention(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width*2, width))
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, value):
        normalized = self.norm1(value)
        delta = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return delta+self.ff(self.norm2(value+delta))


class HorizonAdaptiveForecaster(nn.Module):
    def __init__(self, num_nodes, lag=12, horizon=12, core_width=64,
                 prior_width=48, width=96, dropout=0.1, attention_layers=0, phase_memory=False):
        super().__init__()
        self.lag, self.horizon, self.width = lag, horizon, width
        self.short_encoder = nn.Sequential(nn.Linear(lag*3+3, width), nn.GELU(), nn.Linear(width, width))
        self.trend_encoder = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.periodic_encoder = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.core_proj = nn.Linear(core_width, width)
        self.prior_proj = nn.Linear(prior_width, width)
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, 16)*0.05)
        self.node_proj = nn.Linear(16, width)
        self.horizon_embedding = nn.Parameter(torch.randn(horizon, width)*0.02)
        self.long_attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.phase_memory = phase_memory
        if phase_memory:
            self.phase_attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
            self.phase_project = nn.Linear(width, width)
            nn.init.zeros_(self.phase_project.weight)
            nn.init.zeros_(self.phase_project.bias)
        self.cooperation = nn.Sequential(nn.Linear(width*2, width), nn.Sigmoid())
        self.scale_gate = nn.Sequential(nn.Linear(width*2, width), nn.GELU(), nn.Linear(width, 4))
        self.spatial_proj = nn.Linear(width, width)
        self.fusion_norm = nn.LayerNorm(width)
        self.output = nn.Sequential(nn.Linear(width*2+6, width*2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width*2, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.register_buffer('physical_graph', torch.eye(num_nodes))
        self.attention_layers = attention_layers
        if attention_layers:
            self.recent_embed = nn.Linear(3, width)
            self.recent_position = nn.Parameter(torch.randn(1, lag, width)*0.02)
            layer = nn.TransformerEncoderLayer(width, 4, width*2, dropout,
                                                activation='gelu', batch_first=True, norm_first=True)
            self.recent_temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.recent_project = nn.Linear(width*2, width)
            self.short_spatial = nn.ModuleList([ShortSpatialAttention(width, dropout) for _ in range(attention_layers)])
            self.calendar_encoder = nn.Linear(4, width, bias=False)
            nn.init.zeros_(self.recent_project.weight)
            nn.init.zeros_(self.recent_project.bias)
            nn.init.zeros_(self.calendar_encoder.weight)

    @staticmethod
    def decompose(history):
        flat = history.transpose(1, 2)
        smooth3 = F.avg_pool1d(F.pad(flat, (1, 1), mode='replicate'), 3, stride=1)
        smooth7 = F.avg_pool1d(F.pad(flat, (3, 3), mode='replicate'), 7, stride=1)
        return flat-smooth3, smooth3-smooth7, smooth7

    def forward(self, history, core_state, long_prior, periodic, return_gates=False, calendar=None):
        batch, lag, nodes = history.shape
        short, middle, trend = self.decompose(history)
        x = history.transpose(1, 2)
        summary = torch.stack((x[:, :, -1], x.mean(-1), x.std(-1, unbiased=False)), -1)
        short_state = self.short_encoder(torch.cat((x, short, middle, summary), -1))
        core = self.core_proj(core_state)+self.node_proj(self.node_embedding)[None]
        if self.attention_layers:
            recent = torch.stack((x, short, trend), -1).reshape(batch*nodes, lag, 3)
            recent = self.recent_temporal(self.recent_embed(recent)+self.recent_position)
            temporal = self.recent_project(torch.cat((recent[:, -1], recent.mean(1)), -1)).reshape(batch, nodes, self.width)
            short_state = short_state+temporal
            spatial = short_state+core
            for layer in self.short_spatial:
                short_state = short_state+layer(spatial)
                spatial = short_state+core
        query = (core+short_state).unsqueeze(2)+self.horizon_embedding[None, None]
        if self.attention_layers and calendar is not None:
            day = calendar[..., 0]*2*math.pi
            week = (calendar[..., 1]+calendar[..., 0])*2*math.pi/7
            known_phase = torch.stack((day.sin(), day.cos(), week.sin(), week.cos()), -1)
            query = query+self.calendar_encoder(known_phase)[:, None]
        memory = self.prior_proj(long_prior)
        long_state, _ = self.long_attention(query.reshape(batch*nodes, self.horizon, self.width),
            memory[:, :, :4].reshape(batch*nodes, 4, self.width),
            memory[:, :, :4].reshape(batch*nodes, 4, self.width), need_weights=False)
        if self.phase_memory:
            if memory.shape[2] <= 4:
                raise ValueError('Phase-aligned long-history memory is required')
            aligned = memory[:, :, 4:].reshape(batch*nodes, -1, self.width)
            phase, _ = self.phase_attention(query.reshape(batch*nodes, self.horizon, self.width),
                                            aligned, aligned, need_weights=False)
            long_state = long_state+self.phase_project(phase)
        long_state = long_state.reshape(batch, nodes, self.horizon, self.width)
        cooperation = self.cooperation(torch.cat((query, long_state), -1))
        long_state = long_state*cooperation
        centered_time = torch.arange(lag, device=x.device, dtype=x.dtype)-(lag-1)/2
        slope = (trend*centered_time).sum(-1)/centered_time.square().sum().clamp_min(1)
        steps = torch.arange(1, self.horizon+1, device=x.device, dtype=x.dtype)
        extrapolated = trend[:, :, -1, None]+slope[:, :, None]*steps
        trend_features = torch.stack((trend[:, :, -1, None].expand_as(extrapolated),
            trend.mean(-1)[:, :, None].expand_as(extrapolated),
            slope[:, :, None].expand_as(extrapolated), extrapolated,
            middle[:, :, -1, None].expand_as(extrapolated)), -1)
        trend_state = self.trend_encoder(trend_features)
        periodic = periodic.permute(0, 2, 1, 3)
        periodic_features = torch.cat((periodic, summary[:, :, None, :2].expand(-1, -1, self.horizon, -1)), -1)
        periodic_state = self.periodic_encoder(periodic_features)
        short_expanded = short_state.unsqueeze(2).expand_as(long_state)
        scales = torch.stack((short_expanded, trend_state, periodic_state, long_state), -2)
        fraction = torch.linspace(0, 1, self.horizon, device=x.device, dtype=x.dtype)
        prior_logits = torch.stack((1.5*(1-fraction), 0.5-(2*fraction-1).abs(),
                                    1.5*fraction, 0.2+0.75*fraction), -1)
        gates = torch.softmax(self.scale_gate(torch.cat((query, long_state), -1))+prior_logits, -1)
        fused = (scales*gates.unsqueeze(-1)).sum(-2)
        graph = 0.8*self.physical_graph+0.2*torch.softmax(self.node_embedding@self.node_embedding.T/4, -1)
        neighbors = torch.einsum('ij,bjhc->bihc', graph.to(fused.dtype), fused)
        fused = self.fusion_norm(fused+0.1*self.spatial_proj(neighbors))
        correction = self.output(torch.cat((fused, query, periodic_features), -1))
        output = x[:, :, -1, None, None]+correction
        output = output.permute(0, 2, 1, 3)
        return (output, gates) if return_gates else output
