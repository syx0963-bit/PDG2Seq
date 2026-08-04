import math
import torch
import torch.nn as nn
from model.PDG2Seq_DGCN import PDG2Seq_GCN
from collections import OrderedDict
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FC, self).__init__()
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.mlp = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                ('sigmoid1', nn.Sigmoid()),
                ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                ('sigmoid2', nn.Sigmoid()),
                ('fc3', nn.Linear(self.middle_dim, dim_out))
            ])
        )

    def forward(self, x):
        return self.mlp(x)


class PDG2SeqCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, args=None):
        super(PDG2SeqCell, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.hidden_dim = dim_out
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.use_dgq = getattr(args, 'use_dgq', False) if args is not None else False
        self.dgq_alpha = getattr(args, 'dgq_alpha', 0.1) if args is not None else 0.1
        self.dgq_dim = getattr(args, 'dgq_dim', 16) if args is not None else 16
        self.use_context_graph_refine = getattr(args, 'use_context_graph_refine', False) if args is not None else False
        self.context_graph_lambda = getattr(args, 'context_graph_lambda', 0.05) if args is not None else 0.05
        self.context_graph_dim = getattr(args, 'context_graph_dim', 16) if args is not None else 16
        self.use_signal_decouple = getattr(args, 'use_signal_decouple', False) if args is not None else False
        self.signal_decouple_hidden = getattr(args, 'signal_decouple_hidden', 32) if args is not None else 32
        self.signal_diffusion_bias = getattr(args, 'signal_diffusion_bias', 1.5) if args is not None else 1.5
        self.signal_fuse_diffusion_bias = getattr(args, 'signal_fuse_diffusion_bias', 1.0) if args is not None else 1.0
        self.signal_inherent_scale = getattr(args, 'signal_inherent_scale', 0.3) if args is not None else 0.3
        self.use_meta_reliable_graph = getattr(args, 'use_meta_reliable_graph', False) if args is not None else False
        self.meta_state_dim = getattr(args, 'meta_state_dim', 32) if args is not None else 32
        self.meta_graph_modes = getattr(args, 'meta_graph_modes', 4) if args is not None else 4
        self.meta_graph_alpha = getattr(args, 'meta_graph_alpha', 0.65) if args is not None else 0.65
        self.meta_stable_lambda = getattr(args, 'meta_stable_lambda', 0.8) if args is not None else 0.8
        self.meta_anomaly_lambda = getattr(args, 'meta_anomaly_lambda', 0.7) if args is not None else 0.7
        self.meta_noise_floor = getattr(args, 'meta_noise_floor', 0.2) if args is not None else 0.2
        self.gate = PDG2Seq_GCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, time_dim)
        self.update = PDG2Seq_GCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, time_dim)
        self.fc1 = FC(dim_in + self.hidden_dim, time_dim)
        self.fc2 = FC(dim_in + self.hidden_dim, time_dim)

        if self.use_dgq:
            self.dgq_src_in = nn.Linear(dim_in + self.hidden_dim, self.dgq_dim)
            self.dgq_dst_in = nn.Linear(dim_in + self.hidden_dim, self.dgq_dim)
            self.dgq_src_out = nn.Linear(dim_in + self.hidden_dim, self.dgq_dim)
            self.dgq_dst_out = nn.Linear(dim_in + self.hidden_dim, self.dgq_dim)
        else:
            self.dgq_src_in = None
            self.dgq_dst_in = None
            self.dgq_src_out = None
            self.dgq_dst_out = None

        if self.use_context_graph_refine:
            self.context_cur_proj = nn.Linear(dim_in, self.context_graph_dim)
            self.context_ctx_proj = nn.Linear(dim_in, self.context_graph_dim)
        else:
            self.context_cur_proj = None
            self.context_ctx_proj = None

        if self.use_signal_decouple:
            decouple_in_dim = dim_in + dim_out + time_dim + embed_dim + dim_in + 1
            self.signal_gate = nn.Sequential(
                nn.Linear(decouple_in_dim, self.signal_decouple_hidden),
                nn.ReLU(),
                nn.Linear(self.signal_decouple_hidden, dim_in)
            )
            self.inherent_gru = nn.GRUCell(dim_in, dim_out)
            self.periodic_proj = nn.Linear(dim_in, dim_in)
            self.fuse_gate = nn.Sequential(
                nn.Linear(2 * dim_out + time_dim + embed_dim, self.signal_decouple_hidden),
                nn.ReLU(),
                nn.Linear(self.signal_decouple_hidden, dim_out)
            )
            nn.init.constant_(self.signal_gate[-1].bias, self.signal_diffusion_bias)
            nn.init.constant_(self.fuse_gate[-1].bias, self.signal_fuse_diffusion_bias)
        else:
            self.signal_gate = None
            self.inherent_gru = None
            self.periodic_proj = None
            self.fuse_gate = None

        if self.use_meta_reliable_graph:
            state_in_dim = dim_in + dim_out + time_dim + embed_dim + dim_in + dim_in + 1
            self.traffic_state_encoder = nn.Sequential(
                nn.Linear(state_in_dim, self.meta_state_dim),
                nn.ReLU(),
                nn.Linear(self.meta_state_dim, self.meta_state_dim),
                nn.ReLU()
            )
            self.traffic_state_head = nn.Linear(self.meta_state_dim, 4)
            self.graph_mode_head = nn.Linear(self.meta_state_dim, self.meta_graph_modes)
            self.mode_src_in = nn.Parameter(torch.FloatTensor(self.meta_graph_modes, node_num, self.dgq_dim))
            self.mode_dst_in = nn.Parameter(torch.FloatTensor(self.meta_graph_modes, node_num, self.dgq_dim))
            self.mode_src_out = nn.Parameter(torch.FloatTensor(self.meta_graph_modes, node_num, self.dgq_dim))
            self.mode_dst_out = nn.Parameter(torch.FloatTensor(self.meta_graph_modes, node_num, self.dgq_dim))
            order_dim = cheb_k * 2 + 1
            self.gate_order_head = nn.Linear(self.meta_state_dim, order_dim)
            self.update_order_head = nn.Linear(self.meta_state_dim, order_dim)
            self.gate_meta_scale = nn.Linear(self.meta_state_dim, 2 * dim_out)
            self.gate_meta_bias = nn.Linear(self.meta_state_dim, 2 * dim_out)
            self.update_meta_scale = nn.Linear(self.meta_state_dim, dim_out)
            self.update_meta_bias = nn.Linear(self.meta_state_dim, dim_out)
            self.gate_channel_head = nn.Linear(self.meta_state_dim, 2 * dim_out)
            self.update_channel_head = nn.Linear(self.meta_state_dim, dim_out)
            self.periodic_proto_proj = nn.Linear(dim_in, self.dgq_dim)
            self.current_proto_proj = nn.Linear(dim_in, self.dgq_dim)
        else:
            self.traffic_state_encoder = None
            self.traffic_state_head = None
            self.graph_mode_head = None
            self.gate_order_head = None
            self.update_order_head = None
            self.gate_meta_scale = None
            self.gate_meta_bias = None
            self.update_meta_scale = None
            self.update_meta_bias = None
            self.gate_channel_head = None
            self.update_channel_head = None
            self.periodic_proto_proj = None
            self.current_proto_proj = None

        self._dgq_debug_count = 0
        self._context_debug_count = 0
        self._meta_debug_count = 0
        self._decouple_debug_count = 0

    def forward(self, x, state, node_embeddings, periodic_context=None, context_valid=None):
        state = state.to(x.device)
        if self.use_signal_decouple:
            x_diff, x_inh, time_context, static_embedding = self._decouple_signal(
                x, state, node_embeddings, periodic_context, context_valid
            )
        else:
            x_diff = x
            x_inh = None
            time_context = None
            static_embedding = None

        input_and_state = torch.cat((x_diff, state), dim=-1)
        filter1 = self.fc1(input_and_state)
        filter2 = self.fc2(input_and_state)

        nodevec1 = torch.tanh(torch.einsum('bd,bnd->bnd', node_embeddings[0], filter1))
        nodevec2 = torch.tanh(torch.einsum('bd,bnd->bnd', node_embeddings[1], filter2))

        adj = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1))

        adj_in = PDG2SeqCell.preprocessing(F.relu(adj))
        adj_out = PDG2SeqCell.preprocessing(F.relu(-adj.transpose(-2, -1)))

        if self.use_meta_reliable_graph:
            h_diff = self._forward_meta_reliable(
                x_diff, state, node_embeddings, input_and_state, adj_in, adj_out,
                periodic_context=periodic_context, context_valid=context_valid
            )
            if self.use_signal_decouple:
                return self._fuse_decoupled_state(h_diff, x_inh, state, time_context, static_embedding)
            return h_diff

        adj_in, adj_out = self._maybe_refine_adjs(
            adj_in, adj_out, input_and_state, x_diff, periodic_context=periodic_context, context_valid=context_valid
        )
        adj = [adj_in, adj_out]

        z_r = torch.sigmoid(self.gate(input_and_state, adj, node_embeddings[2]))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x_diff, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, adj, node_embeddings[2]))
        h_diff = r * state + (1 - r) * hc
        if self.use_signal_decouple:
            return self._fuse_decoupled_state(h_diff, x_inh, state, time_context, static_embedding)
        return h_diff

    def _decouple_signal(self, x, state, node_embeddings, periodic_context=None, context_valid=None):
        batch_size = x.shape[0]
        time_context = 0.5 * (node_embeddings[0] + node_embeddings[1])
        if time_context.dim() == 2:
            time_context = time_context.unsqueeze(1).expand(-1, self.node_num, -1)
        static_embedding = node_embeddings[2].unsqueeze(0).expand(batch_size, -1, -1)
        if periodic_context is None:
            periodic_signal = torch.zeros_like(x)
        else:
            periodic_signal = self.periodic_proj(periodic_context)
        if context_valid is None:
            context_flag = x.new_zeros(x.shape[0], x.shape[1], 1)
        else:
            context_flag = context_valid

        gate_input = torch.cat(
            (x, state, time_context, static_embedding, periodic_signal * context_flag, context_flag),
            dim=-1
        )
        diffusion_gate = torch.sigmoid(self.signal_gate(gate_input))
        x_diff = diffusion_gate * x
        x_inh = (1.0 - diffusion_gate) * x

        if self._decouple_debug_count < 3:
            print(
                '[Signal Decouple Debug] '
                f'gate mean/min/max={diffusion_gate.mean().item():.6f}/{diffusion_gate.min().item():.6f}/{diffusion_gate.max().item():.6f}, '
                f'x_diff mean={x_diff.mean().item():.6f}, x_inh mean={x_inh.mean().item():.6f}'
            )
            self._decouple_debug_count += 1

        return x_diff, x_inh, time_context, static_embedding

    def _fuse_decoupled_state(self, h_diff, x_inh, state, time_context, static_embedding):
        h_inh = self._run_inherent_branch(x_inh, state)
        fuse_input = torch.cat((h_diff, h_inh, time_context, static_embedding), dim=-1)
        fuse_gate = torch.sigmoid(self.fuse_gate(fuse_input))
        return h_diff + self.signal_inherent_scale * (1.0 - fuse_gate) * (h_inh - h_diff)

    def _run_inherent_branch(self, x_inh, state):
        batch_size, node_num, _ = x_inh.shape
        x_flat = x_inh.reshape(batch_size * node_num, self.input_dim)
        state_flat = state.reshape(batch_size * node_num, self.hidden_dim)
        h_inh = self.inherent_gru(x_flat, state_flat)
        return h_inh.view(batch_size, node_num, self.hidden_dim)

    def _forward_meta_reliable(self, x, state, node_embeddings, signal, adj_in, adj_out,
                               periodic_context=None, context_valid=None):
        state_emb, state_probs = self._encode_traffic_state(x, state, node_embeddings, periodic_context, context_valid)
        cand_in, cand_out, mode_weights = self._generate_state_candidate_graphs(adj_in, adj_out, state_emb)
        stable_in, stable_out, anomaly_in, anomaly_out, rel_stats = self._split_reliable_graphs(
            cand_in, cand_out, signal, x, periodic_context, context_valid
        )

        gate_order = self._order_weights(self.gate_order_head(state_emb))
        update_order = self._order_weights(self.update_order_head(state_emb))
        gate_scale = 0.2 * torch.tanh(self.gate_meta_scale(state_emb))
        gate_bias = 0.1 * torch.tanh(self.gate_meta_bias(state_emb))
        update_scale = 0.2 * torch.tanh(self.update_meta_scale(state_emb))
        update_bias = 0.1 * torch.tanh(self.update_meta_bias(state_emb))

        gate_stable = self.gate(
            signal, [stable_in, stable_out], node_embeddings[2],
            order_weights=gate_order, meta_scale=gate_scale, meta_bias=gate_bias
        )
        gate_anomaly = self.gate(
            signal, [anomaly_in, anomaly_out], node_embeddings[2],
            order_weights=gate_order, meta_scale=gate_scale, meta_bias=gate_bias
        )
        gate_mix = torch.sigmoid(self.gate_channel_head(state_emb))
        z_r = torch.sigmoid(gate_mix * gate_stable + (1.0 - gate_mix) * gate_anomaly)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)

        candidate = torch.cat((x, z * state), dim=-1)
        update_stable = self.update(
            candidate, [stable_in, stable_out], node_embeddings[2],
            order_weights=update_order, meta_scale=update_scale, meta_bias=update_bias
        )
        update_anomaly = self.update(
            candidate, [anomaly_in, anomaly_out], node_embeddings[2],
            order_weights=update_order, meta_scale=update_scale, meta_bias=update_bias
        )
        update_mix = torch.sigmoid(self.update_channel_head(state_emb))
        hc = torch.tanh(update_mix * update_stable + (1.0 - update_mix) * update_anomaly)
        h = r * state + (1.0 - r) * hc

        if self._meta_debug_count < 3:
            print(
                '[MetaReliableGraph Debug] '
                f'state_probs={state_probs.mean(dim=(0, 1)).detach().cpu().tolist()}, '
                f'mode_entropy={self._entropy(mode_weights).mean().item():.6f}, '
                f'stable_rel={rel_stats[0].mean().item():.6f}, '
                f'anomaly_rel={rel_stats[1].mean().item():.6f}, '
                f'noise_rel={rel_stats[2].mean().item():.6f}, '
                f'gate_order={gate_order.mean(dim=(0, 1)).detach().cpu().tolist()}'
            )
            self._meta_debug_count += 1

        return h

    def _encode_traffic_state(self, x, state, node_embeddings, periodic_context=None, context_valid=None):
        batch_size = x.shape[0]
        time_context = 0.5 * (node_embeddings[0] + node_embeddings[1])
        if time_context.dim() == 2:
            time_context = time_context.unsqueeze(1).expand(-1, self.node_num, -1)
        static_embedding = node_embeddings[2].unsqueeze(0).expand(batch_size, -1, -1)
        if periodic_context is None:
            periodic_context = torch.zeros_like(x)
        if context_valid is None:
            context_valid = x.new_zeros(x.shape[0], x.shape[1], 1)
        mismatch = torch.abs(x - periodic_context) * context_valid
        state_input = torch.cat(
            (x, state, time_context, static_embedding, periodic_context * context_valid, mismatch, context_valid),
            dim=-1
        )
        state_emb = self.traffic_state_encoder(state_input)
        state_probs = F.softmax(self.traffic_state_head(state_emb), dim=-1)
        return state_emb, state_probs

    def _generate_state_candidate_graphs(self, adj_in, adj_out, state_emb):
        mode_weights = F.softmax(self.graph_mode_head(state_emb), dim=-1)
        mode_in = self._mode_graph(self.mode_src_in, self.mode_dst_in)
        mode_out = self._mode_graph(self.mode_src_out, self.mode_dst_out)
        adaptive_in = torch.einsum('bnm,mnk->bnk', mode_weights, mode_in)
        adaptive_out = torch.einsum('bnm,mnk->bnk', mode_weights, mode_out)
        cand_in = self._normalize_graph((1.0 - self.meta_graph_alpha) * adj_in + self.meta_graph_alpha * adaptive_in)
        cand_out = self._normalize_graph((1.0 - self.meta_graph_alpha) * adj_out + self.meta_graph_alpha * adaptive_out)
        return cand_in, cand_out, mode_weights

    def _split_reliable_graphs(self, adj_in, adj_out, signal, x, periodic_context=None, context_valid=None):
        if self.use_dgq:
            q_in = self._compute_dgq_quality(signal, self.dgq_src_in, self.dgq_dst_in)
            q_out = self._compute_dgq_quality(signal, self.dgq_src_out, self.dgq_dst_out)
        else:
            q_in = torch.ones_like(adj_in)
            q_out = torch.ones_like(adj_out)

        if periodic_context is None or context_valid is None:
            stable_rel = 0.5 * (q_in + q_out)
            anomaly_rel = 1.0 - stable_rel
            noise_rel = torch.zeros_like(stable_rel)
        else:
            cur_proto = F.normalize(self.current_proto_proj(x), dim=-1, eps=1.0e-8)
            per_proto = F.normalize(self.periodic_proto_proj(periodic_context), dim=-1, eps=1.0e-8)
            node_consistency = torch.clamp((torch.sum(cur_proto * per_proto, dim=-1) + 1.0) / 2.0, 0.0, 1.0)
            valid = context_valid.squeeze(-1)
            edge_valid = valid.unsqueeze(-1) * valid.unsqueeze(-2)
            consistency = 0.5 * (node_consistency.unsqueeze(-1) + node_consistency.unsqueeze(-2))
            support = 0.5 * (adj_in + adj_out.transpose(-1, -2))
            dgq = 0.5 * (q_in + q_out)
            stable_rel = edge_valid * dgq * (self.meta_stable_lambda * consistency + (1.0 - self.meta_stable_lambda) * support)
            anomaly_rel = edge_valid * dgq * (1.0 - consistency) * (self.meta_anomaly_lambda + (1.0 - self.meta_anomaly_lambda) * support)
            noise_rel = torch.clamp(1.0 - dgq - support, 0.0, 1.0)
            stable_rel = torch.where(edge_valid > 0.0, stable_rel, dgq)

        stable_in = self._normalize_graph(adj_in * torch.clamp(stable_rel, min=self.meta_noise_floor))
        stable_out = self._normalize_graph(adj_out * torch.clamp(stable_rel, min=self.meta_noise_floor))
        anomaly_in = self._normalize_graph(adj_in * torch.clamp(anomaly_rel, min=self.meta_noise_floor * 0.5))
        anomaly_out = self._normalize_graph(adj_out * torch.clamp(anomaly_rel, min=self.meta_noise_floor * 0.5))
        return stable_in, stable_out, anomaly_in, anomaly_out, (stable_rel, anomaly_rel, noise_rel)

    def _mode_graph(self, src, dst):
        score = torch.matmul(src, dst.transpose(-1, -2)) / math.sqrt(float(self.dgq_dim))
        score = score + torch.eye(self.node_num, device=score.device).unsqueeze(0)
        return F.softmax(score, dim=-1)

    def _order_weights(self, logits):
        weights = F.softmax(logits, dim=-1)
        return weights * weights.shape[-1]

    @staticmethod
    def _entropy(weights):
        return -(weights * torch.log(weights.clamp_min(1.0e-8))).sum(dim=-1)

    @staticmethod
    def _normalize_graph(adj):
        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        return adj / adj.sum(-1, keepdim=True).clamp_min(1.0e-8)

    def _maybe_refine_adjs(self, adj_in, adj_out, signal, x, periodic_context=None, context_valid=None):
        if not self.use_dgq and not self.use_context_graph_refine:
            return adj_in, adj_out

        if self.use_dgq:
            q_in_dgq = self._compute_dgq_quality(signal, self.dgq_src_in, self.dgq_dst_in)
            q_out_dgq = self._compute_dgq_quality(signal, self.dgq_src_out, self.dgq_dst_out)
        else:
            q_in_dgq = torch.ones_like(adj_in)
            q_out_dgq = torch.ones_like(adj_out)
        q_in_final = q_in_dgq
        q_out_final = q_out_dgq

        if self.use_context_graph_refine and periodic_context is not None and context_valid is not None:
            q_in_final, q_out_final = self._apply_context_refine(
                q_in_dgq, q_out_dgq, x, periodic_context, context_valid
            )

        refined_in = self._apply_residual_gate(adj_in, q_in_final)
        refined_out = self._apply_residual_gate(adj_out, q_out_final)

        if self._dgq_debug_count < 3:
            delta_in = torch.mean(torch.abs(refined_in - adj_in)).detach().cpu().item()
            delta_out = torch.mean(torch.abs(refined_out - adj_out)).detach().cpu().item()
            max_delta_in = torch.max(torch.abs(refined_in - adj_in)).detach().cpu().item()
            max_delta_out = torch.max(torch.abs(refined_out - adj_out)).detach().cpu().item()
            has_invalid = (not torch.isfinite(refined_in).all().item()) or (not torch.isfinite(refined_out).all().item())
            print(
                '[DGQ Debug] '
                f'Q_in mean/min/max={q_in_final.mean().item():.6f}/{q_in_final.min().item():.6f}/{q_in_final.max().item():.6f}, '
                f'Q_out mean/min/max={q_out_final.mean().item():.6f}/{q_out_final.min().item():.6f}/{q_out_final.max().item():.6f}, '
                f'mean|A_in_refined-A_in|={delta_in:.6e}, max|A_in_refined-A_in|={max_delta_in:.6e}, '
                f'mean|A_out_refined-A_out|={delta_out:.6e}, max|A_out_refined-A_out|={max_delta_out:.6e}, '
                f'A_refined_has_nan_or_inf={has_invalid}'
            )
            self._dgq_debug_count += 1

        return refined_in, refined_out

    def _compute_dgq_quality(self, signal, src_proj, dst_proj):
        src = src_proj(signal)
        dst = dst_proj(signal)
        score = torch.matmul(src, dst.transpose(-1, -2)) / math.sqrt(float(self.dgq_dim))
        quality = torch.sigmoid(score)
        return torch.nan_to_num(quality, nan=0.5, posinf=1.0, neginf=0.0)

    def _apply_context_refine(self, q_in_dgq, q_out_dgq, x, periodic_context, context_valid):
        cur_emb = F.normalize(self.context_cur_proj(x), dim=-1, eps=1.0e-8)
        ctx_emb = F.normalize(self.context_ctx_proj(periodic_context), dim=-1, eps=1.0e-8)
        r = torch.sum(cur_emb * ctx_emb, dim=-1)
        r = torch.clamp((r + 1.0) / 2.0, 0.0, 1.0)
        reliability = (r.unsqueeze(-1) + r.unsqueeze(-2)) / 2.0

        valid = context_valid.squeeze(-1)
        edge_valid = valid.unsqueeze(-1) * valid.unsqueeze(-2)
        offset = self.context_graph_lambda * edge_valid * (reliability - 0.5)

        q_in_final = torch.clamp(q_in_dgq + offset, 0.0, 1.0)
        q_out_final = torch.clamp(q_out_dgq + offset, 0.0, 1.0)
        q_in_final = torch.nan_to_num(q_in_final, nan=0.5, posinf=1.0, neginf=0.0)
        q_out_final = torch.nan_to_num(q_out_final, nan=0.5, posinf=1.0, neginf=0.0)

        if self._context_debug_count < 3:
            invalid_mask = edge_valid == 0
            if invalid_mask.any():
                invalid_diff = torch.max(torch.abs(q_in_final[invalid_mask] - q_in_dgq[invalid_mask])).detach().cpu().item()
            else:
                invalid_diff = 0.0
            has_invalid = (not torch.isfinite(q_in_final).all().item()) or (not torch.isfinite(q_out_final).all().item())
            print(
                '[Context Graph Debug] '
                f'r mean/min/max={r.mean().item():.6f}/{r.min().item():.6f}/{r.max().item():.6f}, '
                f'R mean/min/max={reliability.mean().item():.6f}/{reliability.min().item():.6f}/{reliability.max().item():.6f}, '
                f'valid_mask mean/min/max={edge_valid.mean().item():.6f}/{edge_valid.min().item():.6f}/{edge_valid.max().item():.6f}, '
                f'mean|Q_final-Q_dgq|={torch.mean(torch.abs(q_in_final - q_in_dgq)).item():.6f}, '
                f'context_valid_zero_degenerates={invalid_diff:.6f}, '
                f'Q_final_has_nan_or_inf={has_invalid}'
            )
            self._context_debug_count += 1

        return q_in_final, q_out_final

    def _apply_residual_gate(self, adj, quality):
        scale = 1.0 + self.dgq_alpha * (2.0 * quality - 1.0)
        refined = adj * scale.clamp_min(1.0e-4)
        refined = torch.nan_to_num(refined, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = refined.sum(-1, keepdim=True).clamp_min(1.0e-8)
        return refined / row_sum

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

    @staticmethod
    def preprocessing(adj):
        num_nodes = adj.shape[-1]
        adj = adj + torch.eye(num_nodes, device=adj.device)
        degree = torch.unsqueeze(adj.sum(-1), -1)
        adj = adj / degree.clamp_min(1.0e-8)
        return adj
