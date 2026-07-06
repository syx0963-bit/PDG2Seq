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
        self.hidden_dim = dim_out
        self.use_dgq = getattr(args, 'use_dgq', False) if args is not None else False
        self.dgq_alpha = getattr(args, 'dgq_alpha', 0.1) if args is not None else 0.1
        self.dgq_dim = getattr(args, 'dgq_dim', 16) if args is not None else 16
        self.use_context_graph_refine = getattr(args, 'use_context_graph_refine', False) if args is not None else False
        self.context_graph_lambda = getattr(args, 'context_graph_lambda', 0.05) if args is not None else 0.05
        self.context_graph_dim = getattr(args, 'context_graph_dim', 16) if args is not None else 16
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

        self._dgq_debug_count = 0
        self._context_debug_count = 0

    def forward(self, x, state, node_embeddings, periodic_context=None, context_valid=None):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        filter1 = self.fc1(input_and_state)
        filter2 = self.fc2(input_and_state)

        nodevec1 = torch.tanh(torch.einsum('bd,bnd->bnd', node_embeddings[0], filter1))
        nodevec2 = torch.tanh(torch.einsum('bd,bnd->bnd', node_embeddings[1], filter2))

        adj = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1))

        adj_in = PDG2SeqCell.preprocessing(F.relu(adj))
        adj_out = PDG2SeqCell.preprocessing(F.relu(-adj.transpose(-2, -1)))

        adj_in, adj_out = self._maybe_refine_adjs(
            adj_in, adj_out, input_and_state, x, periodic_context=periodic_context, context_valid=context_valid
        )
        adj = [adj_in, adj_out]

        z_r = torch.sigmoid(self.gate(input_and_state, adj, node_embeddings[2]))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, adj, node_embeddings[2]))
        h = r * state + (1 - r) * hc
        return h

    def _maybe_refine_adjs(self, adj_in, adj_out, signal, x, periodic_context=None, context_valid=None):
        if not self.use_dgq:
            return adj_in, adj_out

        q_in_dgq = self._compute_dgq_quality(signal, self.dgq_src_in, self.dgq_dst_in)
        q_out_dgq = self._compute_dgq_quality(signal, self.dgq_src_out, self.dgq_dst_out)
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
            has_invalid = (not torch.isfinite(refined_in).all().item()) or (not torch.isfinite(refined_out).all().item())
            print(
                '[DGQ Debug] '
                f'Q_in mean/min/max={q_in_final.mean().item():.6f}/{q_in_final.min().item():.6f}/{q_in_final.max().item():.6f}, '
                f'Q_out mean/min/max={q_out_final.mean().item():.6f}/{q_out_final.min().item():.6f}/{q_out_final.max().item():.6f}, '
                f'mean|A_in_refined-A_in|={delta_in:.6f}, mean|A_out_refined-A_out|={delta_out:.6f}, '
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
        refined = adj * ((1.0 - self.dgq_alpha) + self.dgq_alpha * quality)
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
