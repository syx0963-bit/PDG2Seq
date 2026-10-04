import torch
import torch.nn as nn
from model.PDG2SeqCell import PDG2SeqCell
import numpy as np


class LongShortMultiScaleFusion(nn.Module):
    def __init__(self, hidden_dim, horizon, output_dim, residual_scale=0.1):
        super(LongShortMultiScaleFusion, self).__init__()
        self.horizon = horizon
        self.residual_scale = residual_scale
        self.short_encoder = nn.Linear(1, hidden_dim)
        self.trend_encoder = nn.Linear(1, hidden_dim)
        self.periodic_encoder = nn.Linear(1, hidden_dim)
        self.long_encoder = nn.Linear(1, hidden_dim)
        self.horizon_embedding = nn.Parameter(torch.empty(horizon, hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        self.state_fusion = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, source_traffic, base_state, periodic_context=None, context_valid=None):
        short_signal = source_traffic[:, -1, :, :]
        history_mean = source_traffic.mean(dim=1)
        early_mean = source_traffic[:, :max(source_traffic.shape[1] // 2, 1), :, :].mean(dim=1)
        trend_signal = source_traffic[:, -1, :, :] - early_mean

        if periodic_context is not None:
            if context_valid is None:
                context_weight = torch.ones_like(periodic_context)
            else:
                context_weight = context_valid.expand_as(periodic_context)
            valid_count = context_weight.sum(dim=1).clamp_min(1.0)
            long_signal = (periodic_context * context_weight).sum(dim=1) / valid_count
            periodic_signal = periodic_context[:, -1, :, :]
            if context_valid is not None:
                last_valid = context_valid[:, -1, :, :]
                periodic_signal = last_valid * periodic_signal + (1.0 - last_valid) * history_mean
        else:
            long_signal = history_mean
            periodic_signal = history_mean

        scale_repr = torch.stack((
            self.short_encoder(short_signal),
            self.trend_encoder(trend_signal),
            self.periodic_encoder(periodic_signal),
            self.long_encoder(long_signal)
        ), dim=1)

        horizon_state = base_state.unsqueeze(1).expand(-1, self.horizon, -1, -1)
        horizon_emb = self.horizon_embedding.view(1, self.horizon, 1, -1).expand_as(horizon_state)
        gate_logits = self.gate(torch.cat((horizon_state, horizon_emb), dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        fused = torch.sum(gate.permute(0, 1, 3, 2).unsqueeze(-1) * scale_repr.unsqueeze(1), dim=2)
        state_delta = self.state_fusion(fused[:, 0])
        residual = self.residual_scale * self.out_proj(fused)
        return state_delta, residual, gate


class PDG2Seq_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1, args=None):
        super(PDG2Seq_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PDG2Seq_cells = nn.ModuleList()
        self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, args=args))
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim, args=args))

    def forward(self, x, init_state, node_embeddings, periodic_context=None, context_valid=None):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                periodic_context_t = None if periodic_context is None else periodic_context[:, t, :, :]
                context_valid_t = None if context_valid is None else context_valid[:, t, :, :]
                state = self.PDG2Seq_cells[i](
                    current_inputs[:, t, :, :],
                    state,
                    [node_embeddings[0][:, t, :], node_embeddings[1][:, t, :], node_embeddings[2]],
                    periodic_context=periodic_context_t,
                    context_valid=context_valid_t
                )
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.PDG2Seq_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class PDG2Seq_Dncoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1, args=None):
        super(PDG2Seq_Dncoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PDG2Seq_cells = nn.ModuleList()
        self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, args=args))
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim, args=args))

    def forward(self, xt, init_state, node_embeddings, periodic_context=None, context_valid=None):
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.PDG2Seq_cells[i](
                current_inputs,
                init_state[i],
                [node_embeddings[0], node_embeddings[1], node_embeddings[2]],
                periodic_context=periodic_context,
                context_valid=context_valid
            )
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class PDG2Seq(nn.Module):
    def __init__(self, args):
        super(PDG2Seq, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.use_context_graph_refine = args.use_context_graph_refine
        self.use_meta_reliable_graph = getattr(args, 'use_meta_reliable_graph', False)
        self.use_reliable_invariant_learning = getattr(args, 'use_reliable_invariant_learning', False)
        self.use_long_short_multiscale = getattr(args, 'use_long_short_multiscale', False)
        self.use_periodic_graph_context = args.use_periodic_context and (
            args.use_context_graph_refine or self.use_meta_reliable_graph or self.use_long_short_multiscale
        )
        self.use_periodic_consistency = getattr(args, 'use_periodic_consistency', False)
        self.use_decoder_periodic_context = getattr(args, 'use_decoder_periodic_context', False)
        self.steps_per_day = args.steps_per_day
        self.steps_per_week = args.steps_per_week
        self.cl_decay_steps = args.lr_decay_step
        self.node_embeddings1 = nn.Parameter(torch.empty(self.num_node, args.embed_dim))
        self.T_i_D_emb1 = nn.Parameter(torch.empty(self.steps_per_day, args.time_dim))
        self.D_i_W_emb1 = nn.Parameter(torch.empty(self.steps_per_week, args.time_dim))
        self.T_i_D_emb2 = nn.Parameter(torch.empty(self.steps_per_day, args.time_dim))
        self.D_i_W_emb2 = nn.Parameter(torch.empty(self.steps_per_week, args.time_dim))

        self.encoder = PDG2Seq_Encoder(
            args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
            args.embed_dim, args.time_dim, args.num_layers, args=args
        )
        self.decoder = PDG2Seq_Dncoder(
            args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
            args.embed_dim, args.time_dim, args.num_layers, args=args
        )
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        if self.use_reliable_invariant_learning:
            invariant_dim = getattr(args, 'invariant_repr_dim', args.rnn_units)
            env_dim = getattr(args, 'env_repr_dim', args.rnn_units)
            self.invariant_encoder = nn.Sequential(
                nn.Linear(self.hidden_dim, invariant_dim),
                nn.ReLU(),
                nn.Linear(invariant_dim, self.hidden_dim)
            )
            self.env_encoder = nn.Sequential(
                nn.Linear(self.hidden_dim, env_dim),
                nn.ReLU(),
                nn.Linear(env_dim, self.hidden_dim)
            )
            self.repr_fusion = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        else:
            self.invariant_encoder = None
            self.env_encoder = None
            self.repr_fusion = None
        if self.use_long_short_multiscale:
            self.long_short_fusion = LongShortMultiScaleFusion(
                self.hidden_dim,
                self.horizon,
                self.output_dim,
                residual_scale=float(getattr(args, 'long_short_residual_scale', 0.1))
            )
        else:
            self.long_short_fusion = None

    def forward(self, source, traget=None, batches_seen=None, return_representation=False):
        t_i_d_data1 = source[..., 0, -2]
        t_i_d_data2 = traget[..., 0, -2]
        t_i_d_idx1 = torch.clamp((t_i_d_data1 * self.steps_per_day).long(), 0, self.steps_per_day - 1).to(source.device)
        t_i_d_idx2 = torch.clamp((t_i_d_data2 * self.steps_per_day).long(), 0, self.steps_per_day - 1).to(traget.device)
        T_i_D_emb1_en = self.T_i_D_emb1[t_i_d_idx1]
        T_i_D_emb2_en = self.T_i_D_emb2[t_i_d_idx1]

        T_i_D_emb1_de = self.T_i_D_emb1[t_i_d_idx2]
        T_i_D_emb2_de = self.T_i_D_emb2[t_i_d_idx2]
        if self.use_W:
            d_i_w_data1 = source[..., 0, -1]
            d_i_w_data2 = traget[..., 0, -1]
            d_i_w_idx1 = torch.clamp(d_i_w_data1.long(), 0, self.steps_per_week - 1).to(source.device)
            d_i_w_idx2 = torch.clamp(d_i_w_data2.long(), 0, self.steps_per_week - 1).to(traget.device)
            D_i_W_emb1_en = self.D_i_W_emb1[d_i_w_idx1]
            D_i_W_emb2_en = self.D_i_W_emb2[d_i_w_idx1]

            D_i_W_emb1_de = self.D_i_W_emb1[d_i_w_idx2]
            D_i_W_emb2_de = self.D_i_W_emb2[d_i_w_idx2]

            node_embedding_en1 = torch.mul(T_i_D_emb1_en, D_i_W_emb1_en)
            node_embedding_en2 = torch.mul(T_i_D_emb2_en, D_i_W_emb2_en)

            node_embedding_de1 = torch.mul(T_i_D_emb1_de, D_i_W_emb1_de)
            node_embedding_de2 = torch.mul(T_i_D_emb2_de, D_i_W_emb2_de)
        else:
            node_embedding_en1 = T_i_D_emb1_en
            node_embedding_en2 = T_i_D_emb2_en

            node_embedding_de1 = T_i_D_emb1_de
            node_embedding_de2 = T_i_D_emb2_de

        en_node_embeddings = [node_embedding_en1, node_embedding_en2, self.node_embeddings1]

        if self.use_periodic_graph_context:
            source_traffic = source[..., 0:1]
            periodic_context = source[..., 1:2]
            context_valid = source[..., 2:3]
        else:
            source_traffic = source[..., 0:1]
            periodic_context = None
            context_valid = None

        init_state = self.encoder.init_hidden(source_traffic.shape[0]).to(source_traffic.device)
        state, _ = self.encoder(
            source_traffic,
            init_state,
            en_node_embeddings,
            periodic_context=periodic_context,
            context_valid=context_valid
        )
        state = state[:, -1:, :, :].squeeze(1)
        raw_state = state
        invariant_state = None
        env_state = None
        if self.use_reliable_invariant_learning:
            invariant_state = self.invariant_encoder(raw_state)
            env_state = self.env_encoder(raw_state)
            state = self.repr_fusion(torch.cat((invariant_state, env_state), dim=-1))
        multiscale_residual = None
        multiscale_gate = None
        if self.use_long_short_multiscale:
            state_delta, multiscale_residual, multiscale_gate = self.long_short_fusion(
                source_traffic,
                state,
                periodic_context=periodic_context,
                context_valid=context_valid
            )
            state = state + state_delta

        ht_list = [state] * self.num_layers

        go = torch.zeros((source_traffic.shape[0], self.num_node, self.output_dim), device=source_traffic.device)
        out = []
        for t in range(self.horizon):
            decoder_periodic_context = None
            decoder_context_valid = None
            if self.use_decoder_periodic_context and self.use_periodic_consistency:
                decoder_periodic_context = traget[:, t, :, 1:2]
                decoder_context_valid = traget[:, t, :, 2:3]
            state, ht_list = self.decoder(
                go,
                ht_list,
                [node_embedding_de1[:, t, :], node_embedding_de2[:, t, :], self.node_embeddings1],
                periodic_context=decoder_periodic_context,
                context_valid=decoder_context_valid
            )
            go = self.proj(state)
            out.append(go)
            if self.training:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    go = traget[:, t, :, 0].unsqueeze(-1)
        output = torch.stack(out, dim=1)
        if multiscale_residual is not None:
            output = output + multiscale_residual

        if return_representation:
            if invariant_state is None:
                invariant_state = raw_state
                env_state = torch.zeros_like(raw_state)
            return output, {
                'raw': raw_state,
                'invariant': invariant_state,
                'environment': env_state,
                'long_short_gate': multiscale_gate
            }
        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
        return x

    def get_latest_reliable_graph(self):
        graphs = []
        for module in list(self.encoder.PDG2Seq_cells) + list(self.decoder.PDG2Seq_cells):
            graph = getattr(module, 'latest_reliable_graph', None)
            if graph is not None:
                graphs.append(graph)
        if not graphs:
            return None
        graph = torch.stack(graphs, dim=0).mean(dim=0)
        graph = torch.relu(0.5 * (graph + graph.transpose(0, 1)))
        graph = graph + torch.eye(self.num_node, device=graph.device)
        return graph / graph.sum(-1, keepdim=True).clamp_min(1.0e-8)
