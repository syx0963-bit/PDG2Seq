import torch
import torch.nn as nn
from model.PDG2SeqCell import PDG2SeqCell
import numpy as np


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
            self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, args=args))

    def forward(self, xt, init_state, node_embeddings):
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.PDG2Seq_cells[i](current_inputs, init_state[i], [node_embeddings[0], node_embeddings[1], node_embeddings[2]])
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
        self.cl_decay_steps = args.lr_decay_step
        self.node_embeddings1 = nn.Parameter(torch.empty(self.num_node, args.embed_dim))
        self.T_i_D_emb1 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb1 = nn.Parameter(torch.empty(7, args.time_dim))
        self.T_i_D_emb2 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb2 = nn.Parameter(torch.empty(7, args.time_dim))

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

    def forward(self, source, traget=None, batches_seen=None):
        t_i_d_data1 = source[..., 0, -2]
        t_i_d_data2 = traget[..., 0, -2]
        t_i_d_idx1 = (t_i_d_data1 * 288).long().to(source.device)
        t_i_d_idx2 = (t_i_d_data2 * 288).long().to(traget.device)
        T_i_D_emb1_en = self.T_i_D_emb1[t_i_d_idx1]
        T_i_D_emb2_en = self.T_i_D_emb2[t_i_d_idx1]

        T_i_D_emb1_de = self.T_i_D_emb1[t_i_d_idx2]
        T_i_D_emb2_de = self.T_i_D_emb2[t_i_d_idx2]
        if self.use_W:
            d_i_w_data1 = source[..., 0, -1]
            d_i_w_data2 = traget[..., 0, -1]
            d_i_w_idx1 = d_i_w_data1.long().to(source.device)
            d_i_w_idx2 = d_i_w_data2.long().to(traget.device)
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

        if self.use_context_graph_refine:
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

        ht_list = [state] * self.num_layers

        go = torch.zeros((source_traffic.shape[0], self.num_node, self.output_dim), device=source_traffic.device)
        out = []
        for t in range(self.horizon):
            state, ht_list = self.decoder(
                go,
                ht_list,
                [node_embedding_de1[:, t, :], node_embedding_de2[:, t, :], self.node_embeddings1]
            )
            go = self.proj(state)
            out.append(go)
            if self.training:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    go = traget[:, t, :, 0].unsqueeze(-1)
        output = torch.stack(out, dim=1)

        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
        return x
