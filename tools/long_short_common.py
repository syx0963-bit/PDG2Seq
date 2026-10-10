import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lib.long_history import split_origins
from lib.metrics import All_Metrics


BASELINE = dict(rmse=28.1697, mae=17.0095, mape=11.4857)
BASELINE_H12 = dict(rmse=28.4734, mae=17.3106, mape=11.7058)
SIGNAL_CHECKPOINT = 'experiments/PEMSD4/newinnovation-1-2_DGQ-DGQEnsemble-PeriodicContext-ContextGraphRefine-SignalDecouple-DecoderPeriodicContext-MetaReliableGraph-PeriodicConsistency_20260804183437/best_model.pth'


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def traffic_data(device='cpu'):
    flow = np.load('data/PeMS04/PEMS04.npz')['data'][:, :, 0].astype(np.float32)
    splits, train_raw_end = split_origins(len(flow))
    mean, std = float(flow[:train_raw_end].mean()), float(flow[:train_raw_end].std())
    raw = torch.from_numpy(flow).to(device)
    normalized = (raw-mean)/std
    return raw, normalized, splits, dict(mean=mean, std=std, train_raw_end=train_raw_end)


def metric(pred, true):
    mae, rmse, mape, _, _ = All_Metrics(pred, true, None, 0.0)
    return dict(rmse=float(rmse), mae=float(mae), mape=float(mape)*100)


def model_args(cli, nodes):
    return SimpleNamespace(dataset='PEMSD4', mode=cli.mode, model='PDG2Seq', device=cli.device,
        num_nodes=nodes, input_dim=1, output_dim=1, rnn_units=64, num_layers=1,
        time_dim=16, embed_dim=8, cheb_k=1, lag=12, horizon=12,
        use_day=True, use_week=True, steps_per_day=288, steps_per_week=7,
        use_dgq=True, dgq_alpha=0.1, dgq_dim=16, use_context_graph_refine=True,
        context_graph_lambda=0.05, context_graph_dim=16, use_periodic_context=True,
        use_signal_decouple=True, signal_decouple_hidden=32, signal_diffusion_bias=1.8,
        signal_fuse_diffusion_bias=1.2, signal_inherent_scale=0.2,
        use_decoder_periodic_context=True, use_periodic_consistency=True,
        periodic_day_steps=288, periodic_week_steps=2016, context_temperature=1.0,
        lr_decay_step=1500, use_long_short_learning=True,
        long_history_steps=cli.history_steps, long_patch_size=cli.patch_size,
        long_prior_width=cli.prior_width, long_prior_layers=cli.prior_layers,
        long_forecast_width=cli.width, long_dropout=cli.dropout,
        long_short_attention_layers=cli.attention_layers,
        long_phase_memory=cli.phase_memory)
