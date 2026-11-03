"""Load the frozen innovation model with its recorded training configuration."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from lib.dataloader import get_dataloader
from lib.enhanced_eval import collect
from model.PDG2Seq import PDG2Seq


def collect_test(checkpoint, output):
    return collect_split(checkpoint, output, 'test')


def collect_split(checkpoint, output, split):
    checkpoint, output = Path(checkpoint), Path(output)
    # Logger persists the exact arguments beside the checkpoint.
    logs = list(checkpoint.parent.glob('*.log'))
    if not logs:
        root_log = Path(__file__).resolve().parents[1] / 'logs' / (checkpoint.parent.name + '.log')
        if root_log.is_file():
            logs = [root_log]
    if not logs:
        raise FileNotFoundError('No recorded training configuration beside checkpoint')
    args = None
    for line in logs[0].read_text().splitlines():
        if 'Namespace(' in line:
            expression = 'Namespace(' + line.split('Namespace(', 1)[1]
            args = SimpleNamespace(**{k.arg: ast.literal_eval(k.value)
                                      for k in ast.parse(expression, mode='eval').body.keywords})
            break
    if args is None:
        raise ValueError('No training arguments found')
    args.use_periodic_consistency = True
    args.use_decoder_periodic_context = False
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.batch_size = 32
    model = PDG2Seq(args).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    teacher_args = copy.copy(args)
    for flag in ('use_dgq', 'use_periodic_context', 'use_context_graph_refine',
                 'use_signal_decouple', 'use_graph_quality_decouple', 'use_periodic_edge_quality',
                 'use_periodic_residual_decouple', 'use_decoder_periodic_context'):
        setattr(teacher_args, flag, False)
    teacher = PDG2Seq(teacher_args).to(args.device)
    teacher.load_state_dict(torch.load('pre-trained/PEMSD4.pth', map_location=args.device, weights_only=True))
    train_loader, _, test_loader, scaler = get_dataloader(args, normalizer=args.normalizer,
                                         tod=args.tod, dow=False, weather=False, single=False)
    if split == 'train':
        loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=32,
                                             shuffle=False, drop_last=False)
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError('split must be train or test')
    trainer = SimpleNamespace(model=model, scaler=scaler)
    values = collect(trainer, teacher, loader)
    np.savez(output / f'base_{split}_predictions.npz', **{f'a{i}': a for i, a in enumerate(values)})
    (output / 'model_config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    return values
