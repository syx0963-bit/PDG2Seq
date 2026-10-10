"""Freeze fusion using validation only; never score or select using test."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_long_short import select_validation_fusion
from tools.long_short_common import traffic_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    cli = parser.parse_args()
    directory = Path(cli.model_dir)
    model = torch.load(directory/'best_model.pth', map_location='cpu', weights_only=False)
    cache_path = model['training'].get('context_cache') or directory/'frozen_context.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    _, normalized, splits, stats = traffic_data(cli.device)
    if stats != model['normalization']:
        raise ValueError('Normalization changed since training')
    select_validation_fusion(model['state_dict']['node_embeddings1'], cache, normalized,
                             splits, stats, directory)
    print('Fusion parameters frozen. Test metrics have not been read.', flush=True)


if __name__ == '__main__':
    main()
