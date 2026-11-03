"""Per-sensor multihorizon spatial residual ridge, frozen innovation backbone."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.dataloader import split_data_by_ratio
from lib.enhanced_eval import metrics
from tools.optimize_history_calibration import load_values
from tools.train_residual_head import matrix


def build(raw, starts, values, w, pw, neighbors):
    x, _, base, true = matrix(raw, starts, values, w, pw, neighbors)
    n, nodes = len(starts), raw.shape[1]
    x = x.reshape(n, nodes, -1)
    # Individual neighbors retain direction-specific signals lost by averaging.
    parts = [x]
    for offset in (0, 1, 2, 3, 5, 8, 11, 17, 23):
        at = np.maximum(starts + 11 - offset, 0)
        parts.append(raw[at[:, None, None], neighbors[None], 0])
    x = np.concatenate(parts, -1)
    return x, base.reshape(n, nodes, 12), true.reshape(n, nodes, 12)


def covariance(x, y, base, device):
    mean, std = x.mean(0), np.maximum(x.std(0), 0.01)
    z = torch.as_tensor(np.clip((x - mean) / std, -10, 10).transpose(1, 0, 2).copy(), device=device)
    z = torch.cat([z, torch.ones((*z.shape[:2], 1), device=device)], -1)
    target = torch.as_tensor(((y - base) / 100).transpose(1, 0, 2).copy(), device=device)
    xtx = z.transpose(1, 2) @ z / len(x)
    xty = z.transpose(1, 2) @ target / len(x)
    return mean, std, xtx, xty


def correction(x, mean, std, coefficients, device):
    result = []
    for begin in range(0, len(x), 512):
        z = torch.as_tensor(np.clip((x[begin:begin + 512] - mean) / std, -10, 10), device=device)
        z = torch.cat([z, torch.ones((*z.shape[:2], 1), device=device)], -1)
        result.append((100 * torch.einsum('tnf,nfh->tnh', z, coefficients)).cpu().numpy())
    return np.clip(np.concatenate(result), -100, 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--history-run', required=True)
    p.add_argument('--head-run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out, source = Path(args.output), Path(args.source)
    out.mkdir(parents=True, exist_ok=True)
    state = np.load(Path(args.history_run) / 'feature_state.npz')
    w, pw, neighbors = [state[k] for k in ('weights', 'periodic_weights', 'neighbors')]
    raw = np.load('data/PeMS04/PEMS04.npz')['data'].astype(np.float32)
    ts, vs, es = split_data_by_ratio(np.arange(len(raw) - 23), 0.2, 0.2)
    train = load_values(Path(args.head_run) / 'base_train_predictions.npz')
    val = load_values(source / 'validation_predictions.npz')
    print('Building node features', flush=True)
    tx, tb, ty = build(raw, ts, train, w, pw, neighbors)
    vx, vb, vy = build(raw, vs, val, w, pw, neighbors)
    cut, hold = len(vs) // 2, len(vs) // 2 + 12
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best = dict(rmse=metrics(vb[hold:], vy[hold:])['rmse'], mode='none', ridge=1, blends=[0.] * 12)
    for mode in ('train_and_validation', 'validation_only'):
        if mode == 'train_and_validation':
            x, y, base = [np.concatenate([a, b[:cut]]) for a, b in ((tx, vx), (ty, vy), (tb, vb))]
        else:
            x, y, base = vx[:cut], vy[:cut], vb[:cut]
        mean, std, xtx, xty = covariance(x, y, base, device)
        del x, y, base
        identity = torch.eye(xtx.shape[-1], device=device)
        for ridge in (0.001, 0.01, 0.1, 1.0, 10.0):
            c = torch.linalg.solve(xtx + ridge * identity, xty)
            delta = correction(vx[hold:], mean, std, c, device)
            blends, pred = [], vb[hold:].copy()
            for h in range(12):
                blend = min((0, .25, .5, .75, 1.), key=lambda b: np.mean(
                    (np.maximum(vb[hold:, :, h] + b * delta[:, :, h], 0) - vy[hold:, :, h]) ** 2))
                blends.append(blend)
                pred[:, :, h] = np.maximum(vb[hold:, :, h] + blend * delta[:, :, h], 0)
            m = metrics(pred, vy[hold:])
            print(json.dumps(dict(mode=mode, ridge=ridge, blends=blends, validation=m)), flush=True)
            if m['rmse'] < best['rmse']:
                best = dict(**m, mode=mode, ridge=ridge, blends=blends)
    print('Selected:', json.dumps(best), flush=True)
    if best['mode'] == 'train_and_validation':
        x, y, base = [np.concatenate([a, b]) for a, b in ((tx, vx), (ty, vy), (tb, vb))]
    else:
        x, y, base = vx, vy, vb
    mean, std, xtx, xty = covariance(x, y, base, device)
    c = torch.linalg.solve(xtx + best['ridge'] * torch.eye(xtx.shape[-1], device=device), xty)
    # Frozen validation selection is now complete; only then materialize test.
    test = load_values(Path(args.history_run) / 'base_test_predictions.npz')
    ex, eb, ey = build(raw, es, test, w, pw, neighbors)
    pred = np.maximum(eb + np.array(best['blends'])[None, None] * correction(ex, mean, std, c, device), 0)
    result = dict(selection=best, test=metrics(pred, ey), baseline_test=metrics(eb, ey),
                  innovations_unchanged=True, extra_historical_channels=True,
                  protocol='frozen model; validation RMSE selection; individual-neighbor histories')
    np.savez(out / 'spatial_ridge.npz', coefficients=c.cpu().numpy(), mean=mean, std=std,
             neighbors=neighbors, weights=w, periodic_weights=pw, blends=best['blends'])
    np.savez(out / 'test_predictions.npz', prediction=pred, target=ey)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('FINAL', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
