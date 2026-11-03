"""Dense network-history residual calibration of a frozen innovation model."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.dataloader import split_data_by_ratio
from lib.enhanced_eval import fuse, metrics
from tools.optimize_history_calibration import load_values


def features(raw, starts):
    origin = starts + 11
    chunks = [raw[np.maximum(origin - lag, 0), :, 0] for lag in (0, 1, 2, 5, 11, 23, 287)]
    chunks += [raw[origin, :, c] for c in (1, 2)]
    tod = (origin % 288) / 288
    chunks += [np.sin(2 * np.pi * tod)[:, None], np.cos(2 * np.pi * tod)[:, None],
               np.eye(7, dtype=np.float32)[origin // 288 % 7]]
    return np.concatenate(chunks, 1).astype(np.float32)


def fit_statistics(x, y, base, device):
    mean, std = x.mean(0), np.maximum(x.std(0), .01)
    z = np.clip((x - mean) / std, -10, 10)
    z = torch.tensor(np.concatenate([z, np.ones((len(x), 1), np.float32)], 1), device=device)
    residual = torch.tensor(((y - base) / 100).reshape(len(x), -1), device=device)
    return mean, std, z.T @ z / len(x), z.T @ residual / len(x)


def apply(x, mean, std, c, device, shape):
    z = np.clip((x - mean) / std, -10, 10)
    z = torch.tensor(np.concatenate([z, np.ones((len(x), 1), np.float32)], 1), device=device)
    return np.clip((100 * z @ c).cpu().numpy().reshape(shape), -100, 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--history-run', required=True)
    p.add_argument('--head-run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out, source = Path(a.output), Path(a.source)
    out.mkdir(parents=True, exist_ok=True)
    state = np.load(Path(a.history_run) / 'feature_state.npz')
    w, pw = state['weights'], state['periodic_weights']
    raw = np.load('data/PeMS04/PEMS04.npz')['data'].astype(np.float32)
    ts, vs, es = split_data_by_ratio(np.arange(len(raw) - 23), .2, .2)
    train, val = load_values(Path(a.head_run) / 'base_train_predictions.npz'), load_values(source / 'validation_predictions.npz')
    tb, vb, ty, vy = fuse(train, w, pw), fuse(val, w, pw), train[2], val[2]
    tx, vx = features(raw, ts), features(raw, vs)
    cut, hold = len(vs) // 2, len(vs) // 2 + 12
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best = dict(rmse=metrics(vb[hold:], vy[hold:])['rmse'], mode='none', ridge=1, blends=[0.] * 12)
    print('Frozen baseline validation', best, 'network feature width', tx.shape[1], flush=True)
    for mode in ('train_and_validation', 'validation_only'):
        if mode == 'train_and_validation':
            x, y, base = [np.concatenate([u, v[:cut]]) for u, v in ((tx, vx), (ty, vy), (tb, vb))]
        else:
            x, y, base = vx[:cut], vy[:cut], vb[:cut]
        mean, std, xtx, xty = fit_statistics(x, y, base, device)
        eye = torch.eye(xtx.shape[0], device=device)
        for ridge in (.01, .1, 1., 10., 100.):
            coefficient = torch.linalg.solve(xtx + ridge * eye, xty)
            delta = apply(vx[hold:], mean, std, coefficient, device, vb[hold:].shape)
            prediction = vb[hold:].copy()
            blends = []
            for h in range(12):
                b = min((0, .25, .5, .75, 1.), key=lambda b: np.mean(
                    (np.maximum(vb[hold:, h] + b * delta[:, h], 0) - vy[hold:, h]) ** 2))
                blends.append(b)
                prediction[:, h] = np.maximum(vb[hold:, h] + b * delta[:, h], 0)
            m = metrics(prediction, vy[hold:])
            print(json.dumps(dict(mode=mode, ridge=ridge, blends=blends, validation=m)), flush=True)
            if m['rmse'] < best['rmse']:
                best = dict(**m, mode=mode, ridge=ridge, blends=blends)
    print('Selected:', json.dumps(best), flush=True)
    if best['mode'] == 'train_and_validation':
        x, y, base = [np.concatenate([u, v]) for u, v in ((tx, vx), (ty, vy), (tb, vb))]
    else:
        x, y, base = vx, vy, vb
    mean, std, xtx, xty = fit_statistics(x, y, base, device)
    c = torch.linalg.solve(xtx + best['ridge'] * torch.eye(xtx.shape[0], device=device), xty)
    test = load_values(Path(a.history_run) / 'base_test_predictions.npz')
    base = fuse(test, w, pw)
    delta = apply(features(raw, es), mean, std, c, device, base.shape)
    prediction = np.maximum(base + np.array(best['blends'])[None, :, None, None] * delta, 0)
    result = dict(selection=best, test=metrics(prediction, test[2]), baseline_test=metrics(base, test[2]),
                  innovations_unchanged=True, extra_historical_channels=True,
                  protocol='frozen backbone; origin-only full-network histories; validation RMSE selection')
    np.savez(out / 'network_residual.npz', coefficients=c.cpu().numpy(), mean=mean, std=std,
             weights=w, periodic_weights=pw, blends=best['blends'])
    np.savez(out / 'test_predictions.npz', prediction=prediction, target=test[2])
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('FINAL', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
