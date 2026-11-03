"""Train an auxiliary residual head while freezing both innovation modules."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.enhanced_eval import metrics
from lib.dataloader import split_data_by_ratio
from tools.optimize_history_calibration import HistoryFeatures, load_values
from tools.history_calibration_inference import collect_split


class ResidualHead(nn.Module):
    def __init__(self, width, nodes=307):
        super().__init__()
        self.embedding = nn.Embedding(nodes, 16)
        self.net = nn.Sequential(nn.Linear(width + 16, 256), nn.LayerNorm(256), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 12))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, node):
        return self.net(torch.cat([x, self.embedding(node)], -1))


def matrix(raw, starts, values, weights, periodic_weights, neighbors):
    builder = HistoryFeatures(raw, starts, values, weights, periodic_weights, neighbors)
    n, h, nodes = builder.shape
    blocks = []
    for begin in range(0, n * nodes, 100000):
        sample = np.arange(begin, min(begin + 100000, n * nodes))
        t, node = sample // nodes, sample % nodes
        point = t * h * nodes + node  # first horizon; every feature is origin-known
        x = builder.rows(point, True)[:, :-1]
        extra = [a[t, :, node, 0] for a in (builder.base, values[0], values[1], values[3])]
        # Local long history complements the original 12-step encoder.
        origin = starts[t] + 11
        lag = np.arange(12, 97, 6)
        long_history = raw[np.maximum(origin[:, None] - lag, 0), node[:, None], 0]
        blocks.append(np.concatenate([x, *extra, long_history], -1).astype(np.float32))
    x = np.concatenate(blocks)
    base = builder.base[..., 0].transpose(0, 2, 1).reshape(-1, 12)
    true = values[2][..., 0].transpose(0, 2, 1).reshape(-1, 12)
    node = np.tile(np.arange(nodes), n)
    return x, node, base, true


def predict(model, x, node):
    model.eval()
    parts = []
    with torch.no_grad():
        for begin in range(0, len(x), 32768):
            parts.append((100 * model(x[begin:begin + 32768], node[begin:begin + 32768])).cpu().numpy())
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--history-run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=50)
    a = p.parse_args()
    torch.manual_seed(10)
    np.random.seed(10)
    torch.set_num_threads(4)
    out, source, history = Path(a.output), Path(a.source), Path(a.history_run)
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((source / 'enhanced_results.json').read_text())
    checkpoint = Path(settings['checkpoint'])
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    raw = np.load(ROOT / 'data/PeMS04/PEMS04.npz')['data'].astype(np.float32)
    train_starts, val_starts, test_starts = split_data_by_ratio(np.arange(len(raw) - 23), 0.2, 0.2)
    w = np.array(settings['horizon_weights'], np.float32).reshape(1, 12, 1, 1)
    pw = np.array(settings['periodic_weights'], np.float32).reshape(1, 12, 1, 1)
    neighbor_state = history / 'feature_state.npz'
    if neighbor_state.exists():
        neighbors = np.load(neighbor_state)['neighbors']
    else:
        corr = np.nan_to_num(np.corrcoef(raw[:train_starts[-1] + 12, :, 0].T))
        np.fill_diagonal(corr, -np.inf)
        neighbors = np.argsort(corr, axis=1)[:, -8:]
    train_path = out / 'base_train_predictions.npz'
    train = load_values(train_path) if train_path.exists() else collect_split(checkpoint, out, 'train')
    val = load_values(source / 'validation_predictions.npz')
    assert len(train[0]) == len(train_starts)
    assert np.allclose(train[2][:, 0, :, 0], raw[train_starts + 12, :, 0], atol=1e-3)
    print('Preparing training history features', flush=True)
    tx, tn, tb, ty = matrix(raw, train_starts, train, w, pw, neighbors)
    vx, vn, vb, vy = matrix(raw, val_starts, val, w, pw, neighbors)
    # A horizon gap separates auxiliary calibration fit and validation selection.
    cut = len(val[0]) // 2
    fit_rows, select_start = cut * 307, (cut + 12) * 307
    tx, tn, tb, ty = [np.concatenate([u, v[:fit_rows]]) for u, v in zip((tx, tn, tb, ty), (vx, vn, vb, vy))]
    mean, std = tx.mean(0), np.maximum(tx.std(0), 1e-3)
    normalize = lambda x: np.clip((x - mean) / std, -10, 10)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.as_tensor(normalize(tx), device=device)
    node = torch.as_tensor(tn, device=device)
    target = torch.as_tensor((ty - tb) / 100, device=device)
    hx = torch.as_tensor(normalize(vx[select_start:]), device=device)
    hn = torch.as_tensor(vn[select_start:], device=device)
    base, true = vb[select_start:], vy[select_start:]
    model = ResidualHead(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=0.00002)
    best = dict(rmse=metrics(base, true)['rmse'], blend=0, epoch=0)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improvement = 0
    print('Head train samples:', len(x), 'features:', x.shape[1], 'baseline validation:', best, flush=True)
    for epoch in range(1, a.epochs + 1):
        start = time.monotonic()
        model.train()
        order = torch.randperm(len(x), device=device)
        total = 0
        for ids in order.split(8192):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[ids], node[ids])
            loss = (prediction - target[ids]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item() * len(ids)
        scheduler.step()
        correction = np.clip(predict(model, hx, hn), -100, 100)
        improved = False
        for blend in (0.25, 0.5, 0.75, 1.0):
            m = metrics(np.maximum(base + blend * correction, 0), true)
            if m['rmse'] < best['rmse']:
                best = dict(**m, epoch=epoch, blend=blend)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                improved = True
        no_improvement = 0 if improved else no_improvement + 1
        torch.save(dict(state_dict=best_state, width=x.shape[1], mean=mean, std=std,
                        selection=best, neighbors=neighbors, weights=w, periodic_weights=pw,
                        frozen_checkpoint=str(checkpoint), frozen_sha256=original_hash), out / 'residual_head.pth')
        print(json.dumps(dict(epoch=epoch, loss=total / len(x), best_validation=best,
                              seconds=round(time.monotonic() - start, 1), patience=no_improvement)), flush=True)
        if no_improvement >= 7:
            break
    # Final test only after head/strength/checkpoint selection.
    test_cache = history / 'base_test_predictions.npz'
    test = load_values(test_cache) if test_cache.exists() else collect_split(checkpoint, out, 'test')
    ex, en, eb, ey = matrix(raw, test_starts, test, w, pw, neighbors)
    model.load_state_dict(best_state)
    correction = np.clip(predict(model, torch.as_tensor(normalize(ex), device=device),
                                 torch.as_tensor(en, device=device)), -100, 100)
    prediction = np.maximum(eb + best['blend'] * correction, 0)
    result = dict(selection=best, test=metrics(prediction, ey), baseline_test=metrics(eb, ey),
                  checkpoint=str(checkpoint), checkpoint_sha256=original_hash,
                  innovations_unchanged=original_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  extra_historical_channels=True,
                  protocol='frozen innovation model; train+validation-fit head; chronological validation selection')
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez(out / 'test_predictions.npz', prediction=prediction, target=ey)
    print('FINAL', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
