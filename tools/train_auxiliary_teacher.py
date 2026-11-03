"""Train a direct teacher, then validate/freeze fusion and node calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.dataloader import split_data_by_ratio
from lib.enhanced_eval import metrics, fuse, residual_features
from lib.forecast_calibration import fit_node, apply_node
from model.AuxiliaryTeacher import AuxiliaryTeacher
from tools.optimize_history_calibration import load_values


def forecast(model, raw, starts, mean, std, batch_size):
    model.eval()
    parts = []
    with torch.no_grad():
        for index in starts.split(batch_size):
            x = raw[index[:, None] + torch.arange(12, device=raw.device)]
            pred = model(x, index)
            parts.append((pred * std[0] + mean[0]).cpu().numpy())
    return np.concatenate(parts)


def finish(model, raw, val_starts, test_starts, mean, std, batch_size, source, history, out, checkpoint, original_hash):
    settings = json.loads((source / 'enhanced_results.json').read_text())
    val = load_values(source / 'validation_predictions.npz')
    teacher_val = forecast(model, raw, val_starts, mean, std, batch_size)
    cut, hold = len(val[0]) // 2, len(val[0]) // 2 + 12
    # Main contribution is at least 25% at every horizon. Fit fusion on first
    # validation segment; calibration hyperparameters use the later segment.
    main_w, teacher_w, legacy_w = [], [], []
    for h in range(12):
        best = None
        for mw in (.25, .4, .55, .7, .85, 1.):
            for tw in np.linspace(0, 1 - mw, 6):
                lw = 1 - mw - tw
                p = mw * val[0][:cut, h] + tw * teacher_val[:cut, h] + lw * val[1][:cut, h]
                error = float(np.mean((p - val[2][:cut, h]) ** 2))
                if best is None or error < best[0]:
                    best = error, mw, tw, lw
        main_w.append(best[1]); teacher_w.append(best[2]); legacy_w.append(best[3])
    mw, tw, lw = [np.array(v, np.float32).reshape(1, 12, 1, 1) for v in (main_w, teacher_w, legacy_w)]
    base = mw * val[0] + tw * teacher_val + lw * val[1]
    # Reuse origin-known periodic/input features; teacher slot now holds the new teacher.
    cv = list(val)
    cv[1] = teacher_val
    x = residual_features(cv, base)
    best = dict(rmse=metrics(np.maximum(base[hold:], 0), val[2][hold:])['rmse'], ridge=.001, robust=0, blend=0)
    for ridge in (1e-5, .001, .1, 1.):
        for robust in (0, 2):
            c = fit_node(x[:cut], val[2][:cut], base[:cut], ridge, robust)
            for blend in (.25, .5, .75, 1.):
                p = apply_node(x[hold:], base[hold:], c, blend)
                m = metrics(p, val[2][hold:])
                if m['rmse'] < best['rmse']:
                    best = dict(**m, ridge=ridge, robust=robust, blend=blend)
    print('Frozen fusion/calibration selection:', json.dumps(best), flush=True)
    c = fit_node(x, val[2], base, best['ridge'], best['robust'])
    np.savez(out / 'validation_predictions.npz', teacher=teacher_val, base=base, target=val[2])
    np.savez(out / 'fusion_calibration.npz', main_weights=mw, teacher_weights=tw, legacy_weights=lw,
             coefficients=c, blend=best['blend'])
    # Test labels are opened only after model and all fusion/calibration choices freeze.
    test = load_values(history / 'base_test_predictions.npz')
    teacher_test = forecast(model, raw, test_starts, mean, std, batch_size)
    base = mw * test[0] + tw * teacher_test + lw * test[1]
    cv = list(test)
    cv[1] = teacher_test
    prediction = apply_node(residual_features(cv, base), base, c, best['blend'])
    result = dict(test=metrics(prediction, test[2]), teacher_test=metrics(teacher_test, test[2]),
                  fusion_before_calibration=metrics(np.maximum(base, 0), test[2]),
                  validation_selection=best, main_weights=mw.ravel().tolist(),
                  auxiliary_teacher_weights=tw.ravel().tolist(), legacy_weights=lw.ravel().tolist(),
                  innovations_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash,
                  frozen_innovation_checkpoint=str(checkpoint), frozen_sha256=original_hash,
                  target_rmse_below_29=bool(metrics(prediction, test[2])['rmse'] < 29),
                  auxiliary_teacher_input_channels=3,
                  protocol='MSE teacher; validation-only checkpoint/fusion/calibration; frozen original innovations')
    np.savez(out / 'test_predictions.npz', prediction=prediction, target=test[2], teacher=teacher_test)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('FINAL', json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--history-run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    torch.manual_seed(10)
    np.random.seed(10)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    source, history, out = Path(a.source), Path(a.history_run), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((source / 'enhanced_results.json').read_text())
    checkpoint = Path(settings['checkpoint'])
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    values = np.load('data/PeMS04/PEMS04.npz')['data'].astype(np.float32)
    train_starts, val_starts, test_starts = split_data_by_ratio(np.arange(len(values) - 23), .2, .2)
    end = train_starts[-1] + 12
    mean, std = values[:end].mean((0, 1)), values[:end].std((0, 1))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    raw = torch.tensor((values - mean) / std, device=device)
    ts, vs, es = [torch.tensor(s, device=device) for s in (train_starts, val_starts, test_starts)]
    model = AuxiliaryTeacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=.3)
    best, no_improvement = float('inf'), 0
    config = dict(vars(a), mean=mean.tolist(), std=std.tolist(), seed=10,
                  frozen_innovation_checkpoint=str(checkpoint), frozen_sha256=original_hash)
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for epoch in range(1, a.epochs + 1):
        start, total, count = time.monotonic(), 0., 0
        model.train()
        for index in ts[torch.randperm(len(ts), device=device)].split(a.batch_size):
            x = raw[index[:, None] + torch.arange(12, device=device)]
            y = raw[index[:, None] + 12 + torch.arange(12, device=device), :, :1]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, index)
            loss = (prediction - y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item() * len(index)
            count += len(index)
        scheduler.step()
        prediction = forecast(model, raw, vs, mean, std, a.batch_size)
        truth = values[val_starts[:, None] + 12 + np.arange(12), :, :1]
        m = metrics(prediction, truth)
        if m['rmse'] < best:
            best, no_improvement = m['rmse'], 0
            torch.save(model.state_dict(), out / 'best_teacher.pth')
        else:
            no_improvement += 1
        print(json.dumps(dict(epoch=epoch, train_rmse=float(np.sqrt(total / count) * std[0]),
                              validation=m, best_validation_rmse=best, patience=no_improvement,
                              seconds=round(time.monotonic() - start, 1))), flush=True)
        if no_improvement >= 15:
            break
    model.load_state_dict(torch.load(out / 'best_teacher.pth', map_location=device, weights_only=True))
    finish(model, raw, vs, es, mean, std, a.batch_size, source, history, out, checkpoint, original_hash)


if __name__ == '__main__':
    main()
