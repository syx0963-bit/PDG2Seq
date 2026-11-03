"""Validation-selected teacher/periodic fusion and observation-only online bias.

Requires chronological, stride-one loaders. Future target traffic is used only
for validation fitting and reporting metrics, never to update online state.
"""
import json
from pathlib import Path

import numpy as np
import torch


def metrics(pred, true):
    err = pred - true
    valid = true > 0
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))),
                mae=float(np.mean(np.abs(err))),
                mape=float(np.mean(np.abs(err[valid] / true[valid])) * 100))


def score(pred, true):
    m = metrics(pred, true)
    return m['rmse'] + m['mae'] + 1.2 * m['mape']


def online_bias(pred, observations, rate, clip):
    """At origin t, forecast j,h matures only when j+h+1 <= t.

    observations[t] is the last *input* observation for origin t. Each past
    forecast contributes an error once, precisely when its target is observed.
    """
    bias = np.zeros_like(pred[0])
    result = np.empty_like(pred)
    for t in range(len(pred)):
        for h in range(min(t, pred.shape[1])):
            j = t - h - 1
            error = observations[t] - pred[j, h]
            bias[h] = np.clip((1 - rate) * bias[h] + rate * error, -clip, clip)
        result[t] = np.maximum(pred[t] + bias, 0)
    return result


def collect(trainer, teacher, loader):
    parts = [[] for _ in range(8)]
    trainer.model.eval()
    teacher.eval()
    with torch.no_grad():
        for data, target in loader:
            values = [trainer.scaler.inverse_transform(trainer.model(data, target)),
                      trainer.scaler.inverse_transform(teacher(data, target)),
                      trainer.scaler.inverse_transform(target[..., :1]),
                      trainer.scaler.inverse_transform(target[..., 1:2]),
                      target[..., 2:3],
                      trainer.scaler.inverse_transform(data[:, -1, :, :1]),
                      trainer.scaler.inverse_transform(data[:, -3:, :, :1].mean(1)),
                      trainer.scaler.inverse_transform(data[:, :, :, :1].mean(1))]
            for part, value in zip(parts, values):
                part.append(value.detach().cpu().numpy())
    return [np.concatenate(part) for part in parts]


def fuse(values, weights, periodic_weights):
    student, teacher, _, reference, valid = values[:5]
    base = weights * student + (1 - weights) * teacher
    return np.maximum(base + periodic_weights * valid * (reference - base), 0)


def residual_features(values, base):
    """Features available at the forecast origin, in flow units / 100."""
    student, teacher, _, reference, valid, latest, mean3, mean12 = values
    expand = lambda x: np.broadcast_to(x[:, None], base.shape)
    return np.concatenate([np.ones_like(base), base / 100,
                           (student - base) / 100, (teacher - base) / 100,
                           valid * (reference - base) / 100,
                           (expand(latest) - base) / 100,
                           (expand(mean3) - base) / 100,
                           (expand(mean12) - base) / 100], axis=-1)


def fit_residual(features, residual, ridge, weighted=False):
    # Independent coefficients per forecast horizon and sensor, with a common
    # shrinkage prior of zero correction to the existing ensemble.
    x = features.astype(np.float64)
    y = residual.astype(np.float64) / 100
    weight = 1 / np.maximum(np.abs(y * 100), 20) if weighted else np.ones_like(y)
    if weighted:
        weight /= weight.mean(axis=0, keepdims=True)
    xtx = np.einsum('thnf,thng->hnfg', x * weight, x) / len(x)
    xty = np.einsum('thnf,thno->hnfo', x * weight, y) / len(x)
    regularizer = np.eye(x.shape[-1]) * ridge
    return np.linalg.solve(xtx + regularizer, xty).astype(np.float32)


def apply_residual(base, features, coefficients, blend):
    return np.maximum(base + blend * 100 * np.einsum('thnf,hnfo->thno', features, coefficients), 0)


def evaluate(trainer):
    if not trainer.args.use_periodic_consistency:
        raise ValueError('enhanced_eval requires --use_periodic_consistency True')
    teacher = trainer._load_dgq_teacher()
    if teacher is None:
        raise ValueError('A compatible teacher checkpoint is required')
    val = collect(trainer, teacher, trainer.val_loader)
    np.savez(Path(trainer.args.log_dir) / 'validation_predictions.npz', **{f'a{i}': a for i, a in enumerate(val)})
    # Fit on the first validation segment, select on a later segment separated
    # by a horizon to keep overlapping future labels out of selection inputs.
    cut = len(val[0]) // 2
    start = cut + trainer.args.horizon
    if start >= len(val[0]):
        raise ValueError('Validation split is too short')
    fit = [a[:cut] for a in val]
    select = [a[start:] for a in val]
    shape = (1, trainer.args.horizon, 1, 1)
    weights = np.ones(shape, dtype=np.float32)
    periodic_weights = np.zeros(shape, dtype=np.float32)
    for h in range(trainer.args.horizon):
        s, t, y = [a[:, h] for a in fit[:3]]
        weights[0, h] = min(np.linspace(0, 1, 51),
                            key=lambda w: score(np.maximum(w * s + (1 - w) * t, 0), y))
    base = fuse(fit, weights, periodic_weights)
    for h in range(trainer.args.horizon):
        p, y, r, v = base[:, h], fit[2][:, h], fit[3][:, h], fit[4][:, h]
        periodic_weights[0, h] = min(np.linspace(0, 0.5, 26),
                                     key=lambda w: score(np.maximum(p + w * v * (r - p), 0), y))
    # Include the unmodified student so optional processing can be rejected.
    candidates = [('student', np.ones(shape, np.float32), np.zeros(shape, np.float32)),
                  ('teacher', np.zeros(shape, np.float32), np.zeros(shape, np.float32)),
                  ('ensemble', weights, np.zeros(shape, np.float32)),
                  ('ensemble_periodic', weights, periodic_weights)]
    best = None
    for name, w, pw in candidates:
        prediction = fuse(select, w, pw)
        for rate, clip in [(0, 0), (0.02, 4), (0.05, 4), (0.1, 4),
                           (0.02, 8), (0.05, 8), (0.1, 8), (0.2, 8)]:
            corrected = online_bias(prediction, select[5], rate, clip) if rate else prediction
            candidate_score = score(corrected, select[2])
            if best is None or candidate_score < best[0]:
                best = (candidate_score, name, w.copy(), pw.copy(), rate, clip,
                        metrics(corrected, select[2]))
    _, name, w, pw, rate, clip, val_metrics = best
    selected_correction = None
    fit_base = fuse(fit, weights, periodic_weights)
    select_base = fuse(select, weights, periodic_weights)
    fit_features = residual_features(fit, fit_base)
    select_features = residual_features(select, select_base)
    for ridge in (0.0001, 0.001, 0.01, 0.1, 1.0):
        coefficients = fit_residual(fit_features, fit[2] - fit_base, ridge)
        for blend in (0.25, 0.5, 0.75, 1.0):
            corrected = apply_residual(select_base, select_features, coefficients, blend)
            candidate_score = score(corrected, select[2])
            trainer.logger.info('Residual calibration ridge=%s blend=%s validation=%s',
                                ridge, blend, metrics(corrected, select[2]))
            if candidate_score < best[0]:
                best = (candidate_score, 'node_residual_ridge', weights.copy(), periodic_weights.copy(),
                        0, 0, metrics(corrected, select[2]))
                selected_correction = (coefficients, blend, ridge)
    shared_correction = None
    from lib.forecast_calibration import select_calibration, apply_node
    shared_score, shared_coefficient, shared_blend, shared_settings = select_calibration(
        val, weights, periodic_weights, trainer.logger)
    if shared_score < best[0]:
        best = (shared_score, 'shared_node_robust_calibration', weights.copy(), periodic_weights.copy(),
                0, 0, shared_settings['validation'])
        selected_correction = None
        shared_correction = (shared_coefficient, shared_blend)
    _, name, w, pw, rate, clip, val_metrics = best
    trainer.logger.info('Validation-selected evaluation: %s, rate=%s, clip=%s, metrics=%s',
                        name, rate, clip, val_metrics)
    # Test inference starts only after every selection decision is fixed.
    test = collect(trainer, teacher, trainer.test_loader)
    prediction = fuse(test, w, pw)
    if shared_correction is not None:
        coefficient, blend = shared_correction
        prediction = apply_node(residual_features(test, prediction), prediction, coefficient, blend)
        np.savez(Path(trainer.args.log_dir) / 'shared_node_calibration.npz',
                 coefficients=coefficient, blend=blend, weights=w, periodic_weights=pw)
    if selected_correction is not None:
        coefficients, blend, ridge = selected_correction
        prediction = apply_residual(prediction, residual_features(test, prediction), coefficients, blend)
        np.savez(Path(trainer.args.log_dir) / 'residual_calibration.npz',
                 coefficients=coefficients, blend=blend, ridge=ridge)
    if rate:
        prediction = online_bias(prediction, test[5], rate, clip)
    result = dict(checkpoint=trainer.args.checkpoint, method=name,
                  shared_calibration=shared_settings if shared_correction is not None else None,
                  residual_ridge=None if selected_correction is None else selected_correction[2],
                  residual_blend=None if selected_correction is None else selected_correction[1],
                  teacher=trainer.args.dgq_teacher_path,
                  horizon_weights=w.ravel().tolist(), periodic_weights=pw.ravel().tolist(),
                  online_rate=rate, online_clip=clip, validation_selection=val_metrics,
                  raw_test=metrics(test[0], test[2]), test=metrics(prediction, test[2]),
                  protocol='stride-one; online updates use input observations only')
    path = Path(trainer.args.log_dir) / 'enhanced_results.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    trainer.logger.info('Enhanced evaluation results: %s', json.dumps(result))
    trainer.logger.info('Results saved to %s', path)
    return result
