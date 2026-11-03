"""Optional nonlinear calibration, fitted/selected without test labels."""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from lib.enhanced_eval import collect, fuse, residual_features, metrics, score
from lib.forecast_calibration import fit_node, apply_node


def features(values, base):
    x = residual_features(values, base)
    shape = base.shape
    h = np.broadcast_to(np.arange(shape[1], dtype=np.float32)[None, :, None, None] / shape[1], shape)
    node = np.broadcast_to(np.arange(shape[2], dtype=np.float32)[None, None, :, None] / shape[2], shape)
    return np.concatenate([x, h, node], -1).reshape(-1, x.shape[-1] + 2)


def train_booster(x, residual, depth):
    rng = np.random.default_rng(10)
    indices = rng.choice(len(x), min(300000, len(x)), replace=False)
    model = XGBRegressor(n_estimators=200, max_depth=depth, learning_rate=0.035,
                         reg_lambda=100, min_child_weight=100, subsample=0.8,
                         colsample_bytree=0.9, tree_method='hist', device='cuda',
                         n_jobs=4, random_state=10)
    model.fit(x[indices], residual.reshape(-1)[indices])
    return model


def apply_saved(values, directory):
    """Apply frozen calibration to predictions/known inputs without refitting.

    values has the collect() layout; its future-target slot is never read.
    """
    directory = Path(directory)
    settings = json.loads((directory / 'calibrated_results.json').read_text())
    with np.load(directory / 'shared_node_calibration.npz') as saved:
        base = fuse(values, saved['weights'], saved['periodic_weights'])
        prediction = apply_node(residual_features(values, base), base,
                                saved['coefficients'], float(saved['blend']))
    if settings['nonlinear_depth'] is not None:
        model = XGBRegressor()
        model.load_model(directory / 'nonlinear_calibration.ubj')
        correction = model.predict(features(values, base)).reshape(base.shape)
        prediction = np.maximum(prediction + settings['nonlinear_blend'] * correction, 0)
    return prediction


def evaluate(trainer):
    source = Path(trainer.args.calibration_source)
    settings = json.loads((source / 'enhanced_results.json').read_text())
    if Path(settings['checkpoint']).resolve() != Path(trainer.args.checkpoint).resolve():
        raise ValueError('Calibration source checkpoint does not match')
    cache = np.load(source / 'validation_predictions.npz')
    val = [cache[f'a{i}'] for i in range(8)]
    w = np.array(settings['horizon_weights'], np.float32).reshape(1, -1, 1, 1)
    pw = np.array(settings['periodic_weights'], np.float32).reshape(1, -1, 1, 1)
    shared = settings['shared_calibration']
    cut = len(val[0]) // 2
    fit = [a[:cut] for a in val]
    hold = [a[cut + val[0].shape[1]:] for a in val]
    base_fit, base_hold = fuse(fit, w, pw), fuse(hold, w, pw)
    c = fit_node(residual_features(fit, base_fit), fit[2], base_fit,
                 shared['ridge'], shared['robust_steps'])
    p_fit = apply_node(residual_features(fit, base_fit), base_fit, c, shared['blend'])
    p_hold = apply_node(residual_features(hold, base_hold), base_hold, c, shared['blend'])
    x_fit, x_hold = features(fit, base_fit), features(hold, base_hold)
    best = (score(p_hold, hold[2]), None, 0, metrics(p_hold, hold[2]))
    for depth in (3, 5, 7):
        model = train_booster(x_fit, fit[2] - p_fit, depth)
        correction = model.predict(x_hold).reshape(p_hold.shape)
        for blend in (0.25, 0.5, 0.75, 1.0):
            pred = np.maximum(p_hold + blend * correction, 0)
            s = score(pred, hold[2])
            trainer.logger.info('Nonlinear depth=%s blend=%s validation=%s', depth, blend, metrics(pred, hold[2]))
            if s < best[0]:
                best = (s, depth, blend, metrics(pred, hold[2]))
    _, depth, blend, validation = best
    trainer.logger.info('Validation-selected nonlinear calibration: %s', best)
    base_val = fuse(val, w, pw)
    c = fit_node(residual_features(val, base_val), val[2], base_val,
                 shared['ridge'], shared['robust_steps'])
    p_val = apply_node(residual_features(val, base_val), base_val, c, shared['blend'])
    model = None
    if depth is not None:
        model = train_booster(features(val, base_val), val[2] - p_val, depth)
        model.save_model(Path(trainer.args.log_dir) / 'nonlinear_calibration.ubj')
    teacher = trainer._load_dgq_teacher()
    if teacher is None:
        raise ValueError('Teacher not found')
    test = collect(trainer, teacher, trainer.test_loader)
    base = fuse(test, w, pw)
    prediction = apply_node(residual_features(test, base), base, c, shared['blend'])
    if model is not None:
        prediction = np.maximum(prediction + blend * model.predict(features(test, base)).reshape(base.shape), 0)
    np.savez(Path(trainer.args.log_dir) / 'shared_node_calibration.npz',
             coefficients=c, blend=shared['blend'], weights=w, periodic_weights=pw)
    result = dict(checkpoint=trainer.args.checkpoint, source=str(source),
                  method='shared_node_plus_boosted_residual' if model is not None else 'shared_node_robust_calibration',
                  nonlinear_depth=depth, nonlinear_blend=blend, validation=validation,
                  shared_calibration=shared, raw_test=metrics(test[0], test[2]),
                  test=metrics(prediction, test[2]),
                  protocol='validation-only fitting and selection; no online future labels')
    (Path(trainer.args.log_dir) / 'calibrated_results.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez(Path(trainer.args.log_dir) / 'test_predictions.npz', prediction=prediction, target=test[2])
    trainer.logger.info('Final calibrated results: %s', json.dumps(result))
