"""Shared-horizon node residual calibration with validation-only selection."""
import numpy as np

from lib.enhanced_eval import residual_features, score, metrics


def fit_node(features, true, base, ridge, robust_steps):
    x = features.astype(np.float64)
    y = (true - base).astype(np.float64) / 100
    weight = np.ones_like(y)
    coefficient = None
    count = x.shape[0] * x.shape[1]
    for iteration in range(robust_steps + 1):
        xtx = np.einsum('thnf,thng->nfg', x * weight, x, optimize=True) / count
        xty = np.einsum('thnf,thno->nfo', x * weight, y, optimize=True) / count
        coefficient = np.linalg.solve(xtx + ridge * np.eye(x.shape[-1]), xty)
        if iteration < robust_steps:
            error = 100 * np.abs(np.einsum('thnf,nfo->thno', x, coefficient) - y)
            # A least-squares component retains sensitivity to large errors;
            # the bounded robust component emphasizes typical/relative errors.
            weight = 0.3 + 0.7 * 20 / np.maximum(error, 20)
            weight *= 0.75 + 0.25 * 100 / np.maximum(np.abs(true), 20)
            weight /= weight.mean(axis=(0, 1), keepdims=True)
    return coefficient.astype(np.float32)


def apply_node(features, base, coefficient, blend):
    return np.maximum(base + 100 * blend * np.einsum('thnf,nfo->thno', features, coefficient), 0)


def select_calibration(val, weights, periodic_weights, logger):
    from lib.enhanced_eval import fuse
    cut = len(val[0]) // 2
    start = cut + val[0].shape[1]
    fit = [a[:cut] for a in val]
    hold = [a[start:] for a in val]
    base_fit = fuse(fit, weights, periodic_weights)
    base_hold = fuse(hold, weights, periodic_weights)
    x_fit = residual_features(fit, base_fit)
    x_hold = residual_features(hold, base_hold)
    best = None
    for ridge in (1e-6, 1e-4, 0.01, 1.0):
        for robust_steps in (0, 2):
            coefficient = fit_node(x_fit, fit[2], base_fit, ridge, robust_steps)
            for blend in (0.25, 0.5, 0.75, 1.0):
                prediction = apply_node(x_hold, base_hold, coefficient, blend)
                value = score(prediction, hold[2])
                if best is None or value < best[0]:
                    best = (value, ridge, robust_steps, blend, metrics(prediction, hold[2]))
            logger.info('Shared node calibration ridge=%s robust=%s best=%s', ridge, robust_steps, best)
    value, ridge, robust_steps, blend, validation_metrics = best
    base = fuse(val, weights, periodic_weights)
    coefficient = fit_node(residual_features(val, base), val[2], base, ridge, robust_steps)
    return value, coefficient, blend, dict(ridge=ridge, robust_steps=robust_steps,
                                           blend=blend, validation=validation_metrics,
                                           refit='entire validation set after selection')
