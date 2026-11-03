"""RMSE-oriented residual calibration; leaves both model innovations intact.

Fits/chooses using disjoint chronological validation portions, then refits on
validation and reports test once. Every feature uses observations at or before
the forecast origin; future periodic values come from previous days only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.enhanced_eval import fuse, metrics
from lib.dataloader import split_data_by_ratio


class HistoryFeatures:
    def __init__(self, raw, starts, values, weights, periodic_weights, neighbors):
        self.raw, self.starts, self.values = raw, starts, values
        self.base = fuse(values, weights, periodic_weights)
        self.shape = self.base.shape[:3]
        self.neighbors = neighbors

    def rows(self, indices, extra_channels):
        t, h, n = np.unravel_index(indices, self.shape)
        origin = self.starts[t] + 11
        # All offsets are nonnegative: no future traffic is available here.
        offsets = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            17, 23, 35, 47, 71, 143, 287, 575, 2015])
        history = self.raw[np.maximum(origin[:, None] - offsets, 0), n[:, None], 0]
        columns = [self.base[t, h, n], self.values[0][t, h, n],
                   self.values[1][t, h, n], self.values[3][t, h, n],
                   self.values[4][t, h, n], history]
        # Fixed graph derives only from the original training observations.
        near = self.neighbors[n]
        for offset in (0, 1, 2, 5, 11, 23):
            v = self.raw[np.maximum(origin - offset, 0)[:, None], near, 0]
            columns.append(v.mean(1, keepdims=True))
        columns += [history[:, :3].mean(1, keepdims=True),
                    history[:, :12].mean(1, keepdims=True),
                    history[:, :12].std(1, keepdims=True),
                    history[:, :1] - history[:, 3:4],
                    history[:, :1] - history[:, 11:12]]
        target_time = origin + h + 1
        day = target_time % 288 / 288
        for feature in (h / 12, np.sin(2 * np.pi * day), np.cos(2 * np.pi * day),
                        target_time // 288 % 7):
            columns.append(feature[:, None])
        # Periodic reference for the forecast step has already been observed.
        for offset in (288, 2016):
            reference_time = target_time - offset
            columns.append(self.raw[np.maximum(reference_time, 0), n, :1])
            columns.append((reference_time >= 0)[:, None])
        if extra_channels:
            for channel in range(1, self.raw.shape[2]):
                columns.append(self.raw[np.maximum(origin[:, None] - np.arange(12), 0), n[:, None], channel])
                for offset in (0, 5, 11):
                    columns.append(self.raw[np.maximum(origin - offset, 0)[:, None], near, channel].mean(1, keepdims=True))
        columns.append(n[:, None])  # categorical sensor id, not an ordered number
        return np.concatenate(columns, axis=1).astype(np.float32)

    def predict_correction(self, model, indices, extra_channels):
        result = np.empty(len(indices), dtype=np.float32)
        for begin in range(0, len(indices), 150000):
            block = indices[begin:begin + 150000]
            result[begin:begin + len(block)] = model.predict(self.rows(block, extra_channels))
        return result


def fit_model(builder, indices, target, depth, extra_channels):
    x = builder.rows(indices, extra_channels)
    feature_types = ['q'] * (x.shape[1] - 1) + ['c']
    model = XGBRegressor(n_estimators=700, max_depth=depth, learning_rate=0.035,
                         min_child_weight=100, reg_lambda=100, subsample=0.85,
                         colsample_bytree=0.85, tree_method='hist', device='cuda',
                         enable_categorical=True, feature_types=feature_types,
                         n_jobs=4, random_state=10)
    model.fit(x, target[indices] - builder.base.reshape(-1)[indices])
    return model


def load_values(path):
    with np.load(path) as cached:
        return [cached[f'a{i}'] for i in range(8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='shared node calibration evaluation directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit_rows', type=int, default=1000000)
    parser.add_argument('--test-cache', default='', help='Optional frozen predictor cache matching validation source')
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads((source / 'enhanced_results.json').read_text())
    weights = np.array(settings['horizon_weights'], np.float32).reshape(1, 12, 1, 1)
    periodic_weights = np.array(settings['periodic_weights'], np.float32).reshape(1, 12, 1, 1)
    val = load_values(source / 'validation_predictions.npz')
    raw = np.load(ROOT / 'data/PeMS04/PEMS04.npz')['data'].astype(np.float32)
    train_starts, val_starts, test_starts = split_data_by_ratio(np.arange(len(raw) - 23), 0.2, 0.2)
    assert np.allclose(val[2][:, 0, :, 0], raw[val_starts + 12, :, 0], atol=1e-3)
    corr = np.nan_to_num(np.corrcoef(raw[:train_starts[-1] + 12, :, 0].T))
    np.fill_diagonal(corr, -np.inf)
    neighbors = np.argsort(corr, axis=1)[:, -8:]
    builder = HistoryFeatures(raw, val_starts, val, weights, periodic_weights, neighbors)
    target = val[2].reshape(-1)
    cut = len(val[0]) // 2
    row_per_origin = 12 * raw.shape[1]
    fit_count = cut * row_per_origin
    hold_start = (cut + 12) * row_per_origin
    rng = np.random.default_rng(10)
    fit_indices = rng.choice(fit_count, min(args.fit_rows, fit_count), replace=False)
    hold_indices = rng.choice(len(target) - hold_start, min(700000, len(target) - hold_start), replace=False) + hold_start
    baseline = builder.base.reshape(-1)[hold_indices]
    truth = target[hold_indices]
    best = dict(rmse=metrics(baseline, truth)['rmse'], depth=None, blend=0, extra_channels=False)
    for extra in (False, True):
        for depth in (5, 8):
            start = time.monotonic()
            model = fit_model(builder, fit_indices, target, depth, extra)
            correction = builder.predict_correction(model, hold_indices, extra)
            for blend in (0.25, 0.5, 0.75, 1.0):
                prediction = np.maximum(baseline + blend * correction, 0)
                m = metrics(prediction, truth)
                print(json.dumps(dict(stage='validation_sample', depth=depth, extra_channels=extra,
                                      blend=blend, metrics=m)), flush=True)
                if m['rmse'] < best['rmse']:
                    best = dict(rmse=m['rmse'], depth=depth, blend=blend, extra_channels=extra)
            print('Candidate seconds:', round(time.monotonic() - start, 1), flush=True)
    print('Chosen using validation only:', json.dumps(best), flush=True)
    # Confirm on the complete chronological validation holdout before any test use.
    if best['depth'] is not None:
        model = fit_model(builder, fit_indices, target, best['depth'], best['extra_channels'])
        hold_indices = np.arange(hold_start, len(target))
        correction = builder.predict_correction(model, hold_indices, best['extra_channels'])
        hold_prediction = np.maximum(builder.base.reshape(-1)[hold_indices] + best['blend'] * correction, 0)
        validation = metrics(hold_prediction, target[hold_indices])
        baseline_validation = metrics(builder.base.reshape(-1)[hold_indices], target[hold_indices])
        print('Complete validation:', json.dumps(dict(calibrated=validation, baseline=baseline_validation)), flush=True)
        if validation['rmse'] >= baseline_validation['rmse']:
            best['depth'], best['blend'] = None, 0
    else:
        validation = metrics(baseline, truth)
    all_indices = rng.choice(len(target), min(args.fit_rows, len(target)), replace=False)
    model = None
    if best['depth'] is not None:
        model = fit_model(builder, all_indices, target, best['depth'], best['extra_channels'])
        model.save_model(output / 'history_calibration.ubj')
    np.savez(output / 'feature_state.npz', neighbors=neighbors, weights=weights, periodic_weights=periodic_weights)
    (output / 'selection.json').write_text(json.dumps(dict(selection=best, validation=validation,
        source=str(source), checkpoint=settings['checkpoint'], fit_rows=len(all_indices),
        unchanged_innovations=['periodic_edge_quality', 'graph_quality_decouple']), indent=2) + '\n')
    # Materialize test base predictions only after freezing all calibration choices.
    from tools.history_calibration_inference import collect_test
    test = load_values(Path(args.test_cache)) if args.test_cache else collect_test(settings['checkpoint'], output)
    test_builder = HistoryFeatures(raw, test_starts, test, weights, periodic_weights, neighbors)
    prediction = test_builder.base.reshape(-1).copy()
    if model is not None:
        indices = np.arange(len(prediction))
        prediction += best['blend'] * test_builder.predict_correction(model, indices, best['extra_channels'])
    prediction = np.maximum(prediction.reshape(test[2].shape), 0)
    result = dict(selection=best, validation=validation, raw_test=metrics(test[0], test[2]),
                  test=metrics(prediction, test[2]), target_rmse_below_29=bool(metrics(prediction, test[2])['rmse'] < 29),
                  source=str(source), checkpoint=settings['checkpoint'],
                  extra_input_channels=best['extra_channels'], innovations_unchanged=True,
                  protocol='historical features only; validation-only calibration fitting and selection')
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez(output / 'test_predictions.npz', prediction=prediction, target=test[2])
    print('FINAL', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
