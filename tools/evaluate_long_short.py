"""Validation-selected fusion and the explicitly labelled September benchmark."""
import copy
import hashlib

import torch

from lib.long_history import context_batch
from lib.online_benchmark import OnlineBenchmark, SEPTEMBER_SETTINGS, assisted_validation
from tools.long_short_common import BASELINE, BASELINE_H12, atomic_save, metric, save_json


def features_for(pred, origins, cache, normalized, stats, batch_size=128):
    features = []
    for index, batch in enumerate(origins.to(normalized.device).split(batch_size)):
        source, _, periodic = context_batch(normalized, batch)
        start = index*batch_size
        student = (pred[start:start+len(batch)].to(normalized.device)-stats['mean'])/stats['std']
        teacher = cache['teacher'][batch.cpu()-12].to(normalized.device)
        last = source[:, -1:, :, :1].expand(-1, 12, -1, -1)
        day = periodic[..., 0:1]
        week = periodic[..., 1:2]
        day = torch.where(periodic[..., 2:3] > 0, day, student)
        week = torch.where(periodic[..., 3:4] > 0, week, student)
        features.append(torch.cat((student, teacher, last, day, week, torch.ones_like(student)), -1).cpu())
    return torch.cat(features)


def fit_coefficients(features, true, stats, ridge, weighted=False, node_wise=True):
    x = features.double()
    y = ((true.squeeze(-1)-stats['mean'])/stats['std']).double()
    weights = (1/true.squeeze(-1).double().clamp_min(20)) if weighted else torch.ones_like(y)
    weights = weights/weights.mean().clamp_min(1e-10)
    if node_wise:
        xtx = torch.einsum('shnf,shng,shn->hnfg', x, x, weights)/len(x)
        xty = torch.einsum('shnf,shn,shn->hnf', x, y, weights)/len(x)
    else:
        xtx = torch.einsum('shnf,shng,shn->hfg', x, x, weights)/(len(x)*x.shape[2])
        xty = torch.einsum('shnf,shn,shn->hf', x, y, weights)/(len(x)*x.shape[2])
    identity = torch.eye(features.shape[-1], dtype=torch.float64)
    prior = torch.zeros(features.shape[-1], dtype=torch.float64)
    prior[:2] = 0.5
    return torch.linalg.solve(xtx+ridge*identity, xty+ridge*prior).float()


def apply_coefficients(features, coefficients, stats):
    if coefficients.ndim == 2:
        coefficients = coefficients[:, None]
    normalized = (features*coefficients.unsqueeze(0)).sum(-1, keepdim=True)
    return (normalized*stats['std']+stats['mean']).clamp_min(0)


def select_validation_fusion(node_embedding, cache, normalized, splits, stats, directory):
    validation = torch.load(directory/'validation_predictions.pt', map_location='cpu', weights_only=False)
    val_pred, val_true = validation['pred'], validation['true']
    prefix = len(val_pred)*2//3
    features = features_for(val_pred, splits[1], cache, normalized, stats)
    fingerprint = hashlib.sha256(val_pred.numpy().tobytes()).hexdigest()
    if (directory/'fusion.pth').exists():
        stored = torch.load(directory/'fusion.pth', map_location='cpu', weights_only=False)
        if stored.get('validation_prediction_sha256') == fingerprint:
            print('Reusing previously frozen validation fusion.', flush=True)
            return stored['coefficients'], features, validation, stored['selection']
    torch.set_num_threads(1)
    base_adapted, selection_true = assisted_validation(val_pred, val_true, node_embedding)
    reference = metric(base_adapted, selection_true)
    best_score, best, best_metrics, records = 1.0, None, reference, []
    print('FUTURE-LABEL-ASSISTED validation, direct model:', reference, flush=True)
    for node_wise in (False, True):
        for weighted in (False, True):
            for ridge in (0.001, 0.01, 0.1):
                coefficients = fit_coefficients(features[:prefix], val_true[:prefix], stats, ridge, weighted, node_wise)
                # Preserve a material contribution from the new direct model.
                if float(coefficients[..., 0].mean()) < 0.25:
                    continue
                blended = apply_coefficients(features, coefficients, stats)
                adapted, true = assisted_validation(blended, val_true, node_embedding)
                measured = metric(adapted, true)
                ratios = [measured[key]/reference[key] for key in reference]
                score = max(ratios)+0.05*(sum(ratios)/len(ratios)-1)
                record = dict(node_wise=node_wise, weighted=weighted, ridge=ridge,
                              validation=measured, score=score,
                              mean_coefficients=coefficients.reshape(-1, coefficients.shape[-1]).mean(0).tolist())
                records.append(record)
                print('Validation fusion:', record, flush=True)
                if score < best_score and all(value <= 1 for value in ratios):
                    best_score, best_metrics = score, measured
                    best = dict(node_wise=node_wise, weighted=weighted, ridge=ridge)
    coefficients = None if best is None else fit_coefficients(features, val_true, stats, **best)
    selection = dict(configuration=best, future_label_assisted_validation=best_metrics,
        direct_model_future_label_assisted_validation=reference, prefix_count=prefix,
        heldout_tail_count=len(val_pred)-prefix,
        selection='Fit fusion on validation prefix; select on validation tail; refit once on all validation; no test selection',
        candidates=records, uses_future_labels=True, fixed_online_settings=SEPTEMBER_SETTINGS)
    # Persist parameters before loading test targets into any scoring routine.
    save_json(selection, directory/'fusion_selection.json')
    atomic_save(dict(coefficients=coefficients, normalization=stats,
        validation_prediction_sha256=fingerprint,
        feature_names=['new_direct_prediction', 'existing_dgq_teacher', 'last', 'past_day', 'past_week', 'bias'],
        selection=selection), directory/'fusion.pth')
    print('Selected validation fusion:', best, flush=True)
    return coefficients, features, validation, selection


def evaluate(model, args, cache, raw, normalized, splits, stats, directory, predict):
    node_embedding = model.node_embeddings1.detach().cpu()
    coefficients, features, validation, selection = select_validation_fusion(
        node_embedding, cache, normalized, splits, stats, directory)
    val_pred, val_true = validation['pred'], validation['true']
    torch.set_num_threads(4)
    test_pred, test_true, _ = predict(splits[2])
    raw_direct_metrics = metric(test_pred, test_true)
    test_features = features_for(test_pred, splits[2], cache, normalized, stats)
    if coefficients is not None:
        final_val = apply_coefficients(features, coefficients, stats)
        final_test = apply_coefficients(test_features, coefficients, stats)
    else:
        final_val, final_test = val_pred, test_pred
    torch.set_num_threads(1)
    adapter = OnlineBenchmark(node_embedding, settings=SEPTEMBER_SETTINGS)
    warmup = adapter.adapt(final_val, val_true)
    adapter.fit_post_bias(val_true-warmup)
    assisted = adapter.adapt(final_test, test_true)
    average = metric(assisted, test_true)
    gains = {key:100*(BASELINE[key]-average[key])/BASELINE[key] for key in BASELINE}
    causal = OnlineBenchmark(node_embedding, settings=SEPTEMBER_SETTINGS, delay_steps=12)
    past = causal.adapt(final_val, val_true)
    causal.fit_post_bias(val_true-past)
    causal.reset_pending()
    causal_prediction = causal.adapt(final_test, test_true)
    horizons = [metric(assisted[:, horizon], test_true[:, horizon]) for horizon in range(12)]
    saved = torch.load(directory/'best_model.pth', map_location='cpu', weights_only=False)
    report = dict(average=average, gains_percent=gains, target_gain_percent=1.0,
        target_met=all(value >= 1 for value in gains.values()),
        targets={key:value*0.99 for key,value in BASELINE.items()},
        september_average_baseline=BASELINE, september_horizon12_baseline=BASELINE_H12,
        horizon12_gains_percent={key:100*(BASELINE_H12[key]-horizons[-1][key])/BASELINE_H12[key] for key in BASELINE_H12},
        horizons=horizons, raw_direct_model=raw_direct_metrics,
        frozen_fusion_without_label_updates=metric(final_test, test_true),
        causal_same_fusion_and_settings=metric(causal_prediction, test_true),
        selected_epoch=saved['epoch'], validation=selection,
        uses_future_labels=True, label_update_delay_steps=0, overlap_truth_substitution=False,
        protocol='September future-label-assisted benchmark: immediate full-window updates after output; NOT causal deployment performance',
        architecture=dict(original_dgq=True, original_periodic_context=True,
            original_context_graph_refine=True, original_signal_decouple=True,
            original_decoder_periodic_information='Transferred to direct head via observed day/week references',
            long_history_steps=args.long_history_steps, masked_pretraining=True,
            decomposition=['short', 'intermediate band', 'trend'],
            fused_scales=['short', 'trend', 'periodic', 'long_prior'], prediction='direct 12-step output'),
        pretraining_uses_forecast_validation_or_test=False,
        train_test_selection='This run checkpoint and fusion selected using validation only; prior runs have already reported this test split')
    save_json(report, directory/'report.json')
    atomic_save(dict(pred=test_pred, true=test_true, origins=splits[2]), directory/'raw_test_predictions.pt')
    atomic_save(dict(pred=assisted, true=test_true, origins=splits[2], uses_future_labels=True), directory/'assisted_test_predictions.pt')
    print('FUTURE-LABEL-ASSISTED Average Horizon:', average, flush=True)
    print('Numeric gains against September 25:', gains, flush=True)
    print('All three metrics reach >=1%:', report['target_met'], flush=True)
    print('RAW direct model:', raw_direct_metrics, flush=True)
    print('CAUSAL same fusion/settings:', report['causal_same_fusion_and_settings'], flush=True)
    for horizon, measured in enumerate(horizons, 1):
        print('FUTURE-LABEL-ASSISTED Horizon %02d: %s' % (horizon, measured), flush=True)
