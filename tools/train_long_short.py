"""Two-stage masked weekly pretraining and direct horizon-adaptive prediction."""
import argparse
import copy
import hashlib
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.long_history import context_batch, long_windows, physical_graph
from lib.forecast_optimization import ForecastEMA, balanced_forecast_loss, validation_score, configure_forecast_training, chronological_sample_weights
from lib.online_benchmark import assisted_validation
from lib.TrainInits import init_seed
from model.PDG2Seq import PDG2Seq
from tools.long_short_common import SIGNAL_CHECKPOINT, atomic_save, metric, model_args, save_json, traffic_data
from tools.pretrain_long_history import pretrain


def boolean(value):
    return str(value).lower() in ('true', '1', 'yes')


def make_cache(model, args, normalized, cli, directory):
    cache_path = Path(cli.context_cache) if cli.context_cache else directory/'frozen_context.pt'
    signature = dict(core_checkpoint_sha256=hashlib.sha256(Path(cli.core_checkpoint).read_bytes()).hexdigest(),
        prior_checkpoint_sha256=hashlib.sha256(Path(cli.pretrained_prior).read_bytes()).hexdigest(),
        history_steps=cli.history_steps, patch_size=cli.patch_size,
        phase='past-only context; prior anchor floor(origin/12)*12; no forecast labels')
    if cache_path.exists():
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        if cache['signature'] != signature:
            raise ValueError('Frozen context cache belongs to different weights or history settings')
        return add_phase_memory(model, normalized, cli, directory, cache)
    model.eval()
    anchors = torch.arange(0, len(normalized), cli.patch_size, device=cli.device)
    prior = []
    amp = str(cli.device).startswith('cuda')
    with torch.no_grad():
        for index, ends in enumerate(anchors.split(8)):
            history, available, start = long_windows(normalized, ends, cli.history_steps)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                value = model.long_history_encoder(history, start, available)
            prior.append(value.float().cpu())
            if index % 50 == 0:
                print('Past-only long-prior cache %d/%d anchors' % (min((index+1)*8, len(anchors)), len(anchors)), flush=True)
        teacher_args = copy.copy(args)
        teacher_args.use_long_short_learning = False
        for name in ('use_dgq', 'use_context_graph_refine', 'use_periodic_context',
                     'use_signal_decouple', 'use_decoder_periodic_context', 'use_periodic_consistency'):
            setattr(teacher_args, name, False)
        teacher = PDG2Seq(teacher_args).to(cli.device)
        teacher.load_state_dict(torch.load('pre-trained/PEMSD4.pth', map_location=cli.device, weights_only=False), strict=True)
        teacher.eval()
        origins = torch.arange(12, len(normalized)-11, device=cli.device)
        core, teacher_predictions = [], []
        for index, batch in enumerate(origins.split(64)):
            source, target, _ = context_batch(normalized, batch)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                state = model.encode_short(source)
                reference = teacher(source, target)
            core.append(state.float().cpu()); teacher_predictions.append(reference.float().cpu())
            if index % 40 == 0:
                print('Original PDG2Seq core/teacher context %d/%d forecasts' % (min((index+1)*64, len(origins)), len(origins)), flush=True)
    cache = dict(prior=torch.cat(prior), core=torch.cat(core), teacher=torch.cat(teacher_predictions), signature=signature)
    atomic_save(cache, cache_path)
    save_json(dict(signature=signature, prior_shape=list(cache['prior'].shape), core_shape=list(cache['core'].shape),
        teacher_shape=list(cache['teacher'].shape)), directory/'cache_manifest.json')
    print('Frozen context ready; original core and pretrained long encoder remain fixed.', flush=True)
    return add_phase_memory(model, normalized, cli, directory, cache)


def add_phase_memory(model, normalized, cli, directory, cache):
    if not cli.phase_memory:
        return cache
    path = directory/'phase_prior.pt'
    if path.exists():
        stored = torch.load(path, map_location='cpu', weights_only=False)
        if stored['signature'] != cache['signature']:
            raise ValueError('Phase memory belongs to different pretrained weights')
        cache['prior'] = stored['prior']
        return cache
    model.eval()
    priors = []
    anchors = torch.arange(0, len(normalized), cli.patch_size, device=cli.device)
    with torch.no_grad():
        for index, ends in enumerate(anchors.split(8)):
            history, available, start = long_windows(normalized, ends, cli.history_steps)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(cli.device).startswith('cuda')):
                prior = model.long_history_encoder(history, start, available, phase_memory=True)
            priors.append(prior.half().cpu())
            if index % 50 == 0:
                print('Past-only phase memory %d/%d anchors' % (min((index+1)*8, len(anchors)), len(anchors)), flush=True)
    cache['prior'] = torch.cat(priors)
    atomic_save(dict(prior=cache['prior'], signature=cache['signature']), path)
    return cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use_long_short_learning', type=boolean, default=True)
    parser.add_argument('--dataset', default='PEMSD4', choices=['PEMSD4'])
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default='experiments/PEMSD4/newininvation4_long_short')
    parser.add_argument('--pretrained_prior', default='')
    parser.add_argument('--core_checkpoint', default=SIGNAL_CHECKPOINT)
    parser.add_argument('--history_steps', type=int, default=2016)
    parser.add_argument('--patch_size', type=int, default=12)
    parser.add_argument('--prior_width', type=int, default=48)
    parser.add_argument('--prior_layers', type=int, default=2)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--attention_layers', type=int, default=2)
    parser.add_argument('--phase_memory', type=boolean, default=False)
    parser.add_argument('--context_cache', default='')
    parser.add_argument('--warmstart_forecaster', default='')
    parser.add_argument('--finetune_core', type=boolean, default=False,
                        help='Jointly train original short graph encoder and direct head; never use cached core states')
    parser.add_argument('--core_lr_ratio', type=float, default=0.1)
    parser.add_argument('--keep_warmstart_baseline', action='store_true',
                        help='Include the initial warmstarted head as validation candidate epoch zero')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mse_weight', type=float, default=0.35)
    parser.add_argument('--rmse_weight', type=float, default=0.0)
    parser.add_argument('--precision', choices=['bf16', 'float32'], default='bf16')
    parser.add_argument('--recent_fraction', type=float, default=0.0)
    parser.add_argument('--recent_multiplier', type=float, default=1.0)
    parser.add_argument('--mape_weight', type=float, default=0.05)
    parser.add_argument('--distill_weight', type=float, default=0.03)
    parser.add_argument('--near_horizon_weight', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.0)
    parser.add_argument('--selection_reference', default='', help='JSON validation metrics for balanced worst-metric selection')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--pretrain_epochs', type=int, default=6)
    parser.add_argument('--pretrain_steps', type=int, default=160)
    parser.add_argument('--pretrain_batch_size', type=int, default=2)
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume the latest completed epoch, including optimizer and RNG')
    parser.add_argument('--restart_lr_schedule', action='store_true',
                        help='On resume, keep optimizer moments but start a new cosine schedule over remaining epochs')
    cli = parser.parse_args(argv)
    directory = Path(cli.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if cli.mode == 'test':
        trained = torch.load(directory/'best_model.pth', map_location='cpu', weights_only=False)
        for key, value in trained['training'].items():
            if key not in ('mode', 'device', 'output_dir', 'no_test'):
                setattr(cli, key, value)
        cli.attention_layers = trained['args'].get('long_short_attention_layers', 0)
        cli.phase_memory = trained['args'].get('long_phase_memory', False)
    if cli.mode == 'train' and not cli.resume and (directory/'best_model.pth').exists():
        raise FileExistsError('Use a new output directory, or --mode test to evaluate the existing model')
    if not cli.pretrained_prior:
        existing = directory/'pretraining/long_prior.pth'
        if existing.exists():
            cli.pretrained_prior = str(existing)
        else:
            pretrain_cli = copy.copy(cli)
            pretrain_cli.output_dir = str(directory/'pretraining')
            cli.pretrained_prior = str(pretrain(pretrain_cli))
    init_seed(cli.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    raw, normalized, splits, stats = traffic_data(cli.device)
    args = SimpleNamespace(**trained['args']) if cli.mode == 'test' else model_args(cli, raw.shape[1])
    args.device = cli.device
    model = PDG2Seq(args).to(cli.device)
    source = torch.load(cli.core_checkpoint, map_location=cli.device, weights_only=False)
    if 'state_dict' in source:
        source = source['state_dict']
    own = model.state_dict()
    compatible = {key:value for key,value in source.items() if key in own and value.shape == own[key].shape}
    incompatible = model.load_state_dict(compatible, strict=False)
    if any(not key.startswith(('long_history_encoder.', 'horizon_forecaster.')) for key in incompatible.missing_keys):
        raise ValueError('The checkpoint does not cover the original branch modules: %s' % incompatible.missing_keys)
    pretrained = torch.load(cli.pretrained_prior, map_location=cli.device, weights_only=False)
    if pretrained['normalization'] != stats:
        raise ValueError('Prior normalization differs from forecast normalization')
    model.long_history_encoder.load_state_dict(pretrained['state_dict'], strict=True)
    graph = physical_graph('data/PeMS04/PEMS04.csv', raw.shape[1]).to(cli.device)
    model.horizon_forecaster.physical_graph.copy_(graph)
    if cli.warmstart_forecaster and cli.mode == 'train':
        warmstart = torch.load(cli.warmstart_forecaster, map_location=cli.device, weights_only=False)['state_dict']
        own = model.state_dict()
        if cli.finetune_core:
            model.load_state_dict(warmstart, strict=True)
        else:
            old_head = {key:value for key,value in warmstart.items() if key.startswith('horizon_forecaster.')
                        and key in own and value.shape == own[key].shape}
            model.load_state_dict(old_head, strict=False)
    head_parameters, core_parameters = configure_forecast_training(model, cli.finetune_core)
    if not 0 < cli.core_lr_ratio <= 1:
        raise ValueError('Core learning rate ratio must be in (0, 1]')
    save_json(dict(args=vars(args), training=vars(cli), normalization=stats,
        original_core_checkpoint=cli.core_checkpoint, loaded_core_state_tensors=len(compatible),
        ignored_source_state_tensors=[key for key in source if key not in compatible],
        trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
        training_scope='joint original short encoder and direct head' if cli.finetune_core else 'direct head only',
        cached_core_used_for_prediction=not cli.finetune_core,
        frozen_prior_epoch=pretrained['epoch']), directory/'configuration.json')
    cache = make_cache(model, args, normalized, cli, directory)
    offsets = torch.arange(12, device=cli.device)
    amp = str(cli.device).startswith('cuda') and cli.precision == 'bf16'
    node_embedding = model.node_embeddings1.detach().cpu()

    def batch_inputs(origins):
        source, target, periodic = context_batch(normalized, origins)
        cpu_origins = origins.cpu()
        state = None if cli.finetune_core else cache['core'][cpu_origins-12].to(cli.device)
        prior = cache['prior'][cpu_origins//cli.patch_size].to(cli.device, dtype=torch.float32)
        return source, target, periodic, state, prior

    def predict(origins, gates=False):
        model.eval()
        predictions, truths, total_gates, gate_count = [], [], None, 0
        with torch.no_grad():
            for batch in origins.to(cli.device).split(cli.batch_size):
                source, target, periodic, state, prior = batch_inputs(batch)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    result = model(source, target, long_prior=prior, short_state=state,
                                   periodic_features=periodic, return_gates=gates)
                if gates:
                    result, weights = result
                    weights = weights.float().sum((0, 1)).cpu()
                    total_gates = weights if total_gates is None else total_gates+weights
                    gate_count += len(batch)*raw.shape[1]
                predictions.append((result.float()*stats['std']+stats['mean']).clamp_min(0).cpu())
                truths.append(raw[batch[:, None]+offsets].unsqueeze(-1).cpu())
        return torch.cat(predictions), torch.cat(truths), None if not gates else total_gates/gate_count

    if cli.mode == 'train':
        groups = [dict(params=head_parameters, lr=cli.lr, lr_ratio=1.0)]
        if core_parameters:
            groups.append(dict(params=core_parameters, lr=cli.lr*cli.core_lr_ratio, lr_ratio=cli.core_lr_ratio))
        optimizer = torch.optim.AdamW(groups, lr=cli.lr, weight_decay=cli.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cli.epochs, eta_min=cli.lr*0.03)
        best_score, selected, bad = float('inf'), 0, 0
        records = []
        train_origins, val_origins, _ = splits
        sample_weights = chronological_sample_weights(len(train_origins), cli.recent_fraction, cli.recent_multiplier)
        generator = torch.Generator().manual_seed(cli.seed)
        ema_module = model if cli.finetune_core else model.horizon_forecaster
        ema = ForecastEMA(ema_module, cli.ema_decay) if cli.ema_decay else None
        reference = None
        if cli.selection_reference:
            import json
            reference = json.loads(Path(cli.selection_reference).read_text())['validation_metrics']
        start_epoch = 1
        if cli.resume:
            saved = torch.load(directory/'best_model.pth', map_location='cpu', weights_only=False)
            if saved['args'] != vars(args) or saved['normalization'] != stats:
                raise ValueError('Resume requires identical model configuration and normalization')
            latest = torch.load(directory/'last_training_state.pth', map_location=cli.device, weights_only=False)
            if cli.finetune_core:
                model.load_state_dict(latest['model_state'], strict=True)
            else:
                model.horizon_forecaster.load_state_dict(latest['forecaster'], strict=True)
            if ema is not None:
                if latest.get('ema') is None:
                    raise ValueError('EMA resume requires an EMA snapshot')
                ema.load_state_dict(latest['ema'])
            optimizer.load_state_dict(latest['optimizer'])
            if cli.restart_lr_schedule:
                remaining_epochs = cli.epochs-latest['epoch']
                if remaining_epochs <= 0:
                    raise ValueError('Extended epoch limit must exceed the completed epoch')
                for group in optimizer.param_groups:
                    group['lr'] = cli.lr*group.get('lr_ratio', 1.0)
                    group['initial_lr'] = group['lr']
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, remaining_epochs, eta_min=cli.lr*0.03)
                print('New cosine schedule: %d remaining epochs, initial lr %.7f' %
                      (remaining_epochs, cli.lr), flush=True)
            else:
                scheduler.load_state_dict(latest['scheduler'])
            generator.set_state(latest['generator_state'].cpu())
            if 'torch_rng_state' in latest:
                torch.set_rng_state(latest['torch_rng_state'].cpu())
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([value.cpu() for value in latest['cuda_rng_states']])
            start_epoch = latest['epoch']+1
            import json
            report = json.loads((directory/'training_report.json').read_text())
            records = [row for row in report['history'] if row['epoch'] < start_epoch]
            best_score = validation_score(saved['validation']['future_label_assisted_validation_tail'], reference)
            selected = saved['epoch']
            bad = max(0, latest['epoch']-selected)
            print('Resuming after completed epoch %d' % latest['epoch'], flush=True)
        elif cli.keep_warmstart_baseline:
            if not cli.warmstart_forecaster:
                raise ValueError('An initial baseline requires --warmstart_forecaster')
            initial_prediction, initial_true, _ = predict(val_origins)
            torch.set_num_threads(1)
            assisted, assisted_true = assisted_validation(initial_prediction, initial_true, node_embedding)
            initial_metrics = metric(assisted, assisted_true)
            torch.set_num_threads(4)
            best_score = validation_score(initial_metrics, reference)
            initial_record = dict(epoch=0, raw_validation=metric(initial_prediction, initial_true),
                future_label_assisted_validation_tail=initial_metrics, selection_score=best_score,
                initialization_only=True, validation_weights='EMA' if ema is not None else 'instantaneous')
            records.append(initial_record)
            atomic_save(dict(state_dict=model.state_dict(), args=vars(args), training=vars(cli),
                normalization=stats, epoch=0, validation=initial_record,
                uses_future_labels_for_validation_selection=True), directory/'best_model.pth')
            print('Initial warmstart validation candidate:', initial_metrics, flush=True)
        print('Forecast training: train=%d validation=%d, direct horizon=12, core_finetune=%s, pretrained long prior fixed' %
              (len(train_origins), len(val_origins), cli.finetune_core), flush=True)
        for epoch in range(start_epoch, cli.epochs+1):
            model.train()
            if cli.recent_fraction and cli.recent_multiplier > 1:
                order = train_origins[torch.multinomial(sample_weights, len(train_origins),
                                                      replacement=True, generator=generator)]
            else:
                order = train_origins[torch.randperm(len(train_origins), generator=generator)]
            total, batches, started = 0.0, 0, time.monotonic()
            component_totals = dict(mae=0.0, mse=0.0, mape=0.0, distill=0.0, rmse=0.0)
            for batch in order.to(cli.device).split(cli.batch_size):
                source, target, periodic, state, prior = batch_inputs(batch)
                labels = normalized[batch[:, None]+offsets].unsqueeze(-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    pred = model(source, target, long_prior=prior, short_state=state, periodic_features=periodic)
                pred = pred.float()
                true_real = raw[batch[:, None]+offsets].unsqueeze(-1)
                teacher_pred = cache['teacher'][batch.cpu()-12].to(cli.device)
                loss, components = balanced_forecast_loss(pred, labels, true_real, teacher_pred,
                    stats['std'], cli.mse_weight, cli.mape_weight, cli.distill_weight,
                    cli.near_horizon_weight, cli.rmse_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite forecast loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(head_parameters+core_parameters, 3)
                optimizer.step(); total += float(loss.detach()); batches += 1
                if ema is not None:
                    ema.update(ema_module)
                for key,value in components.items():
                    component_totals[key] += float(value.detach())
            instantaneous = None
            if ema is not None:
                instantaneous = {key:value.detach().clone() for key,value in ema_module.state_dict().items()}
                ema_module.load_state_dict(ema.shadow, strict=True)
            prediction, true, _ = predict(val_origins)
            raw_metrics = metric(prediction, true)
            torch.set_num_threads(1)
            node_embedding = model.node_embeddings1.detach().cpu()
            assisted, assisted_true = assisted_validation(prediction, true, node_embedding)
            assisted_metrics = metric(assisted, assisted_true)
            torch.set_num_threads(4)
            score = validation_score(assisted_metrics, reference)
            record = dict(epoch=epoch, train_loss=total/batches, raw_validation=raw_metrics,
                future_label_assisted_validation_tail=assisted_metrics, selection_score=score,
                seconds=time.monotonic()-started, lr=optimizer.param_groups[0]['lr'],
                core_lr=optimizer.param_groups[-1]['lr'] if core_parameters else 0.0,
                loss_components={key:value/batches for key,value in component_totals.items()},
                validation_weights='EMA' if ema is not None else 'instantaneous')
            records.append(record)
            print('Epoch %d %.1fs train %.5f RAW validation %s; FUTURE-LABEL-ASSISTED validation %s' %
                  (epoch, record['seconds'], total/batches, raw_metrics, assisted_metrics), flush=True)
            if score < best_score:
                best_score, selected, bad = score, epoch, 0
                atomic_save(dict(state_dict=model.state_dict(), args=vars(args), training=vars(cli),
                    normalization=stats, epoch=epoch, validation=record,
                    uses_future_labels_for_validation_selection=True), directory/'best_model.pth')
            else:
                bad += 1
            if instantaneous is not None:
                ema_module.load_state_dict(instantaneous, strict=True)
            scheduler.step()
            atomic_save(dict(forecaster=model.horizon_forecaster.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, generator_state=generator.get_state(),
                torch_rng_state=torch.get_rng_state(), cuda_rng_states=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else [], ema=None if ema is None else ema.state_dict(),
                model_state=model.state_dict() if cli.finetune_core else None), directory/'last_training_state.pth')
            save_json(dict(selected_epoch=selected, best_score=best_score, history=records,
                selection='Fixed September immediate-label benchmark on validation tail; no test selection',
                selection_reference=reference, selection_rule='worst normalized metric' if reference else 'legacy sum'), directory/'training_report.json')
            if bad >= cli.patience:
                print('Early stopping; selected epoch %d' % selected, flush=True)
                break
    saved = torch.load(directory/'best_model.pth', map_location=cli.device, weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    prediction, true, gates = predict(splits[1], gates=True)
    atomic_save(dict(pred=prediction, true=true, origins=splits[1],
                     gates_mean_by_horizon=gates), directory/'validation_predictions.pt')
    save_json(dict(scale_order=['short', 'trend', 'periodic', 'long'],
                   average_gates_by_horizon=gates.tolist(), selected_epoch=saved['epoch']), directory/'horizon_gates.json')
    if cli.no_test:
        print('Validation-only stage complete; test metrics not read.', flush=True)
        return
    from tools.evaluate_long_short import evaluate
    evaluate(model, args, cache, raw, normalized, splits, stats, directory, predict)


if __name__ == '__main__':
    main()
