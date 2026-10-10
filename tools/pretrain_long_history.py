"""Train a masked weekly spatial/temporal encoder on TRAIN traffic only."""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.long_history import long_windows, physical_graph, structured_mask
from lib.TrainInits import init_seed
from model.LongShortTemporal import LongHistoryMaskedEncoder
from tools.long_short_common import atomic_save, save_json, traffic_data


def pretrain(cli):
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'pretraining_report.json').exists():
        raise FileExistsError('Completed pretraining exists; load its checkpoint instead')
    init_seed(cli.seed)
    torch.set_num_threads(4)
    _, raw, _, stats = traffic_data(cli.device)
    config = dict(num_nodes=raw.shape[1], history_steps=cli.history_steps,
                  patch_size=cli.patch_size, width=cli.prior_width, layers=cli.prior_layers,
                  dropout=cli.dropout, steps_per_day=288)
    model = LongHistoryMaskedEncoder(**config).to(cli.device)
    model.physical_graph.copy_(physical_graph('data/PeMS04/PEMS04.csv', raw.shape[1]).to(cli.device))
    holdout_start = stats['train_raw_end']-864
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cli.pretrain_epochs, eta_min=0.0001)
    generator = torch.Generator(device=cli.device).manual_seed(cli.seed)
    amp = str(cli.device).startswith('cuda')

    def validate():
        model.eval()
        validation_generator = torch.Generator(device=cli.device).manual_seed(cli.seed+777)
        ends = torch.tensor([stats['train_raw_end']-offset for offset in (0, 96, 192, 288)], device=cli.device)
        total, count = 0.0, 0
        with torch.no_grad():
            for end in ends.split(2):
                history, available, start = long_windows(raw, end, cli.history_steps)
                mask = structured_mask(history, available, cli.patch_size, 288, validation_generator)
                timestamps = start[:, None]+torch.arange(cli.history_steps, device=cli.device)[None]
                score_mask = mask & (timestamps >= holdout_start).unsqueeze(-1)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    prediction = model.reconstruct(history, start, available, mask)
                diff = prediction.float()[score_mask]-history[score_mask]
                total += float(diff.square().sum()); count += diff.numel()
        return total/count

    best = validate()
    history_records = [dict(epoch=0, heldout_masked_mse=best)]
    selected, bad = 0, 0
    def save(epoch):
        atomic_save(dict(state_dict=model.state_dict(), config=config, normalization=stats,
                         epoch=epoch, heldout_masked_mse=best,
                         pretraining_raw_end=holdout_start, uses_forecast_validation_or_test=False),
                    output/'long_prior.pth')
    save(0)
    print('Masked long-history pretraining: train timestamps < %d; held-out reconstruction [%d,%d); initial MSE %.6f' %
          (holdout_start, holdout_start, stats['train_raw_end'], best), flush=True)
    for epoch in range(1, cli.pretrain_epochs+1):
        model.train()
        total, started = 0.0, time.monotonic()
        for step in range(cli.pretrain_steps):
            ends = torch.randint(cli.history_steps, holdout_start+1, (cli.pretrain_batch_size,),
                                 device=cli.device, generator=generator)
            history, available, start = long_windows(raw, ends, cli.history_steps)
            mask = structured_mask(history, available, cli.patch_size, 288, generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                prediction = model.reconstruct(history, start, available, mask)
            diff = prediction.float()[mask]-history[mask]
            loss = diff.square().mean()+0.2*diff.abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite masked reconstruction loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step(); total += float(loss.detach())
            if (step+1) % 40 == 0:
                print('Pretrain epoch %d step %d/%d loss %.6f' % (epoch, step+1, cli.pretrain_steps, total/(step+1)), flush=True)
        validation = validate()
        history_records.append(dict(epoch=epoch, train_loss=total/cli.pretrain_steps,
                                    heldout_masked_mse=validation, seconds=time.monotonic()-started))
        print('Pretrain epoch %d: held-out MSE %.6f; %.1fs' % (epoch, validation, time.monotonic()-started), flush=True)
        if validation < best:
            best, selected, bad = validation, epoch, 0
            save(epoch)
        else:
            bad += 1
        scheduler.step()
        save_json(dict(selected_epoch=selected, history=history_records, config=config,
                       normalization=stats, pretraining_raw_end=holdout_start,
                       masks=['whole nodes', 'contiguous time spans', 'repeated daily fragments'],
                       uses_forecast_validation_or_test=False), output/'pretraining_report.json')
        if bad >= 3:
            break
    return output/'long_prior.pth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--history_steps', type=int, default=2016)
    parser.add_argument('--patch_size', type=int, default=12)
    parser.add_argument('--prior_width', type=int, default=48)
    parser.add_argument('--prior_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pretrain_epochs', type=int, default=6)
    parser.add_argument('--pretrain_steps', type=int, default=160)
    parser.add_argument('--pretrain_batch_size', type=int, default=2)
    pretrain(parser.parse_args())


if __name__ == '__main__':
    main()
