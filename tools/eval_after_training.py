"""Run enhanced evaluation immediately, or after a training PID exits."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--training-log')
    parser.add_argument('--quality-strength', default=0.1, type=float)
    args = parser.parse_args()
    if args.pid:
        # Guard against PID reuse while waiting for the current trainer.
        proc = Path('/proc') / str(args.pid)
        identity = (proc / 'stat').read_text().split()[21]
        print('Waiting for training PID', args.pid, flush=True)
        while proc.exists():
            try:
                stat = (proc / 'stat').read_text().split()
                if stat[2] == 'Z' or stat[21] != identity:
                    break
            except FileNotFoundError:
                break
            time.sleep(20)
    if args.training_log:
        marker = 'Experiment log path in: '
        directories = [line.split(marker, 1)[1].strip()
                       for line in Path(args.training_log).read_text().splitlines() if marker in line]
        if len(directories) != 1:
            raise ValueError('Expected exactly one experiment directory in training log')
        checkpoint = Path(directories[0]) / 'best_model.pth'
    else:
        checkpoint = Path(args.checkpoint)
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    command = [sys.executable, '-u', 'run.py', '--dataset', 'PEMSD4',
               '--mode', 'enhanced_eval', '--checkpoint', str(checkpoint),
               '--use_dgq', 'True', '--dgq_eval_ensemble', 'True',
               '--dgq_teacher_path', './pre-trained/PEMSD4.pth',
               '--use_periodic_context', 'True', '--use_context_graph_refine', 'True',
               '--use_signal_decouple', 'True', '--use_periodic_edge_quality', 'True',
               '--use_graph_quality_decouple', 'True', '--graph_quality_strength',
               str(args.quality_strength), '--use_periodic_consistency', 'True',
               '--use_decoder_periodic_context', 'False', '--batch_size', '32', '--seed', '10']
    print('Evaluating validation-selected checkpoint:', checkpoint, flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == '__main__':
    main()
