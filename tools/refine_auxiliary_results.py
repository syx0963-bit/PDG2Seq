"""After teacher training, refine its validation-selected fusion using history."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.optimize_history_calibration import load_values


def augment(values, teacher, main_weights, legacy_weights):
    values = list(values)
    total = main_weights + legacy_weights
    if np.any(total <= 0):
        raise ValueError('Innovation model contribution must be positive')
    values[0] = (main_weights * values[0] + legacy_weights * values[1]) / total
    values[1] = teacher
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--source', required=True)
    p.add_argument('--history-run', required=True)
    p.add_argument('--teacher-run', required=True)
    a = p.parse_args()
    if a.wait_pid:
        proc = Path('/proc') / str(a.wait_pid)
        identity = (proc / 'stat').read_text().split()[21]
        print('Waiting for teacher PID', a.wait_pid, flush=True)
        while proc.exists():
            try:
                stat = (proc / 'stat').read_text().split()
                if stat[2] == 'Z' or stat[21] != identity:
                    break
            except FileNotFoundError:
                break
            time.sleep(20)
    teacher_dir = Path(a.teacher_run)
    result = json.loads((teacher_dir / 'results.json').read_text())
    mw, lw = [np.array(result[k], np.float32).reshape(1, 12, 1, 1)
              for k in ('main_weights', 'legacy_weights')]
    source = teacher_dir / 'history_source'
    source.mkdir(exist_ok=True)
    val = load_values(Path(a.source) / 'validation_predictions.npz')
    with np.load(teacher_dir / 'validation_predictions.npz') as data:
        val = augment(val, data['teacher'], mw, lw)
    np.savez(source / 'validation_predictions.npz', **{f'a{i}': v for i, v in enumerate(val)})
    # These arrays are only handed to final reporting after fitting/selection;
    # no test target enters any calibration feature or fitting operation.
    test = load_values(Path(a.history_run) / 'base_test_predictions.npz')
    with np.load(teacher_dir / 'test_predictions.npz') as data:
        test = augment(test, data['teacher'], mw, lw)
    test_cache = source / 'base_test_predictions.npz'
    np.savez(test_cache, **{f'a{i}': v for i, v in enumerate(test)})
    settings = dict(checkpoint=result['frozen_innovation_checkpoint'],
                    horizon_weights=(mw + lw).ravel().tolist(), periodic_weights=[0.] * 12,
                    original_main_weights=result['main_weights'],
                    original_legacy_weights=result['legacy_weights'],
                    auxiliary_teacher=str(teacher_dir / 'best_teacher.pth'))
    (source / 'enhanced_results.json').write_text(json.dumps(settings, indent=2) + '\n')
    out = teacher_dir / 'history_refinement'
    subprocess.run([sys.executable, '-u', str(ROOT / 'tools/optimize_history_calibration.py'),
                    '--source', str(source), '--output', str(out), '--fit_rows', '1500000',
                    '--test-cache', str(test_cache)], cwd=ROOT, check=True)
    refined = json.loads((out / 'results.json').read_text())
    # Recommendation depends on validation RMSE, never test metrics.
    use_refined = refined['validation']['rmse'] < result['validation_selection']['rmse']
    selected = dict(pipeline='history_refinement' if use_refined else 'teacher_fusion_node_calibration',
                    selection_basis='chronological validation holdout RMSE',
                    result_path=str(out / 'results.json' if use_refined else teacher_dir / 'results.json'),
                    validation_rmse=refined['validation']['rmse'] if use_refined else result['validation_selection']['rmse'])
    selected_result = refined if use_refined else result
    selected['test'] = selected_result['test']
    selected['target_rmse_below_29'] = selected_result['test']['rmse'] < 29
    (teacher_dir / 'recommended_result.json').write_text(json.dumps(selected, indent=2) + '\n')
    print('VALIDATION-SELECTED FINAL PIPELINE', json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
