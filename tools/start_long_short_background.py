"""Start an independent server process and persist its log and PID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logfile', required=True)
    parser.add_argument('training_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.training_arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    root = Path(__file__).resolve().parents[1]
    log = Path(args.logfile).resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = ['nohup', sys.executable, '-u', str(root/'run.py'),
               '--use_long_short_learning', 'true', *arguments]
    with log.open('ab', buffering=0) as output:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    metadata = dict(pid=process.pid, log=str(log), command=command,
                    started_utc=datetime.now(timezone.utc).isoformat())
    log.with_suffix(log.suffix+'.pid.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
