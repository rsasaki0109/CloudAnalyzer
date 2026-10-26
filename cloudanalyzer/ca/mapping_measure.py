"""Measure a fresh mapping command in a separate process without changing its policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

SCHEMA = 'cloudanalyzer.mapping_measurement.v1'
PROTOCOL = 'process-tree-rss-sampled-v1'


def file_hash(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def output_size(root: Path) -> tuple[int | None, str | None]:
    total = count = 0
    try:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                count += 1
                if count > 100_000:
                    return None, 'Output file count exceeds measurement limit'
                total += path.stat().st_size
        return total, None
    except OSError as error:
        return None, str(error)


def measure(command: list[str], run_dir: str, interval: float = .05) -> dict:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("Install cloudanalyzer[benchmark] to measure process memory") from error

    if not command or not .01 <= interval <= 1:
        raise ValueError('supply a command and a sample interval between .01 and 1 second')
    root = Path(run_dir).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError('measurement requires a new run directory')
    script_hashes = {str(Path(arg).absolute()): file_hash(Path(arg)) for arg in command if arg.endswith('.py') and Path(arg).is_file()}
    started = time.time()
    clock = time.perf_counter()
    peak = None
    samples = missing = 0
    status, code, error = 'launch_failed', None, None
    process = None
    try:
        process = subprocess.Popen(command, start_new_session=os.name == 'posix')
        owner = psutil.Process(process.pid)
        while True:
            rss = 0
            observed = False
            try:
                children = owner.children(recursive=True)
            except psutil.Error:
                children = []
                missing += 1
            for item in [owner, *children]:
                try:
                    value = item.memory_info().rss
                    rss += value
                    observed = observed or value > 0
                except psutil.Error:
                    missing += 1
            if observed:
                samples += 1
                peak = max(peak or 0, rss)
            if process.poll() is not None:
                break
            time.sleep(interval)
        code = process.wait()
        status = 'process_succeeded' if code == 0 else 'process_failed'
    except KeyboardInterrupt:
        status, error = 'interrupted', 'Measurement interrupted'
        if process is not None:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                code = process.wait()
    except (OSError, psutil.Error) as exc:
        error = str(exc)
        if process is not None:
            code = process.wait()
            status = 'measurement_failed'
    elapsed = time.perf_counter() - clock
    if root.is_symlink():
        raise ValueError('run directory became a symbolic link')
    size, size_error = output_size(root)
    result = {
        'schema': SCHEMA, 'protocol': PROTOCOL, 'sample_interval_seconds': interval,
        'status': status, 'exit_code': code, 'error': error,
        'command': command, 'script_sha256': script_hashes, 'cwd': str(Path.cwd()), 'platform': platform.platform(),
        'started_unix_seconds': started, 'wall_time_seconds': elapsed,
        'sampled_peak_tree_rss_bytes': peak, 'rss_samples': samples, 'unavailable_process_reads': missing,
        'logical_output_bytes': size, 'output_size_error': size_error,
        'run_sha256': file_hash(root / 'run.json'), 'job_sha256': file_hash(root / 'job.json'),
        'human_interventions': None,
        'scope': 'Wall time covers process launch through observed exit, excluding final hashing and output inventory. RSS is the observed sum of parent/descendant RSS samples, may double-count shared pages and miss short-lived peaks/processes. Output size excludes this report, symlinks and external inputs; it is not peak disk use. Process success is not map quality or generation completion.',
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'measurement.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--interval', type=float, default=.05)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    result = measure(command, args.run_dir, args.interval)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'process_succeeded' else 1)


if __name__ == '__main__':
    main()
