import json
import sys

import pytest
from typer.testing import CliRunner
from cloudanalyzer_cli.main import app


def test_pass_through_preserves_child_flags_and_spaces(tmp_path):
    root = tmp_path/'run with spaces'
    code = ('import sys,json; from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(); '
            '(p/"args.json").write_text(json.dumps(sys.argv[2:]))')
    args = [sys.executable, '-c', code, str(root), '--timeout', 'child value', '--interval', 'literal;value']
    result = CliRunner().invoke(app, ['mapping-measure', '--run-dir', str(root),
        '--interval', '.01', '--timeout', '10', '--', *args])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report['command'] == args
    assert report['status'] == 'process_succeeded'
    assert report['timeout_seconds'] == 10
    assert report['sample_interval_seconds'] == .01
    assert json.loads((root/'args.json').read_text()) == args[4:]
    assert json.loads((root/'measurement.json').read_text()) == report


@pytest.mark.parametrize('code,timeout,status', [
    ('raise SystemExit(7)', '10', 'process_failed'),
    ('import time; time.sleep(30)', '.05', 'timed_out'),
])
def test_failed_or_timed_out_command_returns_nonzero_receipt(tmp_path, code, timeout, status):
    root = tmp_path/'failed'
    result = CliRunner().invoke(app, ['mapping-measure', '--run-dir', str(root),
        '--timeout', timeout, '--', sys.executable, '-c', code])
    assert result.exit_code == 1, result.output
    report = json.loads(result.stdout)
    assert report['status'] == status
    assert json.loads((root/'measurement.json').read_text()) == report


@pytest.mark.parametrize('options', [['--timeout', '0'], ['--interval', '0'], ['--timeout', 'nan']])
def test_invalid_options_never_launch(tmp_path, monkeypatch, options):
    from ca import mapping_measure
    monkeypatch.setattr(mapping_measure.subprocess, 'Popen', lambda *a, **k: pytest.fail('launched'))
    root = tmp_path/'invalid'
    result = CliRunner().invoke(app, ['mapping-measure', '--run-dir', str(root), *options,
        '--', sys.executable, '-c', 'pass'])
    assert result.exit_code != 0
    assert not root.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    receipt = tmp_path/'measurement.json'; receipt.write_text('keep')
    result = CliRunner().invoke(app, ['mapping-measure', '--run-dir', str(tmp_path),
        '--', sys.executable, '-c', 'pass'])
    assert result.exit_code != 0
    assert receipt.read_text() == 'keep'
