import json
import pytest
from ca.mapping_scorecard import scorecard, compare


def test_missing_measurements_and_input_identity(tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps({'schema': 'cloudanalyzer.mapping_run.v1', 'layout_file': {'sha256': 'b'*64}, 'history': [{}, {}]}))
    (tmp_path / 'job.json').write_text(json.dumps({'source': {'sha256': 'a'*64}}))
    result = scorecard(str(tmp_path))
    assert result['generated_length_m'] is None
    assert result['human_interventions'] is None
    assert result['recorded_actions'] == 2
    assert compare(result, result)['delta']['generated_length_m'] is None
    other = {**result, 'identity': {'source_sha256': 'c'*64}}
    with pytest.raises(ValueError, match='identical source'):
        compare(result, other)


@pytest.fixture
def measured_run(tmp_path):
    from ca.mapping_measure import SCHEMA, PROTOCOL, file_hash
    (tmp_path/'run.json').write_text(json.dumps({'schema':'cloudanalyzer.mapping_run.v1',
        'layout_file':{'sha256':'b'*64}, 'history':[], 'status':'finished'}))
    (tmp_path/'job.json').write_text(json.dumps({'source':{'sha256':'a'*64}}))
    report = {'schema':SCHEMA, 'protocol':PROTOCOL, 'status':'process_succeeded', 'exit_code':0,
        'run_sha256':file_hash(tmp_path/'run.json'), 'job_sha256':file_hash(tmp_path/'job.json'),
        'sample_interval_seconds':.05, 'wall_time_seconds':1.5, 'sampled_peak_tree_rss_bytes':4096,
        'logical_output_bytes':1024}
    return tmp_path, report


def save_card(fixture, **changes):
    root, report = fixture
    (root/'measurement.json').write_text(json.dumps({**report, **changes}))
    return scorecard(str(root))


def test_failed_measurement_exposes_bound_diagnostics_without_success_metrics(measured_run):
    card = save_card(measured_run, status='measurement_failed', exit_code=-9,
                     error='monitor unavailable', cleanup_error='termination unconfirmed')
    assert card['measurement_diagnostics'] == {'exit_code':-9, 'error':'monitor unavailable',
        'cleanup_error':'termination unconfirmed', 'output_size_error':None}
    assert card['wall_time_seconds'] is None
    assert card['sampled_peak_tree_rss_bytes'] is None
    assert 'wall_time_seconds' not in compare(card, card)['delta']


def test_size_failure_explains_null_and_preserves_independent_metrics(measured_run):
    card = save_card(measured_run, output_size_error='unreadable directory')
    assert card['logical_output_bytes'] is None
    assert card['measurement_diagnostics']['output_size_error'] == 'unreadable directory'
    assert card['wall_time_seconds'] == 1.5
    assert compare(card, card)['delta']['logical_output_bytes'] is None


@pytest.mark.parametrize('key', ['error', 'cleanup_error'])
def test_contradictory_success_receipt_does_not_expose_performance(measured_run, key):
    card = save_card(measured_run, **{key:'recorded failure'})
    assert card['wall_time_seconds'] is None
    assert card['measurement_diagnostics'][key] == 'recorded failure'


def test_stale_measurement_diagnostics_are_not_attached_to_current_run(measured_run):
    card = save_card(measured_run, run_sha256='f'*64, error='unrelated failure')
    assert card['measurement_status'] == 'stale_or_unsupported'
    assert all(value is None for value in card['measurement_diagnostics'].values())


@pytest.mark.parametrize('key,reason', [
    ('schema', 'Measurement schema is missing or unsupported.'),
    ('protocol', 'Measurement protocol is missing or unsupported.'),
    ('run_sha256', 'Measurement run_sha256 is missing or does not match the parsed run.json.'),
    ('job_sha256', 'Measurement job_sha256 is missing or does not match the parsed job.json.'),
])
@pytest.mark.parametrize('value', [None, 'unsupported', []])
def test_binding_failure_explains_rejected_field(measured_run, key, reason, value):
    card = save_card(measured_run, **{key:value, 'error':'unrelated process error'})
    assert card['measurement_status'] == 'stale_or_unsupported'
    assert card['measurement_binding_reasons'] == [reason]
    assert card['wall_time_seconds'] is None
    assert all(value is None for value in card['measurement_diagnostics'].values())


@pytest.mark.parametrize('filename', ['run.json', 'job.json'])
def test_symlink_binding_failure_has_explicit_reason(measured_run, filename):
    root, _ = measured_run
    path = root/filename
    target = root/(filename + '.original')
    path.rename(target)
    path.symlink_to(target)
    card = save_card(measured_run)
    assert card['measurement_binding_reasons'] == [filename + ' is a symbolic link.']
    assert card['measurement_status'] == 'stale_or_unsupported'
    assert card['wall_time_seconds'] is None


def test_cli_lists_multiple_binding_failures(measured_run):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    save_card(measured_run, run_sha256=None, job_sha256='f'*64)
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root)])
    assert result.exit_code == 0, result.output
    card = json.loads(result.stdout)
    assert len(card['measurement_binding_reasons']) == 2
    assert 'run_sha256' in card['measurement_binding_reasons'][0]
    assert 'job_sha256' in card['measurement_binding_reasons'][1]
    assert card['status'] == 'finished'


def test_bound_failed_receipt_has_no_binding_failure(measured_run):
    card = save_card(measured_run, status='process_failed', exit_code=3)
    assert card['measurement_binding_reasons'] == []
    assert card['measurement_diagnostics']['exit_code'] == 3


def test_diagnostic_text_is_bounded_and_invalid_types_are_omitted(measured_run):
    card = save_card(measured_run, status='interrupted', exit_code=True,
                     error='x'*5000, cleanup_error=['not text'])
    assert len(card['measurement_diagnostics']['error']) == 4096
    assert card['measurement_diagnostics']['exit_code'] is None
    assert card['measurement_diagnostics']['cleanup_error'] is None


@pytest.mark.parametrize('raw', [b'{truncated', b'[]', b'{"x":"\xff"}',
    b'{"x":' + b'['*2000 + b'0' + b']'*2000 + b'}'], ids=['truncated', 'array', 'encoding', 'nesting'])
def test_invalid_optional_measurement_preserves_generation_summary(measured_run, raw):
    root, _ = measured_run
    path = root/'measurement.json'; path.write_bytes(raw)
    card = scorecard(str(root))
    assert card['status'] == 'finished'
    assert card['identity']['source_sha256'] == 'a'*64
    assert card['measurement_status'] == 'invalid_report'
    assert card['measurement_report_error']
    assert card['wall_time_seconds'] is None
    assert all(value is None for value in card['measurement_diagnostics'].values())
    assert 'wall_time_seconds' not in compare(card, card)['delta']
    assert path.read_bytes() == raw


def test_unreadable_optional_measurement_retains_reason(measured_run, monkeypatch):
    from pathlib import Path
    root, _ = measured_run; path = root/'measurement.json'; path.write_text('{}')
    original = Path.open
    def read(self, *args, **kwargs):
        if self == path:
            raise PermissionError('fixture unreadable measurement')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', read)
    card = scorecard(str(root))
    assert card['measurement_status'] == 'invalid_report'
    assert 'unreadable measurement' in card['measurement_report_error']
    assert card['status'] == 'finished'


def test_oversized_optional_measurement_is_unavailable(measured_run, monkeypatch):
    from ca import mapping_scorecard
    root, _ = measured_run
    monkeypatch.setattr(mapping_scorecard, 'MAX_METADATA_BYTES', 1024)
    (root/'measurement.json').write_bytes(b' '*1025)
    card = scorecard(str(root))
    assert card['measurement_status'] == 'invalid_report'
    assert 'exceeds' in card['measurement_report_error']


def test_invalid_required_run_still_fails(measured_run):
    root, _ = measured_run
    (root/'run.json').write_text('{broken')
    with pytest.raises(ValueError):
        scorecard(str(root))


@pytest.mark.parametrize('extra', [
    '"status":"process_failed"',
    '"nested":{"key":1,"k\\u0065y":2}',
    '"platform":NaN', '"command":[Infinity]',
    '"nested":{"value":-Infinity}', '"nested":{"value":1e999}',
])
def test_ambiguous_optional_json_preserves_cli_summary(measured_run, extra):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, report = measured_run
    raw = json.dumps(report)[:-1] + ',' + extra + '}'
    path = root/'measurement.json'
    path.write_text(raw)
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root)])
    assert result.exit_code == 0, result.output
    card = json.loads(result.stdout)
    assert card['status'] == 'finished'
    assert card['measurement_status'] == 'invalid_report'
    assert card['measurement_report_error']
    assert card['wall_time_seconds'] is None
    assert not compare(card, card)['performance_comparison']['eligible']
    assert path.read_text() == raw


@pytest.mark.parametrize('filename', ['run.json', 'job.json'])
@pytest.mark.parametrize('extra', ['"extra":{"x":1,"x":2}', '"extra":1e999', '"extra":NaN'])
def test_ambiguous_required_json_fails_cleanly(measured_run, filename, extra):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    path = root/filename
    raw = path.read_text()[:-1] + ',' + extra + '}'
    path.write_text(raw)
    with pytest.raises(ValueError, match='metadata contains'):
        scorecard(str(root))
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root)])
    assert result.exit_code != 0
    assert 'metadata contains' in result.output
    assert path.read_text() == raw


def test_large_integer_metrics_do_not_overflow_float_conversion(measured_run):
    huge = 10**400
    card = save_card(measured_run, logical_output_bytes=huge)
    assert card['logical_output_bytes'] == huge
    assert compare(card, card)['delta']['logical_output_bytes'] == 0
    json.dumps(card, allow_nan=False)


@pytest.mark.parametrize('reverse', [False, True])
def test_cli_comparison_preserves_other_metrics_on_delta_overflow(measured_run, reverse):
    from ca.mapping_measure import file_hash
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, report = measured_run
    baseline = root/'baseline'; baseline.mkdir()
    for name in ('run.json', 'job.json'):
        (baseline/name).write_bytes((root/name).read_bytes())
    huge = 10**400
    old_time, new_time = (huge, 1.5) if reverse else (1.5, huge)
    (baseline/'measurement.json').write_text(json.dumps({**report, 'wall_time_seconds':old_time,
        'run_sha256':file_hash(baseline/'run.json'), 'job_sha256':file_hash(baseline/'job.json')}))
    (root/'measurement.json').write_text(json.dumps({**report, 'wall_time_seconds':new_time,
        'logical_output_bytes':2048}))
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root), '--baseline', str(baseline)])
    assert result.exit_code == 0, result.output
    comparison = json.loads(result.stdout)
    assert comparison['before']['wall_time_seconds'] == old_time
    assert comparison['after']['wall_time_seconds'] == new_time
    assert comparison['delta']['wall_time_seconds'] is None
    assert comparison['delta']['logical_output_bytes'] == 1024
    assert comparison['delta_errors']['wall_time_seconds'] == 'Difference exceeds the finite floating-point range.'
    assert 'logical_output_bytes' not in comparison['delta_errors']
    assert comparison['performance_comparison']['eligible']


def test_large_integer_deltas_remain_exact(measured_run):
    card = save_card(measured_run, logical_output_bytes=10**400)
    other = {**card, 'logical_output_bytes':10**400 + 1}
    assert compare(card, other)['delta']['logical_output_bytes'] == 1
    assert compare(other, card)['delta']['logical_output_bytes'] == -1
    assert 'logical_output_bytes' not in compare(card, other)['delta_errors']


@pytest.mark.parametrize('key', ['generated_length_m', 'trajectory_length_m',
    'unexplained_traces', 'wall_time_seconds', 'sampled_peak_tree_rss_bytes', 'logical_output_bytes'])
@pytest.mark.parametrize('left,right,reason', [
    (None, 0, 'Baseline value is unavailable.'),
    (0, None, 'Current value is unavailable.'),
    (None, None, 'Baseline and current values are unavailable.'),
])
def test_missing_delta_identifies_unavailable_side(measured_run, key, left, right, reason):
    card = save_card(measured_run)
    result = compare({**card, key:left}, {**card, key:right})
    assert result['delta'][key] is None
    assert result['delta_errors'][key] == reason
    assert result['delta']['recorded_actions'] == 0
    assert 'recorded_actions' not in result['delta_errors']


def test_ineligible_performance_is_omitted_without_missing_value_errors(measured_run):
    card = save_card(measured_run)
    result = compare(card, {**card, 'status':'different-stage', 'wall_time_seconds':None})
    assert not result['performance_comparison']['eligible']
    for key in ('wall_time_seconds', 'sampled_peak_tree_rss_bytes', 'logical_output_bytes'):
        assert key not in result['delta']
        assert key not in result['delta_errors']


def test_cli_comparison_explains_missing_metrics(measured_run):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    save_card(measured_run, logical_output_bytes=None)
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root), '--baseline', str(root)])
    assert result.exit_code == 0, result.output
    comparison = json.loads(result.stdout)
    assert comparison['delta']['logical_output_bytes'] is None
    assert comparison['delta_errors']['logical_output_bytes'] == 'Baseline and current values are unavailable.'
    assert comparison['delta']['wall_time_seconds'] == 0
    assert 'wall_time_seconds' not in comparison['delta_errors']


def test_explicit_metadata_nesting_boundary(tmp_path):
    from ca.mapping_scorecard import _read
    path = tmp_path/'metadata.json'
    path.write_text('{"x":' + '['*127 + '0' + ']'*127 + '}')
    assert 'x' in _read(path)[0]
    path.write_text('{"x":' + '['*128 + '0' + ']'*128 + '}')
    with pytest.raises(ValueError, match='128 levels'):
        _read(path)


@pytest.mark.parametrize('filename', ['run.json', 'job.json'])
@pytest.mark.parametrize('new_receipt', [False, True])
def test_summary_and_receipt_binding_use_parsed_bytes(measured_run, monkeypatch, filename, new_receipt):
    import hashlib
    from ca import mapping_scorecard
    root, report = measured_run
    original_run = (root/'run.json').read_bytes()
    original_read = mapping_scorecard._read
    reads = []
    def read(path):
        reads.append(path.name)
        if path.name == 'measurement.json':
            target = root/filename
            data = json.loads(target.read_text())
            if filename == 'run.json':
                data['status'] = 'changed'
                data['history'] = [{}]
            else:
                data['source']['sha256'] = 'c'*64
            raw = json.dumps(data).encode()
            target.write_bytes(raw)
            if new_receipt:
                report[filename.removesuffix('.json') + '_sha256'] = hashlib.sha256(raw).hexdigest()
                path.write_text(json.dumps(report))
        return original_read(path)
    (root/'measurement.json').write_text(json.dumps(report))
    monkeypatch.setattr(mapping_scorecard, '_read', read)
    card = scorecard(str(root))
    assert card['run_sha256'] == hashlib.sha256(original_run).hexdigest()
    assert card['status'] == 'finished'
    assert card['recorded_actions'] == 0
    assert card['identity']['source_sha256'] == 'a'*64
    assert reads == ['run.json', 'job.json', 'measurement.json']
    assert card['measurement_status'] == ('stale_or_unsupported' if new_receipt else 'process_succeeded')
    assert card['wall_time_seconds'] == (None if new_receipt else 1.5)


@pytest.mark.parametrize('filename,keys', [
    ('job.json', ['source']), ('run.json', ['layout_file']),
    ('run.json', ['output']), ('run.json', ['output', 'diagnosis']),
    ('run.json', ['output', 'diagnosis', 'extent']),
    ('run.json', ['output', 'diagnosis', 'review_holds']),
])
@pytest.mark.parametrize('value', [[], '', False, 1])
def test_invalid_metadata_objects_report_the_field(measured_run, filename, keys, value):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    path = root/filename
    data = json.loads(path.read_text())
    parent = data
    for key in keys[:-1]:
        parent = parent.setdefault(key, {})
    parent[keys[-1]] = value
    raw = json.dumps(data)
    path.write_text(raw)
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root)])
    assert result.exit_code != 0
    assert filename + '.' + '.'.join(keys) + ' must be an object' in result.output
    assert path.read_text() == raw


@pytest.mark.parametrize('history', [None, 'actions', {}, 3, [None], ['action']])
def test_invalid_history_cannot_be_counted_as_actions(measured_run, history):
    root, _ = measured_run
    path = root/'run.json'
    data = json.loads(path.read_text()); data['history'] = history
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='history must be an array of objects'):
        scorecard(str(root))


@pytest.mark.parametrize('output', [None, {}, {'diagnosis':None},
    {'diagnosis':{'extent':None, 'review_holds':None}}])
def test_unavailable_diagnosis_keeps_partial_run_summary(measured_run, output):
    root, _ = measured_run
    path = root/'run.json'
    data = json.loads(path.read_text()); data['output'] = output
    data.pop('history')
    path.write_text(json.dumps(data))
    card = scorecard(str(root))
    assert card['status'] == 'finished'
    assert card['recorded_actions'] == 0
    assert card['generated_length_m'] is None
    assert card['unexplained_traces'] is None


def test_compatible_comparison_reports_eligibility(measured_run):
    card = save_card(measured_run)
    result = compare(card, card)
    assert result['performance_comparison'] == {'eligible':True, 'reasons':[]}
    assert result['delta']['wall_time_seconds'] == 0


@pytest.mark.parametrize('change,reason', [
    ({'status':None}, 'Generation stage is missing'),
    ({'status':'other-stage'}, 'Generation stages differ'),
    ({'measurement_status':'interrupted'}, 'successful, current measurements'),
    ({'measurement_protocol':'unknown'}, 'Measurement protocols'),
    ({'sample_interval_seconds':None}, 'Sampling intervals are missing'),
    ({'sample_interval_seconds':True}, 'Sampling intervals are missing'),
    ({'sample_interval_seconds':float('nan')}, 'Sampling intervals are missing'),
    ({'sample_interval_seconds':.1}, 'Sampling intervals differ'),
])
def test_incompatible_comparison_explains_omitted_metrics(measured_run, change, reason):
    card = save_card(measured_run)
    result = compare(card, {**card, **change})
    assert result['performance_comparison']['eligible'] is False
    assert any(reason in text for text in result['performance_comparison']['reasons'])
    assert 'wall_time_seconds' not in result['delta']
    assert 'recorded_actions' in result['delta']


def test_two_unknown_generation_stages_are_not_comparable(measured_run):
    card = {**save_card(measured_run), 'status':None}
    result = compare(card, card)
    assert not result['performance_comparison']['eligible']
    assert 'wall_time_seconds' not in result['delta']


def test_timed_out_measurement_cannot_supply_success_deltas(measured_run):
    card = save_card(measured_run, status='timed_out', exit_code=-15, error='timeout', timeout_seconds=1)
    assert card['measurement_status'] == 'timed_out'
    assert card['measurement_diagnostics']['error'] == 'timeout'
    assert card['wall_time_seconds'] is None
    assert not compare(card, card)['performance_comparison']['eligible']


@pytest.mark.parametrize('baseline', [False, True])
def test_cli_scorecard_and_comparison_are_read_only(measured_run, baseline):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    card = save_card(measured_run)
    before = {path.name:path.read_bytes() for path in root.iterdir()}
    args = ['mapping-scorecard', str(root)]
    if baseline:
        args += ['--baseline', str(root)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == (compare(card, card) if baseline else card)
    assert {path.name:path.read_bytes() for path in root.iterdir()} == before


def test_cli_damaged_optional_report_still_returns_summary(measured_run):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    (root/'measurement.json').write_text('{damaged')
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['measurement_status'] == 'invalid_report'


def test_cli_missing_run_returns_nonzero(tmp_path):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    result = CliRunner().invoke(app, ['mapping-scorecard', str(tmp_path/'missing')])
    assert result.exit_code != 0
    assert not (tmp_path/'missing').exists()


def test_cli_text_summary_is_read_only_and_keeps_unknowns(measured_run):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    save_card(measured_run)
    before = {path.name:path.read_bytes() for path in root.iterdir()}
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root), '--text'])
    assert result.exit_code == 0, result.output
    assert 'Mapping scorecard' in result.stdout
    assert 'Wall time (s): 1.5' in result.stdout
    assert 'Generated length (m): unavailable' in result.stdout
    assert 'Recorded actions: 0' in result.stdout
    assert 'independent accuracy and human intervention counts are unknown' in result.stdout
    assert {path.name:path.read_bytes() for path in root.iterdir()} == before


def test_text_comparison_shows_signed_deltas_and_missing_reasons(measured_run):
    from ca.mapping_scorecard import format_text
    card = save_card(measured_run)
    other = {**card, 'wall_time_seconds':2.5, 'logical_output_bytes':512}
    rendered = format_text(compare(card, other))
    assert 'Delta (current - baseline)' in rendered
    assert 'Performance comparison: eligible' in rendered
    assert 'Wall time (s): +1.0' in rendered
    assert 'Logical output (bytes): -512' in rendered
    assert 'Baseline and current values are unavailable.' in rendered
    incompatible = format_text(compare(card, {**other, 'status':'other-stage'}))
    delta_section = incompatible.split('Delta (current - baseline)')[1]
    assert 'Performance comparison: ineligible' in delta_section
    assert 'Generation stages differ.' in delta_section
    assert 'Wall time (s)' not in delta_section


def test_text_diagnostics_escape_terminal_control_characters(measured_run):
    from ca.mapping_scorecard import format_text
    card = save_card(measured_run, status='process_failed', exit_code=7,
                     error='failure\nforged line\x1b[2J')
    rendered = format_text(card)
    assert 'exit_code: 7' in rendered
    assert '\x1b' not in rendered
    assert '\\u001b[2J' in rendered
    assert '\\nforged line' in rendered
    stale = format_text(save_card(measured_run, run_sha256=None))
    assert 'Binding:' in stale and 'run_sha256 is missing' in stale


def test_module_text_output_matches_main_cli(measured_run):
    import subprocess
    import sys
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    save_card(measured_run)
    args = [str(root), '--baseline', str(root), '--text']
    result = subprocess.run([sys.executable, '-m', 'ca.mapping_scorecard', *args],
                            capture_output=True, text=True, timeout=15)
    cli = CliRunner().invoke(app, ['mapping-scorecard', *args])
    assert result.returncode == cli.exit_code == 0
    assert result.stdout == cli.stdout


def test_cli_incompatible_baseline_returns_nonzero(measured_run, tmp_path):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    root, _ = measured_run
    other = tmp_path/'baseline'; other.mkdir()
    (other/'run.json').write_bytes((root/'run.json').read_bytes())
    (other/'job.json').write_text(json.dumps({'source':{'sha256':'c'*64}}))
    result = CliRunner().invoke(app, ['mapping-scorecard', str(root), '--baseline', str(other)])
    assert result.exit_code != 0
    assert 'identical source' in result.output
