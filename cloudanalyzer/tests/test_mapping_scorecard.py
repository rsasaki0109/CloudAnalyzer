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


def test_explicit_metadata_nesting_boundary(tmp_path):
    from ca.mapping_scorecard import _read
    path = tmp_path/'metadata.json'
    path.write_text('{"x":' + '['*127 + '0' + ']'*127 + '}')
    assert 'x' in _read(path)
    path.write_text('{"x":' + '['*128 + '0' + ']'*128 + '}')
    with pytest.raises(ValueError, match='128 levels'):
        _read(path)


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
