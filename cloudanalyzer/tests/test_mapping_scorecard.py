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
    b'{"x":' + b'['*2000 + b'0' + b']'*2000 + b'}'])
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
