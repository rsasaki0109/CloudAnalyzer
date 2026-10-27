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
