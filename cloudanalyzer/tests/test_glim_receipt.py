import json

import pytest

from ca.glim_receipt import MAX_RECEIPT_BYTES, read_glim_receipt


@pytest.mark.parametrize('extra', [
    '"status":"failed","status":"process_succeeded"',
    '"nested":{"sha256":"first","sha256":"second"}',
    '"nested":[{"key":1,"k\\u0065y":2}]',
    '"value":NaN', '"value":Infinity', '"value":-Infinity',
    '"value":1e999', '"value":-1e999',
])
def test_ambiguous_or_nonfinite_json_rejected(tmp_path, extra):
    path = tmp_path/'glim-run.json'
    raw = ('{"schema":"cloudanalyzer.glim_run.v1",' + extra + '}').encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='invalid GLIM run report JSON'):
        read_glim_receipt(path)
    assert path.read_bytes() == raw


def test_valid_receipt_preserves_exact_bytes_and_numeric_text(tmp_path):
    path = tmp_path/'glim-run.json'
    raw = b'{\r\n "schema":"cloudanalyzer.glim_run.v1", "note":"NaN 1e999", "value":1.7e308 }\r\n'
    path.write_bytes(raw)
    saved, report = read_glim_receipt(path)
    assert saved == raw
    assert report == json.loads(raw)


@pytest.mark.parametrize('raw', [b'', b'[]', b'{}', b'{broken', b'{"schema":"\xff"}',
    b'{"deep":' + b'['*2000 + b'0' + b']'*2000 + b'}', b' '*(MAX_RECEIPT_BYTES+1)])
def test_invalid_bounded_reports_raise_value_error(tmp_path, raw):
    path = tmp_path/'glim-run.json'; path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_glim_receipt(path)


def test_receipt_symlink_rejected(tmp_path):
    target = tmp_path/'original'; target.write_text('{"schema":"cloudanalyzer.glim_run.v1"}')
    path = tmp_path/'link'; path.symlink_to(target)
    with pytest.raises(ValueError, match='regular file'):
        read_glim_receipt(path)
