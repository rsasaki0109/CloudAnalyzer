"""Published derivative validation and rejected-policy accounting."""
import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT/"benchmarks/slam/flatwall-gicp-transported"
spec = importlib.util.spec_from_file_location("transported_packet", PACKET/"verify.py")
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def test_rejected_policies_and_native_gradient_state_accounting():
    result = packet.verify()
    assert result["gradient_states"] == 3728
    assert result["rematched_original_convergence_regressions"] == 34
    assert result["native_registration_policy_added"] is False


def test_changed_adoption_claim_is_rejected(tmp_path):
    target = tmp_path/"packet"
    shutil.copytree(PACKET, target)
    path = target/"audit/summary.json"
    path.write_bytes(path.read_bytes().replace(b'"native_registration_policy_added": false', b'"native_registration_policy_added": true', 1))
    with pytest.raises(AssertionError): packet.verify(target)
