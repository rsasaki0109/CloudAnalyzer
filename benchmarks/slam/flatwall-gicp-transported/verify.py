#!/usr/bin/env python3
"""Verify published bytes and reported rejected-policy/gradient accounting."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(p.read_text(encoding="utf-8"))


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    for name, item in read(root/"packet-files.json")["files"].items():
        data = (root/name).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], name
    summaries = {}
    for group in ["rematched", "fixed"]:
        frames, summary = read(root/group/"frames.json"), read(root/group/"summary.json")
        assert len(frames) == summary["histories"] == 72
        assert len({(r["family"], r["case"], r["row"]) for r in frames}) == 72
        assert dict(Counter(r["termination"] for r in frames)) == summary["statuses"]
        assert dict(Counter(r["original_termination"] for r in frames)) == summary["original_statuses"]
        assert dict(Counter(r["termination"] for r in frames if r["original_converged"])) == summary["original_converged_new_statuses"]
        assert dict(Counter(r["termination"] for r in frames if r["original_termination"] == "max_iterations")) == summary["original_limited_new_statuses"]
        for k in ["iterations", "accepted_steps"]: assert sum(r[k] for r in frames) == summary[k]
        assert sum(r["final_cost"] <= r["original_final_cost"] for r in frames) == summary["same_or_lower_reported_final_cost"]
        summaries[group] = summary
    assert summaries["rematched"]["original_converged_new_statuses"] == {"converged_step": 20, "no_descent": 34}
    assert summaries["fixed"]["original_converged_new_statuses"] == {"converged_step": 48, "max_iterations": 6}
    assert summaries["fixed"]["original_limited_new_statuses"] == {"converged_step": 6, "max_iterations": 4}
    frames, audit = read(root/"audit/frames.json"), read(root/"audit/summary.json")
    assert len(frames) == audit["real_histories"] == 216
    assert len({(r["group"], r["family"], r["case"], r["row"]) for r in frames}) == 216
    for group in ["original", "rematched", "fixed"]:
        assert dict(Counter(r["termination"] for r in frames if r["group"] == group)) == audit["group_statuses"][group]
    assert sum(r["gradient_baselines"] for r in frames) == audit["checked_gradient_baselines"]
    assert sum(r["finite_trials"] for r in frames) == audit["checked_gradient_trials"]
    assert audit["checked_gradient_baselines"]+audit["checked_gradient_trials"] == audit["checked_gradient_states"]
    assert audit["default_registrations_bit_equal"] == audit["cycle_enabled_registrations_bit_equal"] == 72
    assert audit["native_registration_policy_added"] is False
    assert audit["existing_frozen_factor_fields_bit_equal"] is True
    assert audit["serial_parallel_gradient_bits_equal"] is True
    return dict(real_histories=216, gradient_states=audit["checked_gradient_states"],
        rematched_original_convergence_regressions=34, fixed_converged_to_limited=6, fixed_limited_to_converged=6,
        native_registration_policy_added=False,
        scope="Published bytes and reported accounting only; no independent real-array rerun, physical accuracy or performance certificate.")


if __name__ == "__main__": print(json.dumps(verify(), indent=2))
