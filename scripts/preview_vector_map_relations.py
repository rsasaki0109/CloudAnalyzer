"""Reproduce geometric previews and explicit adoption on the frozen source scene.

No label shapes or relationships are read. This scene is development evidence;
unsigned normals do not establish a signal's legal control or front face.
"""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
from review_vector_map_relations import ROOT, CONFIG, digest, review


def reproduce(source: Path, cloud: Path, out: Path) -> dict:
    import cloudanalyzer_core as core
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if out.exists():
        raise FileExistsError(str(out))
    if digest(source) != config["input_map_sha256"] or digest(cloud) != config["input_cloud_sha256"]:
        raise ValueError("frozen source/map changed; review this scene again")
    initial = source.read_bytes()
    reports = []
    with tempfile.TemporaryDirectory(prefix="relation-proposals-") as temporary:
        current = source
        for rule_id in (167, 171):
            payload = json.loads(core.propose_vector_map_relations(str(current), rule_id))
            proposal = payload["report"]["proposal"]
            assert proposal["eligible_count"] == 0 and not proposal["limited"]
            assert payload["report"]["edit"] is None
            reports.append(proposal)
        for rule_id, key in ((169, "stop_line:164"), (173, "crosswalk:156")):
            payload = json.loads(core.propose_vector_map_relations(str(current), rule_id))
            proposal = payload["report"]["proposal"]
            assert proposal["eligible_count"] == 1 and not proposal["ambiguous"] and not proposal["limited"]
            eligible = [c for c in proposal["candidates"] if c["eligible"]]
            assert eligible[0]["key"] == key
            # This explicit choice is a reviewed development-scene draft, not
            # automatic adoption of the highest score or a teacher relationship.
            if rule_id == 173:
                held = next(c for c in proposal["candidates"] if c["target_id"] == 152)
                assert not held["eligible"] and held["distance_m"] < eligible[0]["distance_m"]
                assert held["axis_degrees"] > 80 and eligible[0]["axis_degrees"] < 35
            adoption = {"rule_id": rule_id, "map_snapshot": proposal["map_snapshot"], "candidate_key": key}
            payload = json.loads(core.propose_vector_map_relations(str(current), rule_id, json.dumps(adoption)))
            assert payload["report"]["edit"]["changed"]
            reports.append(proposal)
            current = Path(temporary) / f"{rule_id}.json"
            current.write_text(payload["map_json"], encoding="utf-8", newline="\n")
        # The unsupported legacy pedestrian association is explicitly cleared;
        # preview itself never clears or invents control targets.
        edit = next(e for e in config["edits"] if e["rule_id"] == 171)
        payload = json.loads(core.edit_vector_map_relations(str(current), json.dumps(edit)))
        baseline = review(source, cloud, out)
        assert payload["osm"] == (out / "reviewed.osm").read_text(encoding="utf-8")
        assert payload["map_json"] == (out / "reviewed.json").read_text(encoding="utf-8")
    assert source.read_bytes() == initial
    proof = {"reference_inputs": [], "native_sha256": baseline["native_sha256"],
        "input_map_sha256": baseline["input_map_sha256"], "input_cloud_sha256": baseline["input_cloud_sha256"],
        "proposals": reports, "explicit_adoptions": ["stop_line:164", "crosswalk:156"],
        "unsupported_legacy_rule_explicitly_cleared": 171, "unresolved_rules": [167, 171],
        "readonly_preview": True, "reviewed_native_artifacts_exact": True,
        "physical_map_exact": baseline["physical_map_exact"], "source_quality_unchanged": baseline["source_quality_before_after_equal"],
        "limitations": baseline["limitations"]}
    (out / "proposals.json").write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8", newline="\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=ROOT / "notes/supported-intersection-proof/reviewed.json")
    parser.add_argument("--cloud", type=Path, default=ROOT / "notes/hard-intersection-prepared-v2/geometry.las")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = reproduce(args.map, args.cloud, args.out)
    print(json.dumps({"adoptions": report["explicit_adoptions"], "unresolved": report["unresolved_rules"], "physical_map_exact": report["physical_map_exact"]}, indent=2))
