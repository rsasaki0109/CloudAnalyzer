"""Check frozen README stop choices against the original cloud, without references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vector_map_quality_audit import digest, verify_frozen_map


def run(source: Path, proof: Path, output: Path, source_commit: str) -> dict:
    import cloudanalyzer_core as core

    if output.exists():
        raise FileExistsError("choose a new report path")
    _, manifest = verify_frozen_map(proof)
    if digest(source) != manifest["input_cloud_sha256"]:
        raise ValueError("source points differ from the frozen proof")
    discovery = json.loads((proof / "ground-preview.json").read_text())["report"][
        "discovery"
    ]
    candidates = {c["id"]: c for c in discovery["candidates"]}
    options = json.dumps({"scope": "ground_surface"})
    records = []
    for candidate, lanes in [(44, [8]), (110, [24, 25])]:
        confirmation = {
            "candidate": candidate,
            "key": candidates[candidate]["key"],
            "classification": "stop_line",
            "lanes": lanes,
        }
        try:
            core.discover_vector_map_features(
                str(source),
                str(proof / "generated-roads.json"),
                options,
                json.dumps([confirmation]),
            )
        except ValueError as error:
            if "transverse" not in str(error):
                raise
            records.append(
                {
                    "candidate": candidate,
                    "selected_lanes": lanes,
                    "rejected": True,
                    "reason": str(error),
                }
            )
        else:
            raise AssertionError("Longitudinal paint unexpectedly accepted")
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "web/media/vector-map-hard-intersection-inputs.json"
        ).read_text()
    )
    if config != manifest["operator_inputs"]:
        raise ValueError("operator inputs differ from the frozen proof")
    confirmations = [
        {**c, "key": candidates[c["candidate"]]["key"]} for c in config["confirmations"]
    ]
    reviewed = json.loads(
        core.discover_vector_map_features(
            str(source),
            str(proof / "generated-roads.json"),
            options,
            json.dumps(confirmations),
        )
    )
    if json.loads(reviewed["map_json"]) != json.loads(
        (proof / "reviewed.json").read_text()
    ):
        raise AssertionError("Corrected frozen map changed")
    report = {
        "audit_source_commit": source_commit,
        "audit_native_sha256": digest(Path(core._core.__file__)),
        "input_cloud_sha256": manifest["input_cloud_sha256"],
        "generated_osm_sha256": manifest["generated_sha256"],
        "reference_inputs": [],
        "rejected_longitudinal_markings": records,
        "accepted_feature_count": len(confirmations),
        "correct_feature_map_exactly_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("proof", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    report = run(args.source, args.proof, args.output, args.source_commit)
    print(
        f"Both longitudinal choices rejected; {report['accepted_feature_count']} additions preserved exactly."
    )
