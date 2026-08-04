"""Tests for the versioned evaluation protocol contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ca.protocol import (
    PROTOCOL_SCHEMA_VERSION,
    ProtocolValidationError,
    attach_evaluation_protocol,
    load_protocol,
    validate_protocol,
)
from cloudanalyzer_cli.main import app


runner = CliRunner()


def _protocol() -> dict:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "name": "synthetic-trajectory-v1",
        "kind": "trajectory",
        "conventions": {
            "coordinate_frame": "map",
            "length_unit": "m",
            "time_unit": "s",
            "alignment": "none",
            "association": "timestamp_interpolation",
            "quaternion_order": "xyzw",
        },
        "parameters": {"max_time_delta_s": 0.05, "rpe_distance_m": [1.0, 10.0]},
        "provenance": {"dataset": "synthetic", "hash_inputs": True},
    }


def test_validate_protocol_normalizes_defaults_and_hashes() -> None:
    document = validate_protocol(
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "name": "minimal",
            "kind": "map",
        }
    )

    assert document["conventions"]["length_unit"] == "m"
    assert document["parameters"] == {}
    assert document["inputs"] == {}
    assert document["protocol_id"].startswith("sha256:")
    assert len(document["protocol_sha256"]) == 64


def test_protocol_identity_is_order_independent() -> None:
    first = validate_protocol(_protocol())
    second = validate_protocol(
        {
            "provenance": {"hash_inputs": True, "dataset": "synthetic"},
            "parameters": {"rpe_distance_m": [1.0, 10.0], "max_time_delta_s": 0.05},
            "conventions": {
                "association": "timestamp_interpolation",
                "alignment": "none",
                "time_unit": "s",
                "length_unit": "m",
                "coordinate_frame": "map",
                "quaternion_order": "xyzw",
            },
            "kind": "trajectory",
            "name": "synthetic-trajectory-v1",
            "schema_version": PROTOCOL_SCHEMA_VERSION,
        }
    )

    assert first["protocol_sha256"] == second["protocol_sha256"]


def test_invalid_protocol_schema_is_rejected() -> None:
    with pytest.raises(ProtocolValidationError, match="Unsupported schema_version"):
        validate_protocol({"schema_version": "cloudanalyzer.protocol.v0"})


def test_yaml_load_and_run_manifest(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        "\n".join(
            [
                f"schema_version: {PROTOCOL_SCHEMA_VERSION}",
                "name: yaml-test",
                "kind: point_cloud",
                "provenance:",
                "  hash_inputs: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.txt"
    input_path.write_text("fixture\n", encoding="utf-8")

    document = load_protocol(protocol_path)
    result = attach_evaluation_protocol(
        {"ok": True},
        protocol_path,
        command="ca test",
        kind="point_cloud",
        inputs=(input_path,),
        options={"threshold": 0.1},
    )

    assert document["name"] == "yaml-test"
    manifest = result["evaluation_protocol"]["run"]["inputs"][0]
    assert manifest["size_bytes"] > 0
    assert len(manifest["sha256"]) == 64
    assert result["evaluation_protocol"]["run"]["options"] == {"threshold": 0.1}


def test_protocol_validate_cli_outputs_normalized_json(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(_protocol()), encoding="utf-8")

    result = runner.invoke(app, ["protocol", "validate", str(path), "--format-json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert payload["protocol_id"].startswith("sha256:")


def test_evaluate_cli_attaches_protocol(source_and_target_files, tmp_path: Path) -> None:
    source, target = source_and_target_files
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        "\n".join(
            [
                f"schema_version: {PROTOCOL_SCHEMA_VERSION}",
                "name: point-cloud-fixture",
                "kind: point_cloud",
                "conventions:",
                "  coordinate_frame: map",
                "  length_unit: m",
                "  time_unit: s",
                "  alignment: none",
                "  association: not_applicable",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    result = runner.invoke(
        app,
        [
            "evaluate",
            source,
            target,
            "--protocol",
            str(protocol_path),
            "--output-json",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    protocol = payload["evaluation_protocol"]
    assert protocol["name"] == "point-cloud-fixture"
    assert protocol["run"]["command"] == "ca evaluate"
    assert len(protocol["run"]["inputs"]) == 2
    assert all(entry["sha256"] for entry in protocol["run"]["inputs"])
