"""Pipeline-style assembly for starter QA configs."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import yaml  # type: ignore[import-untyped]

from ca.core import CheckScaffoldRequest, CheckScaffoldResult


def _base_document(project: str) -> dict:
    return {
        "version": 1,
        "project": project,
        "summary_output_json": "qa/summary.json",
        "defaults": {
            "report_dir": "qa/reports",
            "json_dir": "qa/results",
        },
        "checks": [],
    }


def _apply_artifact_defaults(document: dict, thresholds: list[float]) -> dict:
    updated = deepcopy(document)
    updated["defaults"]["thresholds"] = thresholds
    return updated


def _apply_trajectory_defaults(document: dict) -> dict:
    updated = deepcopy(document)
    updated["defaults"]["max_time_delta"] = 0.05
    return updated


def _append_mapping_check(document: dict) -> dict:
    updated = deepcopy(document)
    updated["checks"].append(
        {
            "id": "mapping-postprocess",
            "kind": "artifact",
            "source": "outputs/map.pcd",
            "reference": "baselines/map_ref.pcd",
            "gate": {
                "min_auc": 0.95,
                "max_chamfer": 0.02,
            },
        }
    )
    return updated


def _append_localization_check(document: dict) -> dict:
    updated = deepcopy(document)
    updated["checks"].append(
        {
            "id": "localization-run",
            "kind": "trajectory",
            "estimated": "outputs/trajectory.csv",
            "reference": "baselines/trajectory_ref.csv",
            "alignment": "rigid",
            "gate": {
                "max_ate": 0.5,
                "max_rpe": 0.2,
                "max_drift": 1.0,
                "min_coverage": 0.9,
            },
        }
    )
    return updated


def _append_perception_check(document: dict) -> dict:
    updated = deepcopy(document)
    updated["checks"].append(
        {
            "id": "perception-output",
            "kind": "artifact",
            "source": "outputs/reconstruction.pcd",
            "reference": "baselines/reconstruction_ref.pcd",
            "gate": {
                "min_auc": 0.95,
                "max_chamfer": 0.02,
            },
        }
    )
    return updated


def _append_integrated_run_check(document: dict) -> dict:
    updated = deepcopy(document)
    updated["checks"].append(
        {
            "id": "integrated-run",
            "kind": "run",
            "map": "outputs/map.pcd",
            "map_reference": "baselines/map_ref.pcd",
            "trajectory": "outputs/trajectory.csv",
            "trajectory_reference": "baselines/trajectory_ref.csv",
            "alignment": "rigid",
            "gate": {
                "min_auc": 0.95,
                "max_chamfer": 0.02,
                "max_ate": 0.5,
                "max_rpe": 0.2,
                "max_drift": 1.0,
                "min_coverage": 0.9,
            },
        }
    )
    return updated


class PipelineOverlaysStrategy:
    """Render starter configs by composing profile-specific transformation steps."""

    name = "pipeline_overlays"
    design = "pipeline"

    _PIPELINES: dict[str, list[Callable[..., dict]]] = {
        "mapping": [
            lambda: _base_document("mapping-qa"),
            lambda doc: _apply_artifact_defaults(doc, [0.02, 0.05, 0.1]),
            _append_mapping_check,
        ],
        "localization": [
            lambda: _base_document("localization-qa"),
            _apply_trajectory_defaults,
            _append_localization_check,
        ],
        "perception": [
            lambda: _base_document("perception-qa"),
            lambda doc: _apply_artifact_defaults(doc, [0.02, 0.05, 0.1]),
            _append_perception_check,
        ],
        "integrated": [
            lambda: _base_document("localization-mapping-perception"),
            lambda doc: _apply_artifact_defaults(doc, [0.05, 0.1, 0.2]),
            _apply_trajectory_defaults,
            _append_mapping_check,
            _append_localization_check,
            _append_perception_check,
            _append_integrated_run_check,
        ],
    }

    def render(self, request: CheckScaffoldRequest) -> CheckScaffoldResult:
        steps = self._PIPELINES[request.profile]
        document = steps[0]()
        for step in steps[1:]:
            document = step(document)
        yaml_text = yaml.safe_dump(
            document,
            sort_keys=False,
            allow_unicode=False,
        )
        return CheckScaffoldResult(
            profile=request.profile,
            yaml_text=yaml_text,
            strategy=self.name,
            design=self.design,
            metadata={"step_count": len(steps)},
        )
