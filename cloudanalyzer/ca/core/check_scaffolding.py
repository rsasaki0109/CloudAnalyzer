"""Stable, minimal interface for starter `cloudanalyzer.yaml` generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Protocol

SUPPORTED_CHECK_SCAFFOLD_PROFILES = (
    "mapping",
    "localization",
    "perception",
    "integrated",
)


@dataclass(slots=True)
class CheckScaffoldRequest:
    """Input contract shared by config-scaffolding strategies."""

    profile: str = "integrated"

    def __post_init__(self) -> None:
        normalized = self.profile.strip().lower()
        if normalized not in SUPPORTED_CHECK_SCAFFOLD_PROFILES:
            valid = ", ".join(SUPPORTED_CHECK_SCAFFOLD_PROFILES)
            raise ValueError(
                f"Unsupported profile '{self.profile}'. Choose one of: {valid}"
            )
        self.profile = normalized


@dataclass(slots=True)
class CheckScaffoldResult:
    """Output contract shared by config-scaffolding strategies."""

    profile: str
    yaml_text: str
    strategy: str
    design: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckScaffoldingStrategy(Protocol):
    """Protocol kept in core after comparing concrete template generators."""

    name: str
    design: str

    def render(self, request: CheckScaffoldRequest) -> CheckScaffoldResult:
        """Render a starter cloudanalyzer config."""


class StaticProfileCheckScaffoldingStrategy:
    """Stable starter-config renderer selected after experiment comparison."""

    name = "static_profiles"
    design = "functional"

    _TEMPLATES = {
        "mapping": dedent(
            """
            version: 1
            project: mapping-qa
            summary_output_json: qa/summary.json

            defaults:
              thresholds: [0.02, 0.05, 0.1]
              report_dir: qa/reports
              json_dir: qa/results

            checks:
              - id: mapping-postprocess
                kind: artifact
                source: outputs/map.pcd
                reference: baselines/map_ref.pcd
                gate:
                  min_auc: 0.95
                  max_chamfer: 0.02
            """
        ).strip(),
        "localization": dedent(
            """
            version: 1
            project: localization-qa
            summary_output_json: qa/summary.json

            defaults:
              max_time_delta: 0.05
              report_dir: qa/reports
              json_dir: qa/results

            checks:
              - id: localization-run
                kind: trajectory
                estimated: outputs/trajectory.csv
                reference: baselines/trajectory_ref.csv
                alignment: rigid
                gate:
                  max_ate: 0.5
                  max_rpe: 0.2
                  max_drift: 1.0
                  min_coverage: 0.9
            """
        ).strip(),
        "perception": dedent(
            """
            version: 1
            project: perception-qa
            summary_output_json: qa/summary.json

            defaults:
              thresholds: [0.02, 0.05, 0.1]
              report_dir: qa/reports
              json_dir: qa/results

            checks:
              - id: perception-output
                kind: artifact
                source: outputs/reconstruction.pcd
                reference: baselines/reconstruction_ref.pcd
                gate:
                  min_auc: 0.95
                  max_chamfer: 0.02
            """
        ).strip(),
        "integrated": dedent(
            """
            version: 1
            project: localization-mapping-perception
            summary_output_json: qa/summary.json

            defaults:
              thresholds: [0.05, 0.1, 0.2]
              max_time_delta: 0.05
              report_dir: qa/reports
              json_dir: qa/results

            checks:
              - id: mapping-postprocess
                kind: artifact
                source: outputs/map.pcd
                reference: baselines/map_ref.pcd
                gate:
                  min_auc: 0.95
                  max_chamfer: 0.02

              - id: localization-run
                kind: trajectory
                estimated: outputs/trajectory.csv
                reference: baselines/trajectory_ref.csv
                alignment: rigid
                gate:
                  max_ate: 0.5
                  max_rpe: 0.2
                  max_drift: 1.0
                  min_coverage: 0.9

              - id: perception-output
                kind: artifact
                source: outputs/reconstruction.pcd
                reference: baselines/reconstruction_ref.pcd
                gate:
                  min_auc: 0.95
                  max_chamfer: 0.02

              - id: integrated-run
                kind: run
                map: outputs/map.pcd
                map_reference: baselines/map_ref.pcd
                trajectory: outputs/trajectory.csv
                trajectory_reference: baselines/trajectory_ref.csv
                alignment: rigid
                gate:
                  min_auc: 0.95
                  max_chamfer: 0.02
                  max_ate: 0.5
                  max_rpe: 0.2
                  max_drift: 1.0
                  min_coverage: 0.9
            """
        ).strip(),
    }

    def render(self, request: CheckScaffoldRequest) -> CheckScaffoldResult:
        yaml_text = self._TEMPLATES[request.profile] + "\n"
        return CheckScaffoldResult(
            profile=request.profile,
            yaml_text=yaml_text,
            strategy=self.name,
            design=self.design,
            metadata={
                "profiles_supported": list(SUPPORTED_CHECK_SCAFFOLD_PROFILES),
                "line_count": len(yaml_text.splitlines()),
            },
        )


def render_check_scaffold(
    profile: str = "integrated",
    strategy: CheckScaffoldingStrategy | None = None,
) -> CheckScaffoldResult:
    """Render a starter `cloudanalyzer.yaml` via the stabilized strategy."""

    renderer = strategy or StaticProfileCheckScaffoldingStrategy()
    return renderer.render(CheckScaffoldRequest(profile=profile))
