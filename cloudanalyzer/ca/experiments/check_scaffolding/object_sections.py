"""Class-oriented template composition for starter QA configs."""

from __future__ import annotations

from dataclasses import dataclass

from ca.core import CheckScaffoldRequest, CheckScaffoldResult


def _indent(lines: list[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [f"{prefix}{line}" if line else "" for line in lines]


@dataclass(slots=True)
class Gate:
    values: list[tuple[str, str]]

    def render(self, indent: int = 0) -> list[str]:
        lines = ["gate:"]
        lines.extend(_indent([f"{key}: {value}" for key, value in self.values], 2))
        return _indent(lines, indent)


@dataclass(slots=True)
class CheckBlock:
    check_id: str
    kind: str
    fields: list[tuple[str, str]]
    gate: Gate

    def render(self, indent: int = 0) -> list[str]:
        lines = [f"- id: {self.check_id}", f"  kind: {self.kind}"]
        for key, value in self.fields:
            lines.append(f"  {key}: {value}")
        lines.extend(self.gate.render(indent=2))
        return _indent(lines, indent)


@dataclass(slots=True)
class ProfileTemplate:
    project: str
    defaults: list[tuple[str, str]]
    checks: list[CheckBlock]

    def render(self) -> str:
        lines = [
            "version: 1",
            f"project: {self.project}",
            "summary_output_json: qa/summary.json",
            "",
            "defaults:",
        ]
        lines.extend(_indent([f"{key}: {value}" for key, value in self.defaults], 2))
        lines.extend(["", "checks:"])
        for index, check in enumerate(self.checks):
            if index > 0:
                lines.append("")
            lines.extend(_indent(check.render(), 2))
        return "\n".join(lines) + "\n"


class ObjectSectionsStrategy:
    """Render profile templates from explicit section objects."""

    name = "object_sections"
    design = "oop"

    _TEMPLATES = {
        "mapping": ProfileTemplate(
            project="mapping-qa",
            defaults=[
                ("thresholds", "[0.02, 0.05, 0.1]"),
                ("report_dir", "qa/reports"),
                ("json_dir", "qa/results"),
            ],
            checks=[
                CheckBlock(
                    check_id="mapping-postprocess",
                    kind="artifact",
                    fields=[
                        ("source", "outputs/map.pcd"),
                        ("reference", "baselines/map_ref.pcd"),
                    ],
                    gate=Gate(
                        [
                            ("min_auc", "0.95"),
                            ("max_chamfer", "0.02"),
                        ]
                    ),
                )
            ],
        ),
        "localization": ProfileTemplate(
            project="localization-qa",
            defaults=[
                ("max_time_delta", "0.05"),
                ("report_dir", "qa/reports"),
                ("json_dir", "qa/results"),
            ],
            checks=[
                CheckBlock(
                    check_id="localization-run",
                    kind="trajectory",
                    fields=[
                        ("estimated", "outputs/trajectory.csv"),
                        ("reference", "baselines/trajectory_ref.csv"),
                        ("alignment", "rigid"),
                    ],
                    gate=Gate(
                        [
                            ("max_ate", "0.5"),
                            ("max_rpe", "0.2"),
                            ("max_drift", "1.0"),
                            ("min_coverage", "0.9"),
                        ]
                    ),
                )
            ],
        ),
        "perception": ProfileTemplate(
            project="perception-qa",
            defaults=[
                ("thresholds", "[0.02, 0.05, 0.1]"),
                ("report_dir", "qa/reports"),
                ("json_dir", "qa/results"),
            ],
            checks=[
                CheckBlock(
                    check_id="perception-output",
                    kind="artifact",
                    fields=[
                        ("source", "outputs/reconstruction.pcd"),
                        ("reference", "baselines/reconstruction_ref.pcd"),
                    ],
                    gate=Gate(
                        [
                            ("min_auc", "0.95"),
                            ("max_chamfer", "0.02"),
                        ]
                    ),
                )
            ],
        ),
        "integrated": ProfileTemplate(
            project="localization-mapping-perception",
            defaults=[
                ("thresholds", "[0.05, 0.1, 0.2]"),
                ("max_time_delta", "0.05"),
                ("report_dir", "qa/reports"),
                ("json_dir", "qa/results"),
            ],
            checks=[
                CheckBlock(
                    check_id="mapping-postprocess",
                    kind="artifact",
                    fields=[
                        ("source", "outputs/map.pcd"),
                        ("reference", "baselines/map_ref.pcd"),
                    ],
                    gate=Gate([("min_auc", "0.95"), ("max_chamfer", "0.02")]),
                ),
                CheckBlock(
                    check_id="localization-run",
                    kind="trajectory",
                    fields=[
                        ("estimated", "outputs/trajectory.csv"),
                        ("reference", "baselines/trajectory_ref.csv"),
                        ("alignment", "rigid"),
                    ],
                    gate=Gate(
                        [
                            ("max_ate", "0.5"),
                            ("max_rpe", "0.2"),
                            ("max_drift", "1.0"),
                            ("min_coverage", "0.9"),
                        ]
                    ),
                ),
                CheckBlock(
                    check_id="perception-output",
                    kind="artifact",
                    fields=[
                        ("source", "outputs/reconstruction.pcd"),
                        ("reference", "baselines/reconstruction_ref.pcd"),
                    ],
                    gate=Gate([("min_auc", "0.95"), ("max_chamfer", "0.02")]),
                ),
                CheckBlock(
                    check_id="integrated-run",
                    kind="run",
                    fields=[
                        ("map", "outputs/map.pcd"),
                        ("map_reference", "baselines/map_ref.pcd"),
                        ("trajectory", "outputs/trajectory.csv"),
                        ("trajectory_reference", "baselines/trajectory_ref.csv"),
                        ("alignment", "rigid"),
                    ],
                    gate=Gate(
                        [
                            ("min_auc", "0.95"),
                            ("max_chamfer", "0.02"),
                            ("max_ate", "0.5"),
                            ("max_rpe", "0.2"),
                            ("max_drift", "1.0"),
                            ("min_coverage", "0.9"),
                        ]
                    ),
                ),
            ],
        ),
    }

    def render(self, request: CheckScaffoldRequest) -> CheckScaffoldResult:
        yaml_text = self._TEMPLATES[request.profile].render()
        return CheckScaffoldResult(
            profile=request.profile,
            yaml_text=yaml_text,
            strategy=self.name,
            design=self.design,
            metadata={
                "object_count": len(self._TEMPLATES[request.profile].checks) + 2,
            },
        )
