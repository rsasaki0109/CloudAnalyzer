"""Shared experiment inputs for config scaffolding comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from ca.core import CheckScaffoldResult, CheckSuite, load_check_suite


@dataclass(slots=True)
class ProfileCase:
    """Comparable profile input for scaffolding experiments."""

    profile: str
    description: str
    expected_project: str
    expected_check_ids: tuple[str, ...]
    expected_kinds: tuple[str, ...]


def build_default_profile_cases() -> list[ProfileCase]:
    """Return the shared profile set used across all scaffolding strategies."""

    return [
        ProfileCase(
            profile="mapping",
            description="Single artifact QA slice for map post-processing.",
            expected_project="mapping-qa",
            expected_check_ids=("mapping-postprocess",),
            expected_kinds=("artifact",),
        ),
        ProfileCase(
            profile="localization",
            description="Single trajectory QA slice for localization runs.",
            expected_project="localization-qa",
            expected_check_ids=("localization-run",),
            expected_kinds=("trajectory",),
        ),
        ProfileCase(
            profile="perception",
            description="Single artifact QA slice for 3D reconstruction output.",
            expected_project="perception-qa",
            expected_check_ids=("perception-output",),
            expected_kinds=("artifact",),
        ),
        ProfileCase(
            profile="integrated",
            description="Combined mapping, localization, perception, and integrated run gate.",
            expected_project="localization-mapping-perception",
            expected_check_ids=(
                "mapping-postprocess",
                "localization-run",
                "perception-output",
                "integrated-run",
            ),
            expected_kinds=("artifact", "trajectory", "artifact", "run"),
        ),
    ]


def load_suite_from_result(result: CheckScaffoldResult) -> CheckSuite:
    """Materialize a rendered scaffold and load it through the stable parser."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "cloudanalyzer.yaml"
        config_path.write_text(result.yaml_text, encoding="utf-8")
        return load_check_suite(str(config_path))
