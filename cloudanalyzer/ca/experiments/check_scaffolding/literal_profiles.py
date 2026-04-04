"""Literal profile templates kept as a comparable functional baseline."""

from __future__ import annotations

from textwrap import dedent

from ca.core import CheckScaffoldRequest, CheckScaffoldResult


class LiteralProfilesStrategy:
    """Render each profile from a direct YAML literal."""

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
            metadata={"template_count": len(self._TEMPLATES)},
        )
