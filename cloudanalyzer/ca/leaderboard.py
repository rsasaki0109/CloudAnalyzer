"""Static leaderboard builder for benchmark report bundles."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml  # type: ignore[import-untyped]

import ca


LEADERBOARD_SCHEMA_VERSION = "cloudanalyzer.leaderboard.v0.1"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in: {path}")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level YAML object in: {path}")
    return data


def _safe_id(raw: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw.strip())
    return safe.strip("._-") or "run"


def _unique_id(raw: str, seen: set[str]) -> str:
    base = _safe_id(raw)
    candidate = base
    index = 2
    while candidate in seen:
        candidate = f"{base}-{index}"
        index += 1
    seen.add(candidate)
    return candidate


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _list_or_empty(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _gate_status(metrics: Mapping[str, Any]) -> tuple[str, bool | None]:
    gate = metrics.get("overall_quality_gate")
    if not isinstance(gate, Mapping):
        return "unknown", None
    passed = gate.get("passed")
    if passed is True:
        return "pass", True
    if passed is False:
        return "fail", False
    return "unknown", None


def _combined_artifact_hash(inputs: Mapping[str, Any]) -> str | None:
    hashes: list[str] = []
    for key in ("candidate_map", "candidate_trajectory"):
        value = inputs.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("sha256"), str):
            hashes.append(str(value["sha256"]))
    if not hashes:
        return None
    return hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()


def _method_from_bundle_name(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[0]
    return name


def _extract_row(
    bundle_dir: Path,
    run_id: str,
    *,
    method: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path = bundle_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing benchmark bundle metrics.json: {metrics_path}")

    metrics = _load_json(metrics_path)
    provenance = (
        _load_json(bundle_dir / "provenance.json")
        if (bundle_dir / "provenance.json").is_file()
        else {}
    )
    manifest_lock = (
        _load_yaml(bundle_dir / "manifest.lock.yaml")
        if (bundle_dir / "manifest.lock.yaml").is_file()
        else {}
    )
    lock_suite = _mapping_or_empty(manifest_lock.get("suite"))

    benchmark = _mapping_or_empty(metrics.get("benchmark"))
    map_metrics = _mapping_or_empty(metrics.get("map"))
    traj_metrics = _mapping_or_empty(metrics.get("trajectory"))
    ate = _mapping_or_empty(traj_metrics.get("ate"))
    rpe = _mapping_or_empty(traj_metrics.get("rpe_translation"))
    drift = _mapping_or_empty(traj_metrics.get("drift"))
    matching = _mapping_or_empty(traj_metrics.get("matching"))

    lock_inputs = _mapping_or_empty(manifest_lock.get("inputs"))
    gate_status, gate_passed = _gate_status(metrics)
    rel_run = Path("runs") / run_id
    row_method = (
        method
        or provenance.get("method")
        or provenance.get("driver")
        or _method_from_bundle_name(bundle_dir.name)
    )
    if not isinstance(row_method, str):
        row_method = bundle_dir.name

    row = {
        "id": run_id,
        "method": row_method,
        "dataset": benchmark.get("suite") or lock_suite.get("name"),
        "sequence": benchmark.get("sequence") or lock_suite.get("sequence"),
        "suite_version": benchmark.get("version") or lock_suite.get("version"),
        "gate_status": gate_status,
        "gate_passed": gate_passed,
        "metrics": {
            "ate_rmse_m": _float_or_none(ate.get("rmse")),
            "rpe_rmse_m": _float_or_none(rpe.get("rmse")),
            "drift_m": _float_or_none(drift.get("endpoint")),
            "coverage_ratio": _float_or_none(matching.get("coverage_ratio")),
            "map_auc": _float_or_none(map_metrics.get("auc")),
            "chamfer_m": _float_or_none(map_metrics.get("chamfer_distance")),
        },
        "links": {
            "metrics_json": f"{rel_run.as_posix()}/metrics.json",
            "summary_md": f"{rel_run.as_posix()}/summary.md",
            "report_html": f"{rel_run.as_posix()}/report.html",
            "provenance_json": f"{rel_run.as_posix()}/provenance.json",
            "manifest_lock_yaml": f"{rel_run.as_posix()}/manifest.lock.yaml",
        },
        "artifact_hash": _combined_artifact_hash(lock_inputs),
        "artifact_hashes": {
            key: value.get("sha256")
            for key, value in lock_inputs.items()
            if isinstance(value, Mapping) and isinstance(value.get("sha256"), str)
        },
        "cloudanalyzer_version": provenance.get(
            "cloudanalyzer_version",
            getattr(ca, "__version__", "0.0.0"),
        ),
        "bundle_schema_version": provenance.get("schema_version"),
        "parameter_summary": provenance.get("parameters", {}),
        "source_bundle": bundle_dir.name,
    }
    compare_key = {
        "dataset": row["dataset"],
        "sequence": row["sequence"],
        "suite_version": row["suite_version"],
        "gate": benchmark.get("gate") or manifest_lock.get("gate"),
    }
    return row, compare_key


def _copy_bundle(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    shutil.copytree(source, destination, ignore=ignore)


def _warnings_for_compare_keys(compare_keys: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, Any], list[tuple[str, Mapping[str, Any]]]] = {}
    for run_id, key in compare_keys.items():
        grouped.setdefault((key.get("dataset"), key.get("sequence")), []).append((run_id, key))
    for (dataset, sequence), entries in grouped.items():
        signatures = {
            json.dumps(
                {
                    "suite_version": key.get("suite_version"),
                    "gate": key.get("gate"),
                },
                sort_keys=True,
            )
            for _, key in entries
        }
        if len(signatures) > 1:
            warnings.append(
                {
                    "code": "incomparable_rows",
                    "dataset": dataset,
                    "sequence": sequence,
                    "run_ids": [run_id for run_id, _ in entries],
                    "message": "Rows for this dataset/sequence do not share the same suite version and gate.",
                }
            )
    return warnings


def build_leaderboard_from_bundles(
    bundle_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    title: str = "CloudAnalyzer Benchmark Leaderboard",
) -> dict[str, Any]:
    """Build a self-contained static leaderboard from report bundle dirs."""
    if not bundle_dirs:
        raise ValueError("At least one benchmark report bundle directory is required.")

    out = Path(output_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    compare_keys: dict[str, Mapping[str, Any]] = {}
    seen_ids: set[str] = set()
    for raw_bundle in bundle_dirs:
        bundle = Path(raw_bundle).resolve()
        run_id = _unique_id(bundle.name, seen_ids)
        try:
            row, compare_key = _extract_row(bundle, run_id)
            _copy_bundle(bundle, runs_dir / run_id)
            rows.append(row)
            compare_keys[run_id] = compare_key
        except Exception as exc:  # noqa: BLE001 - keep other rows usable
            errors.append({"id": run_id, "bundle": str(bundle), "error": str(exc)})

    warnings = _warnings_for_compare_keys(compare_keys)
    payload = {
        "schema_version": LEADERBOARD_SCHEMA_VERSION,
        "title": title,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cloudanalyzer_version": getattr(ca, "__version__", "0.0.0"),
        "rows": rows,
        "warnings": warnings,
        "errors": errors,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "index.html").write_text(render_leaderboard_html(payload), encoding="utf-8")
    return payload


def render_leaderboard_html(payload: Mapping[str, Any]) -> str:
    """Render a small static HTML table for a leaderboard payload."""
    title = html.escape(str(payload.get("title", "CloudAnalyzer Leaderboard")))
    generated_at = html.escape(str(payload.get("generated_at", "")))
    rows = _list_or_empty(payload.get("rows"))
    warnings = _list_or_empty(payload.get("warnings"))
    errors = _list_or_empty(payload.get("errors"))

    row_html: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        metrics = _mapping_or_empty(row.get("metrics"))
        links = _mapping_or_empty(row.get("links"))
        status = str(row.get("gate_status", "unknown"))
        status_class = "pass" if status == "pass" else "fail" if status == "fail" else "warn"
        row_html.append(
            "<tr>"
            f"<td><strong>{html.escape(str(row.get('method', '')))}</strong></td>"
            f"<td>{html.escape(str(row.get('dataset', '')))}</td>"
            f"<td>{html.escape(str(row.get('sequence', '')))}</td>"
            f"<td class=\"num\">{_format_metric(metrics.get('ate_rmse_m'), 4)}</td>"
            f"<td class=\"num\">{_format_metric(metrics.get('rpe_rmse_m'), 4)}</td>"
            f"<td class=\"num\">{_format_metric(metrics.get('map_auc'), 4)}</td>"
            f"<td class=\"num\">{_format_metric(metrics.get('chamfer_m'), 4)}</td>"
            f"<td><span class=\"badge {status_class}\">{html.escape(status.upper())}</span></td>"
            "<td class=\"links\">"
            f"<a href=\"{html.escape(str(links.get('report_html', '#')))}\">report</a>"
            f"<a href=\"{html.escape(str(links.get('metrics_json', '#')))}\">metrics</a>"
            f"<a href=\"{html.escape(str(links.get('manifest_lock_yaml', '#')))}\">lock</a>"
            "</td>"
            "</tr>"
        )

    warning_html = "".join(
        f"<li>{html.escape(str(item.get('message', item)))}</li>"
        for item in warnings
        if isinstance(item, Mapping)
    )
    error_html = "".join(
        f"<li>{html.escape(str(item.get('id', 'run')))}: {html.escape(str(item.get('error', '')))}</li>"
        for item in errors
        if isinstance(item, Mapping)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; color: #151515; background: #f7f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 2rem; }}
    .meta {{ color: #5f6874; margin-bottom: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee7; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e7eaf0; text-align: left; }}
    th {{ font-size: 0.78rem; color: #5f6874; text-transform: uppercase; letter-spacing: 0.06em; }}
    .num {{ font-variant-numeric: tabular-nums; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }}
    .pass {{ color: #166534; background: #dcfce7; }}
    .fail {{ color: #991b1b; background: #fee2e2; }}
    .warn {{ color: #92400e; background: #fef3c7; }}
    .links {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .links a {{ color: #075985; }}
    .notice {{ margin-top: 18px; padding: 12px 14px; border: 1px solid #fde68a; background: #fffbeb; }}
    .error {{ margin-top: 18px; padding: 12px 14px; border: 1px solid #fecaca; background: #fef2f2; }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <div class="meta">Generated {generated_at}. Rows are built from CloudAnalyzer benchmark report bundles.</div>
  <table>
    <thead>
      <tr><th>Method</th><th>Dataset</th><th>Sequence</th><th>ATE RMSE</th><th>RPE RMSE</th><th>Map AUC</th><th>Chamfer</th><th>Gate</th><th>Links</th></tr>
    </thead>
    <tbody>
      {''.join(row_html)}
    </tbody>
  </table>
  {f'<div class="notice"><strong>Warnings</strong><ul>{warning_html}</ul></div>' if warning_html else ''}
  {f'<div class="error"><strong>Errors</strong><ul>{error_html}</ul></div>' if error_html else ''}
</main>
</body>
</html>
"""


def _format_metric(value: Any, digits: int) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


__all__ = [
    "LEADERBOARD_SCHEMA_VERSION",
    "build_leaderboard_from_bundles",
    "render_leaderboard_html",
]
