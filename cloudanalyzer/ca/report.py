"""Report generation module (JSON and Markdown)."""

import json
from pathlib import Path


def make_json(
    source_points: int,
    target_points: int,
    fitness: float | None,
    rmse: float | None,
    distance_stats: dict,
) -> dict:
    """Build JSON-serializable report dict.

    Args:
        source_points: Number of source points.
        target_points: Number of target points.
        fitness: Registration fitness (None if no registration).
        rmse: Registration RMSE (None if no registration).
        distance_stats: Distance summary statistics.

    Returns:
        Report dict.
    """
    data = {
        "source_points": source_points,
        "target_points": target_points,
        "fitness": fitness,
        "rmse": rmse,
        "distance_stats": distance_stats,
    }
    return data


def save_json(data: dict, path: str) -> None:
    """Write report dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def make_markdown(data: dict, output_path: str) -> None:
    """Generate Markdown report from report dict.

    Args:
        data: Report dict (from make_json).
        output_path: Path to write .md file.
    """
    lines = [
        "# CloudAnalyzer Report",
        "",
    ]

    if data.get("fitness") is not None:
        lines += [
            "## Registration",
            f"- Fitness: {data['fitness']:.4f}",
            f"- RMSE: {data['rmse']:.4f}",
            "",
        ]

    stats = data.get("distance_stats", {})
    lines += [
        "## Distance Stats",
        f"- Mean: {stats.get('mean', 0):.4f}",
        f"- Median: {stats.get('median', 0):.4f}",
        f"- Max: {stats.get('max', 0):.4f}",
        f"- Min: {stats.get('min', 0):.4f}",
        f"- Std: {stats.get('std', 0):.4f}",
        "",
        "## Point Counts",
        f"- Source: {data['source_points']}",
        f"- Target: {data['target_points']}",
        "",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
