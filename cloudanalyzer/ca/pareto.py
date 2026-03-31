"""Pareto frontier helpers for quality-vs-size tradeoff analysis."""

from typing import Any, cast


def _compression(item: dict) -> dict[str, Any] | None:
    """Return compression metadata if present."""
    compression = item.get("compression")
    if compression is None:
        return None
    return cast(dict[str, Any], compression)


def _dominates(candidate: dict, other: dict) -> bool:
    """Return True if candidate dominates other on size ratio and AUC."""
    candidate_compression = _compression(candidate)
    other_compression = _compression(other)
    if candidate_compression is None or other_compression is None:
        return False

    candidate_size_ratio = float(candidate_compression["size_ratio"])
    other_size_ratio = float(other_compression["size_ratio"])
    candidate_auc = float(candidate["auc"])
    other_auc = float(other["auc"])

    size_better_or_equal = candidate_size_ratio <= other_size_ratio
    auc_better_or_equal = candidate_auc >= other_auc
    strictly_better = (
        candidate_size_ratio < other_size_ratio
        or candidate_auc > other_auc
    )
    return size_better_or_equal and auc_better_or_equal and strictly_better


def _pareto_sort_key(item: dict) -> tuple[float, float, str]:
    """Sort by compression ratio, then descending AUC, then path."""
    compression = _compression(item)
    if compression is None:
        return (float("inf"), float("inf"), str(item["path"]))
    return (
        float(compression["size_ratio"]),
        -float(item["auc"]),
        str(item["path"]),
    )


def _recommendation_sort_key(item: dict) -> tuple[float, float, float, str]:
    """Prefer smaller artifacts, then higher quality, then lower drift."""
    compression = _compression(item)
    if compression is None:
        return (float("inf"), float("inf"), float("inf"), str(item["path"]))
    return (
        float(compression["size_ratio"]),
        -float(item["auc"]),
        float(item.get("chamfer_distance", float("inf"))),
        str(item["path"]),
    )


def _passes_quality_gate(
    item: dict,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
) -> bool:
    """Return True if the item satisfies the effective quality gate."""
    if item.get("quality_gate") is not None:
        return bool(item["quality_gate"]["passed"])
    if min_auc is not None and float(item["auc"]) < min_auc:
        return False
    if max_chamfer is not None and float(item["chamfer_distance"]) > max_chamfer:
        return False
    return True


def quality_size_pareto_results(results: list[dict]) -> list[dict]:
    """Return Pareto-optimal results for the size-ratio vs AUC tradeoff."""
    compression_results = [
        item for item in results
        if _compression(item) is not None
    ]

    frontier = []
    for item in compression_results:
        dominated = any(
            other is not item and _dominates(other, item)
            for other in compression_results
        )
        if not dominated:
            frontier.append(item)

    return sorted(frontier, key=_pareto_sort_key)


def mark_quality_size_pareto(results: list[dict]) -> None:
    """Annotate compression results with a Pareto-optimal flag."""
    frontier_ids = {id(item) for item in quality_size_pareto_results(results)}

    for item in results:
        compression = _compression(item)
        if compression is None:
            continue
        compression["pareto_optimal"] = id(item) in frontier_ids


def recommended_quality_size_result(
    results: list[dict],
    min_auc: float | None = None,
    max_chamfer: float | None = None,
) -> dict | None:
    """Return the recommended Pareto candidate for compression tradeoffs."""
    candidate_pool = [
        item for item in results
        if _compression(item) is not None
    ]
    if not candidate_pool:
        return None

    gate_is_active = (
        min_auc is not None
        or max_chamfer is not None
        or any(item.get("quality_gate") is not None for item in candidate_pool)
    )
    if gate_is_active:
        candidate_pool = [
            item for item in candidate_pool
            if _passes_quality_gate(item, min_auc=min_auc, max_chamfer=max_chamfer)
        ]
        if not candidate_pool:
            return None

    frontier = quality_size_pareto_results(candidate_pool)
    return min(frontier, key=_recommendation_sort_key)


def mark_quality_size_recommended(
    results: list[dict],
    min_auc: float | None = None,
    max_chamfer: float | None = None,
) -> dict | None:
    """Annotate compression results with a recommendation flag."""
    mark_quality_size_pareto(results)
    recommended = recommended_quality_size_result(
        results,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
    )
    recommended_id = id(recommended) if recommended is not None else None

    for item in results:
        compression = _compression(item)
        if compression is None:
            continue
        compression["recommended"] = recommended_id is not None and id(item) == recommended_id

    return recommended
