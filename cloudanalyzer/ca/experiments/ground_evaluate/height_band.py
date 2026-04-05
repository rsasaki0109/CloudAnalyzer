"""Pipeline height-band ground evaluation strategy."""

from __future__ import annotations

import numpy as np

from ca.core.ground_evaluate import (
    GroundEvaluateRequest,
    GroundEvaluateResult,
    confusion_metrics,
)


def _assign_height_band(points: np.ndarray, band_edges: np.ndarray) -> np.ndarray:
    """Assign each point to a height band index based on its Z coordinate."""
    return np.clip(np.searchsorted(band_edges, points[:, 2]) - 1, 0, len(band_edges) - 2)


def _build_band_edges(all_points: np.ndarray, num_bands: int) -> np.ndarray:
    """Create equally-spaced height band edges spanning the Z range."""
    z_min = float(np.min(all_points[:, 2]))
    z_max = float(np.max(all_points[:, 2]))
    margin = max((z_max - z_min) * 0.01, 1e-6)
    return np.linspace(z_min - margin, z_max + margin, num_bands + 1)


def _per_band_confusion(
    estimated_ground: np.ndarray,
    estimated_nonground: np.ndarray,
    reference_ground: np.ndarray,
    reference_nonground: np.ndarray,
    band_edges: np.ndarray,
    voxel_size: float,
) -> list[dict]:
    """Compute confusion matrix per height band using voxel matching."""
    from ca.core.ground_evaluate import _voxel_keys

    num_bands = len(band_edges) - 1
    bands: list[dict] = []
    for band_idx in range(num_bands):
        lo, hi = float(band_edges[band_idx]), float(band_edges[band_idx + 1])

        def _filter(pts: np.ndarray) -> np.ndarray:
            if pts.shape[0] == 0:
                return pts
            mask = (pts[:, 2] >= lo) & (pts[:, 2] < hi)
            return pts[mask]

        eg = _voxel_keys(_filter(estimated_ground), voxel_size)
        en = _voxel_keys(_filter(estimated_nonground), voxel_size)
        rg = _voxel_keys(_filter(reference_ground), voxel_size)
        rn = _voxel_keys(_filter(reference_nonground), voxel_size)

        tp = len(eg & rg)
        fp = len(eg & rn)
        fn = len(en & rg)
        tn = len(en & rn)
        metrics = confusion_metrics(tp, fp, fn, tn)
        bands.append({
            "band": band_idx,
            "z_lo": round(lo, 4),
            "z_hi": round(hi, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            **metrics,
        })
    return bands


class HeightBandGroundEvaluateStrategy:
    """Evaluate ground segmentation per height band then aggregate."""

    name = "height_band"
    design = "pipeline"

    _NUM_BANDS = 5

    def evaluate(self, request: GroundEvaluateRequest) -> GroundEvaluateResult:
        all_points = np.vstack([
            request.estimated_ground,
            request.estimated_nonground,
            request.reference_ground,
            request.reference_nonground,
        ])
        band_edges = _build_band_edges(all_points, self._NUM_BANDS)
        bands = _per_band_confusion(
            request.estimated_ground,
            request.estimated_nonground,
            request.reference_ground,
            request.reference_nonground,
            band_edges,
            request.voxel_size,
        )

        # Aggregate across bands
        tp = sum(b["tp"] for b in bands)
        fp = sum(b["fp"] for b in bands)
        fn = sum(b["fn"] for b in bands)
        tn = sum(b["tn"] for b in bands)
        metrics = confusion_metrics(tp, fp, fn, tn)

        return GroundEvaluateResult(
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            iou=metrics["iou"],
            accuracy=metrics["accuracy"],
            strategy=self.name, design=self.design,
            metadata={
                "num_bands": self._NUM_BANDS,
                "bands": bands,
            },
        )
