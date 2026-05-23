"""Concrete functional baseline for voxel-based ground evaluation."""

from __future__ import annotations

from ca.core.ground_evaluate import (
    GroundEvaluateRequest,
    GroundEvaluateResult,
    _voxel_intersection_size,
    _voxel_keys,
    confusion_metrics,
)


class VoxelConfusionExperimentalStrategy:
    """Voxelize both clouds and compare per-voxel ground labels."""

    name = "voxel_confusion"
    design = "functional"

    def evaluate(self, request: GroundEvaluateRequest) -> GroundEvaluateResult:
        est_ground_voxels = _voxel_keys(request.estimated_ground, request.voxel_size)
        est_nonground_voxels = _voxel_keys(request.estimated_nonground, request.voxel_size)
        ref_ground_voxels = _voxel_keys(request.reference_ground, request.voxel_size)
        ref_nonground_voxels = _voxel_keys(request.reference_nonground, request.voxel_size)

        tp = _voxel_intersection_size(est_ground_voxels, ref_ground_voxels)
        fp = _voxel_intersection_size(est_ground_voxels, ref_nonground_voxels)
        fn = _voxel_intersection_size(est_nonground_voxels, ref_ground_voxels)
        tn = _voxel_intersection_size(est_nonground_voxels, ref_nonground_voxels)
        metrics = confusion_metrics(tp, fp, fn, tn)

        return GroundEvaluateResult(
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            iou=metrics["iou"],
            accuracy=metrics["accuracy"],
            strategy=self.name, design=self.design,
        )
