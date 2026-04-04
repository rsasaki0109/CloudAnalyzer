"""Pipeline-oriented implementation of web point-cloud reduction."""

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from ca.core.web_sampling import WebSampleRequest, WebSampleResult


@dataclass(slots=True)
class PipelineState:
    """Mutable state passed through reduction stages."""

    current_cloud: o3d.geometry.PointCloud
    original_points: int
    max_points: int
    label: str
    metadata: dict[str, float | int | str] = field(default_factory=dict)


class HybridPipelineSamplingStrategy:
    """Reducer assembled from distinct stages instead of one loop."""

    name = "hybrid_pipeline"
    design = "pipeline"

    def __init__(self):
        self.stages = (
            self._estimate_voxel_stage,
            self._voxel_stage,
            self._trim_stage,
        )

    def reduce(self, request: WebSampleRequest) -> WebSampleResult:
        state = PipelineState(
            current_cloud=request.point_cloud,
            original_points=len(request.point_cloud.points),
            max_points=request.max_points,
            label=request.label,
            metadata={"stage_count": len(self.stages)},
        )
        for stage in self.stages:
            state = stage(state)
        return WebSampleResult(
            point_cloud=state.current_cloud,
            strategy=self.name,
            design=self.design,
            original_points=state.original_points,
            reduced_points=len(state.current_cloud.points),
            metadata=state.metadata,
        )

    def _estimate_voxel_stage(self, state: PipelineState) -> PipelineState:
        bbox = state.current_cloud.get_axis_aligned_bounding_box()
        extent = np.maximum(np.asarray(bbox.get_extent(), dtype=float), 1e-6)
        volume = float(np.prod(extent))
        estimated_voxel_size = float((volume / max(state.max_points, 1)) ** (1.0 / 3.0))
        state.metadata["estimated_voxel_size"] = estimated_voxel_size
        return state

    def _voxel_stage(self, state: PipelineState) -> PipelineState:
        if len(state.current_cloud.points) <= state.max_points:
            state.metadata["voxel_pass_points"] = len(state.current_cloud.points)
            return state

        voxel_size = max(float(state.metadata["estimated_voxel_size"]) * 0.85, 1e-4)
        state.current_cloud = state.current_cloud.voxel_down_sample(voxel_size)
        state.metadata["voxel_size"] = voxel_size
        state.metadata["voxel_pass_points"] = len(state.current_cloud.points)
        return state

    def _trim_stage(self, state: PipelineState) -> PipelineState:
        total_points = len(state.current_cloud.points)
        if total_points <= state.max_points:
            state.metadata["trimmed_points"] = 0
            return state

        indices = np.linspace(0, total_points - 1, num=state.max_points, dtype=int)
        state.current_cloud = state.current_cloud.select_by_index(indices.tolist())
        state.metadata["trimmed_points"] = total_points - len(state.current_cloud.points)
        return state
