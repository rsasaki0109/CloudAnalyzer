"""Function-first implementation of web point-cloud reduction."""

from ca.core.web_sampling import WebSampleRequest, WebSampleResult


def _grow_voxel_size(
    current_voxel_size: float,
    reduced_points: int,
    previous_points: int,
    growth_factor: float,
) -> float:
    if reduced_points == previous_points:
        return current_voxel_size * growth_factor * 1.25
    return current_voxel_size * growth_factor


def reduce_with_functional_voxels(
    request: WebSampleRequest,
    initial_voxel_size: float = 0.01,
    growth_factor: float = 1.35,
) -> WebSampleResult:
    """Reduce by repeatedly applying voxel downsampling until under budget."""

    original_points = len(request.point_cloud.points)
    if original_points <= request.max_points:
        return WebSampleResult(
            point_cloud=request.point_cloud,
            strategy="functional_voxel",
            design="functional",
            original_points=original_points,
            reduced_points=original_points,
            metadata={
                "label": request.label,
                "iterations": 0,
                "initial_voxel_size": initial_voxel_size,
                "applied_voxel_size": 0.0,
            },
        )

    reduced = request.point_cloud
    voxel_size = initial_voxel_size
    applied_voxel_size = initial_voxel_size
    iterations = 0
    previous_points = original_points

    while len(reduced.points) > request.max_points:
        applied_voxel_size = voxel_size
        reduced = reduced.voxel_down_sample(voxel_size)
        iterations += 1
        reduced_points = len(reduced.points)
        voxel_size = _grow_voxel_size(
            current_voxel_size=voxel_size,
            reduced_points=reduced_points,
            previous_points=previous_points,
            growth_factor=growth_factor,
        )
        previous_points = reduced_points

    return WebSampleResult(
        point_cloud=reduced,
        strategy="functional_voxel",
        design="functional",
        original_points=original_points,
        reduced_points=len(reduced.points),
        metadata={
            "label": request.label,
            "iterations": iterations,
            "initial_voxel_size": initial_voxel_size,
            "applied_voxel_size": applied_voxel_size,
        },
    )


class FunctionalVoxelSamplingStrategy:
    """Thin adapter over the functional reducer."""

    name = "functional_voxel"
    design = "functional"

    def __init__(self, initial_voxel_size: float = 0.01, growth_factor: float = 1.35):
        self.initial_voxel_size = initial_voxel_size
        self.growth_factor = growth_factor

    def reduce(self, request: WebSampleRequest) -> WebSampleResult:
        return reduce_with_functional_voxels(
            request,
            initial_voxel_size=self.initial_voxel_size,
            growth_factor=self.growth_factor,
        )
