# IMU-compensated LiDAR registration

`ImuLidarOdometry` connects a declared nominal `ImuTrajectory` to the existing
ICP odometry. Each call compensates the input points at an explicit integer
reference timestamp, seeds ICP with the IMU motion between references, and
places those same compensated points using the final ICP pose. Ordinary
constant-motion deskew is disabled inside this dedicated frontend.

The optional [independent registration voxel](slam-registration-sampling.md)
controls ICP source sampling while map insertion keeps its own existing thinning.

```python
import numpy as np
from cloudanalyzer_core import ImuLidarOdometry

# Build motion with declared initial state, gravity, bias and acceleration scale.
odometry = ImuLidarOdometry(
    motion, calibrated_imu_from_lidar,
    min_range=1.5, max_range=80.0, map_voxel=1.0, map_points=20,
    parallel=True, diagnostics=True,
)
map_chunks = []
for scan in selected_scans:
    point_ns = scan.point_times_ns(unit="ns", origin="header")
    reference_ns = int(point_ns.max())  # Empty scans need an explicit policy too.
    local_pose = odometry.register(scan.positions(), point_ns, reference_ns)
    map_chunks.append(odometry.placed())

pose_times_ns = odometry.reference_timestamps_ns()
local_poses = odometry.poses()
local_map = np.concatenate(map_chunks) if map_chunks else np.empty((0, 3))
world_from_local = odometry.world_from_first_lidar
```

Initialization, calibration and scan selection are application inputs. See
[nominal IMU motion](slam-imu-trajectory.md) for building `motion`. The frontend
accepts no inferred clock units, initial velocity, gravity or frame transform.
It does not extrapolate or discard uncovered points.

## Pose time and coordinates

The first accepted scan anchors the local map at identity. Returned poses,
`poses()` and `placed()` use the LiDAR frame at that first reference as their
local world. Every row of `poses()` corresponds exactly to the same row of the
copied int64 `reference_timestamps_ns()` array. Reference timestamps must strictly
increase; acquisition times within a scan may decrease or repeat.

`world_from_first_lidar` is `None` before the first successful call. Afterwards it
returns a copied local-to-nominal-world alignment:

```text
T_world_first = T_world_imu(first_reference) * T_imu_lidar
T_world_lidar = T_world_first * T_local_lidar

relative_prediction = inverse(T_world_lidar(previous_reference))
                    * T_world_lidar(current_reference)
ICP_initial_pose = previous_ICP_pose * relative_prediction
```

The relative transform includes the full IMU-from-LiDAR rotation and lever arm.
It uses fixed-anchor rotations and displacements before adding the initial
world position. Large world coordinates affect the final absolute conversion,
not local deskew or relative prediction arithmetic. `last_prediction()` returns
`None` for the first scan.

## Point and state ownership

The frontend shares immutable motion knots and owns its odometry and a copy of
the sensor transform. The source Python trajectory, series and recording may
be released. No per-point pose matrix or covariance cache is retained.

`register()` copies both buffers before releasing the GIL. Noncontiguous and
negative-stride inputs are supported. Range filtering acts on compensated
reference-frame coordinates, followed by existing first-per-voxel thinning.
The default compensation path is serial; `parallel=True` enables the indexed
CPU point transform. ICP retains its existing CPU parallel reductions.

`placed()` returns a copied array of the last scan's transformed insertion points,
after range filtering and thinning. These already use local-world coordinates.
They are passed to the internal map before its voxel-capacity and distance
pruning policies. This is not a snapshot of every point retained by that map;
applications can accumulate the chunks as an output map.

All exported pose, reference, alignment, prediction and placed-point arrays are
independent copies. Coverage, counts, finite input, calibration and chronological
reference checks precede mutation of the map, pose history, threshold statistics
and diagnostics. Failed calls leave these unchanged. Settings require finite
`0 <= min_range < max_range`, positive finite `map_voxel`, and positive `map_points`.

An empty scan with an explicit covered reference produces no placed points and
uses motion prediction without LiDAR alignment. `last_report()` and
`last_iterations()` distinguish empty maps, insufficient points, singular
systems, iteration limits and convergence. Convergence does not certify alignment.

## Estimation scope

Initial pose, velocity, gravity, biases and calibration remain declared inputs.
ICP determines emitted poses; its correction is composed with the next relative
IMU guess. This component does not update trajectory velocity or bias, perform
physical initialization, or optimize coupled LiDAR/IMU factors. A fixed nominal
trajectory may drift. Ordinary `LidarOdometry` and frame-file drivers retain
their existing behavior.

The [real-data packet](../benchmarks/slam/flatwall-imu-odometry/README.md) checks
reference transforms, complete covered scans and map-coordinate consistency.
It establishes no trajectory accuracy, complete SLAM speedup or GPU result.

A separate [conditional trajectory evaluation](slam-flatwall-trajectory.md)
now scores the frozen nominal-state output against the selected reference.
It reports a large covered-subset mismatch and retains all coverage exclusions;
the reference body/time convention and physical initialization remain unresolved.
