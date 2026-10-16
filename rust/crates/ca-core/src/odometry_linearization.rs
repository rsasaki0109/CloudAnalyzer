//! Owned final-pose point-to-point diagnostics, without estimator feedback.

use crate::icp::Rigid;

/// Recomputed at the returned pose against the map before insertion. This is
/// not the last attempted solve. The fixed-match IRLS normal matrix is not a
/// calibrated information matrix or physical observability certificate.
#[derive(Debug, Clone)]
pub struct RegistrationLinearization {
    pub pose: Rigid,
    /// Fixed prediction center used by the solver for world-axis rotation increments.
    pub center_world: [f64; 3],
    pub threshold_sq: f64,
    /// Raw legacy kernel parameter: weight = kernel² / (kernel + distance²)².
    pub kernel: f64,
    pub map_voxel: f64,
    pub map_points: usize,
    pub max_range: f64,
    /// Map and source met the alignment eligibility conditions at begin.
    pub alignment_eligible: bool,
    pub source_sensor: Vec<[f64; 3]>,
    pub source_world: Vec<[f64; 3]>,
    pub matched_source_indices: Vec<usize>,
    pub target_world: Vec<[f64; 3]>,
    pub squared_distances: Vec<f64>,
    pub weights: Vec<f64>,
    /// Translation xyz, then rotation xyz around center_world; fixed weights.
    pub normal_matrix: [[f64; 6]; 6],
    /// -J^T W (source_world - target_world), in the same parameter order.
    pub right_hand_side: [f64; 6],
    pub weight_sum: f64,
    /// None without finite positive support. A zero radius is not regularized.
    pub weighted_rms_radius_m: Option<f64>,
    /// Canonical last-scan insertion points, before voxel capacity/range pruning.
    pub inserted_world: Vec<[f64; 3]>,
    /// Numeric finiteness only; this is not a pose-quality gate.
    pub finite: bool,
}

impl RegistrationLinearization {
    /// Serial diagnostic accumulation. Does not change the parallel solver's
    /// reduction or arithmetic. Internal callers provide corresponding rows.
    pub(crate) fn accumulate(&mut self) {
        let mut radius_sum = 0.0;
        for (row, &index) in self.matched_source_indices.iter().enumerate() {
            let world = self.source_world[index];
            let q: [f64; 3] = std::array::from_fn(|i| world[i] - self.center_world[i]);
            let residual: [f64; 3] = std::array::from_fn(|i| world[i] - self.target_world[row][i]);
            let w = self.kernel * self.kernel / (self.kernel + self.squared_distances[row]).powi(2);
            self.weights.push(w);
            self.weight_sum += w;
            radius_sum += w * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
            let jacobian = [
                [1.0, 0.0, 0.0, 0.0, q[2], -q[1]],
                [0.0, 1.0, 0.0, -q[2], 0.0, q[0]],
                [0.0, 0.0, 1.0, q[1], -q[0], 0.0],
            ];
            for (j, r) in jacobian.iter().zip(residual) {
                for i in 0..6 {
                    self.right_hand_side[i] -= w * j[i] * r;
                    for k in 0..6 {
                        self.normal_matrix[i][k] += w * j[i] * j[k];
                    }
                }
            }
        }
        self.finite &= self
            .normal_matrix
            .iter()
            .flatten()
            .chain(&self.right_hand_side)
            .chain(&self.weights)
            .chain(&self.squared_distances)
            .chain(self.target_world.iter().flatten())
            .all(|v| v.is_finite())
            && self.weight_sum.is_finite()
            && radius_sum.is_finite();
        self.weighted_rms_radius_m = if self.finite && self.weight_sum > 0.0 && radius_sum >= 0.0 {
            let radius = (radius_sum / self.weight_sum).sqrt();
            self.finite &= radius.is_finite();
            radius.is_finite().then_some(radius)
        } else {
            None
        };
    }
}
