//! Owned, bounded inertial windows with explicit oldest-state marginalization.

use ca_core::imu_factor::ImuState;
use ca_core::imu_window::{ImuWindow as Window, MAX_STATES, NavigationState, OptimizeParams};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::imu::{ImuFactor, err, state};

/// Adjacent IMU factors, explicit priors and a bias random walk per interval.
/// The caller may append states and marginalize the oldest state explicitly.
#[pyclass(module = "cloudanalyzer_core._core")]
pub struct ImuWindow {
    window: Window,
}

impl ImuWindow {
    fn state_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let states = self.window.states();
        let poses: Vec<f64> = states
            .iter()
            .flat_map(|s| {
                (0..4).flat_map(move |r| {
                    (0..4).map(move |c| {
                        if r == 3 {
                            f64::from(u8::from(c == 3))
                        } else if c == 3 {
                            s.motion.position[r]
                        } else {
                            s.motion.rotation[r][c]
                        }
                    })
                })
            })
            .collect();
        let velocities: Vec<[f64; 3]> = states.iter().map(|s| s.motion.velocity).collect();
        let biases: Vec<[f64; 6]> = states.iter().map(|s| s.bias).collect();
        let result = PyDict::new(py);
        result.set_item(
            "timestamps_ns",
            self.window.timestamps_ns().to_vec().into_pyarray(py),
        )?;
        result.set_item(
            "poses",
            Array3::from_shape_vec((states.len(), 4, 4), poses)
                .expect("pose dimensions")
                .into_pyarray(py),
        )?;
        result.set_item(
            "velocities",
            numpy::PyArray2::from_vec2(
                py,
                &velocities.iter().map(|v| v.to_vec()).collect::<Vec<_>>(),
            )?,
        )?;
        result.set_item(
            "biases",
            numpy::PyArray2::from_vec2(py, &biases.iter().map(|v| v.to_vec()).collect::<Vec<_>>())?,
        )?;
        Ok(result)
    }
}

#[pymethods]
impl ImuWindow {
    #[new]
    #[pyo3(signature = (timestamps_ns, poses, velocities, biases, *, gravity))]
    fn new(
        timestamps_ns: Vec<i64>,
        poses: PyReadonlyArray3<f64>,
        velocities: PyReadonlyArray2<f64>,
        biases: PyReadonlyArray2<f64>,
        gravity: [f64; 3],
    ) -> PyResult<Self> {
        let n = timestamps_ns.len();
        if !(2..=MAX_STATES).contains(&n)
            || poses.shape() != [n, 4, 4]
            || velocities.shape() != [n, 3]
            || biases.shape() != [n, 6]
        {
            return Err(PyValueError::new_err(format!(
                "expected 2..={MAX_STATES} timestamps, poses (N,4,4), velocities (N,3), biases (N,6)"
            )));
        }
        let p = poses.as_array();
        let v = velocities.as_array();
        let b = biases.as_array();
        let mut states = Vec::with_capacity(n);
        for k in 0..n {
            if !(0..4).all(|j| {
                p[[k, 3, j]].is_finite()
                    && (p[[k, 3, j]] - f64::from(u8::from(j == 3))).abs() <= 1e-12
            }) {
                return Err(PyValueError::new_err(
                    "IMU pose must have homogeneous last row [0, 0, 0, 1]",
                ));
            }
            states.push(NavigationState {
                motion: ImuState {
                    rotation: std::array::from_fn(|r| std::array::from_fn(|c| p[[k, r, c]])),
                    position: std::array::from_fn(|r| p[[k, r, 3]]),
                    velocity: std::array::from_fn(|r| v[[k, r]]),
                },
                bias: std::array::from_fn(|r| b[[k, r]]),
            });
        }
        Ok(Self {
            window: Window::new(timestamps_ns, states, gravity).map_err(err)?,
        })
    }

    /// Copies the factor. Its exact endpoints must match nodes index and index+1.
    #[pyo3(signature = (index, factor, *, gyro_bias_random_walk, accel_bias_random_walk))]
    fn add_imu_factor(
        &mut self,
        index: usize,
        factor: &ImuFactor,
        gyro_bias_random_walk: f64,
        accel_bias_random_walk: f64,
    ) -> PyResult<()> {
        self.window
            .add_imu_factor(
                index,
                factor.factor.clone(),
                gyro_bias_random_walk,
                accel_bias_random_walk,
            )
            .map_err(err)
    }

    /// Append one state and exact adjacent factor; all validation is atomic.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (timestamp_ns, pose, velocity, bias, factor, *, gyro_bias_random_walk, accel_bias_random_walk))]
    fn append_state(
        &mut self,
        timestamp_ns: i64,
        pose: PyReadonlyArray2<f64>,
        velocity: [f64; 3],
        bias: [f64; 6],
        factor: &ImuFactor,
        gyro_bias_random_walk: f64,
        accel_bias_random_walk: f64,
    ) -> PyResult<()> {
        self.window
            .append_state(
                timestamp_ns,
                NavigationState {
                    motion: state(pose, velocity)?,
                    bias,
                },
                factor.factor.clone(),
                gyro_bias_random_walk,
                accel_bias_random_walk,
            )
            .map_err(err)
    }

    /// Freeze and eliminate only the oldest state's incident factors. Requires
    /// three states and a full prior at the oldest state (or a previous marginal).
    fn marginalize_oldest<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = py
            .detach(|| self.window.marginalize_oldest())
            .map_err(err)?;
        let result = PyDict::new(py);
        result.set_item("dropped_timestamp_ns", report.dropped_timestamp_ns)?;
        result.set_item("retained_timestamp_ns", report.retained_timestamp_ns)?;
        result.set_item("removed_priors", report.removed_priors)?;
        result.set_item("replaced_marginal_prior", report.replaced_marginal_prior)?;
        result.set_item(
            "dropped_linearized_constant",
            report.dropped_linearized_constant,
        )?;
        Ok(result)
    }

    /// Independent copies of the frozen linearization point, root and offset.
    fn marginal_prior<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(prior) = self.window.marginal_prior() else {
            return Ok(None);
        };
        let result = PyDict::new(py);
        let motion = &prior.reference.motion;
        let pose = Array2::from_shape_fn((4, 4), |(r, c)| {
            if r == 3 {
                f64::from(c == 3)
            } else if c == 3 {
                motion.position[r]
            } else {
                motion.rotation[r][c]
            }
        });
        result.set_item("timestamp_ns", self.window.timestamps_ns()[0])?;
        result.set_item("reference_pose", pose.into_pyarray(py))?;
        result.set_item(
            "reference_velocity",
            motion.velocity.to_vec().into_pyarray(py),
        )?;
        result.set_item(
            "reference_bias",
            prior.reference.bias.to_vec().into_pyarray(py),
        )?;
        let root = Array2::from_shape_fn((15, 15), |(r, c)| prior.square_root[r][c]);
        result.set_item("sqrt_information", root.into_pyarray(py))?;
        result.set_item(
            "residual_offset",
            prior.residual_offset.to_vec().into_pyarray(py),
        )?;
        Ok(Some(result))
    }

    /// Diagonal standard deviations: rotation, velocity, position, gyro/accel bias.
    #[pyo3(signature = (index, pose, velocity, bias, *, sigmas))]
    fn add_state_prior(
        &mut self,
        index: usize,
        pose: PyReadonlyArray2<f64>,
        velocity: [f64; 3],
        bias: [f64; 6],
        sigmas: [f64; 15],
    ) -> PyResult<()> {
        let mean = NavigationState {
            motion: state(pose, velocity)?,
            bias,
        };
        self.window
            .add_state_prior(index, mean, sigmas)
            .map_err(err)
    }

    /// Pose prior at the IMU origin, with rotation in radians and position in m.
    #[pyo3(signature = (index, pose, *, rotation_sigma, position_sigma))]
    fn add_pose_prior(
        &mut self,
        index: usize,
        pose: PyReadonlyArray2<f64>,
        rotation_sigma: f64,
        position_sigma: f64,
    ) -> PyResult<()> {
        self.window
            .add_pose_prior(
                index,
                state(pose, [0.0; 3])?,
                rotation_sigma,
                position_sigma,
            )
            .map_err(err)
    }

    /// Independent copies; modifying returned arrays does not change the window.
    fn states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.state_dict(py)
    }

    /// Returns copied states and cost/convergence diagnostics. Errors are atomic.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, max_iterations=20, initial_damping=1e-3,
        gradient_tolerance=1e-8, step_tolerance=1e-9, cost_tolerance=1e-10))]
    fn optimize<'py>(
        &mut self,
        py: Python<'py>,
        max_iterations: usize,
        initial_damping: f64,
        gradient_tolerance: f64,
        step_tolerance: f64,
        cost_tolerance: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params = OptimizeParams {
            max_iterations,
            initial_damping,
            gradient_tolerance,
            step_tolerance,
            cost_tolerance,
        };
        let report = py.detach(|| self.window.optimize(&params)).map_err(err)?;
        let result = self.state_dict(py)?;
        result.set_item("initial_cost", report.initial_cost)?;
        result.set_item("final_cost", report.final_cost)?;
        result.set_item("iterations", report.iterations)?;
        result.set_item("accepted_steps", report.accepted_steps)?;
        result.set_item("rejected_steps", report.rejected_steps)?;
        result.set_item("converged", report.converged)?;
        result.set_item("termination", report.termination)?;
        result.set_item("cost_history", report.cost_history.into_pyarray(py))?;
        result.set_item(
            "max_abs_bias_correction",
            report.max_abs_bias_correction.to_vec().into_pyarray(py),
        )?;
        Ok(result)
    }
}
