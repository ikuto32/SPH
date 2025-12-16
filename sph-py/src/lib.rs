use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use sph_core::{World, WorldConfig};

#[cfg(feature = "cuda")]
use sph_cuda;

#[pyclass]
struct PyWorld {
    world: World,
    use_gpu: bool,
}

#[pymethods]
impl PyWorld {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        width: Option<f32>,
        height: Option<f32>,
        smoothing_radius: Option<f32>,
        target_density: Option<f32>,
        pressure_multiplier: Option<f32>,
        drag: Option<f32>,
        gravity: Option<f32>,
        collision_damping: Option<f32>,
        delta: Option<f32>,
        use_gpu: Option<bool>,
        num_particles: Option<usize>,
    ) -> PyResult<Self> {
        let mut config = WorldConfig::default();
        if let Some(w) = width {
            config.world_width = w;
        }
        if let Some(h) = height {
            config.world_height = h;
        }
        if let Some(sr) = smoothing_radius {
            config.smoothing_radius = sr;
        }
        if let Some(td) = target_density {
            config.target_density = td;
        }
        if let Some(pm) = pressure_multiplier {
            config.pressure_multiplier = pm;
        }
        if let Some(d) = drag {
            config.drag = d;
        }
        if let Some(g) = gravity {
            config.gravity = g;
        }
        if let Some(cd) = collision_damping {
            config.collision_damping = cd;
        }
        if let Some(d) = delta {
            config.delta = d;
        }
        if let Some(n) = num_particles {
            config.num_particles = n;
        }

        Ok(Self {
            world: World::new(config),
            use_gpu: use_gpu.unwrap_or(false),
        })
    }

    fn step(&mut self, dt: f32) -> PyResult<()> {
        if self.use_gpu {
            self.world.step_with_cuda(dt);
        } else {
            self.world.step(dt);
        }
        Ok(())
    }

    fn get_positions<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f32>> {
        let flat: Vec<f32> = self
            .world
            .positions()
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        make_matrix(py, flat, self.world.get_num_particles())
    }

    fn get_velocities<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f32>> {
        let flat: Vec<f32> = self
            .world
            .velocities()
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        make_matrix(py, flat, self.world.get_num_particles())
    }

    fn width(&self) -> PyResult<f32> {
        Ok(self.world.get_world_width())
    }

    fn height(&self) -> PyResult<f32> {
        Ok(self.world.get_world_height())
    }

    fn smoothing_radius(&self) -> PyResult<f32> {
        Ok(self.world.get_smoothing_radius())
    }

    fn gravity(&self) -> PyResult<f32> {
        Ok(self.world.get_gravity())
    }

    fn drag(&self) -> PyResult<f32> {
        Ok(self.world.get_drag())
    }

    fn set_interaction_force(
        &mut self,
        x: f32,
        y: f32,
        radius: f32,
        strength: f32,
    ) -> PyResult<()> {
        self.world.set_interaction_force(x, y, radius, strength);
        Ok(())
    }

    fn delete_interaction_force(&mut self) -> PyResult<()> {
        self.world.delete_interaction_force();
        Ok(())
    }

    fn query_neighbors(&self, x: f32, y: f32) -> PyResult<Vec<usize>> {
        Ok(self.world.query_neighbors(x, y))
    }

    fn query_spatial_hash(&self, x: f32, y: f32) -> PyResult<Vec<usize>> {
        Ok(self.world.query_spatial_hash(x, y))
    }
}

#[pyfunction]
#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    sph_cuda::cuda_available()
}

#[pymodule]
fn _sph(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyWorld>()?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    Ok(())
}

fn make_matrix<'py>(py: Python<'py>, data: Vec<f32>, rows: usize) -> PyResult<&'py PyArray2<f32>> {
    let array = Array2::from_shape_vec((rows, 2), data)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
    let bound = array.into_pyarray_bound(py);
    Ok(bound.into_gil_ref())
}
