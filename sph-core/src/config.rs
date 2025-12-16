/// Configuration values for the SPH world.
#[derive(Debug, Clone)]
pub struct WorldConfig {
    pub world_width: f32,
    pub world_height: f32,
    pub smoothing_radius: f32,
    pub target_density: f32,
    pub pressure_multiplier: f32,
    pub delta: f32,
    pub drag: f32,
    pub gravity: f32,
    pub collision_damping: f32,
    pub num_particles: usize,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            world_width: 20.0,
            world_height: 10.0,
            smoothing_radius: 0.8,
            target_density: 32.0,
            pressure_multiplier: 100.0,
            delta: 0.0,
            drag: 0.9999,
            gravity: 9.8,
            collision_damping: 1.0,
            num_particles: 1_000,
        }
    }
}
