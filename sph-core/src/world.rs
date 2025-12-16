use crate::config::WorldConfig;
use crate::grid_map::GridMap;
use crate::kernels::{calc_smoothing_kernel, calc_smoothing_kernel_derivative};
#[cfg(feature = "cuda")]
use sph_cuda;

#[derive(Debug, Clone)]
pub struct ForcePoint {
    pub pos: [f32; 2],
    pub radius: f32,
    pub strength: f32,
}

#[derive(Debug)]
pub struct World {
    _particle_radius: i32,
    num_particle: usize,
    gravity: f32,
    world_size: [f32; 2],
    collision_damping: f32,
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    delta: f32,
    drag: f32,
    mass: Vec<f32>,

    pos: Vec<[f32; 2]>,
    predpos: Vec<[f32; 2]>,
    vel: Vec<[f32; 2]>,
    density: Vec<f32>,
    pressure_accelerations: Vec<[f32; 2]>,
    interaction_force: Vec<[f32; 2]>,
    color: Vec<[u8; 3]>,
    querysize: Vec<Vec<usize>>,

    force_point: ForcePoint,
    gridmap: GridMap,
}

impl World {
    pub const DEFAULT_NUM_PARTICLES: usize = 1_000;

    pub fn new(config: WorldConfig) -> Self {
        let num_particle = config.num_particles.max(1);
        let world_size = [config.world_width, config.world_height];
        let mut world = Self {
            _particle_radius: 5,
            num_particle,
            gravity: config.gravity,
            world_size,
            collision_damping: config.collision_damping,
            smoothing_radius: config.smoothing_radius,
            target_density: config.target_density,
            pressure_multiplier: config.pressure_multiplier,
            delta: config.delta,
            drag: config.drag,
            mass: vec![1.0; num_particle],
            pos: Vec::with_capacity(num_particle),
            predpos: vec![[0.0; 2]; num_particle],
            vel: vec![[0.0; 2]; num_particle],
            density: vec![0.0; num_particle],
            pressure_accelerations: vec![[0.0; 2]; num_particle],
            interaction_force: vec![[0.0; 2]; num_particle],
            color: vec![[255, 255, 255]; num_particle],
            querysize: vec![Vec::new(); num_particle],
            force_point: ForcePoint {
                pos: [0.0, 0.0],
                radius: 0.0,
                strength: 0.0,
            },
            gridmap: GridMap::new(world_size[0], world_size[1], config.smoothing_radius),
        };

        world.initialize_positions();
        world
    }

    fn initialize_positions(&mut self) {
        let side = (self.num_particle as f32).sqrt().ceil() as usize;
        for i in 0..self.num_particle {
            let row = i / side;
            let col = i % side;
            let x = (col as f32 / side as f32) * self.world_size[0];
            let y = (row as f32 / side as f32) * self.world_size[1];
            self.pos.push([x, y]);
        }
    }

    pub fn step(&mut self, delta_time: f32) {
        self.update(delta_time, false);
    }

    pub fn step_with_cuda(&mut self, delta_time: f32) {
        self.update(delta_time, true);
    }

    pub fn set_interaction_force(&mut self, pos_x: f32, pos_y: f32, radius: f32, strength: f32) {
        self.force_point = ForcePoint {
            pos: [pos_x, pos_y],
            radius,
            strength,
        };
    }

    pub fn delete_interaction_force(&mut self) {
        self.force_point = ForcePoint {
            pos: [0.0, 0.0],
            radius: 0.0,
            strength: 0.0,
        };
    }

    pub fn get_world_width(&self) -> f32 {
        self.world_size[0]
    }

    pub fn get_world_height(&self) -> f32 {
        self.world_size[1]
    }

    pub fn get_smoothing_radius(&self) -> f32 {
        self.smoothing_radius
    }

    pub fn get_gravity(&self) -> f32 {
        self.gravity
    }

    pub fn get_drag(&self) -> f32 {
        self.drag
    }

    pub fn get_target_density(&self) -> f32 {
        self.target_density
    }

    pub fn get_pressure_multiplier(&self) -> f32 {
        self.pressure_multiplier
    }

    pub fn get_delta(&self) -> f32 {
        self.delta
    }

    pub fn get_collision_damping(&self) -> f32 {
        self.collision_damping
    }

    pub fn get_num_particles(&self) -> usize {
        self.num_particle
    }

    pub fn positions(&self) -> &[[f32; 2]] {
        &self.pos
    }

    pub fn velocities(&self) -> &[[f32; 2]] {
        &self.vel
    }

    pub fn query_neighbors(&self, x: f32, y: f32) -> Vec<usize> {
        let candidates = self.gridmap.find_neighborhood(x, y, self.smoothing_radius);
        let r2 = self.smoothing_radius * self.smoothing_radius;
        candidates
            .into_iter()
            .filter(|idx| {
                let dx = self.pos[*idx][0] - x;
                let dy = self.pos[*idx][1] - y;
                dx * dx + dy * dy <= r2
            })
            .collect()
    }

    pub fn query_spatial_hash(&self, x: f32, y: f32) -> Vec<usize> {
        self.gridmap.find_neighborhood(x, y, self.smoothing_radius)
    }

    fn update(&mut self, delta_time: f32, prefer_cuda: bool) {
        self.predicted_pos(delta_time);
        self.gridmap.unregister_all();

        for idx in 0..self.num_particle {
            let p = self.pos[idx];
            self.gridmap.register_target(idx, p[0], p[1]);
        }

        self.querysize = (0..self.num_particle)
            .map(|idx| {
                let p = self.pos[idx];
                self.gridmap
                    .find_neighborhood(p[0], p[1], self.smoothing_radius)
            })
            .collect();

        for idx in 0..self.num_particle {
            self.update_density(idx, prefer_cuda);
        }
        for idx in 0..self.num_particle {
            self.update_pressure_force(idx);
        }
        for idx in 0..self.num_particle {
            self.update_interaction_force(idx);
        }

        self.update_position(delta_time);
        for idx in 0..self.num_particle {
            self.fix_position_from_world_size(idx);
        }
        self.update_color();
    }

    fn predicted_pos(&mut self, delta_time: f32) {
        for i in 0..self.num_particle {
            self.vel[i][1] += self.mass[i] * self.gravity * delta_time;
            self.predpos[i][0] = self.pos[i][0] + self.vel[i][0] * (1.0 / 120.0);
            self.predpos[i][1] = self.pos[i][1] + self.vel[i][1] * (1.0 / 120.0);
        }
    }

    fn update_density(&mut self, particle_index: usize, prefer_cuda: bool) {
        self.density[particle_index] = self.calc_density(particle_index, prefer_cuda);
    }

    fn update_pressure_force(&mut self, particle_index: usize) {
        let mut pressure_force = [0.0, 0.0];
        self.calc_pressure_force(&mut pressure_force, particle_index);
        let divisor = self.density[particle_index] + self.delta;
        self.pressure_accelerations[particle_index][0] = pressure_force[0] / divisor;
        self.pressure_accelerations[particle_index][1] = pressure_force[1] / divisor;
    }

    fn update_interaction_force(&mut self, particle_index: usize) {
        let mut force = [0.0, 0.0];
        self.calc_interaction_force(&mut force, particle_index);
        self.interaction_force[particle_index] = force;
    }

    fn update_position(&mut self, delta_time: f32) {
        for i in 0..self.num_particle {
            self.vel[i][0] +=
                (self.pressure_accelerations[i][0] + self.interaction_force[i][0]) * delta_time;
            self.vel[i][1] +=
                (self.pressure_accelerations[i][1] + self.interaction_force[i][1]) * delta_time;
            self.pos[i][0] += self.vel[i][0] * delta_time;
            self.pos[i][1] += self.vel[i][1] * delta_time;
            self.vel[i][0] *= self.drag;
            self.vel[i][1] *= self.drag;
        }
    }

    fn fix_position_from_world_size(&mut self, i: usize) {
        let x = self.pos[i][0];
        let y = self.pos[i][1];
        let vel_x = self.vel[i][0];
        let vel_y = self.vel[i][1];
        let w = self.world_size[0];
        let h = self.world_size[1];
        if x < 0.0 {
            self.pos[i][0] = 0.0;
            self.vel[i][0] = -vel_x * self.collision_damping;
        }
        if w < x {
            self.pos[i][0] = w;
            self.vel[i][0] = -vel_x * self.collision_damping;
        }
        if y < 0.0 {
            self.pos[i][1] = 0.0;
            self.vel[i][1] = -vel_y * self.collision_damping;
        }
        if h < y {
            self.pos[i][1] = h;
            self.vel[i][1] = -vel_y * self.collision_damping;
        }
    }

    fn update_color(&mut self) {
        let mut speeds = Vec::with_capacity(self.num_particle);
        let mut min_speed = f32::MAX;
        let mut max_speed = 0.0_f32;
        for i in 0..self.num_particle {
            let speed = (self.vel[i][0] * self.vel[i][0] + self.vel[i][1] * self.vel[i][1]).sqrt();
            min_speed = min_speed.min(speed);
            max_speed = max_speed.max(speed);
            speeds.push(speed);
        }

        for (i, speed) in speeds.into_iter().enumerate() {
            let denom = (max_speed - min_speed).max(f32::EPSILON);
            let norm_speed = ((speed - min_speed) / denom).clamp(0.0, 1.0);
            let byte_vel = (norm_speed * 255.0) as u8;
            let color1 = [0, 0, 255];
            let color2 = [255, 0, 0];
            self.color[i][0] = (((255 - byte_vel) as u16 * color1[0] as u16
                + byte_vel as u16 * color2[0] as u16)
                / 255) as u8;
            self.color[i][1] = (((255 - byte_vel) as u16 * color1[1] as u16
                + byte_vel as u16 * color2[1] as u16)
                / 255) as u8;
            self.color[i][2] = (((255 - byte_vel) as u16 * color1[2] as u16
                + byte_vel as u16 * color2[2] as u16)
                / 255) as u8;
        }
    }

    fn calc_density(&self, particle_index: usize, prefer_cuda: bool) -> f32 {
        #[cfg(not(feature = "cuda"))]
        let _ = prefer_cuda;

        let other_indexes = &self.querysize[particle_index];
        if other_indexes.is_empty() {
            return 0.0;
        }

        let distances: Vec<f32> = other_indexes
            .iter()
            .map(|&j| {
                let dx = self.predpos[j][0] - self.predpos[particle_index][0];
                let dy = self.predpos[j][1] - self.predpos[particle_index][1];
                (dx * dx + dy * dy).sqrt()
            })
            .collect();

        #[cfg(feature = "cuda")]
        if prefer_cuda {
            if let Ok(values) = sph_cuda::calc_smoothing_kernel(&distances, self.smoothing_radius) {
                return other_indexes
                    .iter()
                    .zip(values.into_iter())
                    .map(|(&j, val)| self.mass[j] * val)
                    .sum();
            }
        }

        other_indexes
            .iter()
            .zip(distances.iter())
            .map(|(&j, &dist)| self.mass[j] * calc_smoothing_kernel(dist, self.smoothing_radius))
            .sum()
    }

    fn calc_pressure_force(&self, pressure_force: &mut [f32; 2], particle_index: usize) {
        for &other_index in &self.querysize[particle_index] {
            if particle_index == other_index {
                continue;
            }
            let offset_x = self.pos[other_index][0] - self.pos[particle_index][0];
            let offset_y = self.pos[other_index][1] - self.pos[particle_index][1];
            let dist = (offset_x * offset_x + offset_y * offset_y).sqrt();
            if dist > self.smoothing_radius {
                continue;
            }
            let (dir_x, dir_y) = if dist <= f32::EPSILON {
                (1.0_f32, 0.0_f32)
            } else {
                (offset_x / dist, offset_y / dist)
            };
            let slope = calc_smoothing_kernel_derivative(dist, self.smoothing_radius);
            let other_density = self.density[other_index];
            let shared_pressure =
                self.calc_shared_pressure(other_density, self.density[particle_index]);
            let a = shared_pressure * slope * self.mass[other_index] / (other_density + self.delta);
            pressure_force[0] += dir_x * a;
            pressure_force[1] += dir_y * a;
        }
    }

    fn calc_shared_pressure(&self, density_left: f32, density_right: f32) -> f32 {
        let pressure_left = self.convert_density_to_pressure(density_left);
        let pressure_right = self.convert_density_to_pressure(density_right);
        (pressure_left + pressure_right) * 0.5
    }

    fn convert_density_to_pressure(&self, density_val: f32) -> f32 {
        let density_error = density_val - self.target_density;
        density_error * self.pressure_multiplier
    }

    fn calc_interaction_force(&self, out_force: &mut [f32; 2], particle_index: usize) {
        out_force[0] = 0.0;
        out_force[1] = 0.0;
        let p = &self.force_point;
        let offset_x = p.pos[0] - self.pos[particle_index][0];
        let offset_y = p.pos[1] - self.pos[particle_index][1];
        let sqr_dst = offset_x * offset_x + offset_y * offset_y;
        if sqr_dst >= p.radius * p.radius {
            return;
        }
        let dist = sqr_dst.sqrt();
        let (dir_to_force_pos_x, dir_to_force_pos_y) = if dist > f32::EPSILON {
            (offset_x / dist, offset_y / dist)
        } else {
            (0.0, 0.0)
        };
        let centre_t = 1.0 - dist / p.radius;
        let vel_x = self.vel[particle_index][0];
        let vel_y = self.vel[particle_index][1];
        out_force[0] = (dir_to_force_pos_x * p.strength - vel_x) * centre_t;
        out_force[1] = (dir_to_force_pos_y * p.strength - vel_y) * centre_t;
    }
}
