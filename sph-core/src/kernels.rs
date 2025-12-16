/// CPU implementation of the smoothing kernel used by the SPH solver.
pub fn calc_smoothing_kernel(dist: f32, radius: f32) -> f32 {
    if dist >= radius {
        return 0.0;
    }
    let volume = std::f32::consts::PI * radius * radius * radius * radius / 6.0;
    let influence = (radius - dist) * (radius - dist) / volume;
    influence
}

/// Derivative of the smoothing kernel with respect to distance.
pub fn calc_smoothing_kernel_derivative(dist: f32, radius: f32) -> f32 {
    if dist >= radius {
        return 0.0;
    }
    let scale = 12.0 / (std::f32::consts::PI * radius * radius * radius * radius);
    let slope = (dist - radius) * scale;
    slope
}
