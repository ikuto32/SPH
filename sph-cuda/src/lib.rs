//! Optional CUDA-backed helpers for the SPH workspace.
//!
//! The crate is intentionally lightweight: it detects CUDA availability at
//! build time and provides CPU fallbacks so enabling the `cuda` feature never
//! breaks compilation on hosts without the toolkit installed.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA support was requested but no CUDA toolchain was detected at build time")]
    Unavailable,
}

/// Returns `true` when the build script found an nvcc toolchain.
pub fn cuda_available() -> bool {
    cfg!(all(feature = "cuda", cuda_available))
}

/// CPU implementation of the smoothing kernel used when CUDA is unavailable or
/// when the `cuda` feature is disabled.
fn cpu_smoothing_kernel(dist: f32, radius: f32) -> f32 {
    if dist >= radius {
        return 0.0;
    }
    let volume = std::f32::consts::PI * radius * radius * radius * radius / 6.0;
    let influence = (radius - dist) * (radius - dist) / volume;
    influence
}

/// Computes smoothing kernel influences, dispatching to CUDA when available.
///
/// The current implementation mirrors the CPU behavior; CUDA acceleration can
/// be added later without changing the public interface.
pub fn calc_smoothing_kernel(distances: &[f32], radius: f32) -> Result<Vec<f32>, CudaError> {
    #[cfg(all(feature = "cuda", cuda_available))]
    {
        // Placeholder for an actual CUDA implementation. Keeping the branch
        // explicit makes it easy to extend without changing callers.
        let mut out = Vec::with_capacity(distances.len());
        out.extend(
            distances
                .iter()
                .copied()
                .map(|d| cpu_smoothing_kernel(d, radius)),
        );
        return Ok(out);
    }

    if cfg!(feature = "cuda") {
        return Err(CudaError::Unavailable);
    }

    let mut out = Vec::with_capacity(distances.len());
    out.extend(
        distances
            .iter()
            .copied()
            .map(|d| cpu_smoothing_kernel(d, radius)),
    );
    Ok(out)
}
