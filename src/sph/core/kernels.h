#ifndef SPH_CORE_KERNELS_H
#define SPH_CORE_KERNELS_H

#include <cmath>

namespace sph {

inline float calcSmoothingKernel(float dist, float radius) {
    if (dist >= radius) {
        return 0.0f;
    }
    float volume = static_cast<float>(M_PI * radius * radius * radius * radius) / 6.0f;
    float influence = (radius - dist) * (radius - dist) / volume;
    return influence;
}

inline float calcSmoothingKernelDerivative(float dist, float radius) {
    if (dist >= radius) {
        return 0.0f;
    }
    float scale = 12.0f / static_cast<float>(M_PI * radius * radius * radius * radius);
    float slope = (dist - radius) * scale;
    return slope;
}

} // namespace sph

#endif // SPH_CORE_KERNELS_H
