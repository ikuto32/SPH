#include <cmath>
#include <iostream>
#include <cassert>

float calcSmoothingKernel(float dist, float radius)
{
    if (dist >= radius)
    {
        return 0.0f;
    }
    float volume = (float)(M_PI * radius * radius * radius * radius) / 6.0f;
    float influence = (radius - dist) * (radius - dist) / volume;
    return influence;
}

int main()
{
    const float radius = 2.0f;
    const float dist = 0.5f;
    float volume = (float)(M_PI * radius * radius * radius * radius) / 6.0f;
    float expected = (radius - dist) * (radius - dist) / volume;

    assert(std::abs(calcSmoothingKernel(dist, radius) - expected) < 1e-6);
    assert(calcSmoothingKernel(radius, radius) == 0.0f);
    std::cout << "All tests passed" << std::endl;
    return 0;
}
