#ifndef SPH_CORE_WORLD_H
#define SPH_CORE_WORLD_H

#include <vector>
#include <array>
#include <cstddef>
#include <algorithm>
#include <execution>
#include <cmath>
#include <cfloat>
#include <cstdint>

#include "kernels.h"
#include "kernels_cuda.h"
#ifdef SPH_ENABLE_HASH2D
#include "sph/gpu/hash_grid_2d.hpp"
#include <cuda_runtime.h>
#endif

namespace sph {

float floatRand();

class GridMap {
private:
    int width;
    int height;
    float chunkRange;
    int numChunkWidth;
    int numChunkHeight;
    int numChunk;
    std::vector<std::vector<int>> chunks;

public:
    GridMap(float width, float height, float radius);
    std::vector<int> getChunk(int chunkX, int chunkY) const;
    void registerTarget(int target, float x, float y);
    void unregisterAll();
    std::vector<int> findNeighborhood(float x, float y, float radius) const;
};

struct ForcePoint {
    float pos[2];
    float radius;
    float strength;
};

struct WorldConfig {
    float worldWidth = 20.0f;
    float worldHeight = 10.0f;
    float smoothingRadius = 0.8f;
    float targetDensity = 32.0f;
    float pressureMultiplier = 100.0f;
    float delta = 0.0f;
    float drag = 0.9999f;
    float gravity = 9.8f;
    float collisionDamping = 1.0f;
    int   numParticles = 1000;
};

class World {
public:
    static constexpr int defaultNumParticles = 1000;

private:
    const int particleRadius = 5;

    int numParticle = defaultNumParticles;
    float gravity = 9.8f;
    float worldSize[2] = {20.0f, 10.0f};
    float collisionDamping = 1.0f;
    float smoothingRadius = 0.8f;
    float targetDensity = 32.0f;
    float pressureMultiplier = 100.0f;
    float delta = 0.0f;
    float drag = 0.9999f;

    std::vector<std::array<float, 2>> pos;
    std::vector<std::array<float, 2>> predpos;
    std::vector<std::array<float, 2>> vel;
    std::vector<float> density;
    std::vector<std::array<float, 2>> pressureAccelerations;
    std::vector<std::array<float, 2>> interactionForce;
    std::vector<std::array<int, 3>> color;
    std::vector<std::vector<int>> querysize;
    std::vector<int> iterator;
    std::vector<float> mass;

    ForcePoint forcePoint;
    GridMap gridmap;
#ifdef SPH_ENABLE_HASH2D
    HashGrid2D grid;
    uint32_t* d_neighbors = nullptr;
    uint32_t* d_counts    = nullptr;
    bool device_allocated = false;
#endif

public:
    World(const WorldConfig& config = WorldConfig());
#ifdef SPH_ENABLE_HASH2D
    ~World();
#endif

    void setInteractionForce(float posX, float posY, float radius, float strength);
    void deleteInteractionForce();

    float getWorldWidth() const;
    float getWorldHeight() const;
    float getSmoothingRadius() const { return smoothingRadius; }
    float getGravity() const { return gravity; }
    float getDrag() const { return drag; }
    float getTargetDensity() const { return targetDensity; }
    float getPressureMultiplier() const { return pressureMultiplier; }
    float getDelta() const { return delta; }
    float getCollisionDamping() const { return collisionDamping; }

    void update(float deltaTime);
    void stepGPU(float deltaTime);
    size_t getNumParticles() const { return numParticle; }
    const std::vector<std::array<float,2>>& getPositions() const { return pos; }
    const std::vector<std::array<float,2>>& getVelocities() const { return vel; }

    // query neighbours around an arbitrary point
    std::vector<int> queryNeighbors(float x, float y) const;
    // return indices from spatial hash before distance filtering
    std::vector<int> querySpatialHash(float x, float y) const;

private:
    void predictedPos(float deltaTime);
    void updateDensity(int particleIndex);
    void updatePressureForce(int particleIndex);
    void updateInteractionForce(int particleIndex);
    void updatePosition(float deltaTime);
    void fixPositionFromWorldSize(int i);
    void updateColor();
    float calcDensity(int particleIndex);
    void calcPressureForce(float pressureForce[], int particleIndex);
    float calcSharedPressure(float densityLeft, float densityRight) const;
    float convertDensityToPressure(float density) const;
    void calcInteractionForce(float outInteractionForce[], int particleIndex) const;
#ifdef SPH_ENABLE_HASH2D
    void allocateDeviceBuffers();
    void freeDeviceBuffers();
#endif
};

} // namespace sph

#endif // SPH_CORE_WORLD_H
