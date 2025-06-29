#ifndef SPH_CORE_WORLD_H
#define SPH_CORE_WORLD_H

#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>

#include "kernels.h"
#include "kernels_cuda.h"

namespace sph {

float floatRand();
#ifdef USE_CUDA
void predictedPosCUDA(float* d_pos, float* d_vel, float* d_predpos,
                      float gravity, float dt, int n);
void updatePositionCUDA(float* d_pos, float* d_vel, float* d_pressure,
                        float* d_interaction, float drag, float dt, int n);
#endif

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
    const std::vector<int>& getChunk(int chunkX, int chunkY) const;
    void registerTarget(int target, float x, float y);
    void unregisterAll();
    std::vector<int> findNeighborhood(float x, float y, float radius);
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
};

class World {
public:
    static const int numParticle = 1000;

private:
    const int particleRadius = 5;

    float gravity = 9.8f;
    float worldSize[2] = {20.0f, 10.0f};
    float collisionDamping = 1.0f;
    float smoothingRadius = 0.8f;
    float targetDensity = 32.0f;
    float pressureMultiplier = 100.0f;
    float delta = 0.0f;
    float drag = 0.9999f;

    float pos[numParticle][2];
    float predpos[numParticle][2];
    float vel[numParticle][2];
    float density[numParticle];
    float pressureAccelerations[numParticle][2];
    float interactionForce[numParticle][2];
    int color[numParticle][3];
    std::vector<std::vector<int>> querysize;
    std::vector<int> iterator;
    float mass[numParticle];

    ForcePoint forcePoint;
    GridMap gridmap;
#ifdef USE_CUDA
    float* d_dist_buffer = nullptr;
    float* d_out_buffer = nullptr;
    float* d_pos = nullptr;
    float* d_predpos = nullptr;
    float* d_vel = nullptr;
    float* d_density = nullptr;
    float* d_pressure = nullptr;
    float* d_interaction = nullptr;
#endif

public:
    World(const WorldConfig& config = WorldConfig());
    ~World();

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
    const float (*getPositions() const)[2] { return pos; }
    const float (*getVelocities() const)[2] { return vel; }

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
};

} // namespace sph

#endif // SPH_CORE_WORLD_H
