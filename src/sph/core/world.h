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
void predictedPosCUDA(float* d_posX, float* d_posY,
                      float* d_velX, float* d_velY,
                      float* d_predX, float* d_predY,
                      float gravity, float dt, int n);
void updatePositionCUDA(float* d_posX, float* d_posY,
                        float* d_velX, float* d_velY,
                        float* d_pressureX, float* d_pressureY,
                        float* d_interactionX, float* d_interactionY,
                        float drag, float dt, int n);
void fixPositionCUDA(float* d_posX, float* d_posY,
                     float* d_velX, float* d_velY,
                     float width, float height,
                     float damping, int n);
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
    void findNeighborhood(float x, float y, float radius, std::vector<int>& out) const;
};

struct ForcePoint {
    float pos[2];
    float radius;
    float strength;
};

struct ProfileInfo {
    double predictedPosMs = 0.0;
    double gridRegisterMs = 0.0;
    double queryMs = 0.0;
    double densityMs = 0.0;
    double pressureMs = 0.0;
    double interactionMs = 0.0;
    double updatePosMs = 0.0;
    double fixPosMs = 0.0;
    double colorMs = 0.0;
    size_t memTransferBytes = 0;
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
    static constexpr int numParticle = 1000;

private:
    int activeParticles = numParticle;
    const int particleRadius = 5;

    float gravity = 9.8f;
    float worldSize[2] = {20.0f, 10.0f};
    float collisionDamping = 1.0f;
    float smoothingRadius = 0.8f;
    float targetDensity = 32.0f;
    float pressureMultiplier = 100.0f;
    float delta = 0.0f;
    float drag = 0.9999f;

    std::vector<float> posX, posY;
    std::vector<float> predPosX, predPosY;
    std::vector<float> velX, velY;
    std::vector<float> density;
    std::vector<float> pressureX, pressureY;
    std::vector<float> interactionX, interactionY;
    std::vector<int> colorR, colorG, colorB;
    std::vector<std::vector<int>> querysize;
    std::vector<int> iterator;
    std::vector<float> mass;

    ForcePoint forcePoint;
    GridMap gridmap;
#ifdef USE_CUDA
    float* d_dist_buffer = nullptr;
    float* d_out_buffer = nullptr;
    float *d_posX = nullptr, *d_posY = nullptr;
    float *d_predPosX = nullptr, *d_predPosY = nullptr;
    float *d_velX = nullptr, *d_velY = nullptr;
    float* d_density = nullptr;
    float *d_pressureX = nullptr, *d_pressureY = nullptr;
    float *d_interactionX = nullptr, *d_interactionY = nullptr;
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
    void setActiveParticleCount(int n);
    int getActiveParticleCount() const { return activeParticles; }

    void update(float deltaTime);
    void updateWithStats(float deltaTime, ProfileInfo& info);
    const std::vector<float>& getPosX() const { return posX; }
    const std::vector<float>& getPosY() const { return posY; }
    const std::vector<float>& getVelX() const { return velX; }
    const std::vector<float>& getVelY() const { return velY; }

private:
    void predictedPos(float deltaTime, ProfileInfo* info = nullptr);
    void updateDensity(int particleIndex);
    void updatePressureForce(int particleIndex);
    void updateInteractionForce(int particleIndex);
    void updatePosition(float deltaTime, ProfileInfo* info = nullptr);
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
