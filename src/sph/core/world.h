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
    std::vector<int> getChunk(int chunkX, int chunkY);
    void registerTarget(int target, float x, float y);
    void unregisterAll();
    std::vector<int> findNeighborhood(float x, float y, float radius);
};

struct ForcePoint {
    float pos[2];
    float radius;
    float strength;
};

class World {
public:
    static const int numParticle = 1000;

private:
    const int particleRadius = 5;
    const float gravity = 9.8F;
    const float worldSize[2] = {20, 10};
    const float collisionDamping = 1.0F;
    const float smoothingRadius = 0.8F;
    const float targetDensity = 32.0F;
    const float pressureMultiplier = 100.0F;
    const float delta = 0.0F;
    const float drag = 0.9999F;

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

public:
    World();

    void setInteractionForce(float posX, float posY, float radius, float strength);
    void deleteInteractionForce();

    float getWorldWidth() const;
    float getWorldHeight() const;

    void update(float deltaTime);
    const float (*getPositions() const)[2] { return pos; }

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
