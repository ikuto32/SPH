#include "world.h"

namespace sph {

float floatRand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

GridMap::GridMap(float width, float height, float radius)
    : width(static_cast<int>(width)),
      height(static_cast<int>(height)),
      chunkRange(radius) {
    numChunkWidth = static_cast<int>(std::ceil(this->width / this->chunkRange)) + 1;
    numChunkHeight = static_cast<int>(std::ceil(this->height / this->chunkRange)) + 1;
    numChunk = numChunkWidth * numChunkHeight;
    chunks.assign(numChunk, std::vector<int>());
}

const std::vector<int>& GridMap::getChunk(int chunkX, int chunkY) const {
    return chunks[chunkY * numChunkWidth + chunkX];
}

void GridMap::registerTarget(int target, float x, float y) {
    int chunkX = static_cast<int>(x / chunkRange);
    int chunkY = static_cast<int>(y / chunkRange);
    if (chunkX < 0 || chunkX >= numChunkWidth) return;
    if (chunkY < 0 || chunkY >= numChunkHeight) return;
    chunks[chunkY * numChunkWidth + chunkX].push_back(target);
}

void GridMap::unregisterAll() {
    for (int i = 0; i < numChunk; ++i) {
        chunks[i].clear();
    }
}

std::vector<int> GridMap::findNeighborhood(float x, float y, float radius) {
    std::vector<int> out;
    int centerChunkX = static_cast<int>(x / chunkRange);
    int centerChunkY = static_cast<int>(y / chunkRange);
    int radiusChunk = static_cast<int>(std::ceil(radius / chunkRange));
    int minChunkX = centerChunkX - radiusChunk;
    int maxChunkX = centerChunkX + radiusChunk;
    int minChunkY = centerChunkY - radiusChunk;
    int maxChunkY = centerChunkY + radiusChunk;

    for (int x1 = minChunkX; x1 <= maxChunkX; ++x1) {
        for (int y1 = minChunkY; y1 <= maxChunkY; ++y1) {
            if (x1 < 0 || x1 >= numChunkWidth) continue;
            if (y1 < 0 || y1 >= numChunkHeight) continue;
            const auto& chunk = getChunk(x1, y1);
            out.insert(out.end(), chunk.begin(), chunk.end());
        }
    }
    return out;
}

World::World(const WorldConfig& config)
    : gravity(config.gravity),
      worldSize{config.worldWidth, config.worldHeight},
      collisionDamping(config.collisionDamping),
      smoothingRadius(config.smoothingRadius),
      targetDensity(config.targetDensity),
      pressureMultiplier(config.pressureMultiplier),
      delta(config.delta),
      drag(config.drag),
      forcePoint{{0,0},0,0},
      gridmap(worldSize[0], worldSize[1], smoothingRadius)
{
    std::memset(pos, 0, sizeof(pos));
    std::memset(predpos, 0, sizeof(predpos));
    std::memset(vel, 0, sizeof(vel));
    std::memset(density, 0, sizeof(density));
    std::memset(pressureAccelerations, 0, sizeof(pressureAccelerations));
    std::memset(interactionForce, 0, sizeof(interactionForce));
    for (int i = 0; i < numParticle; ++i) mass[i] = 1.0f;

    for (int i = 0; i < numParticle; ++i) {
        int a = static_cast<int>(std::sqrt(numParticle));
        int row = i / a;
        int col = i % a;
        pos[i][0] = (col / static_cast<float>(a)) * worldSize[0];
        pos[i][1] = (row / static_cast<float>(a)) * worldSize[1];
    }

    std::memset(color, 255, sizeof(color));
    iterator.resize(numParticle);
    for (int i = 0; i < numParticle; ++i) iterator[i] = i;
#ifdef USE_CUDA
    CUDA_CHECK(cudaMalloc(&d_dist_buffer, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_buffer, numParticle * sizeof(float)));
#endif
}

World::~World()
{
#ifdef USE_CUDA
    CUDA_CHECK(cudaFree(d_dist_buffer));
    CUDA_CHECK(cudaFree(d_out_buffer));
    d_dist_buffer = nullptr;
    d_out_buffer = nullptr;
#endif
}

void World::setInteractionForce(float posX, float posY, float radius, float strength) {
    forcePoint = {{posX, posY}, radius, strength};
}

void World::deleteInteractionForce() {
    forcePoint = {{0,0},0,0};
}

float World::getWorldWidth() const { return worldSize[0]; }
float World::getWorldHeight() const { return worldSize[1]; }

void World::update(float deltaTime) {
    predictedPos(deltaTime);
    gridmap.unregisterAll();

    std::vector<int> v = iterator;
    std::for_each(std::execution::seq, v.begin(), v.end(), [&](int idx){
        gridmap.registerTarget(idx, pos[idx][0], pos[idx][1]);
    });

    querysize.resize(numParticle);
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){
        querysize[idx] = gridmap.findNeighborhood(pos[idx][0], pos[idx][1], smoothingRadius);
    });

    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updateDensity(idx); });
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updatePressureForce(idx); });
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updateInteractionForce(idx); });

    updatePosition(deltaTime);
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ fixPositionFromWorldSize(idx); });
    updateColor();
}

void World::predictedPos(float deltaTime) {
    for (int i = 0; i < numParticle; ++i) {
        vel[i][0] += 0.0f;
        vel[i][1] += mass[i] * gravity * deltaTime;
        predpos[i][0] = pos[i][0] + vel[i][0] * 1.0f / 120.0f;
        predpos[i][1] = pos[i][1] + vel[i][1] * 1.0f / 120.0f;
    }
}

void World::updateDensity(int particleIndex) { density[particleIndex] = calcDensity(particleIndex); }

void World::updatePressureForce(int particleIndex) {
    float pressureForce[] = {0,0};
    calcPressureForce(pressureForce, particleIndex);
    pressureAccelerations[particleIndex][0] = pressureForce[0] / (density[particleIndex] + delta);
    pressureAccelerations[particleIndex][1] = pressureForce[1] / (density[particleIndex] + delta);
}

void World::updateInteractionForce(int i) {
    float outForce[] = {0,0};
    calcInteractionForce(outForce, i);
    interactionForce[i][0] = outForce[0];
    interactionForce[i][1] = outForce[1];
}

void World::updatePosition(float deltaTime) {
    for (int i = 0; i < numParticle; ++i) {
        vel[i][0] += (pressureAccelerations[i][0] + interactionForce[i][0]) * deltaTime;
        vel[i][1] += (pressureAccelerations[i][1] + interactionForce[i][1]) * deltaTime;
        pos[i][0] += vel[i][0] * deltaTime;
        pos[i][1] += vel[i][1] * deltaTime;
        vel[i][0] *= drag;
        vel[i][1] *= drag;
    }
}

void World::fixPositionFromWorldSize(int i) {
    float x = pos[i][0];
    float y = pos[i][1];
    float velX = vel[i][0];
    float velY = vel[i][1];
    int w = static_cast<int>(worldSize[0]);
    int h = static_cast<int>(worldSize[1]);
    if (x < 0) { pos[i][0] = 0; vel[i][0] = -velX * collisionDamping; }
    if (w < x) { pos[i][0] = w; vel[i][0] = -velX * collisionDamping; }
    if (y < 0) { pos[i][1] = 0; vel[i][1] = -velY * collisionDamping; }
    if (h < y) { pos[i][1] = h; vel[i][1] = -velY * collisionDamping; }
}

void World::updateColor() {
    float speeds[numParticle];
    float minSpeed = FLT_MAX;
    float maxSpeed = 0.0f;
    int color1[3] = {0,0,255};
    int color2[3] = {255,0,0};
    for (int i = 0; i < numParticle; ++i) {
        float speed = std::sqrt(vel[i][0]*vel[i][0] + vel[i][1]*vel[i][1]);
        if (minSpeed > speed) minSpeed = speed;
        if (maxSpeed < speed) maxSpeed = speed;
        speeds[i] = speed;
    }
    for (int i = 0; i < numParticle; ++i) {
        float normSpeed = (speeds[i] - minSpeed) / (maxSpeed - minSpeed);
        normSpeed = std::clamp(normSpeed, 0.0f, 1.0f);
        uint8_t byteVel = static_cast<uint8_t>(normSpeed * 255.0f);
        color[i][0] = (255 - byteVel) * color1[0] + byteVel * color2[0];
        color[i][1] = (255 - byteVel) * color1[1] + byteVel * color2[1];
        color[i][2] = (255 - byteVel) * color1[2] + byteVel * color2[2];
    }
}

float World::calcDensity(int particleIndex) {
    float densityVal = 0.0f;
    auto otherIndexes = querysize[particleIndex];
    std::vector<float> distances(otherIndexes.size());
    for (size_t idx = 0; idx < otherIndexes.size(); ++idx) {
        int j = otherIndexes[idx];
        float dx = predpos[j][0] - predpos[particleIndex][0];
        float dy = predpos[j][1] - predpos[particleIndex][1];
        distances[idx] = std::sqrt(dx*dx + dy*dy);
    }
    std::vector<float> influences(otherIndexes.size());
#ifdef USE_CUDA
    sph::calcSmoothingKernelCUDA(distances.data(),
                                influences.data(),
                                smoothingRadius,
                                static_cast<int>(otherIndexes.size()),
                                d_dist_buffer,
                                d_out_buffer);
#else
    sph::calcSmoothingKernelCUDA(distances.data(),
                                influences.data(),
                                smoothingRadius,
                                static_cast<int>(otherIndexes.size()),
                                nullptr,
                                nullptr);
#endif
    for (size_t idx = 0; idx < otherIndexes.size(); ++idx) {
        int j = otherIndexes[idx];
        densityVal += mass[j] * influences[idx];
    }
    return densityVal;
}

void World::calcPressureForce(float pressureForce[], int particleIndex) {
    auto otherIndexes = querysize[particleIndex];
    for (int otherIndex : otherIndexes) {
        if (particleIndex == otherIndex) continue;
        float offsetX = pos[otherIndex][0] - pos[particleIndex][0];
        float offsetY = pos[otherIndex][1] - pos[particleIndex][1];
        float dist = std::sqrt(offsetX*offsetX + offsetY*offsetY);
        if (dist > smoothingRadius) continue;
        float dirX = 0.0f;
        float dirY = 0.0f;
        if (dist <= FLT_EPSILON) {
            dirX = floatRand() - 0.5f;
            dirY = floatRand() - 0.5f;
        } else {
            dirX = offsetX / dist;
            dirY = offsetY / dist;
        }
        float slope = sph::calcSmoothingKernelDerivative(dist, smoothingRadius);
        float otherDensity = density[otherIndex];
        float sharedPressure = calcSharedPressure(otherDensity, density[particleIndex]);
        float a = sharedPressure * slope * mass[otherIndex] / (otherDensity + delta);
        pressureForce[0] += dirX * a;
        pressureForce[1] += dirY * a;
    }
}

float World::calcSharedPressure(float densityLeft, float densityRight) const {
    float pressureLeft = convertDensityToPressure(densityLeft);
    float pressureRight = convertDensityToPressure(densityRight);
    return (pressureLeft + pressureRight) * 0.5f;
}

float World::convertDensityToPressure(float densityVal) const {
    float densityError = densityVal - targetDensity;
    float pressure = densityError * pressureMultiplier;
    return pressure;
}

void World::calcInteractionForce(float outForce[], int particleIndex) const {
    outForce[0] = 0.0f;
    outForce[1] = 0.0f;
    ForcePoint p = forcePoint;
    float offsetX = p.pos[0] - pos[particleIndex][0];
    float offsetY = p.pos[1] - pos[particleIndex][1];
    float sqrDst = offsetX*offsetX + offsetY*offsetY;
    if (!(sqrDst < p.radius * p.radius)) return;
    float dirToForcePosX = 0.0f;
    float dirToForcePosY = 0.0f;
    float dist = std::sqrt(sqrDst);
    if (FLT_EPSILON < dist) {
        dirToForcePosX = offsetX / dist;
        dirToForcePosY = offsetY / dist;
    }
    float centreT = 1 - dist / p.radius;
    float velX = vel[particleIndex][0];
    float velY = vel[particleIndex][1];
    outForce[0] = (dirToForcePosX * p.strength - velX) * centreT;
    outForce[1] = (dirToForcePosY * p.strength - velY) * centreT;
}

} // namespace sph

