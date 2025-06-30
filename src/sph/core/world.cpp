#include "world.h"
#include <chrono>
#include <random>
#ifdef USE_CUDA
#ifdef SPH_ENABLE_HASH2D
#include "sph/gpu/hash_grid_2d.hpp"
#endif
#endif

namespace sph {

const int World::numParticle;
thread_local std::mt19937 rng(std::random_device{}());
thread_local std::uniform_real_distribution<float> dist01(0.0f, 1.0f);


float floatRand() {
    return dist01(rng);
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

void GridMap::findNeighborhood(float x, float y, float radius, std::vector<int>& out) const {
    out.clear();
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
      gridmap(worldSize[0], worldSize[1], smoothingRadius),
      activeParticles(numParticle)
{
    posX.assign(numParticle, 0.0f);
    posY.assign(numParticle, 0.0f);
    predPosX.assign(numParticle, 0.0f);
    predPosY.assign(numParticle, 0.0f);
    velX.assign(numParticle, 0.0f);
    velY.assign(numParticle, 0.0f);
    density.assign(numParticle, 0.0f);
    pressureX.assign(numParticle, 0.0f);
    pressureY.assign(numParticle, 0.0f);
    interactionX.assign(numParticle, 0.0f);
    interactionY.assign(numParticle, 0.0f);
    mass.assign(numParticle, 1.0f);

    for (int i = 0; i < numParticle; ++i) {
        int a = static_cast<int>(std::sqrt(numParticle));
        int row = i / a;
        int col = i % a;
        posX[i] = (col / static_cast<float>(a)) * worldSize[0];
        posY[i] = (row / static_cast<float>(a)) * worldSize[1];
    }

    colorR.assign(numParticle, 255);
    colorG.assign(numParticle, 255);
    colorB.assign(numParticle, 255);
    iterator.resize(activeParticles);
    for (int i = 0; i < activeParticles; ++i) iterator[i] = i;
#ifdef USE_CUDA
    CUDA_CHECK(cudaMalloc(&d_dist_buffer, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_buffer, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_posX, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_posY, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predPosX, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predPosY, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_velX, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_velY, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_density, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressureX, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressureY, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_interactionX, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_interactionY, numParticle * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_posX, posX.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, posY.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, velX.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, velY.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pressureX, pressureX.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pressureY, pressureY.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interactionX, interactionX.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interactionY, interactionY.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
#endif
}

World::~World()
{
#ifdef USE_CUDA
    CUDA_CHECK(cudaFree(d_dist_buffer));
    CUDA_CHECK(cudaFree(d_out_buffer));
    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_predPosX));
    CUDA_CHECK(cudaFree(d_predPosY));
    CUDA_CHECK(cudaFree(d_velX));
    CUDA_CHECK(cudaFree(d_velY));
    CUDA_CHECK(cudaFree(d_density));
    CUDA_CHECK(cudaFree(d_pressureX));
    CUDA_CHECK(cudaFree(d_pressureY));
    CUDA_CHECK(cudaFree(d_interactionX));
    CUDA_CHECK(cudaFree(d_interactionY));
    d_dist_buffer = nullptr;
    d_out_buffer = nullptr;
    d_posX = d_posY = nullptr;
    d_predPosX = d_predPosY = nullptr;
    d_velX = d_velY = nullptr;
    d_density = nullptr;
    d_pressureX = d_pressureY = nullptr;
    d_interactionX = d_interactionY = nullptr;
#endif
}

void World::setActiveParticleCount(int n) {
    activeParticles = std::clamp(n, 1, numParticle);
    iterator.resize(activeParticles);
    for (int i = 0; i < activeParticles; ++i) iterator[i] = i;
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
#ifdef USE_CUDA
#ifdef SPH_ENABLE_HASH2D
    static HashGrid2D grid(worldSize[0], worldSize[1], smoothingRadius);
    static int* d_neighborCount = nullptr;
    if (!d_neighborCount) {
        CUDA_CHECK(cudaMalloc(&d_neighborCount, numParticle * sizeof(int)));
    }
    grid.build(d_posX, d_posY, activeParticles);
    launchNeighbourSearch(d_posX, d_posY, grid, smoothingRadius,
                          activeParticles, d_neighborCount);
#endif
#endif
    gridmap.unregisterAll();

    std::vector<int> v = iterator;
    std::for_each(std::execution::seq, v.begin(), v.end(), [&](int idx){
        gridmap.registerTarget(idx, posX[idx], posY[idx]);
    });

    querysize.resize(activeParticles);
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){
        gridmap.findNeighborhood(posX[idx], posY[idx], smoothingRadius, querysize[idx]);
    });

    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updateDensity(idx); });
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updatePressureForce(idx); });
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updateInteractionForce(idx); });

    updatePosition(deltaTime);
#ifdef USE_CUDA
    fixPositionCUDA(d_posX, d_posY, d_velX, d_velY, worldSize[0], worldSize[1],
                   collisionDamping, activeParticles);
    CUDA_CHECK(cudaMemcpy(posX.data(), d_posX, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posY.data(), d_posY, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velX.data(), d_velX, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velY.data(), d_velY, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
#else
    std::for_each(std::execution::par_unseq, v.begin(), v.end(),
                  [&](int idx) { fixPositionFromWorldSize(idx); });
#endif
    updateColor();
}

void World::updateWithStats(float deltaTime, ProfileInfo& info) {
    auto t0 = std::chrono::high_resolution_clock::now();
    predictedPos(deltaTime, &info);
    auto t1 = std::chrono::high_resolution_clock::now();
    info.predictedPosMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    gridmap.unregisterAll();
    std::vector<int> v = iterator;
    std::for_each(std::execution::seq, v.begin(), v.end(), [&](int idx){
        gridmap.registerTarget(idx, posX[idx], posY[idx]);
    });
    t1 = std::chrono::high_resolution_clock::now();
    info.gridRegisterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    querysize.resize(activeParticles);
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){
        gridmap.findNeighborhood(posX[idx], posY[idx], smoothingRadius, querysize[idx]);
    });
    t1 = std::chrono::high_resolution_clock::now();
    info.queryMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updateDensity(idx); });
    t1 = std::chrono::high_resolution_clock::now();
    info.densityMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updatePressureForce(idx); });
    t1 = std::chrono::high_resolution_clock::now();
    info.pressureMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int idx){ updateInteractionForce(idx); });
    t1 = std::chrono::high_resolution_clock::now();
    info.interactionMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    updatePosition(deltaTime, &info);
    t1 = std::chrono::high_resolution_clock::now();
    info.updatePosMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
#ifdef USE_CUDA
    fixPositionCUDA(d_posX, d_posY, d_velX, d_velY, worldSize[0], worldSize[1],
                   collisionDamping, activeParticles);
    CUDA_CHECK(cudaMemcpy(posX.data(), d_posX, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posY.data(), d_posY, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velX.data(), d_velX, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velY.data(), d_velY, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    info.memTransferBytes += 4 * numParticle * sizeof(float);
#else
    std::for_each(std::execution::par_unseq, v.begin(), v.end(),
                  [&](int idx){ fixPositionFromWorldSize(idx); });
#endif
    t1 = std::chrono::high_resolution_clock::now();
    info.fixPosMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    updateColor();
    t1 = std::chrono::high_resolution_clock::now();
    info.colorMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void World::predictedPos(float deltaTime, ProfileInfo* info) {
#ifdef USE_CUDA
    predictedPosCUDA(d_posX, d_posY, d_velX, d_velY,
                     d_predPosX, d_predPosY, gravity, deltaTime,
                     activeParticles);
    CUDA_CHECK(cudaMemcpy(velX.data(), d_velX, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velY.data(), d_velY, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    if (info) info->memTransferBytes += 2 * numParticle * sizeof(float);
    CUDA_CHECK(cudaMemcpy(predPosX.data(), d_predPosX, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(predPosY.data(), d_predPosY, numParticle * sizeof(float), cudaMemcpyDeviceToHost));
    if (info) info->memTransferBytes += 2 * numParticle * sizeof(float);
#else
    for (int i = 0; i < activeParticles; ++i) {
        velX[i] += 0.0f;
        velY[i] += mass[i] * gravity * deltaTime;
        predPosX[i] = posX[i] + velX[i] * deltaTime;
        predPosY[i] = posY[i] + velY[i] * deltaTime;
    }
#endif
}

void World::updateDensity(int particleIndex) { density[particleIndex] = calcDensity(particleIndex); }

void World::updatePressureForce(int particleIndex) {
    float pressureForce[] = {0,0};
    calcPressureForce(pressureForce, particleIndex);
    pressureX[particleIndex] = pressureForce[0] / (density[particleIndex] + delta);
    pressureY[particleIndex] = pressureForce[1] / (density[particleIndex] + delta);
}

void World::updateInteractionForce(int i) {
    float outForce[] = {0,0};
    calcInteractionForce(outForce, i);
    interactionX[i] = outForce[0];
    interactionY[i] = outForce[1];
}

void World::updatePosition(float deltaTime, ProfileInfo* info) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(d_pressureX, pressureX.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pressureY, pressureY.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    if (info) info->memTransferBytes += 2 * numParticle * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_interactionX, interactionX.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interactionY, interactionY.data(), numParticle * sizeof(float), cudaMemcpyHostToDevice));
    if (info) info->memTransferBytes += 2 * numParticle * sizeof(float);
    updatePositionCUDA(d_posX, d_posY, d_velX, d_velY,
                       d_pressureX, d_pressureY,
                       d_interactionX, d_interactionY,
                       drag, deltaTime, activeParticles);
#else
    for (int i = 0; i < activeParticles; ++i) {
        velX[i] += (pressureX[i] + interactionX[i]) * deltaTime;
        velY[i] += (pressureY[i] + interactionY[i]) * deltaTime;
        posX[i] += velX[i] * deltaTime;
        posY[i] += velY[i] * deltaTime;
        velX[i] *= drag;
        velY[i] *= drag;
    }
#endif
}

void World::fixPositionFromWorldSize(int i) {
    float x = posX[i];
    float y = posY[i];
    float velXv = velX[i];
    float velYv = velY[i];
    int w = static_cast<int>(worldSize[0]);
    int h = static_cast<int>(worldSize[1]);
    if (x < 0) { posX[i] = 0; velX[i] = -velXv * collisionDamping; }
    if (w < x) { posX[i] = w; velX[i] = -velXv * collisionDamping; }
    if (y < 0) { posY[i] = 0; velY[i] = -velYv * collisionDamping; }
    if (h < y) { posY[i] = h; velY[i] = -velYv * collisionDamping; }
}

void World::updateColor() {
    static std::vector<float> speeds;
    speeds.resize(activeParticles);
    float minSpeed = FLT_MAX;
    float maxSpeed = 0.0f;
    int color1[3] = {0,0,255};
    int color2[3] = {255,0,0};
    for (int i = 0; i < activeParticles; ++i) {
        float speed = std::sqrt(velX[i]*velX[i] + velY[i]*velY[i]);
        if (minSpeed > speed) minSpeed = speed;
        if (maxSpeed < speed) maxSpeed = speed;
        speeds[i] = speed;
    }
    for (int i = 0; i < activeParticles; ++i) {
        float normSpeed = (speeds[i] - minSpeed) / (maxSpeed - minSpeed);
        normSpeed = std::clamp(normSpeed, 0.0f, 1.0f);
        uint8_t byteVel = static_cast<uint8_t>(normSpeed * 255.0f);
        colorR[i] = (255 - byteVel) * color1[0] + byteVel * color2[0];
        colorG[i] = (255 - byteVel) * color1[1] + byteVel * color2[1];
        colorB[i] = (255 - byteVel) * color1[2] + byteVel * color2[2];
    }
}

float World::calcDensity(int particleIndex) {
    float densityVal = 0.0f;
    const auto& otherIndexes = querysize[particleIndex];
    thread_local std::vector<float> distances;
    thread_local std::vector<float> influences;
    distances.resize(otherIndexes.size());
    influences.resize(otherIndexes.size());
    for (size_t idx = 0; idx < otherIndexes.size(); ++idx) {
        int j = otherIndexes[idx];
        float dx = predPosX[j] - predPosX[particleIndex];
        float dy = predPosY[j] - predPosY[particleIndex];
        distances[idx] = std::sqrt(dx*dx + dy*dy);
    }
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(d_dist_buffer, distances.data(),
                         otherIndexes.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    sph::calcSmoothingKernelGPU(d_dist_buffer, d_out_buffer,
                                smoothingRadius,
                                static_cast<int>(otherIndexes.size()));
    CUDA_CHECK(cudaMemcpy(influences.data(), d_out_buffer,
                         otherIndexes.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
#else
    sph::calcSmoothingKernelCPU(distances.data(),
                                influences.data(),
                                smoothingRadius,
                                static_cast<int>(otherIndexes.size()));
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
        float offsetX = posX[otherIndex] - posX[particleIndex];
        float offsetY = posY[otherIndex] - posY[particleIndex];
        float dist = std::sqrt(offsetX*offsetX + offsetY*offsetY);
        if (dist > smoothingRadius) continue;
        float dirX = 0.0f;
        float dirY = 0.0f;
        if (dist <= FLT_EPSILON) {
            dirX = dist01(rng) - 0.5f;
            dirY = dist01(rng) - 0.5f;
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
    float offsetX = p.pos[0] - posX[particleIndex];
    float offsetY = p.pos[1] - posY[particleIndex];
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
    float velXv = velX[particleIndex];
    float velYv = velY[particleIndex];
    outForce[0] = (dirToForcePosX * p.strength - velXv) * centreT;
    outForce[1] = (dirToForcePosY * p.strength - velYv) * centreT;
}

} // namespace sph

