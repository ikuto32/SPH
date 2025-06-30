#include <iostream>
#include "sph/core/world.h"

int main() {
    const int counts[] = {100, 500, sph::World::numParticle};
    double totals[3] = {0.0};

    for (int i = 0; i < 3; ++i) {
        int n = counts[i];
        sph::World w;
        w.setActiveParticleCount(n);
        sph::ProfileInfo info;
        w.updateWithStats(1.0f/60.0f, info);
        double total = info.predictedPosMs + info.gridRegisterMs + info.queryMs +
                       info.densityMs + info.pressureMs + info.interactionMs +
                       info.updatePosMs + info.fixPosMs + info.colorMs;
        totals[i] = total;

        auto print = [&](const char* name, double ms) {
            std::cout << name << " " << ms << " ms (";
            if (total > 0) {
                std::cout << (ms / total * 100.0);
            } else {
                std::cout << 0.0;
            }
            std::cout << "%)\n";
        };

        std::cout << "==== Particles: " << n << " ====" << std::endl;
        std::cout << "Total time: " << total << " ms\n";
        print("predictedPos", info.predictedPosMs);
        print("gridRegister", info.gridRegisterMs);
        print("query", info.queryMs);
        print("density", info.densityMs);
        print("pressure", info.pressureMs);
        print("interaction", info.interactionMs);
        print("updatePosition", info.updatePosMs);
        print("fixPosition", info.fixPosMs);
        print("color", info.colorMs);
        std::cout << "Memory transferred: " << info.memTransferBytes << " bytes\n\n";
    }

    std::cout << "=== Total time ratios (relative to " << counts[0] << ") ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "N=" << counts[i] << ": " << (totals[i] / totals[0]) << std::endl;
    }
    return 0;
}
