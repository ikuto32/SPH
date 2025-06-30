#include <iostream>
#include <vector>
#include <cstdlib>
#include "sph/core/world.h"

int main(int argc, char* argv[]) {
    std::vector<int> counts = {100, 500, sph::World::numParticle};
#ifdef USE_CUDA
    if (argc > 1) {
        counts.push_back(std::atoi(argv[1]));
    }
#else
    // Running millions of particles on the CPU is too slow for CI, so ignore
    // any user supplied count and clamp the maximum to a small value.
    if (argc > 1) {
        std::cout << "CUDA support not enabled - ignoring custom particle count" << std::endl;
    }
    if (counts.back() > 1000) {
        counts.back() = 1000;
    }
#endif
    std::vector<double> totals(counts.size(), 0.0);

    for (size_t i = 0; i < counts.size(); ++i) {
        int n = counts[i];
        sph::World w;
        w.setActiveParticleCount(n);
        sph::ProfileInfo info;
        w.updateWithStats(1.0f/60.0f, info);
        double total = info.predictedPosMs + info.gridRegisterMs + info.queryMs +
                       info.densityMs + info.pressureMs + info.interactionMs +
                       info.updatePosMs + info.fixPosMs + info.colorMs;
        totals[i] = total;

        double fps = total > 0 ? 1000.0 / total : 0.0;

        double maxMs = info.predictedPosMs;
        const char* maxLabel = "predictedPos";
        struct Part { const char* name; double ms; } parts[] = {
            {"gridRegister", info.gridRegisterMs},
            {"query", info.queryMs},
            {"density", info.densityMs},
            {"pressure", info.pressureMs},
            {"interaction", info.interactionMs},
            {"updatePosition", info.updatePosMs},
            {"fixPosition", info.fixPosMs},
            {"color", info.colorMs},
        };
        for (const auto& p : parts) {
            if (p.ms > maxMs) {
                maxMs = p.ms;
                maxLabel = p.name;
            }
        }

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
        std::cout << "Processing speed: " << fps << " steps/s\n";
        print("predictedPos", info.predictedPosMs);
        print("gridRegister", info.gridRegisterMs);
        print("query", info.queryMs);
        print("density", info.densityMs);
        print("pressure", info.pressureMs);
        print("interaction", info.interactionMs);
        print("updatePosition", info.updatePosMs);
        print("fixPosition", info.fixPosMs);
        print("color", info.colorMs);
        std::cout << "Bottleneck: " << maxLabel << " (" << maxMs << " ms)\n";
        std::cout << "Memory transferred: " << info.memTransferBytes << " bytes\n\n";
    }

    std::cout << "=== Total time ratios (relative to " << counts[0] << ") ===" << std::endl;
    for (size_t i = 0; i < counts.size(); ++i) {
        std::cout << "N=" << counts[i] << ": " << (totals[i] / totals[0]) << std::endl;
    }
    return 0;
}
