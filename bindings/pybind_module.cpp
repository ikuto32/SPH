#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "sph/core/world.h"

namespace py = pybind11;

class PyWorld {
public:
    PyWorld(
        float width = 20.0f,
        float height = 10.0f,
        float smoothing_radius = 0.8f,
        float target_density = 32.0f,
        float pressure_multiplier = 100.0f,
        float drag = 0.9999f,
        float gravity = 9.8f,
        float collision_damping = 1.0f,
        float delta = 0.0f,
        bool  use_gpu = false,
        int   num_particles = sph::World::defaultNumParticles)
        : world(sph::WorldConfig{width, height, smoothing_radius, target_density,
                               pressure_multiplier, delta, drag, gravity,
                               collision_damping, num_particles}),
          use_gpu(use_gpu) {}

    void step(float dt) {
#ifdef SPH_ENABLE_HASH2D
        if (use_gpu) {
            world.stepGPU(dt);
            return;
        }
#endif
        world.update(dt);
    }

    py::array_t<float> get_positions() const {
        size_t N = world.getNumParticles();
        const auto& pos = world.getPositions();
        return py::array_t<float>({static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(2)},
                                  reinterpret_cast<const float*>(pos.data()));
    }

    py::array_t<float> get_velocities() const {
        size_t N = world.getNumParticles();
        const auto& vel = world.getVelocities();
        return py::array_t<float>({static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(2)},
                                  reinterpret_cast<const float*>(vel.data()));
    }

    float width() const { return world.getWorldWidth(); }
    float height() const { return world.getWorldHeight(); }
    float smoothing_radius() const { return world.getSmoothingRadius(); }
    float gravity() const { return world.getGravity(); }
    float drag() const { return world.getDrag(); }

    void set_interaction_force(float x, float y, float radius, float strength) {
        world.setInteractionForce(x, y, radius, strength);
    }

    void delete_interaction_force() { world.deleteInteractionForce(); }

    std::vector<int> query_neighbors(float x, float y) const {
        return world.queryNeighbors(x, y);
    }

    std::vector<int> query_spatial_hash(float x, float y) const {
        return world.querySpatialHash(x, y);
    }

private:
    sph::World world;
    bool use_gpu = false;
};

PYBIND11_MODULE(_sph, m) {
    py::class_<PyWorld>(m, "PyWorld")
        .def(
            py::init<
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                bool,
                int>(),
            py::arg("width") = 20.0f,
            py::arg("height") = 10.0f,
            py::arg("smoothing_radius") = 0.8f,
            py::arg("target_density") = 32.0f,
            py::arg("pressure_multiplier") = 100.0f,
            py::arg("drag") = 0.9999f,
            py::arg("gravity") = 9.8f,
            py::arg("collision_damping") = 1.0f,
            py::arg("delta") = 0.0f,
            py::arg("use_gpu") = false,
            py::arg("num_particles") = sph::World::defaultNumParticles)
        .def("step", &PyWorld::step, py::arg("dt"))
        .def("get_positions", &PyWorld::get_positions)
        .def("get_velocities", &PyWorld::get_velocities)
        .def("width", &PyWorld::width)
        .def("height", &PyWorld::height)
        .def("smoothing_radius", &PyWorld::smoothing_radius)
        .def("gravity", &PyWorld::gravity)
        .def("drag", &PyWorld::drag)
        .def(
            "set_interaction_force",
            &PyWorld::set_interaction_force,
            py::arg("x"),
            py::arg("y"),
            py::arg("radius"),
            py::arg("strength"))
        .def(
            "query_neighbors",
            &PyWorld::query_neighbors,
            py::arg("x"),
            py::arg("y"))
        .def(
            "query_spatial_hash",
            &PyWorld::query_spatial_hash,
            py::arg("x"),
            py::arg("y"))
        .def("delete_interaction_force", &PyWorld::delete_interaction_force);
}
