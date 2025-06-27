#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
        float delta = 0.0f)
        : world(sph::WorldConfig{width, height, smoothing_radius, target_density,
                               pressure_multiplier, delta, drag, gravity,
                               collision_damping}) {}

    void step(float dt) { world.update(dt); }

    py::array_t<float> get_positions() const {
        constexpr int N = sph::World::numParticle;
        const float (*pos)[2] = world.getPositions();
        return py::array_t<float>({N, 2}, &pos[0][0]);
    }

    py::array_t<float> get_velocities() const {
        constexpr int N = sph::World::numParticle;
        const float (*vel)[2] = world.getVelocities();
        return py::array_t<float>({N, 2}, &vel[0][0]);
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

private:
    sph::World world;
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
                float>(),
            py::arg("width") = 20.0f,
            py::arg("height") = 10.0f,
            py::arg("smoothing_radius") = 0.8f,
            py::arg("target_density") = 32.0f,
            py::arg("pressure_multiplier") = 100.0f,
            py::arg("drag") = 0.9999f,
            py::arg("gravity") = 9.8f,
            py::arg("collision_damping") = 1.0f,
            py::arg("delta") = 0.0f)
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
        .def("delete_interaction_force", &PyWorld::delete_interaction_force);
}
