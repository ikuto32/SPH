#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "sph/core/world.h"

namespace py = pybind11;

class PyWorld {
public:
    PyWorld() = default;

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

    void set_interaction_force(float x, float y, float radius, float strength) {
        world.setInteractionForce(x, y, radius, strength);
    }

    void delete_interaction_force() { world.deleteInteractionForce(); }

private:
    sph::World world;
};

PYBIND11_MODULE(_sph, m) {
    py::class_<PyWorld>(m, "PyWorld")
        .def(py::init<>())
        .def("step", &PyWorld::step, py::arg("dt"))
        .def("get_positions", &PyWorld::get_positions)
        .def("get_velocities", &PyWorld::get_velocities)
        .def("width", &PyWorld::width)
        .def("height", &PyWorld::height)
        .def(
            "set_interaction_force",
            &PyWorld::set_interaction_force,
            py::arg("x"),
            py::arg("y"),
            py::arg("radius"),
            py::arg("strength"))
        .def("delete_interaction_force", &PyWorld::delete_interaction_force);
}
