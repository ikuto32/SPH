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

    float width() const { return world.getWorldWidth(); }
    float height() const { return world.getWorldHeight(); }

private:
    sph::World world;
};

PYBIND11_MODULE(_sph, m) {
    py::class_<PyWorld>(m, "PyWorld")
        .def(py::init<>())
        .def("step", &PyWorld::step, py::arg("dt"))
        .def("get_positions", &PyWorld::get_positions)
        .def_property_readonly("width", &PyWorld::width)
        .def_property_readonly("height", &PyWorld::height);
}
