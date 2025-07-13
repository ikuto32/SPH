import numpy as np
import _sph


def test_world_properties():
    w = _sph.PyWorld()
    assert w.width() > 0
    assert w.height() > 0
    pos = w.get_positions()
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    assert pos.shape[0] > 0


def test_step_updates_positions():
    w = _sph.PyWorld()
    before = w.get_positions().copy()
    w.step(1.0 / 60.0)
    after = w.get_positions()
    assert after.shape == before.shape
    assert np.any(before != after)


def test_interaction_force_methods():
    w = _sph.PyWorld()
    # set an interaction force and step the simulation; just ensure no errors
    w.set_interaction_force(1.0, 1.0, 2.0, 5.0)
    w.step(1.0 / 60.0)
    w.delete_interaction_force()


def test_query_neighbors_returns_indices():
    w = _sph.PyWorld()
    w.step(1.0 / 60.0)
    pos = w.get_positions()
    x, y = pos[0]
    neighbours = w.query_neighbors(float(x), float(y))
    assert isinstance(neighbours, list)
    assert 0 in neighbours


def test_custom_parameters_affect_world_size():
    w = _sph.PyWorld(width=5.0, height=3.0)
    assert w.width() == 5.0
    assert w.height() == 3.0


def test_custom_physics_parameters():
    w = _sph.PyWorld(gravity=20.0, smoothing_radius=1.2, drag=0.5)
    assert np.isclose(w.gravity(), 20.0)
    assert np.isclose(w.smoothing_radius(), 1.2)
    assert np.isclose(w.drag(), 0.5)
    # ensure stepping does not raise
    w.step(1.0 / 60.0)


def test_custom_particle_count():
    w = _sph.PyWorld(num_particles=10)
    assert w.get_positions().shape[0] == 10

