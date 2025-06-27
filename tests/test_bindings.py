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
