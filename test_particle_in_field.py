"""Tests the EMWave and ChargedParticle PhysicsModules"""

import pytest
from turbopy import Simulation
from particle_in_field import EMWave, ChargedParticle

wave_data = {"amplitude": 10, "omega": 18}
particle_data = {"position": 25, "pusher": "ForwardEuler"}


@pytest.fixture(name="sim_example")
def fixture():
    """"Creates a Simulation object to be used as the owner of an
    EMWave object and a ChargedParticle PhysicsModule for testing"""
    input_data = {"Grid": {"N": 10, "min": 20, "max": 30},
                  "PhysicsModules": {"EMWave": wave_data,
                                     "ChargedParticle": particle_data},
                  "Tools": {"ForwardEuler": {}},
                  "Clock": {"start_time": 0, "end_time": 1, "num_steps": 2}}
    sim = Simulation(input_data)
    sim.prepare_simulation()
    return sim


def test_emwave(sim_example):
    """Tests the EMWave PhysicsModule"""
    em_example = EMWave(sim_example, wave_data)
    assert isinstance(em_example, EMWave)
    assert em_example.c == 2.998e8
    assert em_example.E0 == wave_data["amplitude"]
    assert em_example.omega == wave_data["omega"]
    assert em_example.k == wave_data["omega"] / 2.998e8


def test_chargedparticle(sim_example):
    """Tests the ChargedParticle PhysicsModule"""
    charged_example = ChargedParticle(sim_example, particle_data)
    assert isinstance(charged_example, ChargedParticle)
    assert charged_example.E is None
    assert charged_example.x == particle_data["position"]


def test_id_of_shared_attribute_should_not_change_after_update(sim_example):
    """Tests if a shared attribute references the same object after update"""
    wave = EMWave(sim_example, wave_data)
    old_id = id(wave.E)
    wave.update()
    assert old_id == id(wave.E)
