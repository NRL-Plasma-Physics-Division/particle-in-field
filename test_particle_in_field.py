import pytest
from turbopy import Simulation
from particle_in_field import EMWave, ChargedParticle, ForwardEuler

owner_data = {"N": 10, "min": 20, "max": 30, "type": "test_pusher"}
wave_data = {"amplitude": 10, "omega": 18}
particle_data = {"position": 26.12, "pusher": "test_pusher"}


@pytest.fixture(name="EM_example")
def fixture_EM():
    Owner = Simulation({"Grid": owner_data})
    Owner.read_grid_from_input()
    return EMWave(Owner, wave_data)


def test_EMWave(EM_example):
    assert isinstance(EM_example, EMWave)
    assert EM_example.c == 2.998e8
    assert EM_example.E0 == wave_data["amplitude"]
    assert EM_example.omega == wave_data["omega"]
    assert EM_example.k == wave_data["omega"] / 2.998e8


@pytest.fixture(name="Charged_example")
def fixture_Charged():
    Owner = Simulation({"Grid": owner_data})
    Owner.compute_tools.append(ForwardEuler(Owner, owner_data))
    Owner.read_grid_from_input()
    return ChargedParticle(Owner, particle_data)


def test_ChargedParticle(Charged_example):
    assert isinstance(Charged_example, ChargedParticle)
    assert Charged_example.E is None
    assert Charged_example.x == particle_data["position"]

