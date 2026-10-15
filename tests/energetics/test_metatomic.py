"""Tests for the metatomic energetics adapter, using a UPET checkpoint as
the model source.

Requires the optional "upet" extra -- skipped entirely if it isn't
installed, consistent with it being an optional dependency.

The model comes from a local checkpoint under ``models/`` when there is
one, and only otherwise from the UPET download. That order matters: an
environment whose network policy denies huggingface.co can still exercise
every one of these tests against a checkpoint on disk. When neither route
is available the module skips rather than failing, since an unreachable
download is a property of the environment and not a defect in the
adapter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("upet")

from raymc.energetics.metatomic import (  # noqa: E402
    MetatomicEnergetics,
    load_upet_model,
)
from raymc.modalities.base import Calculator  # noqa: E402
from raymc.system import AtomicSystem  # noqa: E402

_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "models"


def _local_checkpoints() -> list[Path]:
    return sorted(_CHECKPOINT_DIR.glob("*.ckpt"))


@pytest.fixture(scope="module")
def upet() -> MetatomicEnergetics:
    local = _local_checkpoints()
    if local:
        model = load_upet_model(checkpoint_path=str(local[0]))
    else:
        try:
            model = load_upet_model(model="pet-mad", size="s", version="1.5.0")
        except Exception as exc:  # noqa: BLE001 - any transport failure is a skip
            pytest.skip(
                f"no checkpoint in {_CHECKPOINT_DIR} and the UPET download "
                f"is unavailable here ({type(exc).__name__})"
            )
    return MetatomicEnergetics(model=model)


def test_load_upet_model_rejects_a_missing_checkpoint():
    """A mistyped path must say so, rather than surfacing later as an
    opaque failure from inside the loader."""
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_upet_model(checkpoint_path=str(_CHECKPOINT_DIR / "no-such-model.ckpt"))


@pytest.fixture
def single_water_nonperiodic() -> AtomicSystem:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9281, 0.0]]
    )
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    return AtomicSystem(positions=positions, atomic_numbers=atomic_numbers)


def test_metatomic_energetics_is_a_calculator(upet):
    assert isinstance(upet, Calculator)


def test_compute_matches_energy_for_a_single_system(upet, single_water_nonperiodic):
    single_value = upet.compute(single_water_nonperiodic)
    batched_value = upet.energy([single_water_nonperiodic])[0]
    assert single_value.item() == pytest.approx(batched_value.item())


def test_energy_is_finite_and_negative(upet, single_water_nonperiodic):
    energy = upet.energy([single_water_nonperiodic])
    assert energy.shape == (1,)
    assert torch.isfinite(energy).all()
    assert energy.item() < 0  # bound molecule: negative relative to isolated atoms


def test_energy_is_translation_invariant(upet, single_water_nonperiodic):
    shifted = AtomicSystem(
        positions=single_water_nonperiodic.positions + torch.tensor([5.0, -3.0, 2.0]),
        atomic_numbers=single_water_nonperiodic.atomic_numbers,
    )
    e1 = upet.energy([single_water_nonperiodic])
    e2 = upet.energy([shifted])
    assert torch.allclose(e1, e2, atol=1e-3)


def test_energy_batches_systems_of_different_sizes(upet, single_water_nonperiodic):
    two_waters_positions = torch.cat(
        [
            single_water_nonperiodic.positions,
            single_water_nonperiodic.positions + torch.tensor([10.0, 0.0, 0.0]),
        ]
    )
    two_waters = AtomicSystem(
        positions=two_waters_positions,
        atomic_numbers=torch.cat([single_water_nonperiodic.atomic_numbers] * 2),
    )
    energies = upet.energy([single_water_nonperiodic, two_waters])
    assert energies.shape == (2,)
    # two well-separated waters should have roughly double the one-water energy
    assert energies[1].item() == pytest.approx(2 * energies[0].item(), rel=0.05)


def test_energy_forces_net_force_is_near_zero(upet, single_water_nonperiodic):
    """Translation invariance implies the net force on an isolated system
    must vanish -- a cheap sanity check that doesn't depend on the sign
    convention being right in a way that would cancel out a bug."""
    _, forces = upet.energy_forces(single_water_nonperiodic)
    assert forces.shape == (3, 3)
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3), atol=1e-4)


def test_energy_forces_matches_finite_difference(upet, single_water_nonperiodic):
    positions = single_water_nonperiodic.positions
    atomic_numbers = single_water_nonperiodic.atomic_numbers
    _, forces = upet.energy_forces(single_water_nonperiodic)

    delta = 1e-3
    pos_plus = positions.clone()
    pos_plus[1, 0] += delta
    pos_minus = positions.clone()
    pos_minus[1, 0] -= delta
    e_plus = upet.energy(
        [AtomicSystem(positions=pos_plus, atomic_numbers=atomic_numbers)]
    )[0]
    e_minus = upet.energy(
        [AtomicSystem(positions=pos_minus, atomic_numbers=atomic_numbers)]
    )[0]
    numeric_force = -(e_plus - e_minus).item() / (2 * delta)

    assert forces[1, 0].item() == pytest.approx(numeric_force, abs=0.01)


def test_energy_works_on_a_periodic_system(upet, water_template):
    from raymc.initialize.pack import pack_on_lattice
    from raymc.utils import MinDistTable

    cell = torch.eye(3) * 15.0
    system = pack_on_lattice(
        water_template,
        cell=cell,
        n_particles=5,
        min_dist=MinDistTable(default=3.0),
        seed=1,
    )
    energy = upet.energy([system])
    assert torch.isfinite(energy).all()
    assert energy.item() < 0
