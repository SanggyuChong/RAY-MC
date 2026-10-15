"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch

from raymc.initialize.pack import pack_on_lattice
from raymc.initialize.templates import GroupTemplate, load_molecule
from raymc.system import AtomicSystem, set_rigid_groups
from raymc.utils import MinDistTable


@pytest.fixture
def water_template() -> GroupTemplate:
    return load_molecule("water")


@pytest.fixture
def benzene_template() -> GroupTemplate:
    return load_molecule("benzene")


@pytest.fixture
def single_water(water_template: GroupTemplate) -> AtomicSystem:
    """One water molecule, non-periodic, tagged as a single rigid group."""
    system = AtomicSystem(
        positions=water_template.positions.clone(),
        atomic_numbers=water_template.atomic_numbers.clone(),
    )
    set_rigid_groups(system, torch.zeros(water_template.num_atoms, dtype=torch.long))
    return system


@pytest.fixture
def packed_water(water_template: GroupTemplate) -> AtomicSystem:
    """A small periodic box of ~10 well-separated water molecules."""
    cell = torch.eye(3) * 15.0
    min_dist = MinDistTable(default=3.0)
    return pack_on_lattice(
        water_template,
        cell=cell,
        n_particles=10,
        min_dist=min_dist,
        seed=3,
    )


def uniform_r_grid(r_min: float, r_max: float, n: int) -> torch.Tensor:
    """Evenly-spaced histogram bin centers over ``[r_min, r_max)``.

    The reference data owns the grid in production (see
    :func:`raymc.residuals.aggregator.make_partial_pdf_term`); tests that
    only care about some other behaviour use this to stand in for a
    well-behaved evenly-sampled reference file.
    """
    dr = (r_max - r_min) / n
    return r_min + dr * (torch.arange(n, dtype=torch.float64) + 0.5)
