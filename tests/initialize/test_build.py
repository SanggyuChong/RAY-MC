from __future__ import annotations

import warnings

import pytest

from raymc.config import EquilibrateConfig, InitConfig, MinDistPairConfig
from raymc.initialize.build import build_initial_system
from raymc.system import check_no_atomic_overlap
from raymc.utils import MinDistTable


def test_build_dispatches_pack_on_lattice():
    config = InitConfig(
        strategy="pack_on_lattice",
        molecule="water",
        n_particles=126,
        cell=(20.0, 20.0, 20.0),
        seed=1,
    )
    system = build_initial_system(config)
    assert system.num_nodes % 3 == 0
    assert system.num_nodes > 0


def test_build_dispatches_load_structure():
    config = InitConfig(
        strategy="load_structure",
        structure_file="src/raymc/database/molecules/water.xyz",
    )
    system = build_initial_system(config)
    assert system.num_nodes == 3


def test_build_accepts_a_bare_atom_species():
    config = InitConfig(
        strategy="pack_on_lattice",
        molecule="Na",
        n_particles=17,
        cell=(15.0, 15.0, 15.0),
        seed=1,
    )
    system = build_initial_system(config, min_dist=MinDistTable(default=3.0))
    assert system.atomic_numbers.unique().tolist() == [11]


def test_build_runs_the_equilibrate_stage_when_configured():
    """init.equilibrate must run and the result must still pass the
    overlap check -- and it must run at the SAME min-dist as the rest of
    the pipeline, so RMC never starts in violation."""
    config = InitConfig(
        strategy="pack_on_lattice",
        molecule="Ar",
        n_particles=64,
        cell=(16.0, 16.0, 16.0),
        seed=1,
        equilibrate=EquilibrateConfig(n_steps=2_000, max_displacement=0.5),
    )
    min_dist = MinDistTable(default=2.5)
    system = build_initial_system(config, min_dist=min_dist)
    check_no_atomic_overlap(system, min_dist)  # should not raise


def test_build_equilibrate_min_dist_override_larger_stays_in_constraint():
    """A pre-equilibration min_dist override LARGER than the RMC
    constraint's own min_dist is a stricter exclusion, so the equilibrated
    result must still satisfy the (smaller) RMC constraint at handoff."""
    config = InitConfig(
        strategy="pack_on_lattice",
        molecule="Ar",
        n_particles=64,
        cell=(16.0, 16.0, 16.0),
        seed=1,
        equilibrate=EquilibrateConfig(
            n_steps=2_000, max_displacement=0.5, min_dist=2.7
        ),
    )
    min_dist = MinDistTable(default=2.5)
    system = build_initial_system(config, min_dist=min_dist)
    check_no_atomic_overlap(system, min_dist)  # should not raise


def test_build_allows_an_equilibrate_min_dist_looser_than_the_rmc_constraint():
    """Equilibrating at a looser exclusion than RMC enforces is allowed.

    The resulting structure starts inside the RMC constraint and pays a
    large finite smooth penalty it can relax out of -- see
    EquilibrateConfig.min_dist. build_initial_system used to refuse this
    outright; it must now just build.
    """
    config = InitConfig(
        strategy="pack_on_lattice",
        molecule="Ar",
        n_particles=64,
        cell=(16.0, 16.0, 16.0),
        seed=1,
        equilibrate=EquilibrateConfig(
            n_steps=2_000, max_displacement=0.5, min_dist=2.0
        ),
    )
    min_dist = MinDistTable(default=2.5)
    system = build_initial_system(config, min_dist=min_dist)  # must not raise
    assert system.num_nodes == 64


def test_build_result_passes_the_default_overlap_check():
    config = InitConfig(
        strategy="pack_on_lattice",
        molecule="water",
        n_particles=10,
        cell=(15.0, 15.0, 15.0),
        seed=1,
    )
    min_dist = MinDistTable(default=3.0)
    system = build_initial_system(config, min_dist=min_dist)
    check_no_atomic_overlap(system, min_dist)  # should not raise


def test_packing_is_checked_against_the_equilibration_table_not_the_rmc_one():
    """The packing is held to whatever the NEXT stage enforces.

    A lattice cannot satisfy a physical O-O minimum before being melted,
    so checking it against the RMC constraint made a strict RMC floor
    impossible to configure at all. The equilibration table is what the
    melt will actually enforce, so that is what packing is checked
    against -- and only when there is no melt does the RMC table apply.
    """
    config = InitConfig(
        strategy="pack_on_lattice",
        molecule="water",
        n_particles=500,
        cell=(25.0, 25.0, 25.0),
        seed=42,
        equilibrate=EquilibrateConfig(
            n_steps=2_000, max_displacement=0.8, default=1.0,
            pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.25)],
        ),
    )
    # RMC asks for floors the raw lattice cannot possibly meet
    strict = MinDistTable({(8, 8): 2.4, (8, 1): 1.9, (1, 1): 2.0}, default=1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        system = build_initial_system(config, min_dist=strict)

    assert system.num_nodes == 1500
    messages = [str(w.message) for w in caught]
    # packing was measured against 2.25/1.0, which the lattice clears
    assert not any("packed lattice" in m for m in messages)
    # and the melt's result is still reported as starting inside the RMC floors
    assert any("starts inside" in m for m in messages)
