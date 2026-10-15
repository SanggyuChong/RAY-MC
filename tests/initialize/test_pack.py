from __future__ import annotations

import pytest
import torch

from raymc.initialize.pack import (
    bragg_peak_max_sk,
    equilibrate_hard_spheres,
    pack_on_lattice,
)
from raymc.initialize.templates import GroupTemplate, load_group_template
from raymc.system import check_no_atomic_overlap, rigid_groups
from raymc.utils import MinDistTable

TARGET_DENSITY = 0.03142  # ~ LDA ice, molecules/A^3 -- reference for the n_particles used below.


# -- pack_on_lattice --------------------------------------------------------


def test_lattice_produces_correct_atom_and_group_counts(water_template):
    cell = torch.eye(3) * 20.0
    system = pack_on_lattice(water_template, cell, n_particles=126, seed=1)
    n_molecules = system.num_nodes // water_template.num_atoms
    assert system.num_nodes == n_molecules * water_template.num_atoms
    assert rigid_groups(system).max().item() == n_molecules - 1


def test_lattice_result_has_no_atomic_overlap(water_template):
    cell = torch.eye(3) * 20.0
    system = pack_on_lattice(water_template, cell, n_particles=126, seed=1)
    check_no_atomic_overlap(system)  # should not raise


def test_lattice_reaches_a_high_site_occupancy(water_template):
    """pack_on_lattice must be able to fill close to every available
    lattice site, not just a sparse subset of them."""
    cell = torch.eye(3) * 20.0
    n_requested = 214
    system = pack_on_lattice(water_template, cell, n_particles=n_requested, seed=1)
    n_molecules = system.num_nodes // water_template.num_atoms
    assert n_molecules == n_requested


def test_lattice_is_reproducible_with_a_seed(water_template):
    cell = torch.eye(3) * 20.0
    a = pack_on_lattice(water_template, cell, n_particles=126, seed=42)
    b = pack_on_lattice(water_template, cell, n_particles=126, seed=42)
    assert torch.allclose(a.positions, b.positions)


def test_lattice_rejects_a_non_diagonal_cell(water_template):
    cell = torch.eye(3) * 20.0
    cell[0, 1] = 1.0
    with pytest.raises(ValueError, match="orthorhombic"):
        pack_on_lattice(water_template, cell, n_particles=100, seed=1)


def test_lattice_handles_a_non_cubic_cell_without_overlap(water_template):
    """Regression test: using one site count for every axis regardless of
    that axis's length (e.g. same n_side on a 24x24x12 A box) squeezes the
    short axis far too densely and produces real atomic overlaps. Spacing
    must be uniform across axes, not the site *count*."""
    cell = torch.tensor([[24.0, 0.0, 0.0], [0.0, 24.0, 0.0], [0.0, 0.0, 12.0]])
    system = pack_on_lattice(water_template, cell, n_particles=217, seed=1)
    check_no_atomic_overlap(system)  # should not raise
    n_molecules = system.num_nodes // water_template.num_atoms
    achieved_density = n_molecules / (24.0 * 24.0 * 12.0)
    assert achieved_density == pytest.approx(TARGET_DENSITY, rel=0.05)


def test_lattice_spacing_is_not_inflated_by_floating_point_rounding():
    """Regression test: when box_length / spacing is exactly an integer
    up to floating-point noise (e.g. a perfect-cube atom count in a cubic
    cell), ceil() must not silently round it up to the next integer --
    that collapses the realized lattice spacing well below the intended
    value (confirmed bug: 1000 atoms in a 36.13 A box intended a spacing
    of 3.613 A with 10 sites/axis, but unrounded ceil() gave 11
    sites/axis and a realized spacing of only 3.28 A)."""
    from raymc.initialize.templates import GroupTemplate, load_group_template
    from raymc.utils import minimum_image_delta

    ar = GroupTemplate.atom("Ar")
    box_length = 36.13167094043123  # chosen so box_length/spacing == 10.000000000000002
    cell = torch.eye(3) * box_length
    system = pack_on_lattice(ar, cell, n_particles=1000, seed=42)
    assert system.num_nodes == 1000

    positions = system.positions.to(torch.float64)
    pbc = torch.tensor([True, True, True])
    delta = minimum_image_delta(positions[0], positions[1:], cell.to(torch.float64), pbc)
    min_dist = torch.linalg.norm(delta, dim=-1).min().item()
    expected_spacing = box_length / 10
    assert min_dist == pytest.approx(expected_spacing, rel=1e-4)


# -- equilibrate_hard_spheres / bragg_peak_max_sk --------------------------


def test_bragg_peak_max_sk_is_high_for_a_perfect_lattice():
    """A perfect simple-cubic lattice should score close to N at its own
    Bragg points."""
    ar = GroupTemplate.atom("Ar")
    n_per_axis = 4
    spacing = 4.0
    box_length = n_per_axis * spacing
    cell = torch.eye(3) * box_length
    system = pack_on_lattice(ar, cell, n_particles=n_per_axis**3, seed=1)
    assert system.num_nodes == n_per_axis**3
    sk = bragg_peak_max_sk(system, n_per_axis=n_per_axis)
    assert sk > 0.9 * system.num_nodes


def test_equilibrate_hard_spheres_melts_the_lattice_and_lowers_sk():
    """After equilibration, the max Bragg S(k) must drop sharply from the
    perfect-lattice value -- the whole point of the melt stage."""
    ar = GroupTemplate.atom("Ar")
    n_per_axis = 4
    spacing = 4.0
    box_length = n_per_axis * spacing
    cell = torch.eye(3) * box_length
    min_dist = MinDistTable(default=2.5)
    system = pack_on_lattice(
        ar, cell, n_particles=n_per_axis**3, min_dist=min_dist, seed=1
    )
    sk_before = bragg_peak_max_sk(system, n_per_axis=n_per_axis)

    equilibrated = equilibrate_hard_spheres(
        system, min_dist=min_dist, n_steps=20_000, max_displacement=0.6, seed=3
    )
    check_no_atomic_overlap(equilibrated, min_dist)  # never violates during the melt
    sk_after = bragg_peak_max_sk(equilibrated, n_per_axis=n_per_axis)

    assert sk_after < sk_before
    assert sk_after < 0.5 * system.num_nodes  # substantially melted


def test_equilibrate_hard_spheres_never_produces_an_overlap_mid_run():
    ar = GroupTemplate.atom("Ar")
    n_per_axis = 3
    spacing = 4.0
    cell = torch.eye(3) * (n_per_axis * spacing)
    min_dist = MinDistTable(default=2.5)
    system = pack_on_lattice(
        ar, cell, n_particles=n_per_axis**3, min_dist=min_dist, seed=2
    )
    equilibrated = equilibrate_hard_spheres(
        system, min_dist=min_dist, n_steps=2_000, max_displacement=0.5, seed=5
    )
    check_no_atomic_overlap(equilibrated, min_dist)


def test_pack_on_lattice_accepts_a_lone_atom_template():
    from raymc.initialize.templates import GroupTemplate, load_group_template

    na = GroupTemplate.atom("Na")
    cell = torch.eye(3) * 15.0
    lattice_system = pack_on_lattice(na, cell, n_particles=17, seed=1)
    assert lattice_system.num_nodes > 0


def test_pack_on_lattice_warns_rather_than_raises_on_a_close_contact():
    """A raw lattice's worst-case contact is geometry, not a defect.

    Water's outermost atom sits 0.783 A from its centroid, so two adjacent
    sites at 3.15 A spacing reach ~1.58 A in the worst orientation --
    and which pair trips first moves from seed to seed. Raising there
    vetoed perfectly good runs before the melt that exists to fix it had
    a chance to run.
    """
    template = load_group_template("water")
    with pytest.warns(UserWarning, match="packed lattice"):
        system = pack_on_lattice(
            template,
            cell=torch.eye(3, dtype=torch.float64) * 25.0,
            n_particles=500,
            pbc=torch.tensor([True, True, True]),
            min_dist=MinDistTable(default=2.5),  # unreachable for this lattice
            seed=42,
        )
    assert system.num_nodes == 1500


def test_equilibrate_relaxes_out_of_a_violating_start():
    """A melt must be able to START in violation and descend.

    A raw lattice frequently cannot satisfy a physical exclusion -- water's
    outermost atom is 0.783 A from its centroid, so adjacent sites reach
    ~1.58 A in the worst orientation. Refusing to begin made any H-H
    minimum above that impossible to ask for, however many steps were
    available to fix it.
    """
    from raymc.constraints.min_dist import MinDistConstraint

    template = load_group_template("water")
    # 9.5 A for 27 molecules is the SAME 3.17 A site spacing as a
    # 500-molecule box at 25 A -- i.e. real LDA density, where the
    # lattice's worst-case contact genuinely violates the strict table.
    lattice = pack_on_lattice(
        template,
        cell=torch.eye(3, dtype=torch.float64) * 9.5,
        n_particles=27,
        pbc=torch.tensor([True, True, True]),
        min_dist=MinDistTable(default=0.5),
        seed=3,
    )
    strict = MinDistTable({(8, 8): 2.4, (8, 1): 1.9, (1, 1): 2.0}, default=1.0)
    scorer = MinDistConstraint(strict)
    before = float(scorer.compute(lattice))
    assert before > 0, "fixture must start in violation for this test to mean anything"

    with pytest.warns(UserWarning, match="already violates min_dist"):
        melted = equilibrate_hard_spheres(
            lattice, min_dist=strict, n_steps=8_000, max_displacement=0.8, seed=3
        )
    assert float(scorer.compute(melted)) < before


def test_equilibrate_from_a_clean_start_never_introduces_a_violation():
    """The relax rule must reduce to the old strict rule when it can.

    "the touched pairs' penalty must not increase" from a zero-penalty
    start can only ever keep it at zero, so a clean run behaves exactly
    as the strict hard-sphere melt always did -- and takes the cheaper
    single-score-per-step path while doing it.
    """
    from raymc.constraints.min_dist import MinDistConstraint

    template = load_group_template("water")
    table = MinDistTable({(8, 8): 2.0}, default=0.9)
    lattice = pack_on_lattice(
        template,
        cell=torch.eye(3, dtype=torch.float64) * 14.0,
        n_particles=27,
        pbc=torch.tensor([True, True, True]),
        min_dist=table,
        seed=5,
    )
    scorer = MinDistConstraint(table)
    assert float(scorer.compute(lattice)) == 0.0

    melted = equilibrate_hard_spheres(
        lattice, min_dist=table, n_steps=4_000, max_displacement=0.6, seed=5
    )
    assert float(scorer.compute(melted)) == 0.0
