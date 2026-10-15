"""Tests for the intramolecular distance-window constraint."""

from __future__ import annotations

import math

import pytest
import torch

from raymc.config import ConstraintConfig, IntramolecularWindowConfig
from raymc.constraints.intramolecular import IntramolecularConstraint
from raymc.constraints.penalty import PENALTY_AT_WINDOW
from raymc.initialize.pack import pack_on_lattice
from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem
from raymc.system.atomic_system import _attach_rigid_groups_unchecked
from raymc.utils import IntraWindowTable, MinDistTable

# The windows the flexible-water runs use: RMCA-lineage values for a
# flexible water molecule (O-D 0.9-1.1 A, D-D 1.5-1.65 A).
WATER_WINDOWS = IntraWindowTable({(8, 1): (0.9, 1.1), (1, 1): (1.5, 1.65)})


def _periodic_water(positions: list[list[float]], box: float = 20.0) -> AtomicSystem:
    """One water molecule at explicit coordinates, as one rigid group.

    Attaches the group id WITHOUT `set_rigid_groups`' connectivity check,
    which infers bonds from geometry (`BOND_TOLERANCE` x summed covalent
    radii) and so rejects exactly the stretched geometries these tests
    need. That matches how a real run reaches such a configuration: every
    move builds its candidate through `_attach_rigid_groups_unchecked`,
    so the check runs once at initialization and never again mid-run.
    """
    system = AtomicSystem(
        positions=torch.tensor(positions, dtype=torch.float64),
        atomic_numbers=torch.tensor([8, 1, 1]),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * box,
        pbc=torch.tensor([[True, True, True]]),
    )
    return _attach_rigid_groups_unchecked(system, torch.zeros(3, dtype=torch.long))


def _template_water() -> AtomicSystem:
    """Water at exact template geometry: O-H 0.9584, H-H 1.5154."""
    return _periodic_water(
        [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9281, 0.0]]
    )


def _stretched_water(oh: float) -> AtomicSystem:
    """Both O-H bonds set to *oh*, H-O-H held at the template 104.5 deg."""
    angle = math.radians(104.5)
    return _periodic_water(
        [
            [0.0, 0.0, 0.0],
            [oh, 0.0, 0.0],
            [oh * math.cos(angle), oh * math.sin(angle), 0.0],
        ]
    )


def test_template_geometry_scores_zero():
    """The geometry every run starts from must cost nothing."""
    constraint = IntramolecularConstraint(WATER_WINDOWS)
    assert constraint.compute(_template_water()).item() == 0.0


def test_a_collapsed_bond_is_penalized():
    constraint = IntramolecularConstraint(WATER_WINDOWS)
    assert constraint.compute(_stretched_water(0.5)).item() > 0.0


def test_a_stretched_bond_is_penalized():
    constraint = IntramolecularConstraint(WATER_WINDOWS)
    assert constraint.compute(_stretched_water(2.0)).item() > 0.0


def test_penalty_is_exactly_penalty_at_window_one_window_outside():
    """The defining property of the smooth ramp, checked through this
    constraint's own violation bookkeeping rather than the penalty's."""
    windows = IntraWindowTable({(8, 1): (0.9, 1.1)})
    constraint = IntramolecularConstraint(
        windows, softening_window=0.1, penalty_at_window=PENALTY_AT_WINDOW
    )
    # Two O-H bonds, each exactly one softening window past the maximum.
    score = constraint.compute(_stretched_water(1.2))
    assert score.item() == pytest.approx(2 * PENALTY_AT_WINDOW)


def test_only_same_group_pairs_are_scored(water_template):
    """The complement of MinDistConstraint: a pair in DIFFERENT groups is
    this constraint's business under no circumstances, however close."""
    system = pack_on_lattice(
        water_template,
        cell=torch.eye(3) * 20.0,
        n_particles=2,
        min_dist=MinDistTable(default=8.0),
        seed=0,
    )
    # A window that every intermolecular O-H in the box violates wildly
    # (they are ~10 A apart), but which every intramolecular one satisfies.
    constraint = IntramolecularConstraint(IntraWindowTable({(8, 1): (0.9, 1.1)}))
    assert constraint.compute(system).item() == 0.0


def test_an_unlisted_element_pair_is_unconstrained():
    """O-H alone leaves H-H free -- an unlisted pair has no window."""
    constraint = IntramolecularConstraint(IntraWindowTable({(8, 1): (0.9, 1.1)}))
    # H-H here is 1.90 A, far outside the H-H window the other tests use.
    system = _periodic_water(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    assert constraint.compute(system).item() == 0.0


def test_an_empty_table_constrains_nothing():
    constraint = IntramolecularConstraint(IntraWindowTable())
    assert constraint.compute(_stretched_water(5.0)).item() == 0.0


def test_check_raises_on_a_violating_system():
    constraint = IntramolecularConstraint(WATER_WINDOWS)
    constraint.check(_template_water())  # does not raise
    with pytest.raises(ValueError, match="intramolecular"):
        constraint.check(_stretched_water(2.0))


def test_incremental_matches_full_recompute_for_one_moved_atom():
    """The property the whole incremental path rests on, for the single-atom
    delta `translate_atom` produces."""
    constraint = IntramolecularConstraint(WATER_WINDOWS)
    system = _template_water()

    for displacement in (0.05, 0.15, 0.3):
        positions = system.positions.clone()
        positions[1, 0] += displacement  # stretch one O-H
        moved = AtomicSystem(
            positions=positions,
            atomic_numbers=system.atomic_numbers,
            cell=system.cell,
            pbc=system.pbc,
        )
        _attach_rigid_groups_unchecked(moved, torch.zeros(3, dtype=torch.long))

        incremental = constraint.compute_increm_change(
            moved, MoveDelta(moved_indices=torch.tensor([1]))
        )
        assert incremental.item() == pytest.approx(constraint.compute(moved).item())


def test_incremental_scores_the_moved_atoms_own_group_only():
    """A moved H in a 2-molecule box must not pick up the other molecule."""
    system = _template_water()
    positions = torch.cat([system.positions, system.positions + 8.0], dim=0)
    two = AtomicSystem(
        positions=positions,
        atomic_numbers=torch.tensor([8, 1, 1, 8, 1, 1]),
        cell=system.cell,
        pbc=system.pbc,
    )
    _attach_rigid_groups_unchecked(two, torch.tensor([0, 0, 0, 1, 1, 1]))

    constraint = IntramolecularConstraint(WATER_WINDOWS)
    score = constraint.compute_increm_change(
        two, MoveDelta(moved_indices=torch.tensor([1]))
    )
    assert score.item() == 0.0


def test_a_group_that_dissolves_is_caught():
    """The failure this constraint exists to prevent: one atom walking away
    from its molecule while the inter-group constraints look elsewhere."""
    constraint = IntramolecularConstraint(WATER_WINDOWS)
    runaway = _periodic_water(
        [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-3.0, 4.0, 0.0]]
    )
    assert not math.isfinite(constraint.compute(runaway).item()) or (
        constraint.compute(runaway).item() > 1e3
    )


def test_from_config_builds_the_configured_windows():
    config = ConstraintConfig(
        intramolecular=[
            IntramolecularWindowConfig(elements=("O", "H"), min=0.9, max=1.1),
            IntramolecularWindowConfig(elements=("H", "H"), min=1.5, max=1.65),
        ],
        softening_window=0.05,
        penalty_at_window=500.0,
    )
    constraint = IntramolecularConstraint.from_config(config)
    assert constraint.softening_window == 0.05
    assert constraint.penalty_at_window == 500.0
    assert constraint.windows.window_for(8, 1) == (0.9, 1.1)
    assert constraint.windows.window_for(1, 1) == (1.5, 1.65)
    assert constraint.compute(_template_water()).item() == 0.0


def test_config_rejects_an_empty_window():
    with pytest.raises(ValueError, match="must exceed min"):
        IntramolecularWindowConfig(elements=("O", "H"), min=1.1, max=0.9)


def test_table_rejects_an_empty_window():
    with pytest.raises(ValueError, match="must exceed min"):
        IntraWindowTable({(8, 1): (1.1, 0.9)})


def test_is_unconstrained_and_constrained_elements():
    assert IntraWindowTable().is_unconstrained()
    assert IntraWindowTable({(8, 1): (0.0, math.inf)}).is_unconstrained()
    table = IntraWindowTable({(8, 1): (0.9, 1.1)})
    assert not table.is_unconstrained()
    assert table.constrained_elements().tolist() == [1, 8]
