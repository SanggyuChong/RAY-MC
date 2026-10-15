"""Tests for the minimum-distance close-contact constraint."""

from __future__ import annotations

import warnings

import pytest
import torch

from raymc.config import ConstraintConfig, MinDistPairConfig
from raymc.constraints.min_dist import MinDistConstraint
from raymc.initialize.pack import pack_on_lattice
from raymc.moves.base import MoveDelta
from raymc.moves.group_based import RelocateGroup, RemoveGroup, RotateGroup, TranslateGroup
from raymc.moves.moveset import MoveSet
from raymc.system import AtomicSystem, group_atom_indices, rigid_groups
from raymc.system.atomic_system import _attach_rigid_groups_unchecked
from raymc.utils import DEFAULT_MIN_DIST, MinDistTable


def _two_far_apart_waters(water_template):
    """Two water molecules, far enough apart that nothing overlaps."""
    cell = torch.eye(3) * 20.0
    return pack_on_lattice(
        water_template,
        cell=cell,
        n_particles=2,
        min_dist=MinDistTable(default=8.0),
        seed=0,
    )


def test_valid_translate_scores_zero(water_template):
    system = _two_far_apart_waters(water_template)
    min_dist = MinDistTable(pair_distances={(8, 8): 2.4, (1, 8): 1.5, (1, 1): 1.0})
    constraint = MinDistConstraint(min_dist)
    constraint.init(system)

    move = TranslateGroup(max_displacement=0.3)
    candidate, delta = move.attempt(system, torch.Generator().manual_seed(0))

    score = constraint.compute_increm_change(candidate, delta)
    assert torch.isfinite(score)
    assert score.item() == 0.0


def test_overlapping_relocate_scores_an_overwhelming_penalty(water_template):
    system = _two_far_apart_waters(water_template)
    constraint = MinDistConstraint(MinDistTable(pair_distances={(8, 8): 2.4}))
    constraint.init(system)

    # Manually build a MoveDelta that drops one water's oxygen directly on
    # top of the other's -- a deliberate, unambiguous overlap.
    other_group_positions = system.positions[system.atomic_numbers == 8][0]
    positions = system.positions.clone()
    relocated_group_atoms = group_atom_indices(system, group_id=1)
    positions[relocated_group_atoms[0]] = other_group_positions  # oxygen on oxygen
    candidate = AtomicSystem(
        positions=positions,
        atomic_numbers=system.atomic_numbers,
        cell=system.cell,
        pbc=system.pbc,
    )
    _attach_rigid_groups_unchecked(candidate, rigid_groups(system))
    delta = MoveDelta(moved_indices=relocated_group_atoms)

    # An oxygen dropped exactly on another is 2.4 A inside the 2.4 A
    # minimum -- 24 softening windows, which the smooth penalty turns into
    # an astronomically large (but still finite) number. It is no longer
    # literally +inf, but it is far beyond anything a chi-squared
    # improvement could ever buy back; see raymc.constraints.penalty.
    score = constraint.compute_increm_change(candidate, delta)
    assert score.item() > 1e200


def test_a_gross_violation_still_overflows_to_infinity(water_template):
    """The smooth penalty must recover the old hard veto exactly in the
    limit: past ~26.6 softening windows it overflows float64 and returns a
    literal +inf, so a wildly violating structure is still uncatchably bad
    rather than merely expensive."""
    system = _two_far_apart_waters(water_template)
    # 2.4 A of overlap against a 4.0 A softening window is only 0.6 windows;
    # against the default 0.1 A width it is 24 windows (finite, see above).
    # A 5 A minimum distance on a pair sitting on top of each other is 50.
    constraint = MinDistConstraint(MinDistTable(pair_distances={(8, 8): 5.0}))

    other_o = system.positions[system.atomic_numbers == 8][0]
    positions = system.positions.clone()
    atoms1 = group_atom_indices(system, group_id=1)
    positions[atoms1[0]] = other_o
    violating = AtomicSystem(
        positions=positions,
        atomic_numbers=system.atomic_numbers,
        cell=system.cell,
        pbc=system.pbc,
    )
    _attach_rigid_groups_unchecked(violating, rigid_groups(system))

    score = constraint.compute_increm_change(violating, MoveDelta(moved_indices=atoms1))
    assert score.item() == float("inf")


def test_penalty_is_zero_at_the_threshold_and_1000_one_window_inside(water_template):
    """The defining property of the softening window: exactly 0 for a pair
    sitting right at its minimum distance, exactly 1000 one
    softening_window inside it."""
    system = _two_far_apart_waters(water_template)
    constraint = MinDistConstraint(
        MinDistTable(pair_distances={(8, 8): 3.0, (1, 8): 1.0, (1, 1): 1.0}),
        softening_window=0.1,
    )
    oxygens = (system.atomic_numbers == 8).nonzero(as_tuple=True)[0]
    atoms1 = group_atom_indices(system, group_id=1)

    def _score_at(separation):
        positions = system.positions.clone()
        # Move the whole second molecule so its oxygen lands `separation`
        # from the first's, keeping its own internal geometry intact.
        shift = system.positions[oxygens[0]] + torch.tensor(
            [separation, 0.0, 0.0], dtype=system.positions.dtype
        ) - system.positions[oxygens[1]]
        positions[atoms1] = positions[atoms1] + shift
        candidate = AtomicSystem(
            positions=positions,
            atomic_numbers=system.atomic_numbers,
            cell=system.cell,
            pbc=system.pbc,
        )
        _attach_rigid_groups_unchecked(candidate, rigid_groups(system))
        return constraint.compute_increm_change(
            candidate, MoveDelta(moved_indices=atoms1)
        ).item()

    # (The template's float32 positions make `separation` good to ~1e-6 A,
    # and the penalty's slope at the window edge is ~3e4 per A, so these
    # compare to a relative tolerance rather than bit-exactly.)
    assert _score_at(3.5) == 0.0  # comfortably outside the threshold
    assert _score_at(3.0) < 1e-3  # right at it
    assert _score_at(2.9) == pytest.approx(1000.0, rel=1e-3)  # one window inside
    # Strictly monotonic in the violation depth -- the gradient the old
    # 0/+inf step function couldn't provide.
    assert 0.0 < _score_at(2.97) < _score_at(2.95) < _score_at(2.9)


def test_penalty_at_window_sets_what_a_violation_costs(water_template):
    """The ramp's height, independent of its width: one softening window
    inside the minimum distance costs exactly penalty_at_window."""
    system = _two_far_apart_waters(water_template)
    other_o = system.positions[system.atomic_numbers == 8][0]
    positions = system.positions.clone()
    atoms1 = group_atom_indices(system, group_id=1)
    # Drop the second molecule's oxygen 2.3 A from the first's, against a
    # 2.4 A minimum -- exactly one 0.1 A window inside it.
    positions[atoms1] = positions[atoms1] + (
        other_o + torch.tensor([2.3, 0.0, 0.0]) - positions[atoms1[0]]
    )
    candidate = AtomicSystem(
        positions=positions,
        atomic_numbers=system.atomic_numbers,
        cell=system.cell,
        pbc=system.pbc,
    )
    _attach_rigid_groups_unchecked(candidate, rigid_groups(system))
    delta = MoveDelta(moved_indices=atoms1)

    for penalty_at_window in (1.0, 250.0, 1000.0, 1e5):
        constraint = MinDistConstraint(
            MinDistTable(pair_distances={(8, 8): 2.4, (1, 8): 1.0, (1, 1): 1.0}),
            softening_window=0.1,
            penalty_at_window=penalty_at_window,
        )
        score = constraint.compute_increm_change(candidate, delta).item()
        assert score == pytest.approx(penalty_at_window, rel=1e-3)


def test_from_config_threads_the_ramp_parameters_through():
    config = ConstraintConfig(
        default=1.2,
        pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.4)],
        softening_window=0.3,
        penalty_at_window=25.0,
    )
    constraint = MinDistConstraint.from_config(config)
    assert constraint.softening_window == 0.3
    assert constraint.penalty_at_window == 25.0


def test_the_default_window_is_a_tenth_of_an_angstrom_and_lives_with_distances():
    """The window is the one parameter of the penalty that carries units,
    so its default belongs to the distance constraints rather than to the
    (unit-agnostic) penalty module they share."""
    from raymc.constraints import pairwise, penalty

    assert pairwise.DEFAULT_SOFTENING_WINDOW == 0.1  # Angstrom
    assert not hasattr(penalty, "DEFAULT_SOFTENING_WINDOW")


def test_from_config_defaults_the_ramp_to_a_tenth_of_an_angstrom_and_a_thousand():
    constraint = MinDistConstraint.from_config(ConstraintConfig())
    assert constraint.softening_window == 0.1
    assert constraint.penalty_at_window == 1000.0


def test_same_group_bonded_pairs_are_never_checked(water_template):
    """An O-H bond (~0.96A) would violate almost any sane non-bonded
    minimum distance -- it must never be checked, since same-group pairs
    are excluded entirely (they're bonded, not a "closeness" concern)."""
    system = _two_far_apart_waters(water_template)
    constraint = MinDistConstraint(MinDistTable(pair_distances={(1, 8): 5.0}))  # huge
    constraint.init(system)

    move = RotateGroup(max_angle=1.0)
    candidate, delta = move.attempt(system, torch.Generator().manual_seed(0))

    score = constraint.compute_increm_change(candidate, delta)
    assert score.item() == 0.0


def test_missing_pair_falls_back_to_default_with_a_warning(water_template):
    system = _two_far_apart_waters(water_template)
    min_dist = MinDistTable()  # no pairs defined at all
    constraint = MinDistConstraint(min_dist)
    assert min_dist.default == DEFAULT_MIN_DIST

    # init() delegates to compute() (a full scan of every pair in the
    # system), so the "no pair defined, defaulting" warning fires here --
    # every element pair present (O-O, O-H, H-H) is unlisted.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        constraint.init(system)
    assert any("defaulting to 1.0" in str(w.message) for w in caught)

    other_o = system.positions[system.atomic_numbers == 8][0]
    positions = system.positions.clone()
    atoms1 = group_atom_indices(system, group_id=1)
    # 1.5 A: closer than the 1.0 A default, so should be flagged
    positions[atoms1[0]] = other_o + torch.tensor([1.5, 0.0, 0.0])
    close_but_not_overlapping = AtomicSystem(
        positions=positions,
        atomic_numbers=system.atomic_numbers,
        cell=system.cell,
        pbc=system.pbc,
    )
    _attach_rigid_groups_unchecked(close_but_not_overlapping, rigid_groups(system))
    delta = MoveDelta(moved_indices=atoms1)

    score = constraint.compute_increm_change(close_but_not_overlapping, delta)
    assert score.item() == 0.0  # 1.5A is fine against the 1.0A default


def test_missing_pair_warning_only_fires_once(water_template):
    system = _two_far_apart_waters(water_template)
    constraint = MinDistConstraint(MinDistTable())

    move = TranslateGroup(max_displacement=0.3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(5):
            candidate, delta = move.attempt(system, torch.Generator().manual_seed(i))
            constraint.compute_increm_change(candidate, delta)

    default_warnings = [w for w in caught if "defaulting to" in str(w.message)]
    # Two element pairs appear (O-O, O-H, H-H across the two molecules) --
    # each should warn exactly once, not once per step.
    assert len(default_warnings) == len({str(w.message) for w in default_warnings})
    assert 1 <= len(default_warnings) <= 3


def test_remove_only_delta_never_flagged(water_template):
    system = _two_far_apart_waters(water_template)
    constraint = MinDistConstraint(MinDistTable(pair_distances={(8, 8): 2.4}))
    constraint.init(system)

    move = RemoveGroup()
    candidate, delta = move.attempt(system, torch.Generator().manual_seed(0))

    score = constraint.compute_increm_change(candidate, delta)
    assert score.item() == 0.0


def test_commit_and_discard_are_no_ops_since_the_constraint_is_stateless(water_template):
    system = _two_far_apart_waters(water_template)
    constraint = MinDistConstraint(MinDistTable(pair_distances={(8, 8): 2.4}))
    constraint.commit()  # should not raise
    constraint.discard()  # should not raise


def test_check_passes_on_a_valid_system_and_raises_on_a_violating_one(water_template):
    system = _two_far_apart_waters(water_template)
    constraint = MinDistConstraint(MinDistTable(pair_distances={(8, 8): 2.4}))
    constraint.check(system)  # must not raise

    other_o = system.positions[system.atomic_numbers == 8][0]
    positions = system.positions.clone()
    atoms1 = group_atom_indices(system, group_id=1)
    positions[atoms1[0]] = other_o
    violating = AtomicSystem(
        positions=positions,
        atomic_numbers=system.atomic_numbers,
        cell=system.cell,
        pbc=system.pbc,
    )
    _attach_rigid_groups_unchecked(violating, rigid_groups(system))

    try:
        constraint.check(violating)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_compute_matches_compute_increm_change_for_the_full_system(water_template):
    """compute() (full O(N) scan) and init() (which delegates to compute())
    must agree for a valid starting structure: both report 0."""
    system = _two_far_apart_waters(water_template)
    constraint = MinDistConstraint(MinDistTable(pair_distances={(8, 8): 2.4}))
    assert constraint.compute(system).item() == 0.0
    assert constraint.init(system).item() == 0.0


def test_from_config_builds_the_right_pair_table():
    config = ConstraintConfig(
        default=1.2,
        pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.4)],
    )
    constraint = MinDistConstraint.from_config(config)
    assert constraint.min_dist.default == 1.2
    assert constraint.min_dist.lookup(8, 8).item() == 2.4
    assert constraint.min_dist.lookup(1, 1).item() == 1.2  # falls back to default


def test_short_run_with_constraint_never_produces_an_overlap(water_template):
    """End-to-end: run several dozen relocate-heavy steps in a tight box
    (where random relocation frequently collides) with the constraint
    active, and verify every ACCEPTED state never violates it."""
    cell = torch.eye(3) * 9.0  # tight -- forces frequent close contacts
    min_dist = MinDistTable(pair_distances={(8, 8): 2.4, (1, 8): 1.5, (1, 1): 1.0})
    system = pack_on_lattice(
        water_template,
        cell=cell,
        n_particles=6,
        min_dist=min_dist,
        seed=2,
    )

    constraint = MinDistConstraint(min_dist)
    constraint.init(system)
    constraint.check(system)
    move_set = MoveSet(
        moves=[
            TranslateGroup(max_displacement=1.0),
            RotateGroup(max_angle=1.0),
            RelocateGroup(template=water_template),
        ],
        weights=[1.0, 1.0, 1.0],
    )
    generator = torch.Generator().manual_seed(5)

    for _ in range(80):
        candidate, _name, delta = move_set.sample(system, generator)
        score = constraint.compute_increm_change(candidate, delta)
        # Strictly zero, not merely finite: the constraint scores a smooth
        # penalty now, so a small violation is a small finite number.
        if score == 0:
            system = candidate
            constraint.check(system)
            constraint.commit()
        else:
            constraint.discard()
