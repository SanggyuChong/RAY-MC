"""Tests for the maximum-separation constraint -- the mirror of
:mod:`raymc.constraints.min_dist`, plus the two things that are genuinely
its own: an infinite default that costs nothing, and the element
pre-filter that keeps it from scanning atoms it could never flag."""

from __future__ import annotations

import pytest
import torch

from raymc.config import ConstraintConfig, MinDistPairConfig
from raymc.constraints.max_dist import MaxDistConstraint
from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem
from raymc.system.atomic_system import _attach_rigid_groups_unchecked
from raymc.utils import DEFAULT_MAX_DIST, MaxDistTable

BOX = 30.0


def _system(offsets, atomic_numbers, groups):
    """Atoms at the given positions, one element and one rigid group each."""
    system = AtomicSystem(
        positions=torch.tensor(offsets, dtype=torch.float64),
        atomic_numbers=torch.tensor(atomic_numbers),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * BOX,
        pbc=torch.tensor([[True, True, True]]),
    )
    return _attach_rigid_groups_unchecked(system, torch.tensor(groups))


def _three_oxygens(spacing=3.0):
    """Three oxygens in a line, *spacing* apart, each its own rigid group --
    so the pairs are at 1x, 1x and 2x spacing."""
    return _system(
        [[0.0, 0.0, 0.0], [spacing, 0.0, 0.0], [2 * spacing, 0.0, 0.0]],
        [8, 8, 8],
        [0, 1, 2],
    )


def _whole_system_delta(system):
    return MoveDelta(moved_indices=torch.arange(system.positions.shape[0]))


def test_the_default_table_constrains_nothing():
    """The headline requirement: no maximum distance unless one is asked
    for. Every pair, however far apart, scores 0."""
    assert DEFAULT_MAX_DIST == float("inf")
    constraint = MaxDistConstraint(MaxDistTable())
    system = _three_oxygens(spacing=12.0)  # as far apart as the box allows
    assert constraint.compute(system).item() == 0.0


def test_the_default_table_short_circuits_before_touching_any_position():
    """And it must reach that 0 without doing the work -- the reason this
    constraint exists in this shape at all. If it ever starts computing
    distances, this system has none to compute them from."""
    constraint = MaxDistConstraint(MaxDistTable())
    assert constraint._unconstrained

    class _NoPositions:
        @property
        def positions(self):
            raise AssertionError("the unconstrained table must not read positions")

        atomic_numbers = torch.tensor([8])

    assert constraint.compute_increm_change(
        _NoPositions(), MoveDelta(moved_indices=torch.tensor([0]))
    ).item() == 0.0


def test_pairs_within_the_maximum_score_zero():
    constraint = MaxDistConstraint(MaxDistTable(pair_distances={(8, 8): 6.5}))
    system = _three_oxygens(spacing=3.0)  # furthest pair is 6.0 A
    assert constraint.compute(system).item() == 0.0


def test_a_pair_beyond_the_maximum_is_penalized():
    """One softening window (0.1 A) past a 5.9 A maximum costs exactly 1000,
    and only the one offending pair (0-2, at 6.0 A) is charged -- the two
    3.0 A pairs are fine."""
    constraint = MaxDistConstraint(
        MaxDistTable(pair_distances={(8, 8): 5.9}), softening_window=0.1
    )
    system = _three_oxygens(spacing=3.0)
    assert constraint.compute(system).item() == pytest.approx(1000.0)


def test_penalty_at_window_sets_what_a_violation_costs():
    """Same geometry, same window, different height: one softening window
    past the maximum costs exactly whatever penalty_at_window says."""
    system = _three_oxygens(spacing=3.0)  # furthest pair at 6.0 A
    for penalty_at_window in (1.0, 250.0, 1000.0, 1e5):
        constraint = MaxDistConstraint(
            MaxDistTable(pair_distances={(8, 8): 5.9}),
            softening_window=0.1,
            penalty_at_window=penalty_at_window,
        )
        assert constraint.compute(system).item() == pytest.approx(penalty_at_window)


def test_every_violating_pair_is_counted_once():
    """Three oxygens all beyond a 1.0 A maximum is three violating pairs,
    not six -- the touched-by-all scan sees each pair from both ends and
    must drop one of the two copies."""
    constraint = MaxDistConstraint(
        MaxDistTable(pair_distances={(8, 8): 2.0}), softening_window=1.0
    )
    # Pairs at 3.0, 3.0 and 6.0 A: violations of 1.0, 1.0 and 4.0 windows.
    system = _three_oxygens(spacing=3.0)
    from raymc.constraints.penalty import smooth_violation_penalty

    expected = smooth_violation_penalty(
        torch.tensor([1.0, 1.0, 4.0], dtype=torch.float64), 1.0
    ).sum()
    assert constraint.compute(system).item() == pytest.approx(expected.item())


def test_same_group_pairs_are_never_checked():
    """A molecule's own atoms are held at a fixed internal geometry by the
    rigid-body moves; a non-bonded separation limit has nothing to say
    about them -- same exclusion as the minimum-distance constraint."""
    constraint = MaxDistConstraint(MaxDistTable(pair_distances={(8, 8): 1.0}))
    one_group = _system(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]], [8, 8, 8], [0, 0, 0]
    )
    assert constraint.compute(one_group).item() == 0.0


def test_elements_outside_the_table_are_skipped_entirely():
    """An O-O-only table must ignore hydrogens completely -- both as the
    moved atom and as a partner. This is the pre-filter that keeps the
    per-step cost proportional to what's actually constrained."""
    constraint = MaxDistConstraint(MaxDistTable(pair_distances={(8, 8): 5.9}))
    with_hydrogens = _system(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],  # an H further from everything than any O-O pair
        ],
        [8, 8, 8, 1],
        [0, 1, 2, 3],
    )
    assert constraint.max_dist.constrained_elements().tolist() == [8]
    # Still exactly the one O-O violation, unchanged by the distant H.
    assert constraint.compute(with_hydrogens).item() == pytest.approx(1000.0)

    # And a move that only touches the hydrogen is free.
    h_only = MoveDelta(moved_indices=torch.tensor([3]))
    assert constraint.compute_increm_change(with_hydrogens, h_only).item() == 0.0


def test_a_finite_default_constrains_every_element_pair():
    """Opting into a finite default is opting every pair in -- there is
    nothing left to pre-filter, so constrained_elements() reports None."""
    table = MaxDistTable(default=5.9)
    assert table.constrained_elements() is None
    constraint = MaxDistConstraint(table)
    assert not constraint._unconstrained

    mixed = _system(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], [8, 1], [0, 1]
    )  # an O-H pair 6.0 A apart, 0.1 A over
    assert constraint.compute(mixed).item() == pytest.approx(1000.0)


def test_compute_matches_compute_increm_change_for_the_full_system():
    constraint = MaxDistConstraint(MaxDistTable(pair_distances={(8, 8): 5.9}))
    system = _three_oxygens(spacing=3.0)
    full = constraint.compute(system)
    incremental = constraint.compute_increm_change(system, _whole_system_delta(system))
    assert incremental.item() == pytest.approx(full.item())


def test_remove_only_delta_is_never_flagged():
    constraint = MaxDistConstraint(MaxDistTable(pair_distances={(8, 8): 1.0}))
    system = _three_oxygens(spacing=3.0)
    removal = MoveDelta(removed_indices=torch.tensor([2]))
    assert constraint.compute_increm_change(system, removal).item() == 0.0


def test_check_passes_within_the_maximum_and_raises_beyond_it():
    system = _three_oxygens(spacing=3.0)
    MaxDistConstraint(MaxDistTable(pair_distances={(8, 8): 6.5})).check(system)
    with pytest.raises(ValueError, match="maximum-distance constraint"):
        MaxDistConstraint(MaxDistTable(pair_distances={(8, 8): 5.9})).check(system)


def test_commit_and_discard_are_no_ops_since_the_constraint_is_stateless():
    constraint = MaxDistConstraint(MaxDistTable(pair_distances={(8, 8): 5.9}))
    constraint.commit()  # should not raise
    constraint.discard()  # should not raise


def test_from_config_builds_the_right_pair_table():
    config = ConstraintConfig(
        default=1.0,
        max_default=8.0,
        max_pairs=[MinDistPairConfig(elements=("O", "O"), distance=3.5)],
        softening_window=0.25,
        penalty_at_window=40.0,
    )
    constraint = MaxDistConstraint.from_config(config)
    assert constraint.max_dist.default == 8.0
    assert constraint.max_dist.lookup(8, 8).item() == 3.5
    assert constraint.max_dist.lookup(1, 1).item() == 8.0  # falls back to max_default
    assert constraint.softening_window == 0.25
    assert constraint.penalty_at_window == 40.0


def test_from_config_defaults_to_no_constraint_at_all():
    constraint = MaxDistConstraint.from_config(ConstraintConfig())
    assert constraint.max_dist.default == float("inf")
    assert constraint._unconstrained
    assert constraint.softening_window == 0.1
    assert constraint.penalty_at_window == 1000.0
