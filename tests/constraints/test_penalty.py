"""Tests for the smooth violation penalty -- the properties the whole
design rests on: exactly 0 when satisfied, exactly *scale* one window past
the threshold, divergent (all the way to a literal +inf) beyond, and
carrying no units of its own, so any constraint can use it and not just
the distance ones that do today.

Nothing here constructs an atom, a pair, or a system: the module under
test takes a plain array of violation depths and knows nothing about
where they came from.
"""

from __future__ import annotations

import math

import pytest
import torch

from raymc.constraints.penalty import (
    PENALTY_AT_WINDOW,
    smooth_violation_penalty,
    total_violation_penalty,
)


def test_a_satisfied_violation_scores_exactly_zero():
    satisfied = torch.tensor([-10.0, -1.0, -1e-9, 0.0], dtype=torch.float64)
    assert torch.equal(
        smooth_violation_penalty(satisfied, 0.1),
        torch.zeros_like(satisfied),
    )


def test_an_unconstrained_entry_scores_zero_not_nan():
    """An entry with an infinite threshold has a violation of -inf. That
    must come out as a clean 0, not a NaN -- it's the normal case for the
    maximum-distance constraint, where the default threshold is inf."""
    penalty = smooth_violation_penalty(
        torch.tensor([-float("inf")], dtype=torch.float64), 0.1
    )
    assert penalty.item() == 0.0


def test_the_default_scale_is_a_thousand():
    assert PENALTY_AT_WINDOW == 1000.0


def test_one_window_of_violation_costs_exactly_the_advertised_penalty():
    for window in (0.05, 0.1, 0.5, 2.0):
        at_window = smooth_violation_penalty(
            torch.tensor([window], dtype=torch.float64), window
        )
        assert at_window.item() == pytest.approx(PENALTY_AT_WINDOW)


def test_scale_sets_what_one_window_of_violation_costs():
    """The other half of the ramp: *window* is how far, *scale* is how
    expensive. At exactly one window the penalty IS the scale, whatever it
    is set to."""
    one_window = torch.tensor([0.1], dtype=torch.float64)
    for scale in (1.0, 50.0, 1000.0, 1e6):
        penalty = smooth_violation_penalty(one_window, 0.1, scale)
        assert penalty.item() == pytest.approx(scale)


def test_scale_rescales_the_whole_curve_without_changing_its_shape():
    """Scale is a pure multiplier -- it changes how much a violation costs
    relative to the run's other chi-squared terms, not where the penalty
    starts to bite or how fast it steepens."""
    depths = torch.tensor([0.0, 0.02, 0.05, 0.1, 0.3], dtype=torch.float64)
    default = smooth_violation_penalty(depths, 0.1, 1000.0)
    tenfold = smooth_violation_penalty(depths, 0.1, 10_000.0)
    assert torch.allclose(tenfold, default * 10.0)
    assert tenfold[0].item() == 0.0  # still exactly free at the threshold


def test_total_violation_penalty_honors_the_scale():
    """The summing wrapper the constraints actually call must pass the
    scale through, not quietly fall back to the default."""
    violations = torch.tensor([0.0, 0.1, 0.1], dtype=torch.float64)
    total = total_violation_penalty(violations, 0.1, 250.0)
    assert total.item() == pytest.approx(500.0)
    # ...and still short-circuit to a clean zero when nothing is violated.
    satisfied = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    assert total_violation_penalty(satisfied, 0.1, 250.0).item() == 0.0


def test_the_penalty_carries_no_units_of_its_own():
    """Only the ratio violation/window is ever used, so the same
    constraint expressed in different units gives identical penalties.
    This is what makes the module reusable for something that isn't a
    distance at all -- an angle, say, whose window would be angular."""
    degrees = torch.tensor([0.0, 2.5, 5.0, 10.0, 20.0], dtype=torch.float64)
    radians = degrees * math.pi / 180.0
    assert torch.allclose(
        smooth_violation_penalty(degrees, 5.0),
        smooth_violation_penalty(radians, 5.0 * math.pi / 180.0),
    )
    # Same story for Angstrom vs nanometre.
    angstrom = torch.tensor([0.0, 0.05, 0.1, 0.4], dtype=torch.float64)
    assert torch.allclose(
        smooth_violation_penalty(angstrom, 0.1),
        smooth_violation_penalty(angstrom * 0.1, 0.01),
    )


def test_penalty_is_strictly_increasing_in_the_violation_depth():
    depths = torch.tensor([0.0, 0.01, 0.05, 0.1, 0.2, 0.4], dtype=torch.float64)
    penalties = smooth_violation_penalty(depths, 0.1)
    assert torch.all(penalties[1:] > penalties[:-1])


def test_a_gross_violation_overflows_to_infinity():
    """The old hard veto, recovered in the limit: past ~26.6 windows the
    float64 exponential overflows and the penalty is literally +inf."""
    depths = torch.tensor([2.6, 2.7, 10.0], dtype=torch.float64)
    penalties = smooth_violation_penalty(depths, 0.1)
    assert torch.isfinite(penalties[0])
    assert torch.isinf(penalties[1])
    assert torch.isinf(penalties[2])


def test_the_penalty_is_differentiable_including_across_the_threshold():
    """The point of replacing the 0/+inf step. The gradient exists
    everywhere, is exactly 0 on the satisfied side, and -- because the ramp
    is quadratic in the reduced violation -- decays to 0 at the threshold
    itself too, so there's no kink where the constraint starts to bite."""
    depths = torch.tensor(
        [-0.5, -1e-6, 0.0, 1e-6, 0.05, 0.1], dtype=torch.float64, requires_grad=True
    )
    smooth_violation_penalty(depths, 0.1).sum().backward()

    grad = depths.grad
    assert torch.isfinite(grad).all()
    assert grad[:3].tolist() == [0.0, 0.0, 0.0]  # satisfied side: flat
    assert grad[5] > grad[4] > 0  # steepening into the violation
    # Continuous across the threshold: the slope just inside it is a
    # vanishing fraction of the slope at the window edge, rather than the
    # step function's infinite jump.
    assert 0 < grad[3] < 1e-4 * grad[5]


def test_window_controls_how_quickly_the_penalty_bites():
    """Same violation, different windows: a narrower width is a harder,
    more veto-like wall, a wider one a gentler ramp."""
    violation = torch.tensor([0.1], dtype=torch.float64)
    narrow = smooth_violation_penalty(violation, 0.05).item()
    default = smooth_violation_penalty(violation, 0.1).item()
    wide = smooth_violation_penalty(violation, 0.5).item()
    assert narrow > default > wide > 0.0


def test_a_nonpositive_window_or_scale_is_rejected():
    violation = torch.tensor([0.1], dtype=torch.float64)
    for bad_window in (0.0, -0.1):
        with pytest.raises(ValueError, match="window must be positive"):
            smooth_violation_penalty(violation, bad_window)
    for bad_scale in (0.0, -1000.0):
        with pytest.raises(ValueError, match="scale must be positive"):
            smooth_violation_penalty(violation, 0.1, bad_scale)


def test_a_bad_scale_is_rejected_even_when_nothing_is_violated():
    """total_violation_penalty short-circuits on a batch with no
    violation, so it has to validate first -- otherwise a nonsensical
    setting would lie dormant until the first violation turned up, which
    could be a hundred thousand steps in."""
    satisfied = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="scale must be positive"):
        total_violation_penalty(satisfied, 0.1, -1.0)
    with pytest.raises(ValueError, match="window must be positive"):
        total_violation_penalty(satisfied, 0.0, 1000.0)
