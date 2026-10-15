"""Smooth, differentiable penalty for a violated constraint.

A constraint expressed as a step -- ``0`` when satisfied, ``+inf`` the
instant it is crossed -- is not differentiable, and gives a Metropolis
walker a wall it can only bounce off rather than a gradient it can be
steered by: every violating move is equally, infinitely bad, so nothing
tells the engine *how badly* it missed.

:func:`smooth_violation_penalty` replaces that step with a function of the
**violation depth** *v* -- how far past its threshold something is, with
``v <= 0`` meaning satisfied::

    penalty(v) = scale * expm1((v / window)^2) / expm1(1)

- exactly ``0`` when satisfied, and C^1 across ``v = 0`` (the derivative
  vanishes on both sides), so there is no kink where the constraint starts
  to bite;
- exactly *scale* at ``v = window``. Those two together are what the
  parameters mean: *window* is how far past the threshold the penalty
  takes to ramp up, and *scale* is what it costs once it has. *scale*
  sets the constraint's hardness relative to whatever else the caller is
  summing it against -- so the number that matters is its size next to
  those other terms, not its absolute value;
- super-exponentially divergent beyond, overflowing float64 to a literal
  ``+inf`` at roughly ``v > 26.6 * window`` -- recovering a hard veto
  exactly, for any constraint whose violation can actually get that deep.
  A violation bounded well below that (an angle, say, which cannot exceed
  half a turn) instead tops out at a merely astronomical number.

**Units.** Only the ratio ``v / window`` is ever used, so this carries no
units of its own: express the violation and the window in the same
quantity and the penalty is identical either way -- a separation in
Angstrom, an angle in degrees or radians, anything else with a threshold.
For the same reason *window* has no default here; a default belongs with
the constraint that knows what is being measured.

Nothing in this module knows what is being constrained, or where the
violations came from. Callers work that out -- see
:mod:`raymc.constraints.pairwise` for the atom-pair case -- and hand over
a plain array of depths.
"""

from __future__ import annotations

import math

import torch

PENALTY_AT_WINDOW = 1e3
"""Default *scale*: what one *window* of violation costs. Dimensionless
(it is a chi-squared magnitude, not a measurement), so unlike the window
it is meaningful as a shared default for any constraint. A constraint that
wants it configurable exposes it itself -- the distance constraints do, as
:attr:`raymc.config.ConstraintConfig.penalty_at_window`."""

_EXPM1_ONE = math.expm1(1.0)


def _validate(window: float, scale: float) -> None:
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")


def smooth_violation_penalty(
    violation: torch.Tensor,
    window: float,
    scale: float = PENALTY_AT_WINDOW,
) -> torch.Tensor:
    """Per-element smooth penalty for a violation depth (see module docstring).

    Parameters
    ----------
    violation : torch.Tensor
        How far past its threshold each entry is, in the same quantity as
        *window*. Zero or negative means satisfied and scores exactly
        ``0``; ``-inf`` (an entry with an infinite threshold, i.e. one
        that is not constrained at all) is handled the same way, not as a
        NaN.
    window : float
        Violation depth at which the penalty equals *scale*. Same
        quantity as *violation*; no default, since it is the one
        parameter here that has units.
    scale : float
        Penalty at ``violation == window``.

    Returns
    -------
    torch.Tensor
        Same shape as *violation*.
    """
    _validate(window, scale)
    reduced = torch.clamp(violation, min=0.0) / window
    return scale * torch.expm1(reduced * reduced) / _EXPM1_ONE


def total_violation_penalty(
    violation: torch.Tensor,
    window: float,
    scale: float = PENALTY_AT_WINDOW,
) -> torch.Tensor:
    """Summed :func:`smooth_violation_penalty` over a whole array of depths.

    Short-circuits when nothing is violated -- the overwhelmingly common
    case for a constraint that is doing its job, since a state carrying a
    real violation is one the caller is trying to avoid in the first
    place. Exact, not an approximation: a satisfied entry's penalty is
    identically 0, so an array with no violation sums to 0 by definition.
    Worth the extra reduction because the alternative is running the
    exponential (and four other elementwise kernels) over every entry to
    be told 0 again.
    """
    # Validated before the short-circuit, so a nonsensical window/scale is
    # caught on the very first call rather than whenever a violation
    # happens to turn up.
    _validate(window, scale)
    if not bool((violation > 0).any()):
        return torch.zeros((), dtype=violation.dtype)
    return smooth_violation_penalty(violation, window, scale).sum()
