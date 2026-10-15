"""Minimum distance constraint.

This constraint enforces the minimum distance criteria defined in
:class:`raymc.utils.MinDistTable`. Minimum distances are not checked
between members of the same atom group (i.e., intra-atomic).

The score is no longer a ``0``/``+inf`` step: each too-close pair
contributes a smooth, differentiable penalty in the violation depth (see
:mod:`raymc.constraints.penalty`), which is exactly ``0`` at the
configured distance, *penalty_at_window* (1000 by default) one
*softening_window* inside it, and diverges (to a literal ``+inf`` in
float64) for a gross violation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from raymc.constraints.pairwise import (
    DEFAULT_SOFTENING_WINDOW,
    touched_pair_distances,
)
from raymc.constraints.penalty import PENALTY_AT_WINDOW, total_violation_penalty
from raymc.modalities.base import Calculator
from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem
from raymc.utils import MinDistTable

if TYPE_CHECKING:
    from raymc.config import ConstraintConfig


class MinDistConstraint(Calculator):
    """Score the total smooth close-contact penalty over every pair of
    atoms from different rigid groups that sits closer than
    :class:`~raymc.utils.MinDistTable` allows; ``0`` when none do.

    Parameters
    ----------
    min_dist : MinDistTable of the system of interest.
    softening_window : float
        Width (Angstrom) of the window over which the penalty ramps from
        ``0`` at the configured minimum distance to
        *penalty_at_window* that far inside it -- see
        :mod:`raymc.constraints.penalty`.
    penalty_at_window : float
        What one *softening_window* of violation costs. Sets how hard the
        constraint is relative to the run's other chi-squared terms.
    """

    def __init__(
        self,
        min_dist: MinDistTable,
        softening_window: float = DEFAULT_SOFTENING_WINDOW,
        penalty_at_window: float = PENALTY_AT_WINDOW,
    ) -> None:
        self.min_dist = min_dist
        self.softening_window = softening_window
        self.penalty_at_window = penalty_at_window

    def compute(self, system: AtomicSystem) -> torch.Tensor:
        """Full compute for every atom and its neighbors."""
        n = system.positions.shape[0]
        all_indices = torch.arange(n)
        return self._score(system, MoveDelta(moved_indices=all_indices))

    def compute_increm_change(
        self, system: AtomicSystem, delta: MoveDelta
    ) -> torch.Tensor:
        """Partial compute for atoms that have moved during RMC step.

        Scores only the pairs involving a touched atom, which equals the
        whole system's penalty exactly as long as the committed state is
        violation-free -- untouched-untouched pairs are unchanged by the
        move, and contribute 0 to a violation-free state. Since a
        violation of even a fraction of *softening_window* already costs
        far more than a typical per-step chi-squared change (at any sane
        *penalty_at_window*), accepted states are violation-free in
        practice.
        """
        return self._score(system, delta)

    def check(self, system: AtomicSystem) -> None:
        """Method for external access that throws error if any violating pair are found."""
        if self.compute(system) > 0:
            raise ValueError("System violates the minimum-distance constraint.")

    def _score(self, system: AtomicSystem, delta: MoveDelta) -> torch.Tensor:
        touched = torch.cat([delta.moved_indices, delta.added_indices])
        if touched.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float64)

        atomic_numbers = system.atomic_numbers.long()
        partners = torch.arange(system.positions.shape[0])

        self.min_dist.warn_about_missing_pairs(atomic_numbers[touched], atomic_numbers)

        dist, skip = touched_pair_distances(system, touched, partners)
        r_min = self.min_dist.lookup(
            atomic_numbers[touched].unsqueeze(1), atomic_numbers.unsqueeze(0)
        )
        # Positive where a pair is closer than allowed; the skipped pairs
        # (same rigid group, self, double-counted mirror) are zeroed so
        # they can never contribute -- see raymc.constraints.pairwise.
        violation = (r_min - dist).masked_fill(skip, 0.0)
        return total_violation_penalty(
            violation, self.softening_window, self.penalty_at_window
        )

    @classmethod
    def from_config(cls, config: "ConstraintConfig") -> "MinDistConstraint":
        """Build a :class:`MinDistConstraint` from a run config's
        ``constraints`` section (see :class:`raymc.config.ConstraintConfig`)."""
        return cls(
            min_dist=MinDistTable.from_config(config),
            softening_window=config.softening_window,
            penalty_at_window=config.penalty_at_window,
        )
