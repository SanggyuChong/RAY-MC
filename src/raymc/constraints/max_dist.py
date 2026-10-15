"""Maximum distance constraint.

The mirror of :mod:`raymc.constraints.min_dist`: where that one penalizes
a pair of atoms from different rigid groups for being *closer* than
:class:`~raymc.utils.MinDistTable` allows, this one penalizes a pair for
being *further apart* than :class:`~raymc.utils.MaxDistTable` allows.
Same exclusions (same rigid group, self, double-counted mirror), same
smooth penalty (:mod:`raymc.constraints.penalty`).

Note what "every pair" means here, because it is not symmetric with the
minimum-distance case in its consequences: a finite entry for an element
pair requires *every* such pair in the box -- not merely the nearest
neighbour -- to stay within that distance. In a bulk periodic system
that's only satisfiable for a pair of elements whose atoms genuinely all
sit within one restrained distance of each other (a molecule that is not
one rigid group, a cluster, a dimer). It is deliberately not a
coordination-shell restraint.

**Cost.** Unlike the minimum-distance constraint, whose default applies
to every element pair, this one defaults to ``inf`` -- no constraint --
so the common case is a table that restrains nothing, or a couple of
element pairs out of many. Paying a full touched-by-N pair scan every RMC
step to rediscover that would be pure waste, so this constraint narrows
the work up front, in ``__init__``, and never per step:

- a table that constrains nothing at all
  (:meth:`~raymc.utils.MaxDistTable.is_unconstrained`) short-circuits to
  ``0`` before looking at a single position -- and
  :func:`raymc.orchestrate.build_chi_sq` goes further and leaves the term
  out of the aggregator entirely, so a default run never even calls this;
- otherwise the scan is restricted to the elements that appear in a
  finite entry (:meth:`~raymc.utils.MaxDistTable.constrained_elements`),
  so e.g. an O-O-only table never looks at a hydrogen.
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
from raymc.utils import MaxDistTable

if TYPE_CHECKING:
    from raymc.config import ConstraintConfig

_ZERO = torch.tensor(0.0, dtype=torch.float64)


class MaxDistConstraint(Calculator):
    """Score the total smooth penalty over every pair of atoms from
    different rigid groups that sits further apart than
    :class:`~raymc.utils.MaxDistTable` allows; ``0`` when none do (which
    includes, for free, the default table that allows everything).

    Parameters
    ----------
    max_dist : MaxDistTable of the system of interest.
    softening_window : float
        Width (Angstrom) of the window over which the penalty ramps from
        ``0`` at the configured maximum distance to
        *penalty_at_window* that far beyond it -- see
        :mod:`raymc.constraints.penalty`.
    penalty_at_window : float
        What one *softening_window* of violation costs. Sets how hard the
        constraint is relative to the run's other chi-squared terms.
    """

    def __init__(
        self,
        max_dist: MaxDistTable,
        softening_window: float = DEFAULT_SOFTENING_WINDOW,
        penalty_at_window: float = PENALTY_AT_WINDOW,
    ) -> None:
        self.max_dist = max_dist
        self.softening_window = softening_window
        self.penalty_at_window = penalty_at_window
        # Both decided once, here, not per RMC step -- see module docstring.
        self._unconstrained = max_dist.is_unconstrained()
        self._constrained_z = max_dist.constrained_elements()

    def compute(self, system: AtomicSystem) -> torch.Tensor:
        """Full compute for every atom and its (relevant) partners."""
        n = system.positions.shape[0]
        all_indices = torch.arange(n)
        return self._score(system, MoveDelta(moved_indices=all_indices))

    def compute_increm_change(
        self, system: AtomicSystem, delta: MoveDelta
    ) -> torch.Tensor:
        """Partial compute for atoms that have moved during RMC step.

        Same caveat as
        :meth:`raymc.constraints.min_dist.MinDistConstraint.compute_increm_change`:
        exact whenever the committed state is violation-free, which a
        *penalty_at_window*-per-*softening_window* penalty makes true in
        practice at any sane setting.
        """
        return self._score(system, delta)

    def check(self, system: AtomicSystem) -> None:
        """Method for external access that throws error if any violating pair are found."""
        if self.compute(system) > 0:
            raise ValueError("System violates the maximum-distance constraint.")

    def _relevant(self, atomic_numbers: torch.Tensor) -> torch.Tensor | None:
        """Boolean mask over all atoms of the ones whose element appears in
        a finite entry, or ``None`` when that's every atom."""
        if self._constrained_z is None:
            return None
        return torch.isin(atomic_numbers, self._constrained_z)

    def _score(self, system: AtomicSystem, delta: MoveDelta) -> torch.Tensor:
        if self._unconstrained:
            return _ZERO.clone()

        touched = torch.cat([delta.moved_indices, delta.added_indices])
        if touched.numel() == 0:
            return _ZERO.clone()

        atomic_numbers = system.atomic_numbers.long()
        relevant = self._relevant(atomic_numbers)
        if relevant is None:
            partners = torch.arange(atomic_numbers.numel())
        else:
            touched = touched[relevant[touched]]
            partners = relevant.nonzero(as_tuple=True)[0]
            if touched.numel() == 0 or partners.numel() == 0:
                return _ZERO.clone()

        dist, skip = touched_pair_distances(system, touched, partners)
        r_max = self.max_dist.lookup(
            atomic_numbers[touched].unsqueeze(1), atomic_numbers[partners].unsqueeze(0)
        )
        # Positive where a pair is further apart than allowed. An
        # unconstrained pair has r_max = inf, so its violation is -inf,
        # which the penalty clamps to a clean 0 (not a NaN).
        violation = (dist - r_max).masked_fill(skip, 0.0)
        return total_violation_penalty(
            violation, self.softening_window, self.penalty_at_window
        )

    @classmethod
    def from_config(cls, config: "ConstraintConfig") -> "MaxDistConstraint":
        """Build a :class:`MaxDistConstraint` from a run config's
        ``constraints`` section (see :class:`raymc.config.ConstraintConfig`)."""
        return cls(
            max_dist=MaxDistTable.from_config(config),
            softening_window=config.softening_window,
            penalty_at_window=config.penalty_at_window,
        )
