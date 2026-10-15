"""Intramolecular distance-window constraint.

The exact complement of :mod:`raymc.constraints.min_dist` and
:mod:`raymc.constraints.max_dist`: where those two score only pairs of
atoms in *different* rigid groups, this one scores only pairs within the
*same* rigid group, against a per-element-pair ``[min, max]`` window (see
:class:`~raymc.utils.IntraWindowTable`). Because the split is exactly
complementary, adding this constraint needs no change at all to the
existing exclusion logic -- the same-group mask that those constraints
throw away is the one this constraint keeps.

This is what lets a molecule flex without dissolving. Every move in
RAY-MC used to be rigid, so an intramolecular distance was constant by
construction and nothing needed to police it. The moment a single atom
can move on its own (:class:`raymc.moves.group_based.TranslateAtom`)
that stops being true, and an unpoliced O-H bond is free to collapse to
zero or stretch to 5 A with only g(r) to object.

Same smooth penalty as both distance constraints
(:mod:`raymc.constraints.penalty`): exactly ``0`` anywhere inside the
window, ramping to *penalty_at_window* one *softening_window* outside it,
and diverging beyond. A flat-bottomed window like this yields a roughly
uniform distribution of bonded distances across its width, not a
physically-shaped one -- the width is a free parameter, and the design
note's decision 2 records that this is the deliberate potential-free
alternative to restraining bonds with an MLIP.

**Cost.** Like :class:`~raymc.constraints.max_dist.MaxDistConstraint`,
this constraint narrows its work once in ``__init__`` rather than every
step: an empty table short-circuits to ``0``, and otherwise only the
elements appearing in a window are ever looked at. It then narrows again
per step, and much more sharply: a same-group pair can only involve
atoms of the touched atoms' own groups, so the partner set is at most a
few atoms out of the whole box -- unlike the inter-group constraints,
which genuinely have to consider every atom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from raymc.constraints.pairwise import (
    DEFAULT_SOFTENING_WINDOW,
    touched_pair_geometry,
)
from raymc.constraints.penalty import PENALTY_AT_WINDOW, total_violation_penalty
from raymc.modalities.base import Calculator
from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem, rigid_groups
from raymc.utils import IntraWindowTable

if TYPE_CHECKING:
    from raymc.config import ConstraintConfig

_ZERO = torch.tensor(0.0, dtype=torch.float64)


class IntramolecularConstraint(Calculator):
    """Score the total smooth penalty over every pair of atoms from the
    SAME rigid group whose separation falls outside the ``[min, max]``
    window :class:`~raymc.utils.IntraWindowTable` allows for its element
    pair; ``0`` when none do.

    Parameters
    ----------
    windows : IntraWindowTable
        Allowed bonded distance window per element pair.
    softening_window : float
        Width (Angstrom) of the window over which the penalty ramps from
        ``0`` at either edge of the allowed window to *penalty_at_window*
        that far outside it -- see :mod:`raymc.constraints.penalty`.
    penalty_at_window : float
        What one *softening_window* of violation costs. Sets how hard the
        constraint is relative to the run's other chi-squared terms.
    """

    def __init__(
        self,
        windows: IntraWindowTable,
        softening_window: float = DEFAULT_SOFTENING_WINDOW,
        penalty_at_window: float = PENALTY_AT_WINDOW,
    ) -> None:
        self.windows = windows
        self.softening_window = softening_window
        self.penalty_at_window = penalty_at_window
        # Decided once, here, not per RMC step -- see the module docstring.
        self._unconstrained = windows.is_unconstrained()
        self._constrained_z = windows.constrained_elements()

    def compute(self, system: AtomicSystem) -> torch.Tensor:
        """Full compute over every atom and its same-group partners."""
        n = system.positions.shape[0]
        return self._score(system, MoveDelta(moved_indices=torch.arange(n)))

    def compute_increm_change(
        self, system: AtomicSystem, delta: MoveDelta
    ) -> torch.Tensor:
        """Partial compute for atoms that have moved during an RMC step.

        Scores only the same-group pairs involving a touched atom. Same
        caveat as
        :meth:`raymc.constraints.min_dist.MinDistConstraint.compute_increm_change`:
        exact whenever the committed state is violation-free, since an
        untouched-untouched pair is unchanged by the move and contributes
        0 to a violation-free state -- which a
        *penalty_at_window*-per-*softening_window* penalty makes true in
        practice at any sane setting.
        """
        return self._score(system, delta)

    def check(self, system: AtomicSystem) -> None:
        """Raise if any same-group pair sits outside its allowed window."""
        if self.compute(system) > 0:
            raise ValueError(
                "System violates the intramolecular distance-window constraint."
            )

    def _score(self, system: AtomicSystem, delta: MoveDelta) -> torch.Tensor:
        if self._unconstrained:
            return _ZERO.clone()

        touched = torch.cat([delta.moved_indices, delta.added_indices])
        if touched.numel() == 0:
            return _ZERO.clone()

        atomic_numbers = system.atomic_numbers.long()
        relevant = torch.isin(atomic_numbers, self._constrained_z)
        touched = touched[relevant[touched]]
        if touched.numel() == 0:
            return _ZERO.clone()

        # The decisive narrowing, and the reason this constraint is cheap
        # even though it is called on every move: a same-group pair can
        # only reach atoms of the touched atoms' own groups, which for a
        # water box is 3 atoms rather than 1500.
        groups = rigid_groups(system)
        partners = (relevant & torch.isin(groups, groups[touched])).nonzero(
            as_tuple=True
        )[0]
        if partners.numel() == 0:
            return _ZERO.clone()

        dist, same_group, redundant = touched_pair_geometry(system, touched, partners)
        r_min, r_max = self.windows.lookup(
            atomic_numbers[touched].unsqueeze(1), atomic_numbers[partners].unsqueeze(0)
        )
        # Inverted relative to the inter-group constraints: there, a
        # same-group pair is what gets skipped; here it is the only thing
        # scored. `redundant` (self-pairs and double-counted mirrors)
        # is skipped either way.
        skip = ~same_group | redundant
        # Two independent ways to leave a window, scored in one penalty
        # call over the stacked depths. An unconstrained pair has
        # r_min = 0 and r_max = inf, so both its depths are <= 0 and it
        # contributes exactly 0.
        violation = torch.stack(
            [
                (r_min - dist).masked_fill(skip, 0.0),
                (dist - r_max).masked_fill(skip, 0.0),
            ]
        )
        return total_violation_penalty(
            violation, self.softening_window, self.penalty_at_window
        )

    @classmethod
    def from_config(cls, config: "ConstraintConfig") -> "IntramolecularConstraint":
        """Build an :class:`IntramolecularConstraint` from a run config's
        ``constraints`` section (see
        :class:`raymc.config.ConstraintConfig`)."""
        return cls(
            windows=IntraWindowTable.from_config(config),
            softening_window=config.softening_window,
            penalty_at_window=config.penalty_at_window,
        )
