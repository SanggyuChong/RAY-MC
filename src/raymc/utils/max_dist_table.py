"""A user-configurable maximum-separation table per element pair.

The mirror image of :class:`~raymc.utils.MinDistTable`, with one
important difference in spirit: a minimum approach distance is something
every pair of atoms has (so its default is a real number, 1.0 A, and a
pair that falls back to it is worth warning about), whereas a *maximum*
separation is an opt-in restraint. Its default is therefore ``inf`` --
"no constraint" -- and falling back to that default is the normal case,
not something to warn about.

That default also makes the table cheap to skip entirely: see
:meth:`MaxDistTable.is_unconstrained` and
:meth:`MaxDistTable.constrained_elements`, which let
:class:`raymc.constraints.max_dist.MaxDistConstraint` avoid touching
most (or all) of the system at RMC runtime.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from raymc.config import ConstraintConfig

DEFAULT_MAX_DIST = float("inf")  # Angstrom -- i.e. no constraint at all
_MAX_ATOMIC_NUMBER = 118


class MaxDistTable:
    """Maximum allowed distance (Angstrom) for a pair of atomic numbers.

    Parameters
    ----------
    pair_distances : dict[tuple[int, int], float] | None
        Maximum allowed separation for an unordered pair of atomic
        numbers, e.g. ``{(8, 8): 3.5}`` for O-O. Order within the tuple
        doesn't matter.
    default : float
        Distance used for any element pair not present in
        *pair_distances*. Defaults to ``inf`` -- no constraint. Unlike
        :class:`~raymc.utils.MinDistTable`, falling back to the default
        never warns: with an infinite default there is nothing to warn
        about, and a run that opts into a finite default is opting every
        pair in deliberately.
    """

    def __init__(
        self,
        pair_distances: dict[tuple[int, int], float] | None = None,
        default: float = DEFAULT_MAX_DIST,
    ) -> None:
        self.default = default
        self._explicit: dict[tuple[int, int], float] = {
            (min(z1, z2), max(z1, z2)): d
            for (z1, z2), d in (pair_distances or {}).items()
        }
        self._table = torch.full(
            (_MAX_ATOMIC_NUMBER + 1, _MAX_ATOMIC_NUMBER + 1),
            default,
            dtype=torch.float64,
        )
        for (z1, z2), d in self._explicit.items():
            self._table[z1, z2] = d
            self._table[z2, z1] = d

    def is_unconstrained(self) -> bool:
        """True when this table constrains nothing at all -- an infinite
        default and no finite pair entry.

        The whole point of the ``inf`` default: a run that never asks for
        a maximum separation shouldn't pay for one, so the constraint (and
        :func:`raymc.orchestrate.build_chi_sq`) skips itself entirely when
        this is True, rather than looping over every atom every step to
        confirm nothing is out of range."""
        return math.isinf(self.default) and not any(
            not math.isinf(d) for d in self._explicit.values()
        )

    def constrained_elements(self) -> torch.Tensor | None:
        """Atomic numbers that appear in at least one finite entry, or
        ``None`` if the default itself is finite (in which case every
        element is constrained and there's nothing to narrow down).

        Lets the constraint shrink the pair scan to just the elements that
        could possibly violate something -- e.g. a table that only
        restrains O-O never has to look at the hydrogens."""
        if not math.isinf(self.default):
            return None
        elements = {
            z
            for (z1, z2), d in self._explicit.items()
            if not math.isinf(d)
            for z in (z1, z2)
        }
        return torch.tensor(sorted(elements), dtype=torch.long)

    def lookup(self, z1: "int | torch.Tensor", z2: "int | torch.Tensor") -> torch.Tensor:
        """Maximum allowed distance for atomic numbers *z1*/*z2*.

        Accepts either plain ints or broadcastable tensors of atomic
        numbers, exactly like :meth:`raymc.utils.MinDistTable.lookup`.
        """
        return self._table[z1, z2]

    @classmethod
    def from_config(cls, config: "ConstraintConfig") -> "MaxDistTable":
        """Build a :class:`MaxDistTable` from a run config's
        ``constraints.max_pairs``/``constraints.max_default`` fields (see
        :class:`raymc.config.ConstraintConfig`)."""
        import periodictable as pt

        pair_distances = {
            (
                pt.elements.symbol(pair.elements[0]).number,
                pt.elements.symbol(pair.elements[1]).number,
            ): pair.distance
            for pair in config.max_pairs
        }
        return cls(pair_distances=pair_distances, default=config.max_default)
