"""A user-configurable bonded [min, max] distance window per element pair.

The third distance table, alongside :class:`~raymc.utils.MinDistTable`
and :class:`~raymc.utils.MaxDistTable`, and the only one that describes a
pair of atoms *inside* one rigid group. Those two tables deliberately say
nothing about such a pair (see :mod:`raymc.constraints.pairwise`), which
was harmless while every move was rigid and every intramolecular distance
was therefore constant by construction. It stops being harmless the
moment a single atom can move on its own: an O-H bond would then be
scored by nothing at all, free to collapse or stretch arbitrarily, with
chi-squared registering only whether the resulting g(r) happened to
improve.

Both bounds live in one table because they describe one physical thing --
the window a bonded distance is allowed to wander in -- and are always
configured, validated, and looked up together. Splitting them across a
MinDistTable and a MaxDistTable would allow the two halves of a single
window to be set inconsistently, and would double the lookups on the hot
path.

Like :class:`~raymc.utils.MaxDistTable`, this is an opt-in restraint: a
pair with no entry is unconstrained (window ``[0, inf)``), falling back
to that is the normal case rather than something to warn about, and a
table with no entries at all is skipped entirely at runtime -- see
:meth:`IntraWindowTable.is_unconstrained`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from raymc.config import ConstraintConfig

DEFAULT_WINDOW_MIN = 0.0  # Angstrom -- no lower bound
DEFAULT_WINDOW_MAX = float("inf")  # Angstrom -- no upper bound
_MAX_ATOMIC_NUMBER = 118


class IntraWindowTable:
    """Allowed ``[min, max]`` distance (Angstrom) for a pair of atomic
    numbers within one rigid group.

    Parameters
    ----------
    windows : dict[tuple[int, int], tuple[float, float]] | None
        Allowed window for an unordered pair of atomic numbers, e.g.
        ``{(8, 1): (0.9, 1.1)}`` for O-H. Order within the key doesn't
        matter -- ``(8, 1)`` and ``(1, 8)`` are the same pair.

    Notes
    -----
    The unconstrained defaults are ``0.0`` and ``inf`` rather than
    ``-inf`` and ``inf`` on purpose: a violation depth is computed as
    ``r_min - dist`` and ``dist - r_max``, and an infinite *lower* bound
    would put an ``-inf`` into the first of those for every unconstrained
    pair. Zero gives the identical (satisfied) answer with no infinity in
    the arithmetic at all, since a distance is never negative. The upper
    bound keeps ``inf``, matching :class:`~raymc.utils.MaxDistTable`,
    whose penalty is documented to clamp the resulting ``-inf`` to a clean
    0.
    """

    def __init__(
        self,
        windows: dict[tuple[int, int], tuple[float, float]] | None = None,
    ) -> None:
        self._explicit: dict[tuple[int, int], tuple[float, float]] = {
            (min(z1, z2), max(z1, z2)): (lo, hi)
            for (z1, z2), (lo, hi) in (windows or {}).items()
        }
        for (z1, z2), (lo, hi) in self._explicit.items():
            if hi <= lo:
                raise ValueError(
                    f"intramolecular window for element pair (Z={z1}, Z={z2}): "
                    f"max ({hi} A) must exceed min ({lo} A)"
                )

        shape = (_MAX_ATOMIC_NUMBER + 1, _MAX_ATOMIC_NUMBER + 1)
        self._min_table = torch.full(shape, DEFAULT_WINDOW_MIN, dtype=torch.float64)
        self._max_table = torch.full(shape, DEFAULT_WINDOW_MAX, dtype=torch.float64)
        for (z1, z2), (lo, hi) in self._explicit.items():
            self._min_table[z1, z2] = self._min_table[z2, z1] = lo
            self._max_table[z1, z2] = self._max_table[z2, z1] = hi

    def is_unconstrained(self) -> bool:
        """True when this table constrains nothing at all -- no entries,
        or only entries that are themselves ``[0, inf)``.

        Lets :class:`raymc.constraints.intramolecular.IntramolecularConstraint`
        (and :func:`raymc.orchestrate.build_chi_sq`) skip the term
        entirely, so a rigid-molecule run pays nothing for a feature it
        isn't using."""
        return not any(
            lo > 0.0 or not math.isinf(hi) for lo, hi in self._explicit.values()
        )

    def constrained_elements(self) -> torch.Tensor:
        """Atomic numbers appearing in at least one restraining entry.

        Lets the constraint shrink its pair scan to the elements that
        could possibly violate something -- with an O-H and an H-H window
        on water there is nothing to narrow, but a single restrained
        fragment inside a large mixed system narrows a lot."""
        elements = {
            z
            for (z1, z2), (lo, hi) in self._explicit.items()
            if lo > 0.0 or not math.isinf(hi)
            for z in (z1, z2)
        }
        return torch.tensor(sorted(elements), dtype=torch.long)

    def lookup(
        self, z1: "int | torch.Tensor", z2: "int | torch.Tensor"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(min, max)`` allowed distance for atomic numbers *z1*/*z2*.

        Accepts either plain ints or broadcastable tensors of atomic
        numbers, exactly like :meth:`raymc.utils.MinDistTable.lookup`.
        """
        return self._min_table[z1, z2], self._max_table[z1, z2]

    def window_for(self, z1: int, z2: int) -> tuple[float, float]:
        """``(min, max)`` for one element pair, as plain floats -- for
        reporting and tests, not the hot path."""
        key = (min(z1, z2), max(z1, z2))
        return self._explicit.get(key, (DEFAULT_WINDOW_MIN, DEFAULT_WINDOW_MAX))

    @property
    def explicit(self) -> dict[tuple[int, int], tuple[float, float]]:
        """The configured entries, keyed by sorted atomic-number pair."""
        return dict(self._explicit)

    @classmethod
    def from_config(cls, config: "ConstraintConfig") -> "IntraWindowTable":
        """Build an :class:`IntraWindowTable` from a run config's
        ``constraints.intramolecular`` list (see
        :class:`raymc.config.IntramolecularWindowConfig`)."""
        import periodictable as pt

        windows = {
            (
                pt.elements.symbol(entry.elements[0]).number,
                pt.elements.symbol(entry.elements[1]).number,
            ): (entry.min, entry.max)
            for entry in config.intramolecular
        }
        return cls(windows=windows)
