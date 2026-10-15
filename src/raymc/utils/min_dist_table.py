"""A user-configurable minimum-approach-distance table per element pair,
from all that can be identified from the initial geometry or initialization
setup.

"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from raymc.config import ConstraintConfig

DEFAULT_MIN_DIST = 1.0  # Angstrom
_MAX_ATOMIC_NUMBER = 118


class MinDistTable:
    """Minimum allowed distance (Angstrom) for a pair of atomic numbers.

    Parameters
    ----------
    pair_distances : dict[tuple[int, int], float] | None
        Minimum allowed distance for an unordered pair of atomic numbers,
        e.g. ``{(8, 8): 2.4}`` for O-O. Order within the tuple doesn't
        matter -- ``(8, 1)`` and ``(1, 8)`` are the same pair.
    default : float
        Distance used for any element pair not present in
        *pair_distances*. A pair that's actually looked up (via
        :meth:`lookup`) triggers a warning the first time it falls back to
        this default -- once per missing pair, not once per call.
    """

    def __init__(
        self,
        pair_distances: dict[tuple[int, int], float] | None = None,
        default: float = DEFAULT_MIN_DIST,
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
        self._warned: set[tuple[int, int]] = set()

    def max_distance(self) -> float:
        """The largest minimum-approach distance in this table, over both
        *default* and every explicit pair entry. Used to validate that an
        override table (e.g. a denser pre-equilibration diameter) is at
        least as strict everywhere as this one -- see
        :func:`raymc.initialize.build.build_initial_system`."""
        return float(self._table.max())

    def lookup(self, z1: "int | torch.Tensor", z2: "int | torch.Tensor") -> torch.Tensor:
        """Minimum allowed distance for atomic numbers *z1*/*z2*.

        Accepts either plain ints or broadcastable tensors of atomic
        numbers (e.g. ``lookup(touched_z.unsqueeze(1), all_z.unsqueeze(0))``
        for a full pairwise table in one call).
        """
        return self._table[z1, z2]

    def warn_about_missing_pairs(
        self, touched_z: torch.Tensor, all_z: torch.Tensor
    ) -> None:
        """Warn (once per pair) about any pair among *touched_z* x *all_z*
        that has no explicit entry and is about to use :attr:`default`."""
        for z1 in torch.unique(touched_z).tolist():
            for z2 in torch.unique(all_z).tolist():
                key = (min(z1, z2), max(z1, z2))
                if key in self._explicit or key in self._warned:
                    continue
                warnings.warn(
                    f"No min-dist defined for element pair (Z={key[0]}, "
                    f"Z={key[1]}); defaulting to {self.default} Å min-dist.",
                    stacklevel=3,
                )
                self._warned.add(key)

    @classmethod
    def from_config(cls, config: "ConstraintConfig") -> "MinDistTable":
        """Build a :class:`MinDistTable` from a run config's
        ``constraints.min_dist`` section (see
        :class:`raymc.config.ConstraintConfig`)."""
        import periodictable as pt

        pair_distances = {
            (
                pt.elements.symbol(pair.elements[0]).number,
                pt.elements.symbol(pair.elements[1]).number,
            ): pair.distance
            for pair in config.pairs
        }
        return cls(pair_distances=pair_distances, default=config.default)
