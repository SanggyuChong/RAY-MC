"""Reciprocal-space total structure factor S(Q), RMCA-style.

Implements RMCA's Eq. 10 fit target: the configuration's partial pair
correlations are sine-transformed to partial structure factors, then
combined with Faber-Ziman weights into the measured total (RMCA
Sections 3, 9.7)::

    A_ab(Q) - 1 = (4 pi rho / Q) INT_0^rmax r [g_ab(r) - 1] W(r) sin(Qr) dr
    S(Q)        = 1 + SUM_{a<=b} w_ab(Q) [A_ab(Q) - 1]

with W(r) the truncation window (:mod:`raymc.modalities.sq.window`) and
w_ab(Q) the Q-dependent Faber-Ziman weights
(:mod:`raymc.modalities.sq.scattering`).

Two forward models, selectable per modality
-------------------------------------------
Both evaluate the *same* expression; they differ in one respect only --
whether the pair term is binned before being transformed. Writing the
integral out on the histogram and cancelling the shell-volume
normalization against the transform's own r factor leaves::

    A_ab(Q) - 1 = (N / (P_ab * Q)) * SUM_pairs W(r) sin(Q r) / r  -  B(Q)

where ``P_ab`` is the ideal number of unordered a-b pairs and ``B(Q)``
is the transform of the uniform ``-1`` background.

- ``"histogram_ft"`` (default, what RMCA does): the pair sum is taken
  over histogram bins, ``SUM_k W(r_k) sin(Q r_k) C_ab[k] / r_k``. Because
  the bin kernel ``K[q,k] = W(r_k) sin(Q_q r_k) / r_k`` depends only on
  the r and Q grids, it is built once at construction and every step is
  a single matvec -- and an accepted move only touches a handful of
  bins, so the incremental path multiplies just those columns.
- ``"debye"``: the same sum taken over individual pair distances,
  ``SUM_ij W(r_ij) sin(Q r_ij) / r_ij``, with no binning at all. Exact
  with respect to the pair distances but O(n_pairs * n_Q) every step.

So the difference between the two is *purely* the binning error, which
is what makes the Debye path useful: it is the reference the histogram
path can be validated against, on identical physics and identical
normalization.

Two modes, as in the PDF modality
---------------------------------
- **Total** (no ``atomic_number_a``): the weighted S(Q) actually
  measured, requiring ``radiation``.
- **Partial** (``atomic_number_a`` given): the unweighted partial
  A_ab(Q) on its own. RMCA reaches a partial by zeroing every
  coefficient but one; here it is an element selector instead, matching
  :class:`~raymc.modalities.pdf.calculator.PDFCalculator`'s
  ``atomic_number_a``/``atomic_number_b`` and keeping the weighting
  surface free of a per-pair coefficient array.

Conventions
-----------
- The r integration necessarily starts at **r = 0**, not at a fitting
  r_min. Below contact g(r) = 0, so that region still contributes
  ``-INT r W(r) sin(Qr) dr`` through the background term; starting the
  grid anywhere above zero would silently drop it. The histogram here
  therefore always spans ``[0, r_max)``, unlike the PDF modality's
  ``[r_min, r_max)`` fitting window.
- Partial normalization matches
  :class:`~raymc.modalities.pdf.calculator.PDFCalculator` exactly
  (``N_a(N_a-1)/2`` ideal self pairs, ``N_a N_b`` cross pairs), so the
  two modalities describe the same structure.
- ``convention="S"`` returns S(Q) -> 1 at large Q (what the files in
  ``data/`` hold); ``convention="F"`` returns F(Q) = S(Q) - 1 -> 0,
  RMCA Eq. 15's own convention.

Incremental support covers moves that only *displace* atoms (translate,
rotate) -- the overwhelmingly common case. A move that inserts or
removes atoms changes N, every concentration, and hence every weight and
prefactor at once, so it falls back to a full recompute rather than
pretending a delta is cheap.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import torch

from raymc.modalities.base import Calculator
from raymc.modalities.histogram import check_cutoff_safety, pairs_within_cutoff
from raymc.modalities.pdf.scattering_weights import Radiation
from raymc.modalities.sq.scattering import (
    faber_ziman_weights,
    species_form_factors,
    unordered_species_pairs,
)
from raymc.modalities.sq.window import WindowKind, window_function
from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem
from raymc.utils import minimum_image_delta

ForwardModel = Literal["histogram_ft", "debye"]

# Pair-chunk size for the Debye path, so the [n_pairs, n_Q] sin() block
# never has to exist all at once for a large configuration.
_DEBYE_CHUNK = 4096


class SQCalculator(Calculator):
    """Total structure factor S(Q) on a fixed Q grid -- see the module
    docstring for the two forward models and the conventions used.

    Parameters
    ----------
    q_grid : torch.Tensor
        ``[n_Q]`` momentum transfer values (inverse Angstrom). Normally
        the experimental data's own Q points, unmodified: unlike the PDF
        modality (whose r grid is fixed by the histogram, so the
        reference is interpolated onto it), nothing here forces a
        particular Q grid, so the data's grid is adopted and the model is
        evaluated on it. That is also what RMCA does, via its nx1/nx2
        index window into the data file.
    r_max : float
        Upper limit of the r integration (Angstrom). Must not exceed half
        the box's narrowest width; validated on every
        :meth:`compute` via
        :func:`raymc.modalities.histogram.check_cutoff_safety`.
    n_r : int
        Number of equal-width histogram bins spanning ``[0, r_max)``.
        RMCA Section 9.4 recommends about five points per ``2 pi / Q_max``,
        i.e. ``dr <= 2 pi / (5 Q_max)``; a warning is issued if the grid
        is coarser than that.
    radiation : Literal["neutron", "xray"]
    species_weights : Mapping[str, float] | None
        Per-species scattering-factor overrides -- the isotopic
        substitution knob. See
        :func:`raymc.modalities.sq.scattering.species_form_factors`.
    window : Literal["none", "lorch", "hann"]
    forward_model : Literal["histogram_ft", "debye"]
    convention : Literal["S", "F"]
    atomic_number_a : int | None
        Omitted for total mode. Given (with *atomic_number_b* optional)
        to fit an unweighted partial A_ab(Q) instead.
    atomic_number_b : int | None
        Partial mode only; ``None`` means the self partial A_aa(Q).
    """

    def __init__(
        self,
        q_grid: torch.Tensor,
        r_max: float,
        n_r: int = 500,
        radiation: Radiation | None = None,
        species_weights: Mapping[str, float] | None = None,
        window: WindowKind = "lorch",
        forward_model: ForwardModel = "histogram_ft",
        convention: Literal["S", "F"] = "S",
        atomic_number_a: int | None = None,
        atomic_number_b: int | None = None,
    ) -> None:
        if r_max <= 0:
            raise ValueError(f"r_max must be > 0, got {r_max}")
        if n_r <= 0:
            raise ValueError(f"n_r must be > 0, got {n_r}")
        q_grid = torch.as_tensor(q_grid, dtype=torch.float64).flatten()
        if q_grid.numel() == 0:
            raise ValueError("q_grid is empty")
        if bool((q_grid <= 0).any()):
            raise ValueError(
                "q_grid must be strictly positive: S(Q) is computed as a "
                "sine transform divided by Q, which is singular at Q = 0. "
                "Drop the Q = 0 point from the data."
            )
        if forward_model not in ("histogram_ft", "debye"):
            raise ValueError(
                f"Unknown forward_model: {forward_model!r} "
                f"(expected 'histogram_ft' or 'debye')"
            )
        if convention not in ("S", "F"):
            raise ValueError(
                f"Unknown convention: {convention!r} (expected 'S' or 'F')"
            )

        self.is_partial = atomic_number_a is not None
        if self.is_partial:
            if radiation is not None:
                raise ValueError("radiation has no meaning in partial mode")
            if species_weights:
                raise ValueError(
                    "species_weights has no meaning in partial mode: a partial "
                    "A_ab(Q) is unweighted by construction"
                )
            self.atomic_number_a = atomic_number_a
            self.atomic_number_b = (
                atomic_number_b if atomic_number_b is not None else atomic_number_a
            )
        else:
            if radiation is None:
                raise ValueError("total mode (no atomic_number_a) requires radiation")

        self.q_grid = q_grid
        self.r_max = float(r_max)
        self.n_r = int(n_r)
        self.dr = self.r_max / self.n_r
        # Bin centers over [0, r_max) -- the grid starts at zero on
        # purpose; see the module docstring's "Conventions".
        self.r_grid = self.dr * (torch.arange(self.n_r, dtype=torch.float64) + 0.5)
        self.radiation = radiation
        self.species_weights = dict(species_weights or {})
        self.window = window
        self.forward_model = forward_model
        self.convention = convention

        self._window_r = window_function(window, self.r_grid, self.r_max)

        dr_recommended = 2 * math.pi / (5 * float(q_grid.max()))
        if self.dr > dr_recommended:
            warnings.warn(
                f"r spacing dr = {self.dr:.4f} A is coarser than the "
                f"{dr_recommended:.4f} A that Q_max = {float(q_grid.max()):.2f} "
                f"A^-1 warrants (RMCA Section 9.4 asks for roughly five r "
                f"points per 2*pi/Q_max). The high-Q end of S(Q) will be "
                f"damped by the binning. Increase n_r.",
                stacklevel=2,
            )

        # K[q, k] = W(r_k) sin(Q_q r_k) / r_k -- composition-independent,
        # so it is built once and reused for every step and every pair
        # type (see the module docstring).
        self._kernel = (
            torch.sin(torch.outer(self.q_grid, self.r_grid))
            * (self._window_r / self.r_grid)
        )
        # Quadrature of INT_0^rmax r W(r) sin(Qr) dr, for the uniform
        # background. Sharing this quadrature with the pair term is what
        # makes S(Q) come out exactly 1 for a perfectly uniform g(r) = 1.
        # The window multiplies BOTH halves of INT r (g-1) W sin(Qr) dr, so
        # it belongs here as much as in the pair kernel above. Omitting it
        # from just this half breaks the shared quadrature and the S(Q) = 1
        # baseline drifts wildly at low Q -- which is what
        # test_uniform_density_still_transforms_to_one_under_a_window pins.
        self._baseline_kernel = self.dr * (
            torch.sin(torch.outer(self.q_grid, self.r_grid))
            * (self.r_grid * self._window_r)
        ).sum(dim=1)

        # -- incremental state (displacement-only moves; see docstring) --
        self._cell: torch.Tensor | None = None
        self._pbc: torch.Tensor | None = None
        self._volume: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._species: tuple[int, ...] = ()
        self._species_of_atom: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None
        self._pending: dict | None = None
        self._weights_cache: dict[tuple, torch.Tensor] = {}

    # -- composition bookkeeping --------------------------------------

    def _species_layout(
        self, system: AtomicSystem
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        """Sorted distinct atomic numbers, and each atom's index into them."""
        z = system.atomic_numbers.cpu()
        species = tuple(int(v) for v in torch.unique(z).tolist())
        lookup = torch.full((max(species) + 1,), -1, dtype=torch.long)
        for i, s in enumerate(species):
            lookup[s] = i
        return species, lookup[z.long()]

    def _pair_lookup(self, n_species: int) -> torch.Tensor:
        """``[n_species, n_species]`` map from a species-index pair to its
        row in the unordered pair list (symmetric)."""
        table = torch.zeros((n_species, n_species), dtype=torch.long)
        for row, (a, b) in enumerate(unordered_species_pairs(n_species)):
            table[a, b] = row
            table[b, a] = row
        return table

    def _pair_weights(
        self, species: tuple[int, ...], counts_by_species: tuple[int, ...]
    ) -> torch.Tensor:
        """Faber-Ziman w_ab(Q), memoized on (species, composition)."""
        key = (species, counts_by_species)
        if key not in self._weights_cache:
            total = sum(counts_by_species)
            concentrations = torch.tensor(
                [c / total for c in counts_by_species], dtype=torch.float64
            )
            form_factors = species_form_factors(
                species, self.q_grid, self.radiation, self.species_weights
            )
            self._weights_cache[key] = faber_ziman_weights(
                concentrations, form_factors
            )
        return self._weights_cache[key]

    # -- assembly ------------------------------------------------------

    def _ideal_pair_counts(self, counts_by_species: Sequence[int]) -> torch.Tensor:
        """``P_ab``: ideal number of unordered a-b pairs, matching
        :class:`~raymc.modalities.pdf.calculator.PDFCalculator`'s
        normalization exactly."""
        values = []
        for a, b in unordered_species_pairs(len(counts_by_species)):
            n_a, n_b = counts_by_species[a], counts_by_species[b]
            values.append(n_a * (n_a - 1) / 2.0 if a == b else float(n_a * n_b))
        return torch.tensor(values, dtype=torch.float64)

    def _assemble(
        self,
        pair_term: torch.Tensor,
        species: tuple[int, ...],
        counts_by_species: tuple[int, ...],
        volume: torch.Tensor,
    ) -> torch.Tensor:
        """Combine the per-pair-type sums into the weighted total.

        *pair_term* is ``[n_pairs, n_Q]``, holding
        ``SUM W(r) sin(Q r) / r`` for each unordered species pair --
        identical in meaning whether it came from the binned or the
        unbinned path.
        """
        n_atoms = sum(counts_by_species)
        rho = n_atoms / volume
        ideal_pairs = self._ideal_pair_counts(counts_by_species)

        background = 4 * math.pi * rho * self._baseline_kernel / self.q_grid  # [n_Q]

        # A_ab(Q) - 1, one row per unordered species pair. A pair type
        # with no ideal pairs at all (a single atom of that species)
        # contributes nothing rather than dividing by zero.
        safe_pairs = torch.where(
            ideal_pairs > 0, ideal_pairs, torch.ones_like(ideal_pairs)
        )
        partial = (
            n_atoms * pair_term / (safe_pairs.unsqueeze(1) * self.q_grid.unsqueeze(0))
            - background.unsqueeze(0)
        )
        partial = torch.where(
            (ideal_pairs > 0).unsqueeze(1), partial, torch.zeros_like(partial)
        )

        if self.is_partial:
            row = self._partial_row(species)
            total = partial[row]
        else:
            weights = self._pair_weights(species, counts_by_species)  # [n_pairs, n_Q]
            total = (weights * partial).sum(dim=0)
        return total + 1.0 if self.convention == "S" else total

    def _partial_row(self, species: tuple[int, ...]) -> int:
        """Row of the requested element pair in the unordered pair list."""
        for element in (self.atomic_number_a, self.atomic_number_b):
            if element not in species:
                import periodictable as pt

                raise ValueError(
                    f"partial S_ab(Q) was asked for element Z={element} "
                    f"({pt.elements[element].symbol}), which this system does "
                    f"not contain."
                )
        a = species.index(self.atomic_number_a)
        b = species.index(self.atomic_number_b)
        return int(self._pair_lookup(len(species))[a, b])

    # -- binned pair term ----------------------------------------------

    def _binned_counts(
        self, distances: torch.Tensor, pair_idx: torch.Tensor, n_pairs: int
    ) -> torch.Tensor:
        """Histogram *distances* into ``[n_pairs, n_r]`` counts, each
        distance routed to its own species-pair row."""
        d = distances.reshape(-1).to(torch.float64)
        p = pair_idx.reshape(-1)
        out = torch.zeros(n_pairs * self.n_r, dtype=torch.float64, device=d.device)
        keep = d < self.r_max
        d, p = d[keep], p[keep]
        if d.numel() == 0:
            return out.reshape(n_pairs, self.n_r)
        k = (d / self.dr).long().clamp(0, self.n_r - 1)
        flat = p.to(d.device) * self.n_r + k
        out.scatter_add_(0, flat, torch.ones_like(d))
        return out.reshape(n_pairs, self.n_r)

    def _pair_term_from_counts(self, counts: torch.Tensor) -> torch.Tensor:
        """``[n_pairs, n_Q]`` pair term from binned counts: one matvec per
        pair type against the shared bin kernel."""
        return counts @ self._kernel.T

    # -- unbinned (Debye) pair term ------------------------------------

    def _pair_term_direct(
        self, distances: torch.Tensor, pair_idx: torch.Tensor, n_pairs: int
    ) -> torch.Tensor:
        """``[n_pairs, n_Q]`` pair term evaluated straight from the pair
        distances, with no binning -- the reference path."""
        d = distances.reshape(-1).to(torch.float64)
        p = pair_idx.reshape(-1).to(d.device)
        keep = d < self.r_max
        d, p = d[keep], p[keep]
        out = torch.zeros(n_pairs, self.q_grid.numel(), dtype=torch.float64)
        if d.numel() == 0:
            return out

        q = self.q_grid.to(d.device)
        for start in range(0, d.numel(), _DEBYE_CHUNK):
            d_chunk = d[start : start + _DEBYE_CHUNK]
            p_chunk = p[start : start + _DEBYE_CHUNK]
            w_chunk = window_function(self.window, d_chunk, self.r_max)
            contrib = torch.sin(torch.outer(d_chunk, q)) * (
                w_chunk / d_chunk
            ).unsqueeze(1)
            out.index_add_(0, p_chunk.cpu(), contrib.cpu().to(torch.float64))
        return out

    # -- full computation ----------------------------------------------

    def compute(self, system: AtomicSystem) -> torch.Tensor:
        """S(Q) (or F(Q)) on :attr:`q_grid`, recomputed from scratch."""
        if system.cell is None or system.pbc is None:
            raise ValueError("SQCalculator requires a periodic system (cell + pbc)")
        check_cutoff_safety(system, self.r_max)

        cell = system.cell.squeeze(0).to(torch.float64)
        volume = torch.det(cell).abs()
        species, species_of_atom = self._species_layout(system)
        counts_by_species = tuple(
            int((species_of_atom == i).sum()) for i in range(len(species))
        )
        n_pairs = len(unordered_species_pairs(len(species)))
        self._warn_if_q_min_too_low(cell)

        src, dst, dist = pairs_within_cutoff(system, self.r_max)
        if src.numel() == 0:
            empty = torch.zeros(n_pairs, self.q_grid.numel(), dtype=torch.float64)
            return self._assemble(empty, species, counts_by_species, volume)

        lookup = self._pair_lookup(len(species))
        pair_idx = lookup[species_of_atom[src.cpu()], species_of_atom[dst.cpu()]]

        if self.forward_model == "debye":
            pair_term = self._pair_term_direct(dist, pair_idx, n_pairs)
        else:
            counts = self._binned_counts(dist, pair_idx, n_pairs)
            pair_term = self._pair_term_from_counts(counts)
        return self._assemble(pair_term, species, counts_by_species, volume)

    def _warn_if_q_min_too_low(self, cell: torch.Tensor) -> None:
        """RMCA Section 9.4: Q below 2*pi/L is not meaningful for a box of
        side L -- fitting there invites spurious density fluctuations of
        exactly that period."""
        side = float(torch.det(cell).abs() ** (1.0 / 3.0))
        q_min_meaningful = 2 * math.pi / side
        if float(self.q_grid.min()) < q_min_meaningful:
            warnings.warn(
                f"Q grid starts at {float(self.q_grid.min()):.3f} A^-1, below "
                f"the {q_min_meaningful:.3f} A^-1 minimum that a box of side "
                f"~{side:.1f} A can represent (RMCA Section 9.4: Q_min = "
                f"2*pi/L). Fitting below it can drive density fluctuations of "
                f"period L into the configuration. Raise q_min or enlarge the "
                f"cell.",
                stacklevel=3,
            )

    # -- incremental path (displacement-only moves) --------------------

    def init(self, system: AtomicSystem) -> torch.Tensor:
        if system.cell is None or system.pbc is None:
            raise ValueError("SQCalculator requires a periodic system (cell + pbc)")
        check_cutoff_safety(system, self.r_max)

        self._cell = system.cell.squeeze(0).to(torch.float64)
        self._pbc = system.pbc.squeeze(0)
        self._volume = torch.det(self._cell).abs()
        self._positions = system.positions.to(torch.float64).cpu()
        self._species, self._species_of_atom = self._species_layout(system)
        self._warn_if_q_min_too_low(self._cell)

        n_pairs = len(unordered_species_pairs(len(self._species)))
        src, dst, dist = pairs_within_cutoff(system, self.r_max)
        if src.numel() == 0:
            self._counts = torch.zeros(n_pairs, self.n_r, dtype=torch.float64)
        else:
            lookup = self._pair_lookup(len(self._species))
            pair_idx = lookup[
                self._species_of_atom[src.cpu()], self._species_of_atom[dst.cpu()]
            ]
            self._counts = self._binned_counts(dist.cpu(), pair_idx, n_pairs)

        self._pending = None
        return self._from_counts()

    def _counts_by_species(self) -> tuple[int, ...]:
        return tuple(
            int((self._species_of_atom == i).sum()) for i in range(len(self._species))
        )

    def _from_counts(self) -> torch.Tensor:
        pair_term = self._pair_term_from_counts(self._counts)
        return self._assemble(
            pair_term, self._species, self._counts_by_species(), self._volume
        )

    def compute_increm_change(
        self, system: AtomicSystem, delta: MoveDelta
    ) -> torch.Tensor:
        """Update the running histogram with only the moved atoms' pairs.

        Falls back to a full :meth:`compute` whenever the move changes the
        atom count (insert/remove), since that shifts every concentration,
        weight and prefactor at once -- there is no cheap delta to take.
        """
        if (
            self._counts is None
            or delta.added_indices.numel() > 0
            or delta.removed_indices.numel() > 0
        ):
            return self.compute(system)

        moved = delta.moved_indices.cpu()
        if moved.numel() == 0:
            self._pending = None
            return self._from_counts()

        n_pairs = self._counts.shape[0]
        lookup = self._pair_lookup(len(self._species))
        n_atoms = self._positions.shape[0]

        other_mask = torch.ones(n_atoms, dtype=torch.bool)
        other_mask[moved] = False
        others = other_mask.nonzero(as_tuple=True)[0]

        old_pos = self._positions[moved]
        new_pos = system.positions.to(torch.float64).cpu()[moved]
        counts = self._counts.clone()

        # -- moved-against-stationary pairs --
        if others.numel() > 0:
            other_pos = self._positions[others]
            pair_idx = lookup[
                self._species_of_atom[moved].unsqueeze(1),
                self._species_of_atom[others].unsqueeze(0),
            ]
            d_old = torch.linalg.norm(
                minimum_image_delta(
                    old_pos.unsqueeze(1), other_pos.unsqueeze(0), self._cell, self._pbc
                ),
                dim=-1,
            )
            d_new = torch.linalg.norm(
                minimum_image_delta(
                    new_pos.unsqueeze(1), other_pos.unsqueeze(0), self._cell, self._pbc
                ),
                dim=-1,
            )
            counts -= self._binned_counts(d_old, pair_idx, n_pairs)
            counts += self._binned_counts(d_new, pair_idx, n_pairs)

        # -- pairs among the moved atoms themselves --
        if moved.numel() > 1:
            i, j = torch.triu_indices(moved.numel(), moved.numel(), offset=1)
            pair_idx = lookup[
                self._species_of_atom[moved][i], self._species_of_atom[moved][j]
            ]
            d_old = torch.linalg.norm(
                minimum_image_delta(
                    old_pos[i], old_pos[j], self._cell, self._pbc
                ),
                dim=-1,
            )
            d_new = torch.linalg.norm(
                minimum_image_delta(
                    new_pos[i], new_pos[j], self._cell, self._pbc
                ),
                dim=-1,
            )
            counts -= self._binned_counts(d_old, pair_idx, n_pairs)
            counts += self._binned_counts(d_new, pair_idx, n_pairs)

        positions = self._positions.clone()
        positions[moved] = new_pos
        self._pending = {"counts": counts, "positions": positions}

        pair_term = self._pair_term_from_counts(counts)
        return self._assemble(
            pair_term, self._species, self._counts_by_species(), self._volume
        )

    def commit(self) -> None:
        if self._pending is None:
            return
        self._counts = self._pending["counts"]
        self._positions = self._pending["positions"]
        self._pending = None

    def discard(self) -> None:
        self._pending = None
