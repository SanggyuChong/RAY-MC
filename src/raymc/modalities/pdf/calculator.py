"""Real-space PDF calculator: total weighted G(r), or partial g_AB(r).

Bins pairs of atoms into a real-space distance histogram -- the classic
RMCProfile approach (Tucker et al. 2007), not a smooth kernel per pair.
For every pair of atoms within [r_min, r_max), increment the histogram
bin its distance falls into; the resulting bin counts, normalized by
shell volume and pair count, are G(r)/g(r) directly. No Gaussian
broadening is applied here -- a real experimental PDF's peak widths come
from genuine ensemble/thermal disorder (many configurations, real atomic
motion), which a single frozen RMC configuration doesn't have on its own.
Established RMC packages (RMCProfile, fullrmc) get realistic peak widths
by averaging this histogram over many accepted configurations along a
run, and treat thermal/instrumental-resolution broadening as separate,
explicit correction layers on top of the raw histogram -- not by lumping
everything into one ad hoc smoothing width applied to a single snapshot,
which is what an earlier version of this module did (see git history)
and which cannot simultaneously get a sharp first-shell peak and a
broader second-shell peak right with one shared width.

Trade-off: this makes ``compute()`` **not differentiable** in atom
positions (hard bin-edge assignment has zero gradient almost everywhere)
-- unlike the previous Gaussian-kernel version. If/when a gradient-based
generative solver needs a differentiable PDF, that will need its own
smooth approximation; this module no longer provides one.

One class, two modes, picked by whether an element is given:

- **Total** (no ``atomic_number_a``): the weighted, all-elements reduced
  PDF G(r) (Egami & Billinge convention), converging to 0 at large r.
  Requires ``radiation``.
- **Partial** (``atomic_number_a`` given): the plain pair-correlation
  g_AB(r), converging to 1 at large r -- the convention used by
  experimental reference data. One element (``atomic_number_b`` omitted
  or equal to ``atomic_number_a``) gives a self partial (e.g. g_OO); two
  distinct elements give a cross partial (e.g. g_OH).

Incremental support (:meth:`PDFCalculator.compute_increm_change`) exists
only for the self-partial case -- total and cross-partial always fall
back to :class:`~raymc.modalities.base.Calculator`'s default (a full
:meth:`compute` every time). The incremental path updates the same
histogram bin-count array a full recompute would produce, just by
adding/removing only the touched atoms' bin contributions instead of
rebinning every pair.

Cutoff safety: pair search reuses ALCHEMI's neighbor list, which is built
for short MLIP cutoffs, not a PDF's much larger r_max. We checked its
source directly -- past a certain box size it can silently miss pairs at
large cutoffs -- so :meth:`compute` calls
:func:`raymc.modalities.histogram.check_cutoff_safety` and raises a clear
error instead of ever returning a silently-wrong PDF.

The binning, pair-search and cutoff-safety primitives themselves live in
:mod:`raymc.modalities.histogram`, shared with
:mod:`raymc.modalities.sq.calculator` so the two modalities can never
disagree about what a histogram bin contains. They are imported here
under their original underscore-prefixed names.
"""

from __future__ import annotations

import math

import torch

from raymc.modalities.base import Calculator
from raymc.modalities.histogram import bin_counts as _bin_counts
from raymc.modalities.histogram import bin_widths as _bin_widths
from raymc.modalities.histogram import edges_from_centers as _edges_from_centers
from raymc.modalities.histogram import check_cutoff_safety as _check_cutoff_safety
from raymc.modalities.histogram import find_slots as _find_slots
from raymc.modalities.histogram import min_perpendicular_width as _min_perpendicular_width
from raymc.modalities.histogram import pairs_within_cutoff as _pairs_within_cutoff
from raymc.modalities.histogram import weighted_bin_counts as _weighted_bin_counts
from raymc.modalities.pdf.scattering_weights import Radiation, atomic_scattering_weights
from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem
from raymc.utils import minimum_image_delta


class PDFCalculator(Calculator):
    """Total G(r) or partial g_AB(r) -- see the module docstring for the
    two modes and the histogram-vs-Gaussian design note.

    Parameters
    ----------
    r_grid : torch.Tensor
        ``[n_bins]`` strictly increasing histogram bin centers (Angstrom),
        all > 0. This is the REFERENCE DATA's own grid -- the model is
        evaluated where the data was measured, never the other way round,
        so no interpolation is applied to experimental values. The grid
        may be unevenly spaced; bin edges are derived by
        :func:`~raymc.modalities.histogram.edges_from_centers` and widths
        are per-bin.
    radiation : Literal["neutron", "xray"] | None
        Required for total mode; must be omitted for partial mode.
    atomic_number_a : int | None
        Omitted for total mode. Given (with `atomic_number_b` optional)
        for partial mode.
    atomic_number_b : int | None
        Partial mode only. ``None`` (default) means the self partial
        g_AA(r); a distinct element gives the cross partial g_AB(r).
    """

    def __init__(
        self,
        r_grid: torch.Tensor,
        radiation: Radiation | None = None,
        atomic_number_a: int | None = None,
        atomic_number_b: int | None = None,
    ) -> None:
        r_grid = torch.as_tensor(r_grid, dtype=torch.float64).flatten()
        if r_grid.numel() < 2:
            raise ValueError(
                "r_grid needs at least two bin centers (it comes from the "
                f"reference data file); got {r_grid.numel()}"
            )
        if bool((r_grid <= 0).any()):
            raise ValueError("every r_grid bin center must be > 0")

        self.is_partial = atomic_number_a is not None
        if self.is_partial:
            if radiation is not None:
                raise ValueError("radiation has no meaning in partial mode")
            self.atomic_number_a = atomic_number_a
            self.atomic_number_b = (
                atomic_number_b if atomic_number_b is not None else atomic_number_a
            )
        else:
            if radiation is None:
                raise ValueError("total mode (no atomic_number_a) requires radiation")
            self.radiation = radiation

        self.r_grid = r_grid
        self.edges = _edges_from_centers(r_grid)
        self.dr = _bin_widths(self.edges)      # [n_bins], NOT a scalar
        self.r_min = float(self.edges[0])
        self.r_max = float(self.edges[-1])
        self.n_points = int(r_grid.numel())
        if self.r_min <= 0:
            raise ValueError(
                f"the lowest bin edge came out at {self.r_min:.4f} A; the "
                "reference grid starts too close to r = 0"
            )

        # -- incremental cache: only ever populated/used for the
        # self-partial case; every other mode leaves these untouched and
        # relies on Calculator's default compute_increm_change/init.
        self._cell: torch.Tensor | None = None
        self._pbc: torch.Tensor | None = None
        self._volume: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._full_index_of_slot: torch.Tensor | None = None
        self._n_species = 0
        self._pending: dict | None = None

    @property
    def is_self_partial(self) -> bool:
        return self.is_partial and self.atomic_number_a == self.atomic_number_b

    # -- compute(): full, stateless, from-scratch histogram -----------

    def compute(self, system: AtomicSystem) -> torch.Tensor:
        """Bin every pair in *system* into the histogram and return
        G(r)/g(r) on :attr:`r_grid`.

        Raises
        ------
        ValueError
            If *system* isn't periodic, or the cutoff is too large for
            the box (see the module docstring's "Cutoff safety" note).
        """
        if system.cell is None or system.pbc is None:
            raise ValueError("PDFCalculator requires a periodic system (cell + pbc)")
        return self._compute_partial(system) if self.is_partial else self._compute_total(system)

    def _compute_total(self, system: AtomicSystem) -> torch.Tensor:
        cell = system.cell.squeeze(0).to(torch.float64)
        _check_cutoff_safety(system, self.r_max)

        n = system.num_nodes
        r_grid = self.r_grid.to(system.device)
        volume = torch.det(cell).abs()
        rho0 = n / volume
        baseline = -4 * math.pi * r_grid * rho0

        if n < 2:
            return baseline

        weights = atomic_scattering_weights(system.atomic_numbers, self.radiation).to(
            device=system.device, dtype=torch.float64
        )
        mean_weight = weights.mean()

        src, dst, r_ij = _pairs_within_cutoff(system, self.r_max)
        if src.numel() == 0:
            return baseline

        pair_weight = (weights[src] * weights[dst]) / (mean_weight**2)
        weighted_hist = _weighted_bin_counts(r_ij, pair_weight, self.edges)
        summed = weighted_hist / self.dr

        g_of_r = (2.0 / (n * r_grid)) * summed
        return g_of_r + baseline

    def _compute_partial(self, system: AtomicSystem) -> torch.Tensor:
        _check_cutoff_safety(system, self.r_max)
        r_grid = self.r_grid.to(system.device)
        cell = system.cell.squeeze(0).to(torch.float64)
        volume = torch.det(cell).abs()

        mask_a = system.atomic_numbers == self.atomic_number_a
        n_a = int(mask_a.sum().item())

        if self.is_self_partial:
            if n_a < 2:
                return torch.ones_like(r_grid)
            idx = mask_a.nonzero(as_tuple=True)[0]
            sub_system = AtomicSystem(
                positions=system.positions[idx],
                atomic_numbers=system.atomic_numbers[idx],
                cell=system.cell,
                pbc=system.pbc,
            )
            _, _, r_ij = _pairs_within_cutoff(sub_system, self.r_max)
            if r_ij.numel() == 0:
                return torch.ones_like(r_grid)
            hist = _bin_counts(r_ij, self.edges) / self.dr
            return volume * hist / (2 * math.pi * n_a * (n_a - 1) * r_grid**2)

        mask_b = system.atomic_numbers == self.atomic_number_b
        n_b = int(mask_b.sum().item())
        if n_a < 1 or n_b < 1:
            return torch.ones_like(r_grid)

        idx_ab = (mask_a | mask_b).nonzero(as_tuple=True)[0]
        sub_system = AtomicSystem(
            positions=system.positions[idx_ab],
            atomic_numbers=system.atomic_numbers[idx_ab],
            cell=system.cell,
            pbc=system.pbc,
        )
        src, dst, r_ij = _pairs_within_cutoff(sub_system, self.r_max)
        if src.numel() == 0:
            return torch.ones_like(r_grid)

        sub_is_a = sub_system.atomic_numbers == self.atomic_number_a
        cross = sub_is_a[src] != sub_is_a[dst]  # keep only A-B pairs, drop A-A/B-B
        r_ab = r_ij[cross]
        if r_ab.numel() == 0:
            return torch.ones_like(r_grid)

        hist = _bin_counts(r_ab, self.edges) / self.dr
        # Every A-B pair is distinct by construction (no i<j double-count
        # to correct for, unlike the self-partial case above), so the
        # normalization uses the plain product n_a * n_b.
        return volume * hist / (4 * math.pi * n_a * n_b * r_grid**2)

    # -- incremental path: self-partial mode only ----------------------
    #
    # Classic RMC (McGreevy) trick: keep a running histogram between
    # steps, and on a proposed move, only remove the old-distance bin
    # contributions and add new-distance ones for the pairs that
    # actually involve a touched atom -- "rebin the touched atom's
    # pairs" instead of "rebin every pair," same final counts, far less
    # repeated work. Species membership of a moved/removed atom is
    # checked against self._full_index_of_slot (this calculator's own
    # cached state as of the last commit) rather than needing the old
    # system passed in -- a move never changes an atom's element, so
    # that's all we need.

    def init(self, system: AtomicSystem) -> torch.Tensor:
        if not self.is_self_partial:
            return self.compute(system)

        if system.cell is None or system.pbc is None:
            raise ValueError("PDFCalculator requires a periodic system (cell + pbc)")
        _check_cutoff_safety(system, self.r_max)

        self._cell = system.cell.squeeze(0).to(torch.float64)
        self._pbc = system.pbc.squeeze(0)
        self._volume = torch.det(self._cell).abs()

        mask = system.atomic_numbers == self.atomic_number_a
        self._full_index_of_slot = mask.nonzero(as_tuple=True)[0]
        self._positions = system.positions[self._full_index_of_slot].to(torch.float64)
        self._n_species = int(self._full_index_of_slot.numel())

        self._counts = self._full_bin_counts(self._positions)
        return self._normalize(self._counts, self._n_species)

    def _pair_distances(self, positions: torch.Tensor) -> torch.Tensor:
        n = positions.shape[0]
        if n < 2:
            return torch.empty(0, dtype=torch.float64)
        i, j = torch.triu_indices(n, n, offset=1)
        delta = minimum_image_delta(positions[i], positions[j], self._cell, self._pbc)
        return torch.linalg.norm(delta, dim=-1)

    def _full_bin_counts(self, positions: torch.Tensor) -> torch.Tensor:
        dist = self._pair_distances(positions)
        dist = dist[dist < self.r_max]
        return _bin_counts(dist, self.edges)

    def _normalize(self, counts: torch.Tensor, n_species: int) -> torch.Tensor:
        if n_species < 2 or bool(counts.abs().max() < 1e-12):
            return torch.ones_like(self.r_grid)
        return (
            self._volume
            * (counts / self.dr)
            / (2 * math.pi * n_species * (n_species - 1) * self.r_grid**2)
        )

    def compute_increm_change(
        self, system: AtomicSystem, delta: MoveDelta
    ) -> torch.Tensor:
        if not self.is_self_partial:
            return self.compute(system)

        counts = self._counts.clone()
        n_species = self._n_species
        new_positions = self._positions.clone()
        new_full_index_of_slot = self._full_index_of_slot.clone()

        # -- moved atoms (translate/rotate): same identity, new position --
        moved = delta.moved_indices
        moved = moved[torch.isin(moved, self._full_index_of_slot)]
        if moved.numel() > 0:
            slots = _find_slots(self._full_index_of_slot, moved)
            old_pos = self._positions[slots]
            new_pos = system.positions[moved].to(torch.float64)

            other_mask = torch.ones(self._n_species, dtype=torch.bool)
            other_mask[slots] = False
            other_slots = other_mask.nonzero(as_tuple=True)[0]
            other_pos = self._positions[other_slots]

            if other_slots.numel() > 0:
                old_cross = minimum_image_delta(
                    old_pos.unsqueeze(1), other_pos.unsqueeze(0), self._cell, self._pbc
                )
                new_cross = minimum_image_delta(
                    new_pos.unsqueeze(1), other_pos.unsqueeze(0), self._cell, self._pbc
                )
                d_old = torch.linalg.norm(old_cross, dim=-1).reshape(-1)
                d_new = torch.linalg.norm(new_cross, dim=-1).reshape(-1)
                counts = counts - _bin_counts(
                    d_old[d_old < self.r_max], self.edges
                )
                counts = counts + _bin_counts(
                    d_new[d_new < self.r_max], self.edges
                )

            if moved.numel() > 1:
                d_old_int = self._pair_distances(old_pos)
                d_new_int = self._pair_distances(new_pos)
                counts = counts - _bin_counts(
                    d_old_int[d_old_int < self.r_max], self.edges
                )
                counts = counts + _bin_counts(
                    d_new_int[d_new_int < self.r_max], self.edges
                )

            new_positions[slots] = new_pos

        # -- removed atoms --
        removed_full = delta.removed_indices
        removed = removed_full[torch.isin(removed_full, self._full_index_of_slot)]
        if removed.numel() > 0:
            slots = _find_slots(self._full_index_of_slot, removed)
            removed_pos = self._positions[slots]

            other_mask = torch.ones(self._n_species, dtype=torch.bool)
            other_mask[slots] = False
            other_slots = other_mask.nonzero(as_tuple=True)[0]
            other_pos = self._positions[other_slots]

            if other_slots.numel() > 0:
                cross = minimum_image_delta(
                    removed_pos.unsqueeze(1), other_pos.unsqueeze(0), self._cell, self._pbc
                )
                d_old = torch.linalg.norm(cross, dim=-1).reshape(-1)
                counts = counts - _bin_counts(
                    d_old[d_old < self.r_max], self.edges
                )
            if removed_pos.shape[0] > 1:
                d_old_int = self._pair_distances(removed_pos)
                counts = counts - _bin_counts(
                    d_old_int[d_old_int < self.r_max], self.edges
                )

            n_species -= int(slots.numel())
            new_positions = new_positions[other_slots]
            new_full_index_of_slot = new_full_index_of_slot[other_slots]

        if removed_full.numel() > 0:
            # RemoveGroup filters positions (not just masks them), so every
            # surviving atom after the removed block shifts down in the new
            # system's indexing -- species-blind, regardless of whether any
            # of this calculator's own species atoms were among those removed.
            sorted_removed, _ = torch.sort(removed_full)
            shift = torch.searchsorted(sorted_removed, new_full_index_of_slot, right=True)
            new_full_index_of_slot = new_full_index_of_slot - shift

        # -- added atoms (insert) --
        added = delta.added_indices
        if added.numel() > 0:
            species_mask = system.atomic_numbers[added] == self.atomic_number_a
            added = added[species_mask]
        if added.numel() > 0:
            added_pos = system.positions[added].to(torch.float64)

            if new_positions.shape[0] > 0:
                cross = minimum_image_delta(
                    added_pos.unsqueeze(1), new_positions.unsqueeze(0), self._cell, self._pbc
                )
                d_new = torch.linalg.norm(cross, dim=-1).reshape(-1)
                counts = counts + _bin_counts(
                    d_new[d_new < self.r_max], self.edges
                )
            if added_pos.shape[0] > 1:
                d_new_int = self._pair_distances(added_pos)
                counts = counts + _bin_counts(
                    d_new_int[d_new_int < self.r_max], self.edges
                )

            n_species += int(added.numel())
            new_positions = torch.cat([new_positions, added_pos], dim=0)
            new_full_index_of_slot = torch.cat([new_full_index_of_slot, added])

        self._pending = {
            "counts": counts,
            "n_species": n_species,
            "positions": new_positions,
            "full_index_of_slot": new_full_index_of_slot,
        }
        return self._normalize(counts, n_species)

    def commit(self) -> None:
        if self._pending is None:
            return
        self._counts = self._pending["counts"]
        self._n_species = self._pending["n_species"]
        self._positions = self._pending["positions"]
        self._full_index_of_slot = self._pending["full_index_of_slot"]
        self._pending = None

    def discard(self) -> None:
        self._pending = None
