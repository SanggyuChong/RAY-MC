"""Shared real-space pair-histogram machinery.

Both structural modalities are built on the same primitive: bin every
pair of atoms within a cutoff into a radial histogram, then normalize.
The bins need not be equally wide: the PDF's grid is whatever the
reference data file provides, and experimental data is routinely sampled
more finely across a peak than in the tails. :mod:`raymc.modalities.pdf.calculator` reads that histogram in
real space as g(r)/G(r); :mod:`raymc.modalities.sq.calculator` sine-
transforms it into S(Q). Keeping one implementation here means the two
modalities can never disagree about what a bin *is* -- which pairs land
in it, how the neighbor search is bounded, or how a moved atom's
contribution is added and removed again.

Nothing in here knows about either observable. It is pure binning,
pair-finding, and the geometric safety check that bounds both.
"""

from __future__ import annotations

import torch
from nvalchemi.data import Batch
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.neighbors import compute_neighbors

from raymc.system import AtomicSystem


def edges_from_centers(centers: torch.Tensor) -> torch.Tensor:
    """Bin edges bracketing *centers*, for a grid that may be unevenly spaced.

    Interior edges are the midpoints between consecutive centers; the two
    outer edges are extrapolated by half the adjacent spacing, so every
    center sits inside its own bin and the bins tile without gaps. For an
    evenly-spaced grid this reduces exactly to ``center -/+ dr/2``.

    The reference data owns the grid (see
    :func:`raymc.residuals.aggregator.make_partial_pdf_term`), and
    experimental files are routinely sampled more finely across a peak
    than in the tails -- so nothing here may assume a constant ``dr``.

    Parameters
    ----------
    centers : torch.Tensor
        ``[n_bins]`` strictly increasing bin centers (Angstrom).

    Returns
    -------
    torch.Tensor
        ``[n_bins + 1]`` edges.
    """
    c = centers.to(torch.float64)
    if c.numel() == 0:
        raise ValueError("centers is empty")
    if c.numel() == 1:
        raise ValueError(
            "a single bin center gives no spacing to derive edges from; "
            "supply at least two reference points"
        )
    if not bool((torch.diff(c) > 0).all()):
        raise ValueError("centers must be strictly increasing")
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return torch.cat([first.reshape(1), mid, last.reshape(1)])


def bin_widths(edges: torch.Tensor) -> torch.Tensor:
    """Per-bin width ``dr`` from *edges* -- a vector, not a scalar, since
    the grid may be unevenly spaced."""
    return torch.diff(edges.to(torch.float64))


def bin_counts(distances: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Histogram *distances* into the bins delimited by *edges*.

    Distances outside ``[edges[0], edges[-1])`` are dropped. Plain counts
    (weight 1 per distance); see :func:`weighted_bin_counts` for the
    weighted variant. Bins need not be equally wide -- indexing goes
    through :func:`torch.bucketize` rather than arithmetic on a constant
    ``dr``.
    """
    return weighted_bin_counts(distances, None, edges)


def weighted_bin_counts(
    distances: torch.Tensor, weights: torch.Tensor | None, edges: torch.Tensor
) -> torch.Tensor:
    """Same as :func:`bin_counts`, but each distance contributes its own
    *weights* entry instead of a flat 1 -- used by the PDF total mode's
    scattering-weighted histogram. ``weights=None`` means flat 1.
    """
    edges = edges.to(torch.float64)
    n_bins = edges.numel() - 1
    zeros = torch.zeros(n_bins, dtype=torch.float64)
    if distances.numel() == 0:
        return zeros

    d = distances.to(torch.float64)
    in_range = (d >= edges[0]) & (d < edges[-1])
    d = d[in_range]
    if d.numel() == 0:
        return zeros
    w = (
        torch.ones_like(d)
        if weights is None
        else weights.to(torch.float64)[in_range]
    )

    # bucketize returns 1..n_bins for values inside the edges; -1 maps
    # those onto 0-based bin indices.
    idx = (torch.bucketize(d, edges.to(d.device), right=False) - 1).clamp(0, n_bins - 1)
    counts = torch.zeros(n_bins, dtype=torch.float64, device=d.device)
    counts.scatter_add_(0, idx, w)
    return counts.to(zeros.device)


def pairs_within_cutoff(
    system: AtomicSystem, cutoff: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All atom pairs within *cutoff*, with distances.

    Returns ``(src, dst, distance)`` with each unordered pair appearing
    exactly once (``src < dst``).
    """
    batch = Batch.from_data_list([system], device=system.device)
    compute_neighbors(batch, cutoff=cutoff, format=NeighborListFormat.COO)
    edges = batch.neighbor_list
    empty = torch.empty(0, dtype=torch.long, device=system.device)
    if edges.numel() == 0:
        return empty, empty, torch.empty(0, dtype=torch.float64, device=system.device)

    keep = edges[:, 0] < edges[:, 1]
    edges = edges[keep].long()
    shifts = getattr(batch, "neighbor_list_shifts", None)
    if shifts is not None:
        shifts = shifts[keep]

    positions = system.positions.to(torch.float64)
    src, dst = edges[:, 0], edges[:, 1]
    delta = positions[dst] - positions[src]
    if shifts is not None:
        cell = system.cell.squeeze(0).to(torch.float64)
        delta = delta + shifts.to(torch.float64) @ cell
    dist = torch.linalg.norm(delta, dim=-1)
    return src, dst, dist


def check_cutoff_safety(system: AtomicSystem, search_cutoff: float) -> None:
    """Raise if *search_cutoff* is too large for the box.

    Pair search reuses ALCHEMI's neighbor list, which is built for short
    MLIP cutoffs, not for a PDF's or structure factor's much larger
    r_max: past a certain box size it can silently miss pairs at large
    cutoffs. Callers check this up front so a modality raises a clear
    error rather than ever returning a silently-wrong observable.
    """
    cell = system.cell.squeeze(0).to(torch.float64)
    min_width = float(min_perpendicular_width(cell))
    if 2 * search_cutoff > min_width:
        raise ValueError(
            f"search cutoff ({search_cutoff:.2f} A) is too large for this "
            f"box: the box is only {min_width:.2f} A wide at its "
            f"narrowest, so anything past {min_width / 2:.2f} A can't be "
            f"found reliably. Use a smaller r_max or a bigger simulation cell."
        )


def min_perpendicular_width(cell: torch.Tensor) -> torch.Tensor:
    """The box's narrowest width, measured straight across (not diagonally).

    A pair search only gives an unambiguous answer if the cutoff is under
    half of this -- otherwise the same neighbor could be found twice,
    through two different sides of the box.
    """
    a, b, c = cell[0], cell[1], cell[2]
    volume = torch.det(cell).abs()
    width_a = volume / torch.linalg.norm(torch.linalg.cross(b, c))
    width_b = volume / torch.linalg.norm(torch.linalg.cross(c, a))
    width_c = volume / torch.linalg.norm(torch.linalg.cross(a, b))
    return torch.min(torch.stack([width_a, width_b, width_c]))


def find_slots(full_index_of_slot: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """For each index in *query* (a full-system atom index), find its slot
    in *full_index_of_slot* (species-local order). Every entry of *query*
    must actually be present."""
    if query.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    matches = full_index_of_slot.unsqueeze(0) == query.unsqueeze(1)  # [Q, n_species]
    return matches.long().argmax(dim=1)
