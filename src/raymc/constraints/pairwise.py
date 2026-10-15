"""Atom-pair machinery shared by the distance constraints.

Both *inter*-group distance constraints score the same set of pairs --
every touched atom against every (relevant) atom in the system, minus the
pairs that are meaningless to check -- and differ only in which threshold
they compare the distance against. That enumeration lives here so the two
constraints can't drift apart on the details that are easy to get
subtly wrong (minimum-image convention, the same-rigid-group exclusion,
and double counting).

Excluded pairs:

- **same rigid group**: those atoms are bonded, and a table of
  *non-bonded* approach distances has nothing to say about them (see
  :mod:`raymc.constraints.min_dist`). The complementary case -- scoring
  *only* the same-group pairs, against a bonded [min, max] window -- is
  :mod:`raymc.constraints.intramolecular`, which is why this module
  hands the two masks back separately (see
  :func:`touched_pair_geometry`) rather than only their union;
- **an atom against itself**;
- **the mirror of a pair where both atoms were touched**: pair ``(i, j)``
  appears in row *i* and again in row *j*, and the constraints sum over
  pairs, so one of the two copies has to go.

The last two collapse into a single ``j <= i, j touched`` test -- see
:func:`touched_pair_geometry`. Kept to a minimum number of tensor ops on
purpose: this runs on every proposed move, over a
``[touched, partners]`` matrix, and at these sizes per-op dispatch
overhead is most of the cost.
"""

from __future__ import annotations

import torch

from raymc.system import AtomicSystem, rigid_groups
from raymc.utils import minimum_image_delta

DEFAULT_SOFTENING_WINDOW = 1e-1  # Angstrom
"""Default *window* for the distance constraints: a pair violating its
threshold by this much costs
:data:`~raymc.constraints.penalty.PENALTY_AT_WINDOW`.

Lives here rather than with the penalty itself because it is the one
parameter of that penalty carrying units -- 0.1 means 0.1 Angstrom, and
means nothing to a constraint measuring something else. See
:mod:`raymc.constraints.penalty`."""


def touched_pair_geometry(
    system: AtomicSystem, touched: torch.Tensor, partners: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distances from every atom in *touched* to every atom in *partners*,
    with the two exclusion masks kept separate.

    The separation is what lets one enumeration serve both directions:
    the inter-group constraints skip *same_group* pairs, the
    intramolecular one keeps only those, and both skip *redundant*.

    Parameters
    ----------
    system : AtomicSystem
    touched : torch.Tensor
        Atom indices the move affected (moved + added).
    partners : torch.Tensor
        Atom indices to check *touched* against -- ``arange(n)`` for a
        constraint that has to consider every atom, or a subset for one
        that can rule most of the system out up front (see
        :class:`raymc.constraints.max_dist.MaxDistConstraint` and
        :class:`raymc.constraints.intramolecular.IntramolecularConstraint`).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(dist, same_group, redundant)``, all
        ``[len(touched), len(partners)]``: the minimum-image pair
        distances, a boolean mask of the pairs whose two atoms share a
        rigid group, and a boolean mask of the self-pairs and
        double-counted mirrors (see the module docstring).
    """
    cell = system.cell.squeeze(0).to(torch.float64)
    pbc = system.pbc.squeeze(0)
    positions = system.positions.to(torch.float64)

    pair_delta = minimum_image_delta(
        positions[touched].unsqueeze(1), positions[partners].unsqueeze(0), cell, pbc
    )
    dist = torch.linalg.norm(pair_delta, dim=-1)

    groups = rigid_groups(system)
    is_touched = torch.zeros(positions.shape[0], dtype=torch.bool)
    is_touched[touched] = True

    same_group = groups[touched].unsqueeze(1) == groups[partners].unsqueeze(0)
    # One mask covers both remaining exclusions: for a touched atom i, drop
    # partner j whenever j is itself touched and j <= i. j == i is the
    # self-pair (i is touched by definition); j < i is the mirrored copy of
    # a pair whose other end will score it as (j, i).
    redundant = is_touched[partners].unsqueeze(0) & (
        partners.unsqueeze(0) <= touched.unsqueeze(1)
    )
    return dist, same_group, redundant


def touched_pair_distances(
    system: AtomicSystem, touched: torch.Tensor, partners: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distances from every atom in *touched* to every atom in *partners*,
    with one combined "do not score this pair" mask -- the form the
    inter-group distance constraints want.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(dist, skip)``, both ``[len(touched), len(partners)]``: the
        minimum-image pair distances, and a boolean mask marking the
        pairs that must not be scored (see the module docstring).
    """
    dist, same_group, redundant = touched_pair_geometry(system, touched, partners)
    return dist, same_group | redundant
