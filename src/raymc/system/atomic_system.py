"""RAY-MC's shared atomistic system object.

RAY-MC relies on :class:`~nvalchemi.data.AtomicData`. Additional
modularities add the required features for RAY-MC's RMC setup.
"""

from __future__ import annotations

import periodictable as pt
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.neighbors import compute_neighbors

from raymc.utils import MinDistTable, minimum_image_delta

RIGID_GROUP_KEY = "molecule_id"
"""Historical alias for :data:`raymc.topology.MOLECULE_ID_KEY`. The stored
key itself was renamed: nothing reads a saved system back in yet, so
there is no on-disk compatibility to preserve, and leaving two different
key strings alive would let a system carry both."""

BOOKKEEPING_KEYS = (RIGID_GROUP_KEY,)
"""Every per-atom property RAY-MC attaches to a system itself, as opposed
to the physical ones (positions, atomic numbers, masses) that
:class:`~nvalchemi.data.AtomicData`'s constructor already carries.

A move builds its candidate by constructing a *new* AtomicData, which
carries none of these: each one has to be copied across deliberately (see
:func:`carry_node_labels`). Listing them in one place is what keeps a new
label from being silently dropped by the five moves that predate it --
which is not a crash but a system that quietly forgets what its atoms
are, several thousand steps into a run.

"""

# Standard tolerance factor applied to summed covalent radii to decide
# whether a pair of atoms counts as "bonded" for the connectivity check
# below. Matches common conventions (e.g. ASE's natural_cutoffs default).
# Unrelated to MinDistTable: this is about "is this a molecule's own
# internal bond," not "are two non-bonded atoms too close."
BOND_TOLERANCE = 1.2

# Fraction of summed covalent radii below which two atoms are considered an
# unphysical clash, for check_no_unphysical_clash below. Deliberately well
# under 1.0 (and under BOND_TOLERANCE): a real bond sits close to 1x the
# summed radii, so this only fires on genuine atomic overlap -- not on
# ordinary bonded contacts.
OVERLAP_TOLERANCE = 0.7

AtomicSystem = AtomicData
"""Alias for a single atomic configuration within RAY-MC.
See :class:`nvalchemi.data.AtomicData`."""


def _covalent_radii(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Look up covalent radii (Angstroms) for a tensor of atomic numbers."""
    radii = [pt.elements[int(z)].covalent_radius for z in atomic_numbers.cpu()]
    return torch.tensor(radii, device=atomic_numbers.device, dtype=torch.float64)


def _close_pairs(
    system: AtomicSystem, tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find atom pairs closer than ``tolerance`` times their summed covalent
    radii. Uses ALCHEMI's native cutoff-based neighbor list to avoid an
    O(n^2) dense distance scan.

    Used only for the bonding-distance connectivity check below --
    :data:`BOND_TOLERANCE`. The unphysical-overlap check
    (:func:`check_no_atomic_overlap`) uses a user-configurable
    :class:`~raymc.utils.MinDistTable` instead, via :func:`_min_dist_violations`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(src, dst)`` int64 atom-index tensors, one entry per close pair
        (each unordered pair listed once, ``src < dst``).
    """
    radii = _covalent_radii(system.atomic_numbers)
    empty = torch.empty(0, dtype=torch.long, device=system.device)
    if system.num_nodes < 2:
        return empty, empty

    # Generous single cutoff for neighbor search
    search_cutoff = float(2 * radii.max() * tolerance)

    # Pipe through batch for alchemi
    batch = Batch.from_data_list([system], device=system.device)
    compute_neighbors(batch, cutoff=search_cutoff, format=NeighborListFormat.COO)
    edges = batch.neighbor_list  # [E, 2] int32, both directions included
    if edges.numel() == 0:
        return empty, empty

    keep = edges[:, 0] < edges[:, 1]  # each unordered pair once
    edges = edges[keep].long()
    shifts = getattr(batch, "neighbor_list_shifts", None)
    if shifts is not None:
        shifts = shifts[keep]

    positions = system.positions.to(torch.float64)
    src, dst = edges[:, 0], edges[:, 1]
    delta = positions[dst] - positions[src]
    if system.cell is not None and shifts is not None:
        cell = system.cell.squeeze(0).to(torch.float64)
        delta = delta + shifts.to(torch.float64) @ cell
    dist = torch.linalg.norm(delta, dim=-1)

    pair_cutoff = (radii[src] + radii[dst]) * tolerance
    mask = dist < pair_cutoff
    return src[mask], dst[mask]


def _min_dist_violations(
    system: AtomicSystem, min_dist: MinDistTable
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find atom pairs closer than *min_dist* allows for their element pair.

    Same neighbor-list-based approach as :func:`_close_pairs`, but the
    "too close" threshold comes from a user-configurable
    :class:`~raymc.utils.MinDistTable` (per-element-pair) instead of a
    single covalent-radius multiplier. Same-rigid-group pairs are never
    checked -- they're meant to be bonded (e.g. an O-H within one water
    molecule), and a table of *non-bonded* minimum approach distances has
    nothing meaningful to say about a molecule's own fixed internal
    geometry (matches :class:`raymc.constraints.min_dist.MinDistConstraint`'s
    same exclusion).
    """
    empty = torch.empty(0, dtype=torch.long, device=system.device)
    if system.num_nodes < 2:
        return empty, empty

    atomic_numbers = system.atomic_numbers.long()
    unique_z = torch.unique(atomic_numbers)
    thresholds = min_dist.lookup(unique_z.unsqueeze(1), unique_z.unsqueeze(0))
    search_cutoff = float(thresholds.max())
    if search_cutoff <= 0:
        return empty, empty

    batch = Batch.from_data_list([system], device=system.device)
    compute_neighbors(batch, cutoff=search_cutoff, format=NeighborListFormat.COO)
    edges = batch.neighbor_list
    if edges.numel() == 0:
        return empty, empty

    keep = edges[:, 0] < edges[:, 1]
    edges = edges[keep].long()
    shifts = getattr(batch, "neighbor_list_shifts", None)
    if shifts is not None:
        shifts = shifts[keep]

    positions = system.positions.to(torch.float64)
    src, dst = edges[:, 0], edges[:, 1]
    delta = positions[dst] - positions[src]
    if system.cell is not None and shifts is not None:
        cell = system.cell.squeeze(0).to(torch.float64)
        delta = delta + shifts.to(torch.float64) @ cell
    dist = torch.linalg.norm(delta, dim=-1)

    groups = rigid_groups(system)
    same_group = groups[src] == groups[dst]
    src, dst, dist = src[~same_group], dst[~same_group], dist[~same_group]
    if src.numel() == 0:
        return empty, empty

    min_dist.warn_about_missing_pairs(atomic_numbers, atomic_numbers)
    pair_min_dist = min_dist.lookup(atomic_numbers[src], atomic_numbers[dst])
    mask = dist < pair_min_dist
    return src[mask], dst[mask]


def check_no_atomic_overlap(
    system: AtomicSystem, min_dist: MinDistTable | None = None
) -> None:
    """Raise if any two atoms are closer than *min_dist* allows.

    A general sanity check, unrelated to rigid grouping -- works on any
    structure, whether it came from a packing strategy or a file someone
    handed us.

    Parameters
    ----------
    system : AtomicSystem
    min_dist : MinDistTable | None
        Per-element-pair minimum allowed distance. Defaults to a bare
        :class:`~raymc.utils.MinDistTable` (1.0 Å for every pair) when not
        supplied -- e.g. for a standalone call with no run config in
        scope. When a run is assembled from a
        :class:`~raymc.config.RunConfig`, its ``constraints.min_dist``
        section should be built into a table once and passed here, so the
        same numbers back both this check and
        :class:`raymc.constraints.min_dist.MinDistConstraint`.

    Raises
    ------
    ValueError
        If any atom pair is closer than *min_dist* allows for its element
        pair.
    """
    table = min_dist if min_dist is not None else MinDistTable()
    src, dst = _min_dist_violations(system, table)
    if src.numel() > 0:
        i, j = int(src[0]), int(dst[0])
        raise ValueError(
            f"Atom {i} and atom {j} are closer than the configured "
            f"minimum distance for their element pair -- unphysical "
            f"atomic overlap."
        )


def check_no_unphysical_clash(
    system: AtomicSystem, tolerance: float = OVERLAP_TOLERANCE
) -> None:
    """Raise if any two atoms are unphysically close, using covalent radii
    rather than :class:`~raymc.utils.MinDistTable`.

    Unlike :func:`check_no_atomic_overlap`, this needs no rigid-group
    information to tell a real bond from an overlap -- :data:`OVERLAP_TOLERANCE`
    is well under 1x the summed covalent radii, so an ordinary bond (close
    to 1x) never trips it, regardless of whether the two atoms happen to
    share a rigid group. That makes it the right check for a structure
    with no meaningful grouping yet -- e.g. a freshly
    :func:`~raymc.initialize.load.load_structure`'d file, before any
    molecule detection has run -- where :func:`check_no_atomic_overlap`'s
    same-group exclusion can't help, since every atom is trivially its own
    group.

    Parameters
    ----------
    system : AtomicSystem
    tolerance : float
        Multiplier on summed covalent radii below which a pair counts as
        an overlap. Defaults to :data:`OVERLAP_TOLERANCE`.

    Raises
    ------
    ValueError
        If any atom pair is closer than *tolerance* times their summed
        covalent radii.
    """
    src, dst = _close_pairs(system, tolerance)
    if src.numel() > 0:
        i, j = int(src[0]), int(dst[0])
        raise ValueError(
            f"Atom {i} and atom {j} are closer than {tolerance} times the "
            f"sum of their covalent radii -- unphysical atomic overlap."
        )


def set_rigid_groups(system: AtomicSystem, groups: torch.Tensor) -> AtomicSystem:
    """Tag each atom with an integer rigid-body group id.

    .. deprecated::
        The historical spelling of :func:`raymc.topology.set_molecule_ids`,
        kept so existing callers and configs keep working while they
        migrate. One integer tensor used to answer three unrelated
        questions -- which atoms move together, which pairs skip the
        distance constraints, and which atoms form a molecule -- and those
        coincide only while every move is rigid. See
        :mod:`raymc.topology`.
    """
    from raymc.topology import set_molecule_ids

    return set_molecule_ids(system, groups, validate=True)


def rigid_groups(system: AtomicSystem) -> torch.Tensor:
    """Per-atom group ids.

    .. deprecated::
        The historical spelling of :func:`raymc.topology.molecule_ids`.
    """
    from raymc.topology import molecule_ids

    return molecule_ids(system)


def _attach_rigid_groups_unchecked(
    system: AtomicSystem, groups: torch.Tensor
) -> AtomicSystem:
    """Tag *system* with *groups* without running the connectivity check.

    .. deprecated::
        Prefer :func:`raymc.topology.carry_topology`, which also carries
        the bond list -- this only carries membership.
    """
    from raymc.topology import set_molecule_ids

    return set_molecule_ids(system, groups, validate=False)



def carry_node_labels(
    source: AtomicSystem,
    destination: AtomicSystem,
    index: torch.Tensor | None = None,
) -> AtomicSystem:
    """Copy RAY-MC's per-atom bookkeeping (:data:`BOOKKEEPING_KEYS`) from
    *source* onto *destination*, reindexing along the way.

    Parameters
    ----------
    source : AtomicSystem
        The system whose labels to copy. Keys it doesn't carry are skipped.
    destination : AtomicSystem
        Mutated in place and returned.
    index : torch.Tensor | None
        [n_destination] int64 giving, for each destination atom, the index
        of the source atom it came from. ``-1`` marks an atom with no
        source (freshly inserted), which gets the key's documented default:
        ``-1`` for the integer labels, and its own atom index for the
        rigid group. ``None`` means the two
        systems are aligned one-to-one, which is the common case (a move
        that displaces atoms without adding or removing any).

    Returns
    -------
    AtomicSystem
        *destination*.
    """
    n = destination.num_nodes
    if index is None:
        index = torch.arange(n, device=destination.device)
    index = index.to(device=destination.device, dtype=torch.long)
    if index.shape[0] != n:
        raise ValueError(
            f"index has {index.shape[0]} entries, expected {n} (one per "
            f"destination atom)"
        )
    missing = index < 0
    gather = index.clamp(min=0)

    keys = getattr(source, "__node_keys__", ())
    for key in BOOKKEEPING_KEYS:
        if key not in keys:
            continue
        values = getattr(source, key)[gather].clone()
        if missing.any():
            if key == RIGID_GROUP_KEY:
                # A brand-new atom belongs to no existing group: give it a
                # group of its own, the same convention rigid_groups() uses
                # for an ungrouped system.
                fresh = torch.arange(n, device=values.device, dtype=values.dtype)
                offset = int(values[~missing].max()) + 1 if (~missing).any() else 0
                values[missing] = fresh[missing] + offset
            else:
                values[missing] = -1
        destination.add_node_property(key, values)
    return destination


def group_atom_indices(system: AtomicSystem, group_id: int) -> torch.Tensor:
    """Atom indices belonging to rigid group *group_id*."""
    return (rigid_groups(system) == group_id).nonzero(as_tuple=True)[0]


def unwrap_group_positions(
    system: AtomicSystem, atom_indices: torch.Tensor
) -> torch.Tensor:
    """PBC-unwrap one rigid group's atoms relative to its first atom.

    Needed to safely average or rotate a group's atoms as a single rigid
    body even when it straddles a periodic boundary. Public (not
    module-private) because :mod:`raymc.moves.base` also needs it for
    ``apply_rigid_transform``.
    """
    positions = system.positions[atom_indices].to(torch.float64)
    if system.cell is None or system.pbc is None:
        return positions

    cell = system.cell.squeeze(0).to(torch.float64)
    pbc = system.pbc.squeeze(0)
    reference = positions[0]
    return reference + minimum_image_delta(positions, reference, cell, pbc)


def group_centroid(system: AtomicSystem, group_id: int) -> torch.Tensor:
    """PBC-aware centroid of rigid group *group_id*.

    Parameters
    ----------
    system : AtomicSystem
    group_id : int

    Returns
    -------
    torch.Tensor
        [3] centroid position (Angstrom, float64).
    """
    atom_indices = group_atom_indices(system, group_id)
    if atom_indices.numel() == 0:
        raise ValueError(f"No atoms found in rigid group {group_id}")
    return unwrap_group_positions(system, atom_indices).mean(dim=0)
