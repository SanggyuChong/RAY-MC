"""Molecular topology: what is bonded to what, and which atoms form a unit.

RAY-MC used to store one ``[n_atoms]`` integer tensor called
``rigid_group`` that was silently doing three unrelated jobs: deciding
which atoms displace together, deciding which atom pairs skip the
distance constraints, and recording which atoms form one molecule. Those
three coincide only because every move was rigid. This module separates
the third job -- membership and connectivity -- from the other two, and
gives it a home with accessors rather than a raw tensor everyone reaches
into.

Two things are stored, in two different tiers, because their shapes
genuinely differ:

- ``molecule_id``, one integer per atom, as a **node** property. Exactly
  one per atom, so it is a plain ``[n_atoms]`` tensor.
- ``bonds``, a ``[n_bonds, 2]`` index pair list plus ``[n_bonds, 3]``
  periodic image shifts, as a **system** property.

Bonds are deliberately *not* an edge property, even though
:class:`~nvalchemi.data.AtomicData` offers one and it looks like the
natural home. ``check_edge_consistency`` requires every edge property to
match ``neighbor_list`` in length, and ``neighbor_list`` is overwritten by
``compute_neighbors`` at whatever cutoff each caller wants -- which is
every constraint evaluation, at a cutoff that changes with the run. A
bond list parked in the edge tier is either rejected by the validator or
silently invalidated on the next distance check.

The system tier is not unchecked either, though. ``level_storage`` requires every system property to
share a first dimension -- the number of systems -- and
``Batch.from_data_list`` enforces it, so a bare ``[n_bonds, 2]`` tensor
raises ``Inconsistent first dimension: expected 1, got n_bonds`` the next
time anything builds a batch (which is every neighbor search). Bonds are
therefore stored with an explicit leading system dimension,
``[1, n_bonds, 2]``, and :func:`bonds` strips it on the way out so
callers never see it. That is also why these are accessors rather than
raw attribute reads.

**Bonds are data, not a re-derived inference.** The old code perceived
bonds from geometry every time groups were set, comparing distances
against ``BOND_TOLERANCE`` x summed covalent radii. That is incompatible
with flexibility on its own terms: a bond stretched past 1.2x stops being
perceived, the molecule silently dissolves into loose atoms, and the
same-group exclusion switches off underneath a running simulation.
Perception still exists here -- :func:`perceive_bonds` -- but it runs
**once**, at initialization, and what it produces is then carried forward
as data by :func:`carry_topology`.

Nothing in this module changes behaviour. ``rigid_groups`` and friends
remain, as thin aliases over ``molecule_id``, so every existing caller and
test keeps working while callers migrate.
"""

from __future__ import annotations

import warnings

import torch

from raymc.system.atomic_system import (
    BOND_TOLERANCE,
    AtomicSystem,
    _close_pairs,
)

MOLECULE_ID_KEY = "molecule_id"
"""Node-property key: which molecule each atom belongs to."""

BONDS_KEY = "bonds"
"""System-property key. Stored ``[1, n_bonds, 2]`` int64 (see the module
docstring on the leading system dimension); :func:`bonds` returns
``[n_bonds, 2]``."""

BOND_SHIFTS_KEY = "bond_shifts"
"""System-property key. Stored ``[1, n_bonds, 3]``, returned
``[n_bonds, 3]``: the periodic image shift for each bond, in units of
cell vectors, applied to the pair's SECOND atom relative to its first --
the same convention the neighbor list uses, so
``r_j - r_i + shift @ cell`` is the bonded displacement."""

_warned_missing = False


def _warn_missing_topology(what: str) -> None:
    """Complain once per process that topology was never established.

    The old accessor returned ``arange(n)`` -- one group per atom -- with
    no indication that it was guessing, which is the fallback the design
    note asks to give teeth. A guess is survivable (a monatomic system
    really is one atom per group) but it must not be silent: when it is
    wrong, every move unit and every constraint exclusion is wrong too,
    and the only symptom is a strange g(r).

    Warned once rather than per call because these accessors run inside
    the Metropolis loop, several times per step.
    """
    global _warned_missing
    if _warned_missing:
        return
    _warned_missing = True
    warnings.warn(
        f"{what} was never set on this system, falling back to one "
        "molecule per atom. That is correct for a monatomic system and "
        "wrong for anything with molecules in it -- every move unit and "
        "every same-molecule constraint exclusion silently follows this "
        "guess. Call set_molecule_ids() (or a packing strategy that "
        "does) before running.",
        UserWarning,
        stacklevel=3,
    )


# -- membership ------------------------------------------------------------


def molecule_ids(system: AtomicSystem) -> torch.Tensor:
    """Per-atom molecule id, ``[n_atoms]`` int64.

    Falls back to one molecule per atom if never set -- with a warning,
    unlike the old ``rigid_groups``. See :func:`_warn_missing_topology`.
    """
    if MOLECULE_ID_KEY in getattr(system, "__node_keys__", ()):
        return getattr(system, MOLECULE_ID_KEY)
    _warn_missing_topology("molecule_id")
    return torch.arange(system.num_nodes, device=system.device)


def has_molecule_ids(system: AtomicSystem) -> bool:
    """Whether molecule membership was actually established, as opposed to
    being about to be guessed. Lets a caller check without tripping the
    warning."""
    return MOLECULE_ID_KEY in getattr(system, "__node_keys__", ())


def set_molecule_ids(
    system: AtomicSystem, ids: torch.Tensor, validate: bool = True
) -> AtomicSystem:
    """Tag each atom with its molecule id. Mutates and returns *system*.

    Parameters
    ----------
    system : AtomicSystem
    ids : torch.Tensor
        ``[n_atoms]`` integer molecule ids.
    validate : bool
        Check that every molecule is a single connected bonded fragment
        (see :func:`check_molecules_are_connected`). Skipped by
        :func:`carry_topology`, where validity is guaranteed by
        construction and where re-checking would additionally be *wrong*:
        the check perceives bonds from geometry, so a legitimately
        stretched bond stops being seen.
    """
    if ids.shape[0] != system.num_nodes:
        raise ValueError(
            f"ids has {ids.shape[0]} entries, expected {system.num_nodes} "
            f"(one per atom)"
        )
    ids = ids.to(device=system.device, dtype=torch.long)
    if validate:
        check_molecules_are_connected(system, ids)
    system.add_node_property(MOLECULE_ID_KEY, ids)
    return system


def molecule_atom_indices(system: AtomicSystem, molecule_id: int) -> torch.Tensor:
    """Atom indices belonging to *molecule_id*."""
    return (molecule_ids(system) == molecule_id).nonzero(as_tuple=True)[0]


def molecule_sizes(system: AtomicSystem) -> tuple[torch.Tensor, torch.Tensor]:
    """``(unique_ids, atom_counts)`` for every molecule present."""
    return torch.unique(molecule_ids(system), return_counts=True)


# -- bonds -----------------------------------------------------------------


def bonds(system: AtomicSystem) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(pairs, shifts)`` if bonds were established, else ``None``.

    ``None`` rather than an empty tensor, so "no bonds were ever recorded"
    stays distinguishable from "this system genuinely has none" (a
    monatomic box).
    """
    keys = getattr(system, "__system_keys__", ())
    if BONDS_KEY not in keys:
        return None
    # Stored with a leading system dimension; callers want it stripped.
    return (
        getattr(system, BONDS_KEY)[0],
        getattr(system, BOND_SHIFTS_KEY)[0],
    )


def set_bonds(
    system: AtomicSystem, pairs: torch.Tensor, shifts: torch.Tensor | None = None
) -> AtomicSystem:
    """Record *pairs* (``[n_bonds, 2]``) as this system's bonds.

    *shifts* defaults to all-zero, i.e. no bond crosses a periodic
    boundary -- correct for a freshly-packed molecule placed whole.
    """
    pairs = pairs.to(device=system.device, dtype=torch.long).reshape(-1, 2)
    if shifts is None:
        shifts = torch.zeros((pairs.shape[0], 3), dtype=torch.long)
    shifts = shifts.to(device=system.device, dtype=torch.long).reshape(-1, 3)
    if shifts.shape[0] != pairs.shape[0]:
        raise ValueError(
            f"shifts has {shifts.shape[0]} rows, expected {pairs.shape[0]} "
            "(one per bond)"
        )
    system.add_system_property(BONDS_KEY, pairs.unsqueeze(0))
    system.add_system_property(BOND_SHIFTS_KEY, shifts.unsqueeze(0))
    return system


def perceive_bonds(
    system: AtomicSystem, tolerance: float = BOND_TOLERANCE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer bonds from geometry: pairs closer than *tolerance* times their
    summed covalent radii.

    The ONLY place perception belongs -- run once, at initialization, on a
    structure whose geometry is still trustworthy. Everything afterwards
    carries the result forward as data. See the module docstring.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(pairs, shifts)``: ``[n_bonds, 2]`` and ``[n_bonds, 3]``.
    """
    src, dst = _close_pairs(system, tolerance)
    pairs = torch.stack([src, dst], dim=1) if src.numel() else torch.empty(
        (0, 2), dtype=torch.long
    )
    shifts = _bond_shifts(system, pairs)
    return pairs, shifts


def _bond_shifts(system: AtomicSystem, pairs: torch.Tensor) -> torch.Tensor:
    """Which periodic image of each bond's second atom is the bonded one.

    Recovered from the minimum-image displacement rather than carried out
    of the neighbor list, so this works for any pair list regardless of
    where it came from.
    """
    if pairs.numel() == 0 or system.cell is None or system.pbc is None:
        return torch.zeros((pairs.shape[0], 3), dtype=torch.long)
    cell = system.cell.squeeze(0).to(torch.float64)
    pbc = system.pbc.squeeze(0)
    positions = system.positions.to(torch.float64)
    raw = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    frac = raw @ torch.linalg.inv(cell)
    shift = -torch.round(frac) * pbc.to(frac.dtype)
    return shift.to(torch.long)


# -- connectivity ----------------------------------------------------------


def connected_components(
    n_atoms: int, pairs: torch.Tensor, shifts: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flood-fill the bond graph, detecting components that wrap through
    their own periodic images.

    A component that reaches one of its own atoms at a *different*
    cumulative image shift has no finite molecule at all -- it percolates
    through the box. That case is reported rather than guessed at, and it
    is what correctly tells insert/remove moves they do not apply.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(component_id, is_periodic)``: ``[n_atoms]`` int64 component
        labels, and ``[n_components]`` bool flags.
    """
    if shifts is None:
        shifts = torch.zeros((pairs.shape[0], 3), dtype=torch.long)

    adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] = {}
    for (i, j), shift in zip(pairs.tolist(), shifts.tolist()):
        s = (shift[0], shift[1], shift[2])
        adjacency.setdefault(i, []).append((j, s))
        adjacency.setdefault(j, []).append((i, (-s[0], -s[1], -s[2])))

    component = torch.full((n_atoms,), -1, dtype=torch.long)
    periodic: list[bool] = []
    offset: dict[int, tuple[int, int, int]] = {}

    label = 0
    for start in range(n_atoms):
        if component[start] >= 0:
            continue
        component[start] = label
        offset[start] = (0, 0, 0)
        wraps = False
        stack = [start]
        while stack:
            u = stack.pop()
            ou = offset[u]
            for v, s in adjacency.get(u, ()):
                ov = (ou[0] + s[0], ou[1] + s[1], ou[2] + s[2])
                if component[v] < 0:
                    component[v] = label
                    offset[v] = ov
                    stack.append(v)
                elif offset.get(v) != ov:
                    # Reached an already-placed atom at a different image:
                    # the component closes on itself through the boundary.
                    wraps = True
        periodic.append(wraps)
        label += 1

    return component, torch.tensor(periodic, dtype=torch.bool)


def check_molecules_are_connected(
    system: AtomicSystem, ids: torch.Tensor | None = None
) -> None:
    """Raise if any molecule is not a single connected bonded fragment.

    Uses recorded bonds when the system has them, and perceives them from
    geometry only when it does not -- so a flexible molecule whose bonds
    are already data is checked against those bonds rather than against
    whatever its current geometry happens to look like.
    """
    if system.num_nodes < 2:
        return
    ids = molecule_ids(system) if ids is None else ids

    recorded = bonds(system)
    if recorded is not None:
        pairs = recorded[0]
        src, dst = pairs[:, 0], pairs[:, 1]
    else:
        src, dst = _close_pairs(system, BOND_TOLERANCE)

    same = ids[src] == ids[dst]
    src, dst = src[same], dst[same]

    adjacency: dict[int, set[int]] = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        adjacency.setdefault(s, set()).add(d)
        adjacency.setdefault(d, set()).add(s)

    members: dict[int, list[int]] = {}
    for atom_idx, molecule in enumerate(ids.tolist()):
        members.setdefault(molecule, []).append(atom_idx)

    for molecule, atom_indices in members.items():
        if len(atom_indices) <= 1:
            continue
        remaining = set(atom_indices)
        start = atom_indices[0]
        remaining.discard(start)
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adjacency.get(u, ()):
                if v in remaining:
                    remaining.discard(v)
                    stack.append(v)
        if remaining:
            raise ValueError(
                f"Molecule {molecule} is not a single connected bonded "
                f"fragment: atom(s) {sorted(remaining)} are not reachable "
                f"via bonding-distance contacts from the rest of the "
                f"molecule (atom {start})."
            )


# -- carrying topology across a move ---------------------------------------


def carry_topology(
    new_system: AtomicSystem,
    old_system: AtomicSystem,
    keep_mask: torch.Tensor | None = None,
    added_molecule_ids: torch.Tensor | None = None,
    added_bonds: torch.Tensor | None = None,
) -> AtomicSystem:
    """Copy *old_system*'s topology onto *new_system*.

    Node properties do not survive move construction -- every move builds
    a fresh :class:`AtomicSystem` and re-attaches by hand. That was five
    separate call sites, and forgetting one would not raise: the old
    fallback silently returned one group per atom. One helper, called by
    every move, is what makes that impossible to get subtly wrong.

    Parameters
    ----------
    new_system, old_system : AtomicSystem
    keep_mask : torch.Tensor | None
        ``[n_old_atoms]`` bool, for a move that removes atoms. Surviving
        atoms are renumbered, and bonds touching a removed atom are
        dropped -- see :func:`remap_indices`.
    added_molecule_ids : torch.Tensor | None
        Molecule ids for atoms appended by an insert, in order.
    added_bonds : torch.Tensor | None
        Bonds among the appended atoms, indexed from 0 within the added
        block; offset into the new system's numbering here.
    """
    ids = molecule_ids(old_system) if has_molecule_ids(old_system) else None
    recorded = bonds(old_system)
    pairs = recorded[0] if recorded is not None else None
    shifts = recorded[1] if recorded is not None else None

    if keep_mask is not None:
        if ids is not None:
            ids = ids[keep_mask]
        if pairs is not None:
            pairs, shifts = remap_indices(pairs, shifts, keep_mask)

    n_existing = 0 if ids is None else ids.shape[0]
    if added_molecule_ids is not None:
        ids = (
            added_molecule_ids
            if ids is None
            else torch.cat([ids, added_molecule_ids.to(ids.dtype)])
        )
    if added_bonds is not None and added_bonds.numel():
        offset_bonds = added_bonds.to(torch.long) + n_existing
        pairs = offset_bonds if pairs is None else torch.cat([pairs, offset_bonds])
        extra = torch.zeros((offset_bonds.shape[0], 3), dtype=torch.long)
        shifts = extra if shifts is None else torch.cat([shifts, extra])

    if ids is not None:
        set_molecule_ids(new_system, ids, validate=False)
    if pairs is not None:
        set_bonds(new_system, pairs, shifts)
    return new_system


def remap_indices(
    pairs: torch.Tensor, shifts: torch.Tensor, keep_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Renumber a bond list after atoms are removed.

    Removal filters positions rather than masking them, so every surviving
    atom after a removed one shifts down. Bonds touching a removed atom
    are dropped -- a dangling bond is not a bond.
    """
    keep_mask = keep_mask.to(torch.bool)
    new_index = torch.cumsum(keep_mask.to(torch.long), dim=0) - 1
    survives = keep_mask[pairs[:, 0]] & keep_mask[pairs[:, 1]]
    kept = pairs[survives]
    return (
        torch.stack([new_index[kept[:, 0]], new_index[kept[:, 1]]], dim=1),
        shifts[survives],
    )


def establish_topology(
    system: AtomicSystem, tolerance: float = BOND_TOLERANCE
) -> AtomicSystem:
    """Perceive bonds, label connected components, and record both.

    The load-from-file path: a structure someone handed us carries no
    topology, so it is perceived once here and carried as data from then
    on. Raises if any component percolates through the box, rather than
    labelling it a molecule.
    """
    pairs, shifts = perceive_bonds(system, tolerance)
    component, periodic = connected_components(system.num_nodes, pairs, shifts)
    if bool(periodic.any()):
        which = periodic.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"component(s) {which} wrap through their own periodic images: "
            "this is a percolating network, not a set of molecules. Insert "
            "and remove moves do not apply to it, and it has no finite "
            "molecular unit to translate or rotate."
        )
    set_bonds(system, pairs, shifts)
    set_molecule_ids(system, component, validate=False)
    return system


__all__ = [
    "MOLECULE_ID_KEY",
    "BONDS_KEY",
    "BOND_SHIFTS_KEY",
    "molecule_ids",
    "has_molecule_ids",
    "set_molecule_ids",
    "molecule_atom_indices",
    "molecule_sizes",
    "bonds",
    "set_bonds",
    "perceive_bonds",
    "connected_components",
    "check_molecules_are_connected",
    "carry_topology",
    "remap_indices",
    "establish_topology",
]
