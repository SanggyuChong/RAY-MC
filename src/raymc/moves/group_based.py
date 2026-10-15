"""Monte Carlo moves that act on a rigid group of atoms at once.

Translate, rotate, insert, and remove a rigid group (as defined by
:func:`~raymc.system.set_rigid_groups`) -- a multi-atom molecule or a lone
atom alike, both represented as a
:class:`~raymc.initialize.templates.GroupTemplate`. These moves do not
apply to building blocks of polymers or framework materials.

Each move's :meth:`~raymc.moves.base.Move.attempt` returns a *new* system
with the change applied (inputs are never mutated), plus a
:class:`~raymc.moves.base.MoveDelta` describing exactly which atoms it
touched -- so a solver can freely discard a rejected proposal and keep the
original state, and a residual can update incrementally instead of
recomputing from scratch.
"""

from __future__ import annotations

import torch

from raymc.initialize.templates import GroupTemplate
from raymc.moves.base import Move, MoveDelta, MoveNotApplicable
from raymc.system import (
    AtomicSystem,
    carry_node_labels,
    group_atom_indices,
    rigid_groups,
    unwrap_group_positions,
)
from raymc.system.atomic_system import _attach_rigid_groups_unchecked
from raymc.utils import random_rotation_matrices, rotation_about_random_axis


def apply_rigid_transform(
    system: AtomicSystem,
    group_id: int,
    translation: torch.Tensor | None = None,
    rotation: torch.Tensor | None = None,
) -> AtomicSystem:
    """Return a new system with rigid group *group_id* translated and/or
    rotated about its own centroid.

    Afterwards the whole group is moved back inside the box together, as
    one piece, so it doesn't end up split across a periodic edge.

    Doesn't re-check that the group is still one bonded molecule: a rigid
    move can't break that, since every distance inside the group stays
    exactly the same.

    Parameters
    ----------
    system : AtomicSystem
    group_id : int
        Rigid group to transform.
    translation : torch.Tensor | None
        [3] Cartesian displacement (Angstrom) applied to the group's centroid.
    rotation : torch.Tensor | None
        [3, 3] rotation matrix applied about the group's own centroid.

    Returns
    -------
    AtomicSystem
        A new system (the input is not mutated).
    """
    atom_indices = group_atom_indices(system, group_id)
    if atom_indices.numel() == 0:
        raise ValueError(f"No atoms found in rigid group {group_id}")

    unwrapped = unwrap_group_positions(system, atom_indices)
    centroid = unwrapped.mean(dim=0)
    local = unwrapped - centroid

    if rotation is not None:
        local = local @ rotation.to(torch.float64).T

    new_centroid = centroid
    if translation is not None:
        new_centroid = centroid + translation.to(torch.float64)

    new_positions = local + new_centroid

    if system.cell is not None and system.pbc is not None:
        cell = system.cell.squeeze(0).to(torch.float64)
        pbc = system.pbc.squeeze(0)
        inv_cell = torch.linalg.inv(cell)
        frac_centroid = new_centroid @ inv_cell
        shift = torch.floor(frac_centroid) * pbc.to(frac_centroid.dtype)
        new_positions = new_positions - shift @ cell

    positions = system.positions.clone()
    positions[atom_indices] = new_positions.to(positions.dtype)

    new_system = AtomicSystem(
        positions=positions,
        atomic_numbers=system.atomic_numbers,
        atomic_masses=system.atomic_masses,
        cell=system.cell,
        pbc=system.pbc,
    )
    # Carries the rigid groups and any other per-atom bookkeeping: a rigid
    # transform changes where a group is, never what it is -- see
    # BOOKKEEPING_KEYS.
    carry_node_labels(system, new_system)
    return new_system


def _random_existing_group(
    system: AtomicSystem, generator: torch.Generator, min_group_size: int = 1
) -> int:
    """Pick a uniformly-random existing rigid-group id from *system*,
    considering only groups with at least *min_group_size* atoms.

    Shared target-selection logic for the moves below that act on an
    existing molecule (translate, rotate, remove) rather than creating one
    (insert). *min_group_size* lets a move opt out of single-atom groups
    when they wouldn't make sense as a target (see :class:`RotateGroup`,
    where rotating a lone atom about its own centroid is a no-op) --
    translate/remove/relocate leave it at the default and treat a lone
    atom as a perfectly ordinary one-atom group.
    """
    groups = rigid_groups(system)
    unique, counts = torch.unique(groups, return_counts=True)
    eligible = unique[counts >= min_group_size]
    if eligible.numel() == 0:
        raise MoveNotApplicable(
            f"system has no rigid group with at least {min_group_size} atom(s)"
        )
    idx = torch.randint(0, eligible.numel(), (1,), generator=generator).item()
    return int(eligible[idx])


class TranslateGroup(Move):
    """Rigidly translate one randomly-chosen group by a random vector.

    The displacement direction is uniform on the sphere and its magnitude
    is uniform in ``[0, max_displacement]``.
    """

    name = "translate"

    def __init__(self, max_displacement: float):
        self.max_displacement = max_displacement

    def is_applicable(self, system: AtomicSystem) -> bool:
        return bool(rigid_groups(system).numel() > 0)

    def attempt(
        self, system: AtomicSystem, generator: torch.Generator
    ) -> tuple[AtomicSystem, MoveDelta]:
        group_id = _random_existing_group(system, generator)
        atom_indices = group_atom_indices(system, group_id)
        direction = torch.randn(3, generator=generator, dtype=torch.float64)
        direction = direction / direction.norm()
        magnitude = (
            torch.rand(1, generator=generator, dtype=torch.float64)
            * self.max_displacement
        )
        translation = direction * magnitude
        new_system = apply_rigid_transform(system, group_id, translation=translation)
        return new_system, MoveDelta(moved_indices=atom_indices)


class RotateGroup(Move):
    """Rigidly rotate one randomly-chosen multi-atom group about its own
    centroid by a random, bounded angle.

    Only ever targets a group with 2+ atoms -- a lone atom has no
    orientation to change, so rotating it about its own centroid is a
    no-op. If the system currently has no multi-atom rigid group at all,
    this move raises :class:`~raymc.moves.base.MoveNotApplicable`, the
    same way ``RemoveGroup`` does on a system with nothing to remove.
    """

    name = "rotate"

    def __init__(self, max_angle: float):
        self.max_angle = max_angle

    def is_applicable(self, system: AtomicSystem) -> bool:
        groups = rigid_groups(system)
        _, counts = torch.unique(groups, return_counts=True)
        return bool((counts >= 2).any())

    def attempt(
        self, system: AtomicSystem, generator: torch.Generator
    ) -> tuple[AtomicSystem, MoveDelta]:
        group_id = _random_existing_group(system, generator, min_group_size=2)
        atom_indices = group_atom_indices(system, group_id)
        rotation = rotation_about_random_axis(self.max_angle, generator)
        new_system = apply_rigid_transform(system, group_id, rotation=rotation)
        return new_system, MoveDelta(moved_indices=atom_indices)


class InsertGroup(Move):
    """Insert a new copy of a fixed template at a random position and
    orientation, as a new rigid group.

    The new group's id is one greater than the current maximum. Whether
    the insertion lands in a reasonable spot is determined by the
    chi-squared value (which often includes a min-dist constraint or
    energetics for HRMC).
    """

    name = "insert"

    def __init__(self, template: GroupTemplate):
        self.template = template

    def attempt(
        self, system: AtomicSystem, generator: torch.Generator
    ) -> tuple[AtomicSystem, MoveDelta]:
        if system.cell is None or system.pbc is None:
            raise ValueError("InsertGroup requires a periodic system (cell + pbc)")

        n_before = system.num_nodes
        cell = system.cell.squeeze(0).to(torch.float64)
        frac = torch.rand(3, generator=generator, dtype=torch.float64)
        center = frac @ cell
        rotation = random_rotation_matrices(1, generator)[0]

        new_positions = (
            self.template.positions.to(torch.float64) @ rotation.T
        ) + center

        positions = torch.cat(
            [system.positions, new_positions.to(system.positions.dtype)], dim=0
        )
        atomic_numbers = torch.cat(
            [
                system.atomic_numbers,
                self.template.atomic_numbers.to(system.atomic_numbers.dtype),
            ]
        )

        existing_groups = rigid_groups(system)
        new_group_id = (
            int(existing_groups.max().item()) + 1 if existing_groups.numel() > 0 else 0
        )
        new_groups = torch.cat(
            [
                existing_groups,
                torch.full((self.template.num_atoms,), new_group_id, dtype=torch.long),
            ]
        )

        new_system = AtomicSystem(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=system.cell,
            pbc=system.pbc,
        )
        carry_node_labels(
            system,
            new_system,
            index=torch.cat(
                [
                    torch.arange(n_before),
                    torch.full((self.template.num_atoms,), -1, dtype=torch.long),
                ]
            ),
        )
        _attach_rigid_groups_unchecked(new_system, new_groups)
        # Always appended at the end, so no existing atom's index shifts.
        added_indices = torch.arange(n_before, n_before + self.template.num_atoms)
        return new_system, MoveDelta(added_indices=added_indices)


class RemoveGroup(Move):
    """Remove one randomly-chosen rigid group (every atom in it).

    Other groups keep their existing ids (unchanged, possibly leaving gaps
    in the id sequence) -- group ids are just labels, not required to be
    contiguous.
    """

    name = "remove"

    def is_applicable(self, system: AtomicSystem) -> bool:
        return bool(rigid_groups(system).numel() > 0)

    def attempt(
        self, system: AtomicSystem, generator: torch.Generator
    ) -> tuple[AtomicSystem, MoveDelta]:
        group_id = _random_existing_group(system, generator)
        groups = rigid_groups(system)
        keep_mask = groups != group_id
        removed_indices = (~keep_mask).nonzero(as_tuple=True)[0]

        new_system = AtomicSystem(
            positions=system.positions[keep_mask],
            atomic_numbers=system.atomic_numbers[keep_mask],
            cell=system.cell,
            pbc=system.pbc,
        )
        carry_node_labels(
            system, new_system, index=keep_mask.nonzero(as_tuple=True)[0]
        )
        return new_system, MoveDelta(removed_indices=removed_indices)


class RelocateGroup(Move):
    """Remove one randomly-chosen group and immediately reinsert a copy of
    the same template at a new random position and orientation as a single
    atomic move.
    """

    name = "relocate"

    def __init__(self, template: GroupTemplate):
        self.template = template

    def is_applicable(self, system: AtomicSystem) -> bool:
        return bool(rigid_groups(system).numel() > 0)

    def attempt(
        self, system: AtomicSystem, generator: torch.Generator
    ) -> tuple[AtomicSystem, MoveDelta]:
        if system.cell is None or system.pbc is None:
            raise ValueError("RelocateGroup requires a periodic system (cell + pbc)")

        group_id = _random_existing_group(system, generator)
        groups = rigid_groups(system)
        keep_mask = groups != group_id
        removed_indices = (~keep_mask).nonzero(as_tuple=True)[0]

        remaining_positions = system.positions[keep_mask]
        remaining_atomic_numbers = system.atomic_numbers[keep_mask]
        remaining_groups = groups[keep_mask]

        cell = system.cell.squeeze(0).to(torch.float64)
        frac = torch.rand(3, generator=generator, dtype=torch.float64)
        center = frac @ cell
        rotation = random_rotation_matrices(1, generator)[0]
        new_positions = (
            self.template.positions.to(torch.float64) @ rotation.T
        ) + center

        positions = torch.cat(
            [remaining_positions, new_positions.to(system.positions.dtype)], dim=0
        )
        atomic_numbers = torch.cat(
            [
                remaining_atomic_numbers,
                self.template.atomic_numbers.to(system.atomic_numbers.dtype),
            ]
        )

        new_group_id = int(groups.max().item()) + 1 if groups.numel() > 0 else 0
        new_groups = torch.cat(
            [
                remaining_groups,
                torch.full((self.template.num_atoms,), new_group_id, dtype=torch.long),
            ]
        )

        new_system = AtomicSystem(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=system.cell,
            pbc=system.pbc,
        )
        carry_node_labels(
            system,
            new_system,
            index=torch.cat(
                [
                    keep_mask.nonzero(as_tuple=True)[0],
                    torch.full((self.template.num_atoms,), -1, dtype=torch.long),
                ]
            ),
        )
        _attach_rigid_groups_unchecked(new_system, new_groups)

        n_remaining = remaining_positions.shape[0]
        added_indices = torch.arange(n_remaining, n_remaining + self.template.num_atoms)
        return new_system, MoveDelta(
            added_indices=added_indices, removed_indices=removed_indices
        )
