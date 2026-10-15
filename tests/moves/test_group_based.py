from __future__ import annotations

import torch

from raymc.moves.base import MoveNotApplicable
from raymc.moves.group_based import (
    InsertGroup,
    RelocateGroup,
    RemoveGroup,
    RotateGroup,
    TranslateGroup,
    apply_rigid_transform,
)
from raymc.system import AtomicSystem, group_atom_indices, group_centroid, rigid_groups, set_rigid_groups


def _bond_lengths(system, group_id):
    idx = group_atom_indices(system, group_id)
    pos = system.positions[idx]
    return torch.norm(pos[1] - pos[0]), torch.norm(pos[2] - pos[0])


# -- apply_rigid_transform (now lives in this module) ----------------------


def test_apply_rigid_transform_translate_preserves_bond_lengths(single_water):
    before = single_water.positions.clone()
    oh1_before = torch.norm(before[1] - before[0])
    oh2_before = torch.norm(before[2] - before[0])

    moved = apply_rigid_transform(
        single_water, group_id=0, translation=torch.tensor([1.0, 2.0, 3.0])
    )

    oh1_after = torch.norm(moved.positions[1] - moved.positions[0])
    oh2_after = torch.norm(moved.positions[2] - moved.positions[0])
    assert torch.allclose(oh1_before, oh1_after, atol=1e-5)
    assert torch.allclose(oh2_before, oh2_after, atol=1e-5)
    # original system must not be mutated
    assert torch.allclose(single_water.positions, before)


def test_apply_rigid_transform_rotate_preserves_centroid(single_water):
    before_centroid = group_centroid(single_water, group_id=0)
    rotation = torch.eye(3)  # identity rotation: centroid must stay exactly put
    rotated = apply_rigid_transform(single_water, group_id=0, rotation=rotation)
    after_centroid = group_centroid(rotated, group_id=0)
    assert torch.allclose(before_centroid, after_centroid, atol=1e-6)


# -- TranslateGroup ----------------------------------------------------------


def test_translate_group_preserves_internal_geometry(packed_water):
    generator = torch.Generator().manual_seed(0)
    before = _bond_lengths(packed_water, group_id=0)

    move = TranslateGroup(max_displacement=1.0)
    moved, _delta = move.attempt(packed_water, generator)

    after = _bond_lengths(moved, group_id=0)
    assert torch.allclose(before[0], after[0], atol=1e-4)
    assert torch.allclose(before[1], after[1], atol=1e-4)


def test_translate_group_does_not_touch_other_molecules(packed_water):
    generator = torch.Generator().manual_seed(0)
    other_before = packed_water.positions[group_atom_indices(packed_water, 1)].clone()

    move = TranslateGroup(max_displacement=1.0)
    moved, _delta = move.attempt(packed_water, generator)

    other_after = moved.positions[group_atom_indices(moved, 1)]
    assert torch.allclose(other_before, other_after)


def test_translate_group_delta_reports_exactly_the_moved_group(packed_water):
    generator = torch.Generator().manual_seed(0)

    move = TranslateGroup(max_displacement=1.0)
    moved, delta = move.attempt(packed_water, generator)

    actually_moved = (
        (moved.positions - packed_water.positions).abs().sum(dim=-1) > 1e-9
    ).nonzero(as_tuple=True)[0]
    assert torch.equal(
        torch.sort(delta.moved_indices).values, torch.sort(actually_moved).values
    )
    assert delta.moved_indices.numel() == 3  # one water molecule
    assert delta.added_indices.numel() == 0
    assert delta.removed_indices.numel() == 0


def test_translate_group_is_always_applicable(packed_water):
    assert TranslateGroup(max_displacement=1.0).is_applicable(packed_water)


# -- RotateGroup --------------------------------------------------------------


def test_rotate_group_preserves_bond_lengths_and_centroid(packed_water):
    generator = torch.Generator().manual_seed(0)
    before_bonds = _bond_lengths(packed_water, group_id=0)
    before_centroid = group_centroid(packed_water, group_id=0)

    move = RotateGroup(max_angle=0.5)
    rotated, _delta = move.attempt(packed_water, generator)

    after_bonds = _bond_lengths(rotated, group_id=0)
    after_centroid = group_centroid(rotated, group_id=0)
    assert torch.allclose(before_bonds[0], after_bonds[0], atol=1e-4)
    assert torch.allclose(before_bonds[1], after_bonds[1], atol=1e-4)
    assert torch.allclose(before_centroid, after_centroid, atol=1e-5)


def test_rotate_group_is_applicable_when_a_multi_atom_group_exists(packed_water):
    assert RotateGroup(max_angle=0.5).is_applicable(packed_water)


def test_rotate_group_is_not_applicable_on_an_atoms_only_system():
    system = AtomicSystem(
        positions=torch.rand(5, 3) * 10,
        atomic_numbers=torch.full((5,), 11, dtype=torch.long),
        cell=(torch.eye(3) * 15.0).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    set_rigid_groups(system, torch.arange(5))
    assert not RotateGroup(max_angle=0.2).is_applicable(system)


def test_rotate_group_raises_on_an_atoms_only_system():
    system = AtomicSystem(
        positions=torch.rand(5, 3) * 10,
        atomic_numbers=torch.full((5,), 11, dtype=torch.long),
        cell=(torch.eye(3) * 15.0).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    set_rigid_groups(system, torch.arange(5))
    move = RotateGroup(max_angle=0.2)
    try:
        move.attempt(system, torch.Generator())
        raise AssertionError("expected MoveNotApplicable")
    except MoveNotApplicable:
        pass


def test_rotate_group_targets_only_multi_atom_groups_in_a_mixture():
    """A mixture of one water molecule and several lone atoms: rotate must
    always land on the molecule, never a single-atom group."""
    water_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9281, 0.0]]
    )
    atom_positions = torch.tensor([[5.0, 5.0, 5.0], [8.0, 8.0, 8.0], [2.0, 8.0, 2.0]])
    positions = torch.cat([water_positions, atom_positions])
    atomic_numbers = torch.tensor([8, 1, 1, 11, 11, 11], dtype=torch.long)
    system = AtomicSystem(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=(torch.eye(3) * 15.0).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    set_rigid_groups(system, torch.tensor([0, 0, 0, 1, 2, 3]))

    move = RotateGroup(max_angle=0.3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        rotated, delta = move.attempt(system, generator)
        assert torch.equal(torch.sort(delta.moved_indices).values, torch.tensor([0, 1, 2]))


# -- InsertGroup ---------------------------------------------------------------


def test_insert_group_adds_one_molecule(packed_water, water_template):
    generator = torch.Generator().manual_seed(0)
    n_before = packed_water.num_nodes
    n_groups_before = int(rigid_groups(packed_water).max().item())

    move = InsertGroup(template=water_template)
    inserted, delta = move.attempt(packed_water, generator)

    assert inserted.num_nodes == n_before + water_template.num_atoms
    assert int(rigid_groups(inserted).max().item()) == n_groups_before + 1
    assert torch.equal(
        delta.added_indices,
        torch.arange(n_before, n_before + water_template.num_atoms),
    )
    assert delta.removed_indices.numel() == 0
    assert delta.moved_indices.numel() == 0


def test_insert_group_is_always_applicable(packed_water, water_template):
    assert InsertGroup(template=water_template).is_applicable(packed_water)


# -- RemoveGroup ----------------------------------------------------------------


def test_remove_group_deletes_exactly_one_molecule_and_leaves_others_untouched(
    packed_water,
):
    groups_before = set(rigid_groups(packed_water).tolist())

    move = RemoveGroup()
    removed, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))

    groups_after = set(rigid_groups(removed).tolist())
    assert removed.num_nodes == packed_water.num_nodes - 3  # one water molecule
    assert len(groups_before - groups_after) == 1  # exactly one group is gone

    removed_group_id = next(iter(groups_before - groups_after))
    for group_id in groups_after:
        before_atoms = packed_water.positions[
            group_atom_indices(packed_water, group_id)
        ]
        after_atoms = removed.positions[group_atom_indices(removed, group_id)]
        assert torch.allclose(before_atoms, after_atoms)
    assert removed_group_id not in groups_after

    expected_removed = group_atom_indices(packed_water, removed_group_id)
    assert torch.equal(
        torch.sort(delta.removed_indices).values, torch.sort(expected_removed).values
    )
    assert delta.added_indices.numel() == 0
    assert delta.moved_indices.numel() == 0


def test_remove_group_is_not_applicable_on_an_empty_system():
    empty = AtomicSystem(
        positions=torch.zeros((0, 3)),
        atomic_numbers=torch.zeros((0,), dtype=torch.long),
    )
    set_rigid_groups(empty, torch.zeros((0,), dtype=torch.long))
    assert not RemoveGroup().is_applicable(empty)


def test_remove_group_raises_when_no_groups_exist():
    empty = AtomicSystem(
        positions=torch.zeros((0, 3)),
        atomic_numbers=torch.zeros((0,), dtype=torch.long),
    )
    set_rigid_groups(empty, torch.zeros((0,), dtype=torch.long))
    move = RemoveGroup()
    try:
        move.attempt(empty, torch.Generator())
        raise AssertionError("expected MoveNotApplicable")
    except MoveNotApplicable:
        pass


# -- RelocateGroup ---------------------------------------------------------------


def test_relocate_group_preserves_molecule_count(packed_water, water_template):
    generator = torch.Generator().manual_seed(0)
    n_before = packed_water.num_nodes
    n_groups_before = len(set(rigid_groups(packed_water).tolist()))

    move = RelocateGroup(template=water_template)
    relocated, _delta = move.attempt(packed_water, generator)

    assert relocated.num_nodes == n_before  # exact molecule count conservation
    assert len(set(rigid_groups(relocated).tolist())) == n_groups_before


def test_relocate_group_delta_reports_both_removed_and_added(
    packed_water, water_template
):
    groups_before = set(rigid_groups(packed_water).tolist())

    move = RelocateGroup(template=water_template)
    relocated, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))

    groups_after = set(rigid_groups(relocated).tolist())
    assert len(groups_before - groups_after) == 1  # one group id is gone
    assert len(groups_after - groups_before) == 1  # replaced by a new one

    removed_group_id = next(iter(groups_before - groups_after))
    expected_removed = group_atom_indices(packed_water, removed_group_id)
    assert torch.equal(
        torch.sort(delta.removed_indices).values, torch.sort(expected_removed).values
    )
    assert delta.added_indices.numel() == water_template.num_atoms
    assert delta.moved_indices.numel() == 0

    # every other molecule's atoms are untouched
    for group_id in groups_before & groups_after:
        before_atoms = packed_water.positions[
            group_atom_indices(packed_water, group_id)
        ]
        after_atoms = relocated.positions[group_atom_indices(relocated, group_id)]
        assert torch.allclose(before_atoms, after_atoms)


def test_relocate_group_is_not_applicable_on_an_empty_system(water_template):
    empty = AtomicSystem(
        positions=torch.zeros((0, 3)),
        atomic_numbers=torch.zeros((0,), dtype=torch.long),
        cell=(torch.eye(3) * 15.0).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    set_rigid_groups(empty, torch.zeros((0,), dtype=torch.long))
    assert not RelocateGroup(template=water_template).is_applicable(empty)


def test_relocate_group_raises_when_no_groups_exist(water_template):
    empty = AtomicSystem(
        positions=torch.zeros((0, 3)),
        atomic_numbers=torch.zeros((0,), dtype=torch.long),
        cell=(torch.eye(3) * 15.0).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    set_rigid_groups(empty, torch.zeros((0,), dtype=torch.long))
    move = RelocateGroup(template=water_template)
    try:
        move.attempt(empty, torch.Generator())
        raise AssertionError("expected MoveNotApplicable")
    except MoveNotApplicable:
        pass
