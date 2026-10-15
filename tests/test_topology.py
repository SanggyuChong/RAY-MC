"""Tests for the topology module."""

from __future__ import annotations

import pytest
import torch

from raymc.system import AtomicSystem
from raymc.topology import (
    bonds,
    carry_topology,
    check_molecules_are_connected,
    connected_components,
    establish_topology,
    has_molecule_ids,
    molecule_atom_indices,
    molecule_ids,
    perceive_bonds,
    remap_indices,
    set_bonds,
    set_molecule_ids,
)


def _box(positions, numbers, box=20.0):
    return AtomicSystem(
        positions=torch.tensor(positions, dtype=torch.float64),
        atomic_numbers=torch.tensor(numbers),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * box,
        pbc=torch.tensor([[True, True, True]]),
    )


def _two_waters(water_template):
    offset = torch.tensor([8.0, 0.0, 0.0])
    positions = torch.cat(
        [water_template.positions, water_template.positions + offset]
    )
    system = AtomicSystem(
        positions=positions.to(torch.float64),
        atomic_numbers=torch.cat([water_template.atomic_numbers] * 2),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 20.0,
        pbc=torch.tensor([[True, True, True]]),
    )
    return set_molecule_ids(system, torch.tensor([0, 0, 0, 1, 1, 1]))


# -- membership ------------------------------------------------------------


def test_membership_round_trips(water_template):
    system = _two_waters(water_template)
    assert has_molecule_ids(system)
    assert molecule_ids(system).tolist() == [0, 0, 0, 1, 1, 1]
    assert molecule_atom_indices(system, 1).tolist() == [3, 4, 5]


def test_missing_topology_warns_instead_of_guessing_silently():
    """The fallback gains teeth: it still guesses one molecule per atom,
    but it says so."""
    import raymc.topology as topology

    topology._warned_missing = False
    system = _box([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [8, 1])
    with pytest.warns(UserWarning, match="never set"):
        ids = molecule_ids(system)
    assert ids.tolist() == [0, 1]
    assert not has_molecule_ids(system)


def test_set_molecule_ids_rejects_a_wrong_length():
    system = _box([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [8, 1])
    with pytest.raises(ValueError, match="expected 2"):
        set_molecule_ids(system, torch.tensor([0, 0, 0]), validate=False)


# -- bonds -----------------------------------------------------------------


def test_bonds_absent_is_distinguishable_from_no_bonds(water_template):
    system = _two_waters(water_template)
    assert bonds(system) is None
    set_bonds(system, torch.empty((0, 2), dtype=torch.long))
    pairs, shifts = bonds(system)
    assert pairs.shape == (0, 2)
    assert shifts.shape == (0, 3)


def test_perceive_bonds_finds_a_waters_two_OH_bonds(water_template):
    system = _two_waters(water_template)
    pairs, shifts = perceive_bonds(system)
    assert pairs.shape[0] == 4  # two O-H per molecule
    assert shifts.shape == (4, 3)
    for i, j in pairs.tolist():
        assert molecule_ids(system)[i] == molecule_ids(system)[j]


def test_set_bonds_rejects_mismatched_shifts():
    system = _box([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [8, 1])
    with pytest.raises(ValueError, match="one per bond"):
        set_bonds(system, torch.tensor([[0, 1]]), torch.zeros((2, 3), dtype=torch.long))


def test_bonds_survive_a_stretch_that_perception_would_lose(water_template):
    """The whole reason bonds become data: a bond stretched past
    BOND_TOLERANCE stops being *perceived*, but must not stop existing."""
    system = _two_waters(water_template)
    pairs, shifts = perceive_bonds(system)
    set_bonds(system, pairs, shifts)

    stretched = AtomicSystem(
        positions=system.positions.clone(),
        atomic_numbers=system.atomic_numbers,
        cell=system.cell,
        pbc=system.pbc,
    )
    stretched.positions[1] = stretched.positions[0] + torch.tensor(
        [2.5, 0.0, 0.0], dtype=torch.float64
    )
    carry_topology(stretched, system)

    # Perception on the stretched geometry loses the bond ...
    assert perceive_bonds(stretched)[0].shape[0] < pairs.shape[0]
    # ... but the recorded bonds still hold the molecule together, so the
    # connectivity check passes rather than declaring it dissolved.
    check_molecules_are_connected(stretched)


# -- connected components --------------------------------------------------


def test_components_split_two_molecules(water_template):
    system = _two_waters(water_template)
    pairs, shifts = perceive_bonds(system)
    component, periodic = connected_components(6, pairs, shifts)
    assert len(set(component.tolist())) == 2
    assert not bool(periodic.any())


def test_a_chain_that_wraps_through_the_box_is_flagged_periodic():
    """A component reaching one of its own atoms at a different image has
    no finite molecule -- it percolates."""
    # 0-1, 1-2, and 2-0 closing through the +x image: a wrapped ring.
    pairs = torch.tensor([[0, 1], [1, 2], [2, 0]])
    shifts = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    component, periodic = connected_components(3, pairs, shifts)
    assert component.tolist() == [0, 0, 0]
    assert periodic.tolist() == [True]


def test_an_unwrapped_ring_is_not_flagged_periodic():
    pairs = torch.tensor([[0, 1], [1, 2], [2, 0]])
    shifts = torch.zeros((3, 3), dtype=torch.long)
    _, periodic = connected_components(3, pairs, shifts)
    assert periodic.tolist() == [False]


def test_isolated_atoms_are_their_own_components():
    component, periodic = connected_components(3, torch.empty((0, 2), dtype=torch.long))
    assert component.tolist() == [0, 1, 2]
    assert periodic.tolist() == [False, False, False]


def test_establish_topology_labels_a_packed_box(water_template):
    system = _two_waters(water_template)
    establish_topology(system)
    assert bonds(system)[0].shape[0] == 4
    assert sorted(set(molecule_ids(system).tolist())) == [0, 1]


def test_establish_topology_refuses_a_percolating_structure():
    """A chain of atoms bonded all the way around the box is not a set of
    molecules, and is reported rather than guessed at."""
    box = 6.0
    positions = [[x * 1.5, 0.0, 0.0] for x in range(4)]
    system = _box(positions, [6, 6, 6, 6], box=box)
    with pytest.raises(ValueError, match="percolating network"):
        establish_topology(system)


# -- carrying topology -----------------------------------------------------


def test_carry_topology_moves_membership_and_bonds(water_template):
    system = _two_waters(water_template)
    establish_topology(system)
    new = AtomicSystem(
        positions=system.positions.clone(),
        atomic_numbers=system.atomic_numbers,
        cell=system.cell,
        pbc=system.pbc,
    )
    assert bonds(new) is None
    carry_topology(new, system)
    assert molecule_ids(new).tolist() == molecule_ids(system).tolist()
    assert torch.equal(bonds(new)[0], bonds(system)[0])


def test_carry_topology_renumbers_after_a_removal(water_template):
    system = _two_waters(water_template)
    establish_topology(system)
    keep = torch.tensor([False, False, False, True, True, True])
    new = AtomicSystem(
        positions=system.positions[keep],
        atomic_numbers=system.atomic_numbers[keep],
        cell=system.cell,
        pbc=system.pbc,
    )
    carry_topology(new, system, keep_mask=keep)
    assert molecule_ids(new).tolist() == [1, 1, 1]
    # The surviving molecule's two bonds, renumbered into 0..2.
    pairs = bonds(new)[0]
    assert pairs.shape[0] == 2
    assert int(pairs.max()) <= 2


def test_carry_topology_appends_an_inserted_molecule(water_template):
    system = _two_waters(water_template)
    establish_topology(system)
    added_positions = system.positions[:3] + torch.tensor(
        [0.0, 8.0, 0.0], dtype=torch.float64
    )
    new = AtomicSystem(
        positions=torch.cat([system.positions, added_positions]),
        atomic_numbers=torch.cat([system.atomic_numbers, system.atomic_numbers[:3]]),
        cell=system.cell,
        pbc=system.pbc,
    )
    carry_topology(
        new,
        system,
        added_molecule_ids=torch.tensor([2, 2, 2]),
        added_bonds=torch.tensor([[0, 1], [0, 2]]),
    )
    assert molecule_ids(new).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    pairs = bonds(new)[0]
    assert [6, 7] in pairs.tolist() and [6, 8] in pairs.tolist()


def test_remap_indices_drops_dangling_bonds():
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3]])
    shifts = torch.zeros((3, 3), dtype=torch.long)
    keep = torch.tensor([False, True, True, True])
    new_pairs, new_shifts = remap_indices(pairs, shifts, keep)
    # (0, 1) touched a removed atom and is gone; the rest shift down by 1.
    assert new_pairs.tolist() == [[0, 1], [1, 2]]
    assert new_shifts.shape[0] == 2


def test_carry_topology_is_a_no_op_when_there_is_nothing_to_carry():
    system = _box([[0.0, 0.0, 0.0]], [8])
    new = _box([[0.0, 0.0, 0.0]], [8])
    carry_topology(new, system)
    assert not has_molecule_ids(new)
    assert bonds(new) is None
