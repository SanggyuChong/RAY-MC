from __future__ import annotations

import pytest
import torch

from raymc.system import (
    AtomicSystem,
    check_no_atomic_overlap,
    check_no_unphysical_clash,
    group_centroid,
    set_rigid_groups,
)
from raymc.topology import molecule_ids
from raymc.utils import MinDistTable


def test_set_rigid_groups_accepts_a_valid_single_molecule(single_water):
    # fixture setup already calls set_rigid_groups; no exception means pass
    assert molecule_ids(single_water).tolist() == [0, 0, 0]


def test_set_rigid_groups_accepts_two_well_separated_molecules(water_template):
    positions = torch.cat(
        [
            water_template.positions,
            water_template.positions + torch.tensor([10.0, 0.0, 0.0]),
        ]
    )
    atomic_numbers = torch.cat([water_template.atomic_numbers] * 2)
    system = AtomicSystem(positions=positions, atomic_numbers=atomic_numbers)
    set_rigid_groups(system, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert molecule_ids(system).tolist() == [0, 0, 0, 1, 1, 1]


def test_set_rigid_groups_allows_a_bond_crossing_group_boundaries(water_template):
    """A real bond crossing a rigid-group boundary is a legitimate modeling
    choice (e.g. MOF linker vs. metal cluster) and must not be rejected."""
    system = AtomicSystem(
        positions=water_template.positions.clone(),
        atomic_numbers=water_template.atomic_numbers.clone(),
    )
    # 3rd atom (bonded to the rest) put in its own group -- should not raise.
    set_rigid_groups(system, torch.tensor([0, 0, 1]))
    assert molecule_ids(system).tolist() == [0, 0, 1]


def test_set_rigid_groups_rejects_a_disconnected_group(water_template):
    """Two separate, well-separated molecules mistakenly sharing one group
    id must be rejected: they don't form a single bonded fragment."""
    positions = torch.cat(
        [
            water_template.positions,
            water_template.positions + torch.tensor([10.0, 0.0, 0.0]),
        ]
    )
    atomic_numbers = torch.cat([water_template.atomic_numbers] * 2)
    system = AtomicSystem(positions=positions, atomic_numbers=atomic_numbers)
    with pytest.raises(ValueError, match="not a single connected bonded fragment"):
        set_rigid_groups(system, torch.tensor([0, 0, 0, 0, 0, 0]))


def test_set_rigid_groups_handles_a_molecule_straddling_a_periodic_boundary():
    """PBC minimum-image handling: a molecule split across the cell edge in
    raw coordinates must still be recognized as one bonded fragment."""
    cell = torch.eye(3).unsqueeze(0) * 5.0
    pbc = torch.tensor([[True, True, True]])
    positions = torch.tensor(
        [[4.9, 2.5, 2.5], [0.1, 2.5, 2.5], [4.9, 3.3, 2.5]]  # O, H, H
    )
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    system = AtomicSystem(
        positions=positions, atomic_numbers=atomic_numbers, cell=cell, pbc=pbc
    )
    set_rigid_groups(system, torch.tensor([0, 0, 0]))  # should not raise


def test_check_no_atomic_overlap_passes_for_a_real_molecule(single_water):
    check_no_atomic_overlap(single_water)  # should not raise, default MinDistTable


def test_check_no_atomic_overlap_rejects_a_clash():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )  # far closer than any sane O-O min-dist
    atomic_numbers = torch.tensor([8, 8], dtype=torch.long)
    system = AtomicSystem(positions=positions, atomic_numbers=atomic_numbers)
    set_rigid_groups(system, torch.tensor([0, 1]))  # different groups: non-bonded
    with pytest.raises(ValueError, match="unphysical atomic overlap"):
        check_no_atomic_overlap(system)


def test_check_no_atomic_overlap_uses_the_supplied_min_dist_table():
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atomic_numbers = torch.tensor([8, 8], dtype=torch.long)
    system = AtomicSystem(positions=positions, atomic_numbers=atomic_numbers)
    set_rigid_groups(system, torch.tensor([0, 1]))

    check_no_atomic_overlap(system, MinDistTable(default=1.0))  # 2.0 > 1.0: fine
    with pytest.raises(ValueError, match="unphysical atomic overlap"):
        check_no_atomic_overlap(system, MinDistTable(default=2.5))  # 2.0 < 2.5: violates


def test_check_no_atomic_overlap_never_checks_bonded_same_group_pairs(single_water):
    """A water molecule's own O-H bond (~0.96A) would violate almost any
    sane non-bonded min-dist default -- same-rigid-group pairs must never
    be checked, or every real molecule would look like unphysical overlap."""
    check_no_atomic_overlap(single_water, MinDistTable(default=1.0))  # should not raise


def test_group_centroid_matches_mean_position_for_a_non_periodic_system(single_water):
    expected = single_water.positions.to(torch.float64).mean(dim=0)
    assert torch.allclose(group_centroid(single_water, group_id=0), expected, atol=1e-6)


def test_check_no_unphysical_clash_needs_no_rigid_groups_to_accept_a_real_bond():
    """Every atom defaults to its own rigid group with no explicit
    grouping -- unlike check_no_atomic_overlap's MinDistTable, the
    covalent-radius check must still accept a real bonded molecule."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9281, 0.0]]
    )
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    system = AtomicSystem(positions=positions, atomic_numbers=atomic_numbers)
    check_no_unphysical_clash(system)  # should not raise


def test_check_no_unphysical_clash_rejects_a_genuine_clash():
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    atomic_numbers = torch.tensor([8, 8], dtype=torch.long)
    system = AtomicSystem(positions=positions, atomic_numbers=atomic_numbers)
    with pytest.raises(ValueError, match="unphysical atomic overlap"):
        check_no_unphysical_clash(system)


# -- carry_node_labels -----------------------------------------------------


def test_carry_node_labels_copies_every_bookkeeping_key(packed_water):
    from raymc.system import BOOKKEEPING_KEYS, AtomicSystem, carry_node_labels

    fresh = AtomicSystem(
        positions=packed_water.positions.clone(),
        atomic_numbers=packed_water.atomic_numbers,
        cell=packed_water.cell,
        pbc=packed_water.pbc,
    )
    carry_node_labels(packed_water, fresh)
    for key in BOOKKEEPING_KEYS:
        assert torch.equal(getattr(fresh, key), getattr(packed_water, key))


def test_carry_node_labels_reindexes_when_atoms_are_dropped(packed_water):
    from raymc.system import AtomicSystem, carry_node_labels, rigid_groups

    keep = torch.arange(0, packed_water.num_nodes, 2)
    fewer = AtomicSystem(
        positions=packed_water.positions[keep],
        atomic_numbers=packed_water.atomic_numbers[keep],
        cell=packed_water.cell,
        pbc=packed_water.pbc,
    )
    carry_node_labels(packed_water, fewer, index=keep)
    assert torch.equal(rigid_groups(fewer), rigid_groups(packed_water)[keep])


def test_a_new_atom_gets_a_rigid_group_of_its_own(packed_water):
    """An inserted atom belongs to a rigid group of its own -- so the
    distance constraints treat it as a free atom rather than silently
    excluding it from whichever group it inherited."""
    from raymc.system import AtomicSystem, carry_node_labels, rigid_groups

    n = packed_water.num_nodes
    index = torch.cat([torch.arange(n), torch.tensor([-1])])
    bigger = AtomicSystem(
        positions=torch.cat(
            [
                packed_water.positions,
                torch.zeros(1, 3, dtype=packed_water.positions.dtype),
            ]
        ),
        atomic_numbers=torch.cat(
            [
                packed_water.atomic_numbers,
                torch.tensor([19], dtype=packed_water.atomic_numbers.dtype),
            ]
        ),
        cell=packed_water.cell,
        pbc=packed_water.pbc,
    )
    carry_node_labels(packed_water, bigger, index=index)

    groups = rigid_groups(bigger)
    assert int((groups == groups[-1]).sum()) == 1


def test_an_index_of_the_wrong_length_is_rejected(packed_water):
    from raymc.system import AtomicSystem, carry_node_labels

    fresh = AtomicSystem(
        positions=packed_water.positions.clone(),
        atomic_numbers=packed_water.atomic_numbers,
    )
    with pytest.raises(ValueError, match="one per destination atom"):
        carry_node_labels(packed_water, fresh, index=torch.arange(3))
