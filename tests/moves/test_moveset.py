from __future__ import annotations

import pytest
import torch

from raymc.config import MoveConfig
from raymc.moves.group_based import InsertGroup, RemoveGroup, RotateGroup, TranslateGroup
from raymc.moves.moveset import MoveSet
from raymc.system import AtomicSystem, set_rigid_groups


def test_from_config_builds_the_right_move_types():
    configs = [
        MoveConfig(type="translate_group", params={"max_displacement": 0.3}),
        MoveConfig(type="rotate_group", params={"max_angle": 0.3}),
        MoveConfig(type="insert_group", params={"molecule": "water"}),
        MoveConfig(type="remove_group"),
    ]
    move_set = MoveSet.from_config(configs)
    types = [type(m) for m in move_set.moves]
    assert types == [TranslateGroup, RotateGroup, InsertGroup, RemoveGroup]


def test_from_config_rejects_unknown_move_type():
    with pytest.raises(ValueError, match="Unknown move type"):
        MoveSet.from_config([MoveConfig(type="not_a_real_move")])


def test_moveset_requires_at_least_one_move():
    with pytest.raises(ValueError, match="at least one move"):
        MoveSet(moves=[], weights=[])


def test_sample_returns_a_candidate_the_move_name_and_a_delta(packed_water):
    move_set = MoveSet(moves=[TranslateGroup(max_displacement=0.3)], weights=[1.0])
    candidate, name, delta = move_set.sample(
        packed_water, torch.Generator().manual_seed(0)
    )
    assert name == "translate"
    assert candidate.num_nodes == packed_water.num_nodes
    assert delta.moved_indices.numel() == 3  # one water molecule


def test_sample_never_draws_an_inapplicable_move():
    """A RemoveGroup mixed in with an atoms-only, single-atom-groups system
    would previously need retries to avoid RotateGroup (never applicable
    there); now it must simply never be drawn at all."""
    system = AtomicSystem(
        positions=torch.rand(5, 3) * 10,
        atomic_numbers=torch.full((5,), 11, dtype=torch.long),
        cell=(torch.eye(3) * 15.0).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    set_rigid_groups(system, torch.arange(5))

    move_set = MoveSet(
        moves=[TranslateGroup(max_displacement=0.5), RotateGroup(max_angle=0.3)],
        weights=[1.0, 1.0],
    )
    generator = torch.Generator().manual_seed(0)
    for _ in range(30):
        _candidate, name, _delta = move_set.sample(system, generator)
        assert name == "translate"  # rotate is never applicable here


def test_sample_raises_when_no_move_is_applicable():
    empty = AtomicSystem(
        positions=torch.zeros((0, 3)),
        atomic_numbers=torch.zeros((0,), dtype=torch.long),
    )
    set_rigid_groups(empty, torch.zeros((0,), dtype=torch.long))
    move_set = MoveSet(moves=[RemoveGroup()], weights=[1.0])
    with pytest.raises(RuntimeError, match="No applicable move"):
        move_set.sample(empty, torch.Generator())
