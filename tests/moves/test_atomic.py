"""Tests for the single-atom translation move."""

from __future__ import annotations

import pytest
import torch

from raymc.config import (
    ConstraintConfig,
    IntramolecularWindowConfig,
    MinDistPairConfig,
    MoveConfig,
)
from raymc.initialize.pack import pack_on_lattice
from raymc.initialize.templates import GroupTemplate
from raymc.modalities.pdf.calculator import PDFCalculator
from raymc.moves.atomic import TranslateAtom
from raymc.moves.moveset import MoveSet
from raymc.system import rigid_groups
from raymc.utils import MinDistTable, minimum_image_delta
from tests.conftest import uniform_r_grid


def _water_box(water_template, n=8, box=18.0, seed=1):
    return pack_on_lattice(
        water_template,
        cell=torch.eye(3) * box,
        n_particles=n,
        min_dist=MinDistTable(default=2.0),
        seed=seed,
    )


def test_moves_exactly_one_atom(water_template):
    system = _water_box(water_template)
    move = TranslateAtom(max_displacement=0.1)
    candidate, delta = move.attempt(system, torch.Generator().manual_seed(0))

    assert delta.moved_indices.numel() == 1
    assert delta.added_indices.numel() == 0
    assert delta.removed_indices.numel() == 0

    changed = (candidate.positions != system.positions).any(dim=1).nonzero().flatten()
    assert changed.tolist() == delta.moved_indices.tolist()


def test_displacement_never_exceeds_max(water_template):
    system = _water_box(water_template)
    move = TranslateAtom(max_displacement=0.05)
    generator = torch.Generator().manual_seed(7)
    cell = system.cell.squeeze(0).to(torch.float64)
    pbc = system.pbc.squeeze(0)

    for _ in range(200):
        candidate, delta = move.attempt(system, generator)
        index = delta.moved_indices
        shift = minimum_image_delta(
            candidate.positions[index].to(torch.float64),
            system.positions[index].to(torch.float64),
            cell,
            pbc,
        )
        assert torch.linalg.norm(shift, dim=-1).max().item() <= 0.05 + 1e-9


def test_input_is_not_mutated_and_topology_is_carried(water_template):
    system = _water_box(water_template)
    before = system.positions.clone()
    move = TranslateAtom(max_displacement=0.3)
    candidate, _ = move.attempt(system, torch.Generator().manual_seed(3))

    assert torch.equal(system.positions, before)
    assert candidate.num_nodes == system.num_nodes
    assert torch.equal(candidate.atomic_numbers, system.atomic_numbers)
    assert torch.equal(rigid_groups(candidate), rigid_groups(system))


def test_moved_atom_is_wrapped_back_into_the_box(water_template):
    """A big enough step from near an edge must land inside the cell, not
    drift outside it -- positions would otherwise wander over a long run."""
    system = _water_box(water_template)
    move = TranslateAtom(max_displacement=3.0)
    generator = torch.Generator().manual_seed(11)
    box = 18.0
    for _ in range(300):
        candidate, _ = move.attempt(system, generator)
        assert candidate.positions.min().item() >= -1e-9
        assert candidate.positions.max().item() < box + 1e-9


def test_is_applicable_on_a_monatomic_system():
    """Nothing about this move needs a molecule."""
    system = pack_on_lattice(
        GroupTemplate.atom("Ar"),
        cell=torch.eye(3) * 20.0,
        n_particles=8,
        min_dist=MinDistTable(default=3.0),
        seed=0,
    )
    assert TranslateAtom(max_displacement=0.1).is_applicable(system)


# -- the handoff's explicit "verify, don't assume" ---------------------------
#
# A single-index MoveDelta has to be supported end to end by the PDF
# calculator's incremental path, in both of the cases a water box produces:
# the moved atom is the partial's own species, or it is not.


@pytest.mark.parametrize("element", ["O", "H"])
def test_single_atom_delta_matches_a_full_pdf_recompute(water_template, element):
    system = _water_box(water_template, n=12, box=20.0)
    calculator = PDFCalculator(r_grid=uniform_r_grid(1.0, 8.0, 140), atomic_number_a=8)
    calculator.init(system)

    target_z = 8 if element == "O" else 1
    move = TranslateAtom(max_displacement=0.2)
    generator = torch.Generator().manual_seed(5)

    checked = 0
    state = system
    for _ in range(400):
        candidate, delta = move.attempt(state, generator)
        if int(candidate.atomic_numbers[delta.moved_indices[0]]) != target_z:
            continue
        incremental = calculator.compute_increm_change(candidate, delta)
        full = calculator.compute(candidate)
        assert torch.allclose(incremental, full, atol=1e-10), (
            f"incremental PDF diverged from a full recompute after moving "
            f"a single {element} atom"
        )
        calculator.commit()
        state = candidate
        checked += 1
        if checked == 15:
            break
    assert checked == 15, f"only exercised {checked} single-{element} moves"


# -- the guard rail ---------------------------------------------------------


_ATOMIC_MOVE = [MoveConfig(type="translate_atom", params={"max_displacement": 0.05})]


def test_from_config_warns_when_nothing_restrains_a_molecule():
    with pytest.warns(UserWarning, match="NO intramolecular constraint"):
        MoveSet.from_config(_ATOMIC_MOVE, ConstraintConfig(default=1.0))


def test_from_config_is_quiet_when_windows_are_configured(recwarn):
    MoveSet.from_config(
        _ATOMIC_MOVE,
        ConstraintConfig(
            intramolecular=[
                IntramolecularWindowConfig(elements=("O", "H"), min=0.9, max=1.1)
            ]
        ),
    )
    assert not [w for w in recwarn if "intramolecular" in str(w.message)]


def test_from_config_is_quiet_for_group_moves_alone(recwarn):
    """An inter-group min-dist table is not a substitute, but a run with no
    atomic move has nothing to warn about either."""
    MoveSet.from_config(
        [MoveConfig(type="translate_group", params={"max_displacement": 0.3})],
        ConstraintConfig(pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.4)]),
    )
    assert not [w for w in recwarn if "intramolecular" in str(w.message)]


def test_from_config_without_constraints_skips_the_check(recwarn):
    """Callers assembling a move set by hand need not supply constraints."""
    MoveSet.from_config(_ATOMIC_MOVE)
    assert not [w for w in recwarn if "intramolecular" in str(w.message)]


def test_registry_builds_the_move_with_its_configured_step():
    move_set = MoveSet.from_config(
        [MoveConfig(type="translate_atom", params={"max_displacement": 0.07})]
    )
    assert isinstance(move_set.moves[0], TranslateAtom)
    assert move_set.moves[0].max_displacement == 0.07
    assert move_set.moves[0].name == "translate_atom"


def test_name_is_distinct_from_the_group_translate():
    """RunHistory keys on move.name, so the two must not merge in the
    per-move acceptance breakdown."""
    from raymc.moves.group_based import TranslateGroup

    assert TranslateAtom(0.1).name != TranslateGroup(0.1).name


@pytest.mark.parametrize("element", ["O", "H"])
def test_single_atom_delta_matches_a_full_sq_recompute(water_template, element):
    """Same verification for the S(Q) modality, which the flexible-vs-rigid
    comparison is measured through."""
    from raymc.modalities.sq.calculator import SQCalculator

    system = _water_box(water_template, n=12, box=20.0)
    calculator = SQCalculator(
        q_grid=torch.linspace(0.5, 12.0, 60, dtype=torch.float64),
        r_max=9.0,
        n_r=180,
        radiation="neutron",
    )
    calculator.init(system)

    target_z = 8 if element == "O" else 1
    move = TranslateAtom(max_displacement=0.2)
    generator = torch.Generator().manual_seed(9)

    checked = 0
    state = system
    for _ in range(400):
        candidate, delta = move.attempt(state, generator)
        if int(candidate.atomic_numbers[delta.moved_indices[0]]) != target_z:
            continue
        incremental = calculator.compute_increm_change(candidate, delta)
        full = calculator.compute(candidate)
        assert torch.allclose(incremental, full, atol=1e-8), (
            f"incremental S(Q) diverged from a full recompute after moving "
            f"a single {element} atom"
        )
        calculator.commit()
        state = candidate
        checked += 1
        if checked == 10:
            break
    assert checked == 10, f"only exercised {checked} single-{element} moves"
