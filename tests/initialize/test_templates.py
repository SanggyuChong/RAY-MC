from __future__ import annotations

import pytest

from raymc.initialize.templates import GroupTemplate, load_group_template, load_molecule


def test_load_molecule_water(water_template):
    assert water_template.symbols == ("O", "H", "H")
    assert water_template.num_atoms == 3
    assert water_template.atomic_numbers.tolist() == [8, 1, 1]


def test_load_molecule_benzene(benzene_template):
    assert benzene_template.num_atoms == 12
    assert benzene_template.symbols.count("C") == 6
    assert benzene_template.symbols.count("H") == 6


def test_load_molecule_centers_on_its_own_centroid(water_template):
    assert water_template.positions.mean(dim=0).abs().max().item() < 1e-5


def test_load_molecule_unknown_name_raises():
    with pytest.raises(FileNotFoundError):
        load_molecule("not_a_real_molecule")


def test_load_molecule_accepts_a_user_supplied_xyz_path():
    template = load_molecule("src/raymc/database/molecules/water.xyz")
    assert template.name == "water"
    assert template.symbols == ("O", "H", "H")


def test_group_template_atom_builds_a_single_atom_template():
    na = GroupTemplate.atom("Na")
    assert na.name == "Na"
    assert na.num_atoms == 1
    assert na.symbols == ("Na",)
    assert na.atomic_numbers.tolist() == [11]
    assert na.positions.tolist() == [[0.0, 0.0, 0.0]]


def test_load_group_template_resolves_a_molecule_name():
    template = load_group_template("water")
    assert template.num_atoms == 3


def test_load_group_template_resolves_a_bare_element_symbol():
    template = load_group_template("Cl")
    assert template.num_atoms == 1
    assert template.atomic_numbers.tolist() == [17]


def test_load_group_template_raises_for_neither_molecule_nor_element():
    with pytest.raises(FileNotFoundError):
        load_group_template("not_a_molecule_or_element")
