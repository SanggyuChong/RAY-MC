from __future__ import annotations

from raymc.initialize.load import load_structure


def test_load_structure_reads_the_bundled_water_xyz():
    system = load_structure("src/raymc/database/molecules/water.xyz")
    assert system.num_nodes == 3
    assert system.atomic_numbers.tolist() == [8, 1, 1]
    assert system.cell is None
    assert system.pbc is None
