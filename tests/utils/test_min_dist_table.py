from __future__ import annotations

import warnings

import torch

from raymc.config import ConstraintConfig, MinDistPairConfig
from raymc.utils import DEFAULT_MIN_DIST, MinDistTable


def test_lookup_returns_explicit_pair_distance():
    table = MinDistTable(pair_distances={(8, 8): 2.4})
    assert table.lookup(8, 8).item() == 2.4


def test_lookup_is_order_independent():
    table = MinDistTable(pair_distances={(8, 1): 1.5})
    assert table.lookup(8, 1).item() == 1.5
    assert table.lookup(1, 8).item() == 1.5


def test_lookup_falls_back_to_default():
    table = MinDistTable(pair_distances={(8, 8): 2.4}, default=1.2)
    assert table.lookup(1, 1).item() == 1.2


def test_default_min_dist_constant_is_one_angstrom():
    table = MinDistTable()
    assert table.default == DEFAULT_MIN_DIST == 1.0


def test_lookup_broadcasts_over_tensors():
    table = MinDistTable(pair_distances={(8, 8): 2.4}, default=1.0)
    z = torch.tensor([8, 1])
    result = table.lookup(z.unsqueeze(1), z.unsqueeze(0))
    assert result.shape == (2, 2)
    assert result[0, 0].item() == 2.4  # O-O
    assert result[0, 1].item() == 1.0  # O-H, unlisted -> default
    assert result[1, 1].item() == 1.0  # H-H, unlisted -> default


def test_warn_about_missing_pairs_fires_once_per_pair():
    """touched=H against all_z={O, H} means two missing pairs, (H,O) and
    (H,H) -- both must warn once on the first call, and neither again on
    a repeated call with the same atoms."""
    table = MinDistTable(pair_distances={(8, 8): 2.4})
    touched = torch.tensor([1, 1, 1])
    all_z = torch.tensor([8, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table.warn_about_missing_pairs(touched, all_z)
    first_call = [w for w in caught if "defaulting to" in str(w.message)]
    assert len(first_call) == 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table.warn_about_missing_pairs(touched, all_z)
    second_call = [w for w in caught if "defaulting to" in str(w.message)]
    assert len(second_call) == 0


def test_warn_about_missing_pairs_never_fires_for_an_explicit_pair():
    table = MinDistTable(pair_distances={(8, 8): 2.4})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table.warn_about_missing_pairs(torch.tensor([8]), torch.tensor([8]))
    assert not any("defaulting to" in str(w.message) for w in caught)


def test_from_config_builds_the_right_pair_table():
    config = ConstraintConfig(
        default=1.2,
        pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.4)],
    )
    table = MinDistTable.from_config(config)
    assert table.default == 1.2
    assert table.lookup(8, 8).item() == 2.4
    assert table.lookup(1, 1).item() == 1.2  # falls back to default
