"""Tests for MaxDistTable -- mostly the two queries that let the
constraint skip work it doesn't need to do."""

from __future__ import annotations

import torch

from raymc.config import ConstraintConfig, MinDistPairConfig
from raymc.utils import DEFAULT_MAX_DIST, MaxDistTable


def test_the_default_is_infinite():
    """Unlike MinDistTable's 1.0 A, the default here is "no constraint" --
    a maximum separation is an opt-in restraint, not something every pair
    inherently has."""
    assert DEFAULT_MAX_DIST == float("inf")
    assert MaxDistTable().default == float("inf")
    assert MaxDistTable().lookup(8, 8).item() == float("inf")


def test_pair_order_does_not_matter():
    table = MaxDistTable(pair_distances={(1, 8): 3.5})
    assert table.lookup(8, 1).item() == 3.5
    assert table.lookup(1, 8).item() == 3.5


def test_lookup_broadcasts_over_tensors():
    table = MaxDistTable(pair_distances={(8, 8): 3.5})
    z = torch.tensor([8, 1, 8])
    looked_up = table.lookup(z.unsqueeze(1), z.unsqueeze(0))
    assert looked_up.shape == (3, 3)
    assert looked_up[0, 2].item() == 3.5
    assert looked_up[0, 1].item() == float("inf")


def test_falling_back_to_the_default_never_warns(recwarn):
    """MinDistTable warns about a pair with no explicit entry; this one
    must not. With an infinite default there is nothing to warn about, and
    the fallback is the normal case rather than a likely oversight."""
    table = MaxDistTable()
    table.lookup(torch.tensor([[8]]), torch.tensor([[1]]))
    assert len(recwarn) == 0
    assert not hasattr(table, "warn_about_missing_pairs")


def test_is_unconstrained_recognizes_a_table_that_restrains_nothing():
    assert MaxDistTable().is_unconstrained()
    assert MaxDistTable(pair_distances={}).is_unconstrained()
    # An explicit "inf" entry is still no constraint.
    assert MaxDistTable(pair_distances={(8, 8): float("inf")}).is_unconstrained()

    assert not MaxDistTable(pair_distances={(8, 8): 3.5}).is_unconstrained()
    assert not MaxDistTable(default=6.0).is_unconstrained()


def test_constrained_elements_lists_only_elements_with_a_finite_entry():
    table = MaxDistTable(
        pair_distances={(8, 8): 3.5, (1, 6): 2.0, (7, 7): float("inf")}
    )
    assert table.constrained_elements().tolist() == [1, 6, 8]


def test_constrained_elements_is_none_when_the_default_itself_is_finite():
    """Nothing to narrow down: a finite default constrains every pair."""
    assert MaxDistTable(default=6.0).constrained_elements() is None
    mixed = MaxDistTable(default=6.0, pair_distances={(8, 8): 3.5})
    assert mixed.constrained_elements() is None


def test_from_config_reads_the_max_fields_not_the_min_ones():
    config = ConstraintConfig(
        default=1.0,
        pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.4)],
        max_default=8.0,
        max_pairs=[MinDistPairConfig(elements=("O", "H"), distance=3.5)],
    )
    table = MaxDistTable.from_config(config)
    assert table.default == 8.0
    assert table.lookup(1, 8).item() == 3.5
    assert table.lookup(8, 8).item() == 8.0  # the min-dist O-O entry is not read here
