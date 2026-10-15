"""Tests for make_sq_term: how experimental Q points reach the model."""

import pytest
import torch

from raymc.modalities.pdf.reference import ReferenceCurve
from raymc.residuals.aggregator import make_sq_term


def _reference(with_error: bool = False) -> ReferenceCurve:
    q = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0, 9.0], dtype=torch.float64)
    s = torch.tensor([0.1, 0.4, 0.8, 1.5, 1.1, 1.0], dtype=torch.float64)
    err = torch.full_like(q, 0.02) if with_error else None
    return ReferenceCurve(q, s, err)


def test_experimental_q_points_are_adopted_verbatim_not_resampled():
    term = make_sq_term(
        "sq", _reference(), r_max=10.0, n_r=500, radiation="neutron", sigma=0.02
    )
    # Q = 0 dropped (S(Q) is singular there); the rest kept exactly.
    assert torch.allclose(
        term.calculator.q_grid,
        torch.tensor([0.5, 1.0, 2.0, 5.0, 9.0], dtype=torch.float64),
    )
    assert torch.allclose(
        term.target, torch.tensor([0.4, 0.8, 1.5, 1.1, 1.0], dtype=torch.float64)
    )


def test_q_min_and_q_max_select_a_subrange_rmca_nx1_nx2_style():
    term = make_sq_term(
        "sq", _reference(), r_max=10.0, n_r=500, radiation="neutron",
        sigma=0.02, q_min=1.0, q_max=5.0,
    )
    assert torch.allclose(
        term.calculator.q_grid, torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
    )
    assert term.target.numel() == 3


def test_selection_that_keeps_nothing_raises():
    with pytest.raises(ValueError, match="no experimental Q points survive"):
        make_sq_term(
            "sq", _reference(), r_max=10.0, radiation="neutron", sigma=0.02,
            q_min=50.0,
        )


def test_error_column_is_used_when_no_sigma_given():
    term = make_sq_term(
        "sq", _reference(with_error=True), r_max=10.0, radiation="neutron"
    )
    assert torch.allclose(term.sigma, torch.full((5,), 0.02, dtype=torch.float64))


def test_missing_sigma_and_missing_error_column_raises():
    with pytest.raises(ValueError, match="no sigma given"):
        make_sq_term("sq", _reference(), r_max=10.0, radiation="neutron")
