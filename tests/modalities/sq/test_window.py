import math

import pytest
import torch

from raymc.modalities.sq.window import window_function


def test_none_window_is_all_ones():
    r = torch.linspace(0.1, 10.0, 50, dtype=torch.float64)
    assert torch.allclose(window_function("none", r, 10.0), torch.ones_like(r))


def test_lorch_starts_at_one_and_vanishes_at_r_max():
    r = torch.tensor([1e-12, 5.0, 10.0], dtype=torch.float64)
    w = window_function("lorch", r, 10.0)
    assert w[0].item() == pytest.approx(1.0)
    # sinc(pi/2) = 2/pi
    assert w[1].item() == pytest.approx(2.0 / math.pi)
    assert w[2].item() == pytest.approx(0.0, abs=1e-12)


def test_hann_is_a_raised_cosine_reaching_zero_at_r_max():
    r = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
    w = window_function("hann", r, 10.0)
    assert w[0].item() == pytest.approx(1.0)
    assert w[1].item() == pytest.approx(0.5)
    assert w[2].item() == pytest.approx(0.0, abs=1e-12)


def test_hann_damps_harder_than_lorch():
    r = torch.linspace(0.5, 9.5, 40, dtype=torch.float64)
    assert torch.all(
        window_function("hann", r, 10.0) <= window_function("lorch", r, 10.0) + 1e-12
    )


def test_unknown_window_raises():
    with pytest.raises(ValueError, match="Unknown window kind"):
        window_function("gaussian", torch.ones(3, dtype=torch.float64), 10.0)
