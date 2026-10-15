from __future__ import annotations

import pytest
import torch

from raymc.modalities.base import Calculator
from raymc.moves.base import MoveDelta


class _ConstantCalculator(Calculator):
    """Minimal concrete Calculator overriding only the abstract method,
    to exercise the ABC's default init/compute_increm_change/commit/discard."""

    def __init__(self, value: float):
        self.value = value
        self.compute_calls = 0

    def compute(self, system) -> torch.Tensor:
        self.compute_calls += 1
        return torch.tensor(self.value, dtype=torch.float64)


def test_calculator_cannot_be_instantiated_without_compute():
    with pytest.raises(TypeError):
        Calculator()


def test_default_init_delegates_to_compute():
    calc = _ConstantCalculator(3.0)
    result = calc.init(system=None)
    assert result.item() == 3.0
    assert calc.compute_calls == 1


def test_default_compute_increm_change_falls_back_to_compute():
    calc = _ConstantCalculator(5.0)
    delta = MoveDelta(moved_indices=torch.tensor([0]))
    result = calc.compute_increm_change(system=None, delta=delta)
    assert result.item() == 5.0
    assert calc.compute_calls == 1


def test_default_commit_and_discard_are_no_ops():
    calc = _ConstantCalculator(1.0)
    calc.commit()  # should not raise
    calc.discard()  # should not raise
