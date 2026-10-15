from __future__ import annotations

import pytest
import torch

from raymc.utils.geometry import (
    minimum_image_delta,
    random_rotation_matrices,
    rotation_about_random_axis,
)


def test_random_rotation_matrices_are_orthogonal_with_unit_determinant():
    rotations = random_rotation_matrices(50, torch.Generator().manual_seed(0))
    identity = torch.eye(3, dtype=torch.float64).expand(50, 3, 3)
    assert torch.allclose(
        rotations @ rotations.transpose(-1, -2), identity, atol=1e-5
    )
    dets = torch.linalg.det(rotations)
    assert torch.allclose(dets, torch.ones(50, dtype=dets.dtype), atol=1e-5)


def test_random_rotation_matrices_are_reproducible_with_a_seed():
    a = random_rotation_matrices(5, torch.Generator().manual_seed(1))
    b = random_rotation_matrices(5, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_rotation_about_random_axis_is_a_valid_rotation():
    rotation = rotation_about_random_axis(0.5, torch.Generator().manual_seed(0))
    assert rotation.shape == (3, 3)
    assert torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=torch.float64), atol=1e-6)
    assert torch.linalg.det(rotation).item() == pytest.approx(1.0, abs=1e-5)


def test_rotation_about_random_axis_respects_max_angle():
    """Vectors under a small max_angle rotation shouldn't move far -- crude
    bound: displacement magnitude <= 2*sin(max_angle/2) per unit vector,
    with slack for floating point."""
    max_angle = 0.1
    generator = torch.Generator().manual_seed(0)
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    for _ in range(20):
        rotation = rotation_about_random_axis(max_angle, generator)
        rotated = rotation @ v
        displacement = torch.linalg.norm(rotated - v).item()
        assert displacement <= 2 * abs(max_angle) + 1e-6


def test_minimum_image_delta_wraps_across_a_periodic_boundary():
    cell = torch.eye(3) * 10.0
    pbc = torch.tensor([True, True, True])
    a = torch.tensor([0.5, 0.0, 0.0])
    b = torch.tensor([9.5, 0.0, 0.0])
    delta = minimum_image_delta(a, b, cell, pbc)
    # true minimum-image separation is 1.0 (wrapping across the boundary),
    # not the raw 9.0 difference
    assert torch.allclose(delta, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_minimum_image_delta_matches_raw_difference_when_pbc_is_off():
    cell = torch.eye(3) * 10.0
    pbc = torch.tensor([False, False, False])
    a = torch.tensor([0.5, 0.0, 0.0])
    b = torch.tensor([9.5, 0.0, 0.0])
    delta = minimum_image_delta(a, b, cell, pbc)
    assert torch.allclose(delta, a - b)


def test_minimum_image_delta_broadcasts():
    cell = torch.eye(3) * 10.0
    pbc = torch.tensor([True, True, True])
    a = torch.tensor([[0.5, 0.0, 0.0], [1.0, 1.0, 1.0]]).unsqueeze(1)  # [2, 1, 3]
    b = torch.tensor([9.5, 0.0, 0.0]).unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
    delta = minimum_image_delta(a, b, cell, pbc)
    assert delta.shape == (2, 1, 3)
