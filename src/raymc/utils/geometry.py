"""Generic, tensor-only geometry math shared across RAY-MC.

Nothing here knows about :class:`~raymc.system.atomic_system.AtomicSystem`
or any other RAY-MC type -- just positions/cells/rotations as plain
tensors, so this module has no internal RAY-MC dependencies and everything
else can depend on it freely.
"""

from __future__ import annotations

import torch


def random_rotation_matrices(n: int, generator: torch.Generator) -> torch.Tensor:
    """Draw *n* uniformly-random 3x3 rotation matrices via random quaternions.

    Uses Shoemake's method (uniform quaternion sampling), which is uniform
    over SO(3) -- unlike, e.g., independently randomizing three Euler angles.
    Used both for fully-random molecule orientation at initialization and
    (via :func:`rotation_about_random_axis`) for small-angle MC rotation
    moves.
    """
    u1 = torch.rand(n, generator=generator, dtype=torch.float64)
    u2 = torch.rand(n, generator=generator, dtype=torch.float64)
    u3 = torch.rand(n, generator=generator, dtype=torch.float64)

    w = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    x = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    y = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    z = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    r = torch.empty(n, 3, 3, dtype=torch.float64)
    r[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return r


def rotation_about_random_axis(
    max_angle: float, generator: torch.Generator
) -> torch.Tensor:
    """Draw one small-angle rotation matrix about a uniformly-random axis.

    Unlike :func:`random_rotation_matrices` (fully uniform over SO(3), used
    for initialization), this draws a rotation angle uniformly in
    ``[-max_angle, max_angle]`` radians about a random axis -- the usual
    shape of an MC rotation move, where a bounded step size keeps the
    Metropolis acceptance rate reasonable.

    Parameters
    ----------
    max_angle : float
        Maximum rotation magnitude, in radians.
    generator : torch.Generator

    Returns
    -------
    torch.Tensor
        [3, 3] rotation matrix.
    """
    axis = torch.randn(3, generator=generator, dtype=torch.float64)
    axis = axis / axis.norm()
    angle = (
        torch.rand(1, generator=generator, dtype=torch.float64) * 2 - 1
    ) * max_angle

    k = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=torch.float64,
    )
    return (
        torch.eye(3, dtype=torch.float64)
        + torch.sin(angle) * k
        + (1 - torch.cos(angle)) * (k @ k)
    )


def minimum_image_delta(
    a: torch.Tensor, b: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor
) -> torch.Tensor:
    """``a - b`` (broadcastable), wrapped to the nearest periodic image.

    Shared by anything that needs a raw pairwise displacement under PBC
    without going through ALCHEMI's neighbor list (e.g. checking a handful
    of atoms touched by one move against everyone else) --
    :mod:`raymc.constraints.min_dist`, :mod:`raymc.modalities.pdf.calculator`,
    and :mod:`raymc.moves.base` (rigid-group unwrapping) all use this. Only
    correct up to a single unambiguous minimum image, i.e. under the same
    "cutoff no more than half the box's narrowest width" precondition
    already enforced by
    :func:`raymc.modalities.histogram.check_cutoff_safety`.
    """
    delta = a - b
    inv_cell = torch.linalg.inv(cell)
    frac = delta @ inv_cell
    shift = torch.round(frac) * pbc.to(frac.dtype)
    return delta - shift @ cell
