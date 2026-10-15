"""Real-space window functions that damp finite-box truncation ripple.

The model's g(r) is only knowable out to r_max <= L/2 (RMCA Section 9.1).
Sine-transforming a function that is cut off abruptly at r_max convolves
the result with the transform of a step function, putting spurious
oscillations into the calculated S(Q) -- most visibly at low Q, exactly
where a first sharp diffraction peak would live.

RMCA's own answer (Eq. 23, its CONVOL program) is to apply the matching
convolution to the *experimental data* instead, leaving the model raw.
RAY-MC takes the other route: damp the model's own (g(r) - 1) with a
window that goes smoothly to zero at r_max, so the transform never sees
a discontinuity in the first place. This is the RMCProfile/PDFgui-side
convention, and it keeps the experimental data untouched -- what you fit
is the numbers the instrument produced.

Consistency note: a window applied to the model and not to the data is
only rigorous when the data's own transform was treated the same way.
For total-scattering S(Q) measured directly in reciprocal space (the
usual case for experimental total-scattering files) the data carries
no such truncation at all, and the window is purely a model-side
artifact suppressant -- it removes ripple the real system does not have.
Prefer the narrowest damping that removes the ripple; ``"none"`` is
correct whenever the box is large enough that g(r_max) is already flat.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

WindowKind = Literal["none", "lorch", "hann"]


def window_function(kind: WindowKind, r_grid: torch.Tensor, r_max: float) -> torch.Tensor:
    """Window W(r) evaluated on *r_grid*, going to zero at *r_max*.

    Parameters
    ----------
    kind : Literal["none", "lorch", "hann"]
        - ``"none"``: W(r) = 1. No damping; truncation ripple is left in.
          Correct when the box is large enough that g(r) has already gone
          flat well before r_max.
        - ``"lorch"``: W(r) = sinc(pi r / r_max) = sin(pi r/r_max)/(pi r/r_max).
          The standard total-scattering window (Lorch 1969). Mildest
          damping that reaches zero with zero slope discontinuity; costs
          some real-space resolution, broadening genuine peaks slightly.
        - ``"hann"``: W(r) = (1 + cos(pi r / r_max)) / 2. A raised cosine,
          noticeably more aggressive than Lorch -- suppresses ripple
          harder at the price of more peak broadening.
    r_grid : torch.Tensor
        ``[n_r]`` bin centers (Angstrom).
    r_max : float
        Where the window reaches zero -- the histogram's own upper edge.

    Returns
    -------
    torch.Tensor
        ``[n_r]`` float64 window values in [0, 1].
    """
    r = r_grid.to(torch.float64)

    if kind == "none":
        return torch.ones_like(r)

    if kind == "lorch":
        x = math.pi * r / r_max
        # sinc(0) = 1 by continuity; guard the r = 0 bin explicitly rather
        # than dividing by zero and patching a NaN afterwards.
        return torch.where(x.abs() < 1e-12, torch.ones_like(x), torch.sin(x) / x)

    if kind == "hann":
        return 0.5 * (1.0 + torch.cos(math.pi * r / r_max))

    raise ValueError(
        f"Unknown window kind: {kind!r} (expected 'none', 'lorch' or 'hann')"
    )
