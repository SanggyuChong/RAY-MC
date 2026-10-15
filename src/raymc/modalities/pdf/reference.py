"""Loading experimental/reference g(r) and S(Q) data for benchmarking.

Reads the plain whitespace-separated, ``#``-commented text files common
for total-scattering reference data (e.g. supplementary information from
a paper): two or three columns (x, y[, error]), comment lines starting
with ``#``.
"""

from __future__ import annotations

from pathlib import Path

import torch


class ReferenceCurve:
    """One loaded reference curve: x-values, y-values, and optional error bars."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, error: torch.Tensor | None):
        self.x = x
        self.y = y
        self.error = error

    def interpolate(self, new_x: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate y onto *new_x* (must lie within the data's range)."""
        return _interp(new_x, self.x, self.y)

    def interpolate_error(self, new_x: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate the error bars onto *new_x*.

        Raises
        ------
        ValueError
            If this curve has no error column -- callers must supply a
            constant sigma explicitly instead of silently getting an
            all-ones (meaningless) error bar.
        """
        if self.error is None:
            raise ValueError(
                "reference curve has no error column to interpolate -- pass "
                "a constant sigma instead, or use reference data with a "
                "third (error) column."
            )
        return _interp(new_x, self.x, self.error)


def _interp(new_x: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    new_x = new_x.to(torch.float64)
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    idx = torch.searchsorted(x, new_x).clamp(1, len(x) - 1)
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    weight = (new_x - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def load_reference_curve(path: str | Path) -> ReferenceCurve:
    """Load a two- or three-column ``#``-commented reference data file.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    ReferenceCurve
    """
    xs, ys, errs = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
            if len(parts) > 2:
                errs.append(float(parts[2]))

    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    error = torch.tensor(errs, dtype=torch.float64) if errs else None
    return ReferenceCurve(x, y, error)
