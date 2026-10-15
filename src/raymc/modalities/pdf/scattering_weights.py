"""Per-atom scattering weights for the real-space PDF forward model.

Neutron and X-ray PDFs use the same formula (see
:mod:`raymc.modalities.pdf.calculator`); only each atom's weight
differs:

- Neutron: coherent scattering length (doesn't depend on Q).
- X-ray: atomic number Z, used as a stand-in for the real (Q-dependent)
  form factor. This is a known approximation -- RMCProfile and fullrmc
  both do the same thing for real-space X-ray PDFs, since there's no
  single Q value a real-space calculation could use instead.

Electron scattering isn't implemented yet.
"""

from __future__ import annotations

from typing import Literal

import periodictable as pt
import torch

Radiation = Literal["neutron", "xray"]


def atomic_scattering_weights(atomic_numbers: torch.Tensor, radiation: Radiation) -> torch.Tensor:
    """Per-atom scattering weight ``b_i``, for the given *radiation*.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        [n_atoms] integer atomic numbers.
    radiation : Literal["neutron", "xray"]

    Returns
    -------
    torch.Tensor
        [n_atoms] float64 weights: coherent scattering length (fm) for
        ``"neutron"``, atomic number for ``"xray"``.

    Raises
    ------
    ValueError
        If *radiation* is not recognized, or (neutron only) an element has
        no tabulated coherent scattering length.
    """
    if radiation == "xray":
        return atomic_numbers.to(torch.float64)

    if radiation == "neutron":
        weights = []
        for z in atomic_numbers.cpu().tolist():
            b_c = pt.elements[z].neutron.b_c
            if b_c is None:
                raise ValueError(
                    f"No tabulated coherent neutron scattering length for "
                    f"element Z={z} ({pt.elements[z].symbol})."
                )
            weights.append(b_c)
        return torch.tensor(weights, device=atomic_numbers.device, dtype=torch.float64)

    raise ValueError(
        f"Unknown radiation type: {radiation!r} (expected 'neutron' or 'xray')"
    )
