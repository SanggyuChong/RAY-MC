"""Reciprocal-space scattering factors and Faber-Ziman pair weights.

The real-space PDF modality gets away with one scalar weight per atom
(:mod:`raymc.modalities.pdf.scattering_weights`), and approximates the
X-ray form factor by the atomic number Z because a real-space
calculation has no single Q at which to evaluate f(Q). In reciprocal
space that excuse disappears: we are already on a Q grid, so the true
Q-dependent form factor is both correct and free, and it is what RMCA
Eq. 29 specifies. This module therefore returns a **[n_species, n_Q]**
array of scattering factors rather than one number per species.

Weighting surface follows modern practice (fullrmc's ``weighting`` +
``elementsWeight``, RMCProfile's internal tables, Dissolve's
isotopologues) rather than RMCA's per-pair coefficient array:

- The primary knob is the radiation type -- ``"neutron"`` uses tabulated
  coherent scattering lengths, ``"xray"`` uses tabulated form factors.
- ``overrides`` is the per-species escape hatch, which is what makes
  isotopic substitution expressible at all. RMCA's gamma_ab array bought
  exactly this and nothing more; a per-*pair* matrix is not needed,
  because a scattering length belongs to a species, not to a pair.

Normalization convention
------------------------
Faber-Ziman throughout::

    S(Q) - 1 = sum_ab w_ab(Q) [A_ab(Q) - 1]
    w_ab(Q)  = (2 - delta_ab) c_a c_b f_a(Q) f_b(Q) / (sum_g c_g f_g(Q))^2

so the weights sum to 1 at every Q and S(Q) -> 1 at large Q. The
``(2 - delta_ab)`` multiplicity is there because the pair sum runs over
*unordered* species pairs (a <= b), matching RMCA Section 9.7's stated
ordering 1-1, 1-2, ..., n-n.

Note that RMCA Eq. 17 writes the X-ray normalization with a denominator
that reads as ``sum_a c_a f_a(Q)^2`` in the scanned manual, which would
be the mean-square rather than the square-of-mean. We use the
square-of-mean (<f>^2) above: it is the standard Faber-Ziman definition,
it is the convention the experimental S(Q) files in ``data/`` follow
(their X-ray sets tend to 1 at high Q), and it makes the neutron and
X-ray cases consistent with each other. If a data set turns out to use
the Ashcroft-Langreth <f^2> normalization instead, that is a property of
the *data*, and the modality's ``renormalize``/``offset`` options exist
to absorb it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import periodictable as pt
import torch

from raymc.modalities.pdf.scattering_weights import Radiation

# Waasmaier & Kirfel's f0 fit (what periodictable tabulates) is defined
# for Q in [0, 24*pi]; outside that it returns NaN rather than raising.
MAX_XRAY_Q = 24.0 * torch.pi


def species_form_factors(
    species: Sequence[int],
    q_grid: torch.Tensor,
    radiation: Radiation,
    overrides: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Per-species scattering factor f_a(Q), as a ``[n_species, n_Q]`` tensor.

    Parameters
    ----------
    species : Sequence[int]
        Atomic numbers, one per distinct species, in a fixed order that
        the caller reuses for concentrations and pair weights.
    q_grid : torch.Tensor
        ``[n_Q]`` momentum transfer values (inverse Angstrom), using the
        standard ``Q = 4 pi sin(theta) / lambda`` convention -- the same
        one :mod:`periodictable`'s ``xray.f0`` expects.
    radiation : Literal["neutron", "xray"]
        ``"neutron"``: tabulated coherent scattering length b_c (fm),
        constant in Q, broadcast across the grid. ``"xray"``: tabulated
        Q-dependent form factor f0(Q) (electrons), which equals Z at
        Q = 0 and decays with Q. Note f0 carries no anomalous dispersion
        terms (f', f''), so it is wavelength-independent -- adequate away
        from an absorption edge, which is the usual total-scattering case.
    overrides : Mapping[str, float] | None
        Element symbol -> constant scattering factor, replacing the
        tabulated value for that species at every Q. This is how
        isotopic substitution is expressed: e.g. ``{"H": 6.671}`` for
        deuterium, or ``{"Ni": 10.3}`` for a null-scattering Ni alloy.
        An override is Q-independent by nature (a scattering length is a
        constant), so overriding a species in an X-ray run deliberately
        flattens its form factor -- allowed, but rarely what you want.

    Returns
    -------
    torch.Tensor
        ``[n_species, n_Q]`` float64 scattering factors.

    Raises
    ------
    ValueError
        If *radiation* is unknown, an element has no tabulated value, an
        override names an element not present in *species*, or (X-ray
        only) the Q grid runs past the tabulated fit's valid range.
    """
    q_grid = q_grid.to(torch.float64)
    n_q = q_grid.numel()

    override_by_z: dict[int, float] = {}
    for symbol, value in (overrides or {}).items():
        z = pt.elements.symbol(symbol).number
        if z not in set(species):
            present = ", ".join(sorted(pt.elements[s].symbol for s in species))
            raise ValueError(
                f"scattering-factor override for {symbol!r} (Z={z}), which is "
                f"not present in this system (species: {present}). Overrides "
                f"are matched by element, so a typo here would otherwise be "
                f"silently ignored."
            )
        override_by_z[z] = float(value)

    rows = []
    for z in species:
        if z in override_by_z:
            rows.append(torch.full((n_q,), override_by_z[z], dtype=torch.float64))
        elif radiation == "neutron":
            b_c = pt.elements[z].neutron.b_c
            if b_c is None:
                raise ValueError(
                    f"No tabulated coherent neutron scattering length for "
                    f"element Z={z} ({pt.elements[z].symbol}). Supply one "
                    f"explicitly via the per-species override."
                )
            rows.append(torch.full((n_q,), float(b_c), dtype=torch.float64))
        elif radiation == "xray":
            if float(q_grid.max()) > MAX_XRAY_Q:
                raise ValueError(
                    f"Q grid reaches {float(q_grid.max()):.2f} A^-1, past the "
                    f"{MAX_XRAY_Q:.1f} A^-1 upper limit of the tabulated "
                    f"Waasmaier-Kirfel f0 fit -- beyond it the form factor is "
                    f"undefined (NaN), which would poison every chi-squared. "
                    f"Reduce q_max."
                )
            f0 = pt.elements[z].xray.f0(q_grid.cpu().numpy())
            rows.append(torch.as_tensor(f0, dtype=torch.float64))
        else:
            raise ValueError(
                f"Unknown radiation type: {radiation!r} (expected 'neutron' "
                f"or 'xray')"
            )

    return torch.stack(rows, dim=0)


def unordered_species_pairs(n_species: int) -> list[tuple[int, int]]:
    """Index pairs ``(a, b)`` with ``a <= b``, in RMCA Section 9.7's order
    (1-1, 1-2, ..., 1-n, 2-2, ..., n-n)."""
    return [(a, b) for a in range(n_species) for b in range(a, n_species)]


def faber_ziman_weights(
    concentrations: torch.Tensor, form_factors: torch.Tensor
) -> torch.Tensor:
    """Faber-Ziman pair weights w_ab(Q) -- see the module docstring.

    Parameters
    ----------
    concentrations : torch.Tensor
        ``[n_species]`` number fractions c_a, summing to 1.
    form_factors : torch.Tensor
        ``[n_species, n_Q]`` from :func:`species_form_factors`.

    Returns
    -------
    torch.Tensor
        ``[n_pairs, n_Q]``, rows ordered by :func:`unordered_species_pairs`.
        Each column sums to 1.
    """
    concentrations = concentrations.to(torch.float64)
    mean_f = (concentrations.unsqueeze(1) * form_factors).sum(dim=0)  # [n_Q]

    if bool((mean_f.abs() < 1e-12).any()):
        raise ValueError(
            "the mean scattering factor <f(Q)> vanishes at some Q, so the "
            "Faber-Ziman normalization 1/<f>^2 is singular there. This "
            "happens for a genuinely null-scattering composition; such a "
            "data set cannot be normalized this way."
        )

    rows = []
    for a, b in unordered_species_pairs(concentrations.numel()):
        multiplicity = 1.0 if a == b else 2.0
        rows.append(
            multiplicity
            * concentrations[a]
            * concentrations[b]
            * form_factors[a]
            * form_factors[b]
            / mean_f**2
        )
    return torch.stack(rows, dim=0)
