"""Aggregator that combines the residuals, constraints, and energetics
into one chi-squared total for the RMC engine's Metropolis algorithm.

:class:`ChiSqTerm` owns everything about one named term: it holds a
single :class:`~raymc.modalities.base.Calculator` and does the term's own
residual math itself -- calling ``calculator.compute(system)`` (full
recompute) or ``calculator.compute_increm_change(system, delta)`` (the
calculator's own optional fast path; falls back to a full ``compute`` for
a calculator that doesn't override it -- see
:class:`raymc.modalities.base.Calculator`) is entirely the calculator's
concern, not this module's.

Two term shapes are supported, both through the same class:

- **Residual** (``target`` given): a sum-reduced chi-squared,
  ``sum(((raw - target) / sigma) ** 2)`` -- the usual case for a modality
  fit against reference data (PDF).
- **Direct** (``target=None``): the raw value is used as-is, weighted but
  not squared -- for a term that isn't being fit to a target with an
  error bar at all, e.g. the energetics term, which should contribute
  ``weight * E`` linearly to the Metropolis criterion, not
  ``weight * E**2`` -- and likewise the distance constraints
  (:mod:`raymc.constraints`), whose smooth violation penalty is already
  on the chi-squared scale and must not be squared again. Still tracked
  as part of the run's "chi-squared" in
  the RMC-community sense of the word -- the acceptance-driving total --
  even though this particular term isn't literally squared.

One sigma, per McGreevy & Pusztai (1988), Mol. Sim. 1, 359, Eq. 8/11/12:
``sigma``, the per-point experimental error, used both for the REPORTED
chi-squared and to drive the Metropolis acceptance decision --
``chi^2 = sum(((g_calc - g_exp) / sigma) ** 2)``. The paper uses the same
quantity for both roles (there is no separate acceptance-width
parameter), so this is the single number you'd quote as "how well does
this fit the data" *and* the value that ``chi_n^2 - chi_o^2`` in Eq. 12
is computed from. A residual term must be given this explicitly; there
is no default of 1 (silently defaulting to 1 makes chi^2 meaningless as
a fit-quality number).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from raymc.constraints.pairwise import DEFAULT_SOFTENING_WINDOW
from raymc.constraints.penalty import PENALTY_AT_WINDOW
from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem

if TYPE_CHECKING:
    from raymc.modalities.base import Calculator


def _broadcast_sigma(name: str, label: str, sigma, target: torch.Tensor) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigma, dtype=target.dtype)
    try:
        return torch.broadcast_to(sigma_tensor, target.shape).clone()
    except RuntimeError:
        raise ValueError(
            f"term {name!r}: {label} shape {tuple(sigma_tensor.shape)} "
            f"does not match target shape {tuple(target.shape)}"
        ) from None


class ChiSqTerm:
    """One named, weighted contribution to the engine's total chi-squared.

    Parameters
    ----------
    name : str
    calculator : Calculator
        Produces this term's raw observable -- see
        :mod:`raymc.modalities.base`.
    target : torch.Tensor | None
        Reference value(s) this term's raw output is compared against.
        ``None`` means "direct" mode -- see the module docstring.
    sigma : torch.Tensor | float | None
        Per-point experimental error, used for both the reported
        chi-squared and the Metropolis acceptance decision (McGreevy &
        Pusztai (1988), Eq. 8/11/12 -- a single sigma plays both roles).
        Required (no default) whenever *target* is given -- see the
        module docstring. Must be ``None`` (and is ignored) in "direct"
        mode.
    weight : float
        Contribution of this term to the aggregate total.
    data_scale, data_offset : float
        The normalization correction ALREADY APPLIED to *target* by the
        factory that built this term -- recorded here so runs can report
        it, never re-applied. RMCA Section 9.6 (Eq. 24/25) solves for
        these each step; RAY-MC does not, deliberately. A target that
        changes between proposals makes the Metropolis criterion compare
        two different objectives, and in practice the solve is
        ill-conditioned exactly when it is most tempting to reach for:
        against a model of the wrong sign it returns a large negative
        scale and injects noise far larger than the structural signal.
        Correct the data once, declare what you did, and keep the
        objective stationary.
    """

    def __init__(
        self,
        name: str,
        calculator: "Calculator",
        target: torch.Tensor | None = None,
        sigma: "torch.Tensor | float | None" = None,
        weight: float = 1.0,
        data_scale: float = 1.0,
        data_offset: float = 0.0,
    ) -> None:
        self.name = name
        self.calculator = calculator
        self.weight = weight
        self.target = target
        # Recorded for reporting only: the correction is already baked
        # into *target* by the factory that built this term.
        self.data_scale = data_scale
        self.data_offset = data_offset

        if target is None:
            if sigma is not None:
                raise ValueError(
                    f"term {name!r}: sigma has no meaning without a target "
                    f"('direct' mode)"
                )
            self.sigma = None
        else:
            if sigma is None:
                raise ValueError(
                    f"term {name!r}: a residual term requires an explicit "
                    f"sigma (per-point experimental error) -- there is no "
                    f"default. Pass a constant, or supply an error column in "
                    f"the reference data file."
                )
            self.sigma = _broadcast_sigma(name, "sigma", sigma, target)

        self.current_value: torch.Tensor | None = None
        self._pending_raw: torch.Tensor | None = None

    def _reduce(self, raw: torch.Tensor) -> torch.Tensor:
        """Sum-reduced chi-squared, ``sum(((raw - target) / sigma) ** 2)``
        (McGreevy & Pusztai (1988), Eq. 8) -- both the reported fit-quality
        number and what drives the Metropolis acceptance decision (Eq. 12)."""
        if self.target is None:
            return raw
        return ((raw - self.target) / self.sigma).pow(2).sum()

    def r_w(self, raw: torch.Tensor) -> torch.Tensor:
        """Weighted-profile R-factor: ``sqrt(sum(Δg^2) / sum(g_exp^2))``."""
        if self.target is None:
            raise ValueError(f"term {self.name!r} is in 'direct' mode: no r_w")
        delta = raw - self.target
        return torch.sqrt((delta**2).sum() / (self.target**2).sum())

    def report_from_raw(self, raw: torch.Tensor) -> dict[str, float]:
        """Same as :meth:`report`, but from an already-computed raw
        observable (e.g. a g(r) averaged over several saved configurations
        -- see :mod:`raymc.engine.rmc`'s production phase) instead of
        recomputing from a live system."""
        chi2 = self._reduce(raw)
        n_r = self.target.numel()
        report = {
            "chi2": float(chi2),
            "chi2_per_nr": float(chi2) / n_r,
            "n_r": n_r,
            "Rw": float(self.r_w(raw)),
        }
        if self.data_scale != 1.0 or self.data_offset != 0.0:
            report["data_scale"] = self.data_scale
            report["data_offset"] = self.data_offset
        return report

    def report(self, system: AtomicSystem) -> dict[str, float]:
        """Full-recompute report: chi^2, chi^2/n_r, n_r, and R_w --
        for logging/inspection, not the hot loop."""
        return self.report_from_raw(self.calculator.compute(system))

    def init(self, system: AtomicSystem) -> torch.Tensor:
        """Score *system*'s starting state and record it as current."""
        raw = self.calculator.init(system)
        self.current_value = self._reduce(raw)
        return self.current_value

    def evaluate(self, candidate: AtomicSystem, delta: MoveDelta) -> torch.Tensor:
        """Score *candidate*, given what the move touched. Does not mutate
        committed state -- see :meth:`accept`.

        Uses :meth:`_reduce` (sigma-scaled chi-squared, Eq. 8/12) -- the
        Metropolis acceptance decision must always be driven by the
        instantaneous chi-squared, never by a running average."""
        raw = self.calculator.compute_increm_change(candidate, delta)
        self._pending_raw = raw
        return self._reduce(raw)

    def accept(self, keep: bool) -> None:
        """Told whether the last :meth:`evaluate`'d candidate was kept."""
        if keep:
            self.calculator.commit()
            self.current_value = self._reduce(self._pending_raw)
        else:
            self.calculator.discard()
        self._pending_raw = None


class ChiSqAggregator:
    """Weighted sum of several :class:`ChiSqTerm`\\ s (modality residuals,
    hard constraints, and energetics for HRMC).

    Parameters
    ----------
    terms : list[ChiSqTerm]
        At least one term.
    """

    def __init__(self, terms: list[ChiSqTerm]) -> None:
        if not terms:
            raise ValueError("ChiSqAggregator requires at least one chi-squared term")
        self.terms = terms

    def init(self, system: AtomicSystem) -> torch.Tensor:
        values = [term.init(system) for term in self.terms]
        return _weighted_sum(self.terms, values)

    def evaluate(self, candidate: AtomicSystem, delta: MoveDelta) -> torch.Tensor:
        values = [term.evaluate(candidate, delta) for term in self.terms]
        return _weighted_sum(self.terms, values)

    def accept(self, keep: bool) -> None:
        for term in self.terms:
            term.accept(keep)

    def current_values(self) -> dict[str, float]:
        """Each term's committed chi-squared for the state the engine is
        currently sitting on -- UNWEIGHTED, so the numbers are comparable
        to the term's own point count and R_w rather than to each other's
        contribution to the acceptance decision.

        Recorded per step by :class:`raymc.engine.rmc.RunHistory` so a
        multi-modality run can be read back apart: a total that falls
        while one data set climbs and another dives is the signature of
        two experiments pulling against each other, and the total alone
        cannot show it.
        """
        return {
            term.name: float(term.current_value)
            for term in self.terms
            if term.current_value is not None
        }

    def breakdown(self, system: AtomicSystem) -> dict[str, torch.Tensor]:
        """Per-term (unweighted) acceptance-criterion values (sigma-scaled)
        via a full recompute, for inspection/logging -- ignores any running
        incremental state."""
        return {
            term.name: term._reduce(term.calculator.compute(system))
            for term in self.terms
        }

    def report(self, system: AtomicSystem) -> dict[str, dict[str, float]]:
        """Per-term reported chi^2/n_r and R_w for every residual term --
        skips 'direct'-mode terms (e.g. energetics), which have no target
        to report a fit quality against."""
        return {
            term.name: term.report(system)
            for term in self.terms
            if term.target is not None
        }


def _weighted_sum(terms: list[ChiSqTerm], values: list[torch.Tensor]) -> torch.Tensor:
    total = None
    for term, value in zip(terms, values):
        contribution = term.weight * value
        total = contribution if total is None else total + contribution
    return total


def _corrected_target(
    y: torch.Tensor, data_scale: float, data_offset: float
) -> torch.Tensor:
    """Apply the user-declared normalization correction to reference values.

    ``data_scale * y + data_offset``, applied ONCE at construction. See
    :class:`ChiSqTerm` for why this is a declared constant rather than a
    per-step least-squares solve.
    """
    return data_scale * y.to(torch.float64) + data_offset


def _select_range(
    x: torch.Tensor,
    lo: float | None,
    hi: float | None,
    name: str,
    axis: str,
    unit: str,
) -> torch.Tensor:
    """Boolean mask selecting reference points within ``[lo, hi]``."""
    keep = x > 0
    if lo is not None:
        keep &= x >= lo
    if hi is not None:
        keep &= x <= hi
    if not bool(keep.any()):
        raise ValueError(
            f"term {name!r}: no experimental {axis} points survive the "
            f"{axis.lower()}_min/{axis.lower()}_max selection (data spans "
            f"{float(x.min()):.3f}-{float(x.max()):.3f} {unit})"
        )
    return keep


def make_partial_pdf_term(
    name: str,
    reference: "ReferenceCurve",  # noqa: F821
    atomic_number_a: int,
    atomic_number_b: int | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
    sigma: float | None = None,
    weight: float = 1.0,
    data_scale: float = 1.0,
    data_offset: float = 0.0,
) -> ChiSqTerm:
    """Build a partial-PDF residual term (see
    :class:`raymc.modalities.pdf.calculator.PDFCalculator`, partial mode --
    a real-space histogram, not a Gaussian-smeared kernel).

    **The reference data owns the grid.** The histogram's bin centers are
    the reference file's own r values, so the model is evaluated exactly
    where the data was measured and no interpolation is ever applied to
    experimental values. This is the same contract :func:`make_sq_term`
    has always had in Q, and it replaces an earlier arrangement in which
    the reference was interpolated onto a ``linspace`` the model chose --
    which silently made fit quality a function of how the file had been
    digitized, and made a reference with uneven spacing unusable.

    Two consequences worth knowing. The grid may be unevenly spaced, so
    bin widths are per-bin. And because chi-squared sums over points,
    a region the data samples more densely carries proportionally more
    weight -- that is now the data provider's choice to make, but it is
    still a choice.

    *r_min*/*r_max* select a sub-range of the reference's own points by
    value; they never resample.

    Gets the fast incremental path for free for a self partial (e.g.
    g_OO); a cross partial (*atomic_number_b* set and different from
    *atomic_number_a*, e.g. g_OH) is always a full recompute -- the
    calculator itself decides that, this factory doesn't need to know.

    Parameters
    ----------
    sigma : float | None
        Constant per-point sigma. If omitted, *reference* must carry an
        error column (:attr:`ReferenceCurve.error`); raises otherwise --
        see the module docstring, "no default of 1" is not negotiable.
    data_scale, data_offset : float
        Declared normalization correction applied once to the reference
        values -- see :class:`ChiSqTerm`.
    """
    from raymc.modalities.pdf.calculator import PDFCalculator

    r_all = reference.x.to(torch.float64)
    keep = _select_range(r_all, r_min, r_max, name, "r", "A")
    r_grid = r_all[keep]

    calculator = PDFCalculator(
        r_grid=r_grid,
        atomic_number_a=atomic_number_a,
        atomic_number_b=atomic_number_b,
    )
    target_g = _corrected_target(reference.y[keep], data_scale, data_offset)

    if sigma is not None:
        target_err = torch.full_like(target_g, sigma)
    elif reference.error is not None:
        target_err = reference.error.to(torch.float64)[keep]
    else:
        raise ValueError(
            f"term {name!r}: no sigma given and the reference data has no "
            f"error column -- pass a constant sigma instead. There is no "
            f"default of 1 (see the module docstring)."
        )

    return ChiSqTerm(
        name=name,
        calculator=calculator,
        target=target_g,
        sigma=target_err,
        weight=weight,
        data_scale=data_scale,
        data_offset=data_offset,
    )


def make_sq_term(
    name: str,
    reference: "ReferenceCurve",  # noqa: F821
    r_max: float,
    n_r: int = 500,
    radiation: str | None = None,
    species_weights: dict[str, float] | None = None,
    q_min: float | None = None,
    q_max: float | None = None,
    window: str = "lorch",
    forward_model: str = "histogram_ft",
    convention: str = "S",
    atomic_number_a: int | None = None,
    atomic_number_b: int | None = None,
    sigma: float | None = None,
    weight: float = 1.0,
    data_scale: float = 1.0,
    data_offset: float = 0.0,
) -> ChiSqTerm:
    """Build an S(Q) residual term (see
    :class:`raymc.modalities.sq.calculator.SQCalculator`).

    The reference data owns the Q grid: the experimental Q points are
    adopted verbatim and the model is evaluated there, never the other
    way round. :func:`make_partial_pdf_term` now works the same way in r.
    That avoids introducing interpolation error into the data, and matches
    RMCA, which fits a chosen index range of the data file's own points
    (its nx1/nx2).

    Note that *n_r* is the sine transform's own internal integration grid
    over ``[0, r_max)`` -- an implementation detail of the forward model,
    unrelated to the data's Q grid and to the PDF modality's r bins.

    *q_min*/*q_max* select that range by value. Q <= 0 is always dropped:
    S(Q) is a sine transform divided by Q, singular at the origin.

    Parameters
    ----------
    sigma : float | None
        Constant per-point sigma. If omitted, *reference* must carry an
        error column; raises otherwise (see the module docstring).
    """
    from raymc.modalities.sq.calculator import SQCalculator

    q_all = reference.x.to(torch.float64)
    keep = _select_range(q_all, q_min, q_max, name, "Q", "A^-1")
    q_grid = q_all[keep]
    target = _corrected_target(reference.y[keep], data_scale, data_offset)

    calculator = SQCalculator(
        q_grid=q_grid,
        r_max=r_max,
        n_r=n_r,
        radiation=radiation,
        species_weights=species_weights,
        window=window,
        forward_model=forward_model,
        convention=convention,
        atomic_number_a=atomic_number_a,
        atomic_number_b=atomic_number_b,
    )

    if sigma is not None:
        target_err = torch.full_like(target, sigma)
    elif reference.error is not None:
        target_err = reference.error.to(torch.float64)[keep]
    else:
        raise ValueError(
            f"term {name!r}: no sigma given and the reference data has no "
            f"error column -- pass a constant sigma instead. There is no "
            f"default of 1 (see the module docstring)."
        )

    return ChiSqTerm(
        name=name,
        calculator=calculator,
        target=target,
        sigma=target_err,
        weight=weight,
        data_scale=data_scale,
        data_offset=data_offset,
    )


def make_min_dist_term(
    name: str,
    min_dist: "MinDistTable",  # noqa: F821
    weight: float = 1.0,
    softening_window: float = DEFAULT_SOFTENING_WINDOW,
    penalty_at_window: float = PENALTY_AT_WINDOW,
) -> ChiSqTerm:
    """Build the minimum-distance constraint as a :class:`ChiSqTerm` (see
    :class:`raymc.constraints.min_dist.MinDistConstraint`).

    "Direct" mode (``target=None``), not a residual: the constraint's
    output is already a penalty on the chi-squared scale --
    *penalty_at_window* for a pair one ``softening_window`` inside its
    minimum distance -- so it contributes ``weight * penalty`` as-is.
    Squaring it against a target of 0 (which is what this term used to do)
    would silently turn that penalty into its square, defeating the point
    of making it configurable, and there is no experimental error bar here
    for a sigma to mean anything.
    """
    from raymc.constraints.min_dist import MinDistConstraint

    calculator = MinDistConstraint(
        min_dist,
        softening_window=softening_window,
        penalty_at_window=penalty_at_window,
    )
    return ChiSqTerm(name=name, calculator=calculator, target=None, weight=weight)


def make_max_dist_term(
    name: str,
    max_dist: "MaxDistTable",  # noqa: F821
    weight: float = 1.0,
    softening_window: float = DEFAULT_SOFTENING_WINDOW,
    penalty_at_window: float = PENALTY_AT_WINDOW,
) -> ChiSqTerm:
    """Build the maximum-distance constraint as a :class:`ChiSqTerm` (see
    :class:`raymc.constraints.max_dist.MaxDistConstraint`). "Direct" mode,
    for the same reasons as :func:`make_min_dist_term`.

    Note that a table constraining nothing (the default, an infinite
    maximum separation everywhere) makes this term a constant 0 --
    :func:`raymc.orchestrate.build_chi_sq` leaves it out of the aggregator
    entirely in that case rather than build it.
    """
    from raymc.constraints.max_dist import MaxDistConstraint

    calculator = MaxDistConstraint(
        max_dist,
        softening_window=softening_window,
        penalty_at_window=penalty_at_window,
    )
    return ChiSqTerm(name=name, calculator=calculator, target=None, weight=weight)


def make_intramolecular_term(
    name: str,
    windows: "IntraWindowTable",  # noqa: F821
    weight: float = 1.0,
    softening_window: float = DEFAULT_SOFTENING_WINDOW,
    penalty_at_window: float = PENALTY_AT_WINDOW,
) -> ChiSqTerm:
    """Build the intramolecular distance-window constraint as a
    :class:`ChiSqTerm` (see
    :class:`raymc.constraints.intramolecular.IntramolecularConstraint`).
    "Direct" mode, for the same reasons as :func:`make_min_dist_term`.

    Like :func:`make_max_dist_term`, an empty window table makes this term
    a constant 0 and :func:`raymc.orchestrate.build_chi_sq` leaves it out
    of the aggregator entirely -- which is every rigid-molecule run.
    """
    from raymc.constraints.intramolecular import IntramolecularConstraint

    calculator = IntramolecularConstraint(
        windows,
        softening_window=softening_window,
        penalty_at_window=penalty_at_window,
    )
    return ChiSqTerm(name=name, calculator=calculator, target=None, weight=weight)


#: Boltzmann constant in eV/K -- the energetics term's energies are in eV
#: (see :class:`raymc.energetics.metatomic.MetatomicEnergetics`).
BOLTZMANN_EV_PER_K = 8.617333262e-5


def energetics_chi_sq_weight(temperature: float, weight: float = 1.0) -> float:
    """The chi-squared weight that makes an energy in eV enter the
    Metropolis criterion as ``ΔE / (k_B T)``.

    The engine accepts with ``min(1, exp(-(chi'^2 - chi^2) / 2))`` (see
    :class:`raymc.engine.rmc.ReverseMonteCarlo`), and a "direct"-mode
    term contributes ``w * E`` to chi-squared, hence ``-w * ΔE / 2`` to
    that exponent. Hybrid RMC wants ``-ΔE / (k_B T)`` there, so
    ``w = 2 / (k_B T)``, times a dimensionless *weight* for anyone who
    wants to deliberately over- or under-weight the potential relative to
    the data.

    This factor is the whole reason ``temperature`` exists in
    :class:`raymc.config.EnergeticsConfig`: without it the term's weight
    silently carries units of inverse energy, and the seemingly
    innocuous ``weight=1.0`` means an effective temperature of ~23,000 K,
    at which the energy barely influences acceptance at all.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return weight * 2.0 / (BOLTZMANN_EV_PER_K * temperature)


def make_metatomic_term(
    name: str,
    model: "mta.AtomisticModel",  # noqa: F821
    temperature: float,
    device: str = "cpu",
    weight: float = 1.0,
) -> ChiSqTerm:
    """Build a metatomic energetics term (see
    :class:`raymc.energetics.metatomic.MetatomicEnergetics`), in "direct"
    mode (``target=None``) -- the energy contributes to the total
    linearly, not as a residual against a reference.

    *temperature* (K) and the dimensionless *weight* are combined into
    the term's actual chi-squared weight by
    :func:`energetics_chi_sq_weight`; see there for why a temperature is
    required rather than optional.

    Imports :mod:`raymc.energetics.metatomic` lazily, so this module
    doesn't force the optional "metatomic" extra on anyone who only wants
    the PDF term.
    """
    from raymc.energetics.metatomic import MetatomicEnergetics

    calculator = MetatomicEnergetics(model=model, device=device)
    return ChiSqTerm(
        name=name,
        calculator=calculator,
        target=None,
        weight=energetics_chi_sq_weight(temperature, weight),
    )

