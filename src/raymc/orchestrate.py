"""Wires a :class:`~raymc.config.RunConfig` up to a real, runnable
:class:`~raymc.engine.rmc.ReverseMonteCarlo`.

This is the orchestration layer :mod:`raymc.cli` runs on: build the initial system, build the min-dist table, build a
:class:`~raymc.residuals.aggregator.ChiSqTerm` for each configured
modality/constraint/energetics term, and assemble the
:class:`~raymc.moves.moveset.MoveSet` and engine. Kept separate from
:mod:`raymc.cli` so this logic is usable from a plain Python script too.
"""

from __future__ import annotations

import periodictable as pt

from raymc.config import ModalityConfig, RunConfig
from raymc.engine.rmc import ReverseMonteCarlo
from raymc.initialize.build import build_initial_system
from raymc.modalities.pdf.reference import load_reference_curve
from raymc.moves.moveset import MoveSet
from raymc.residuals.aggregator import (
    ChiSqAggregator,
    ChiSqTerm,
    make_intramolecular_term,
    make_max_dist_term,
    make_metatomic_term,
    make_min_dist_term,
    make_partial_pdf_term,
    make_sq_term,
)
from raymc.system import AtomicSystem
from raymc.utils import IntraWindowTable, MaxDistTable, MinDistTable


def _term_name(modality: ModalityConfig) -> str:
    """What this modality is called in logs, errors and reports.

    The label when there is one, otherwise the modality type. A combined
    run has two modalities both named "sq", so the type alone cannot say
    which of them a message is about -- and these messages are mostly
    read when something has gone wrong with exactly one of them.
    """
    return modality.label or modality.name


def _build_pdf_term(modality: ModalityConfig) -> ChiSqTerm:
    atomic_number_a = pt.elements.symbol(modality.element_a).number
    atomic_number_b = (
        pt.elements.symbol(modality.element_b).number
        if modality.element_b is not None
        else None
    )
    ref_curve = load_reference_curve(modality.target_file)
    # The reference file owns the r grid: make_partial_pdf_term builds the
    # histogram on the data's own points, so model and target are evaluated
    # at exactly the same r and no interpolation touches experimental
    # values. r_min/r_max only select a sub-range of those points.
    return make_partial_pdf_term(
        _term_name(modality),
        ref_curve,
        atomic_number_a=atomic_number_a,
        atomic_number_b=atomic_number_b,
        r_min=modality.r_min,
        r_max=modality.r_max,
        data_scale=modality.data_scale,
        data_offset=modality.data_offset,
        sigma=modality.sigma,
        weight=modality.weight,
    )


def _build_sq_term(
    modality: ModalityConfig, cell: tuple[float, float, float] | None
) -> ChiSqTerm:
    """Build an S(Q) term, resolving the transform cutoff against the box.

    ``r_max`` bounds the sine transform's r integration and must not
    exceed half the box's narrowest width -- past that the pair search
    cannot resolve an unambiguous minimum image (RMCA Section 9.1 makes
    the same point about g(r) being knowable only to L/2). Left unset, it
    defaults to exactly that limit, which is both the largest usable
    value and the one that minimizes truncation ripple.
    """
    atomic_number_a = (
        pt.elements.symbol(modality.element_a).number
        if modality.element_a is not None
        else None
    )
    atomic_number_b = (
        pt.elements.symbol(modality.element_b).number
        if modality.element_b is not None
        else None
    )

    r_max = modality.r_max
    if "r_max" not in modality.model_fields_set:
        if cell is None:
            raise ValueError(
                f"modality {_term_name(modality)!r}: 'r_max' must be set explicitly "
                "when the run has no declared cell (init.strategy "
                "'load_structure'), since it cannot be derived from the box."
            )
        r_max = min(cell) / 2.0
        print(
            f"[raymc] modality {_term_name(modality)!r}: r_max not set, using "
            f"{r_max:.3f} A (half the box's narrowest width)"
        )

    ref_curve = load_reference_curve(modality.target_file)
    return make_sq_term(
        _term_name(modality),
        ref_curve,
        r_max=r_max,
        n_r=modality.n_r,
        radiation=modality.radiation,
        species_weights=modality.species_weights,
        q_min=modality.q_min,
        q_max=modality.q_max,
        window=modality.window,
        forward_model=modality.forward_model or "histogram_ft",
        convention=modality.convention,
        atomic_number_a=atomic_number_a,
        atomic_number_b=atomic_number_b,
        sigma=modality.sigma,
        weight=modality.weight,
        data_scale=modality.data_scale,
        data_offset=modality.data_offset,
    )


def build_chi_sq(config: RunConfig, min_dist: MinDistTable) -> ChiSqAggregator:
    """Build the run's :class:`~raymc.residuals.aggregator.ChiSqAggregator`
    from ``config.modalities``, ``config.constraints``,
    and ``config.energetics`` -- the min-dist
    constraint is always included; the energetics term only if
    ``config.energetics.enabled``.

    The max-dist and intramolecular-window constraints are included
    only when ``config.constraints`` actually restrains something (see
    :meth:`raymc.utils.MaxDistTable.is_unconstrained` and
    :meth:`raymc.utils.IntraWindowTable.is_unconstrained`). With its default
    of an infinite maximum separation the term would be a constant 0, and
    leaving it out is what keeps the default run from paying anything at
    all for a feature it isn't using -- rather than calling a calculator
    every step to be told 0 again.

    Raises
    ------
    ValueError
        If a modality's ``name`` is neither ``"pdf"`` nor ``"sq"``.
    """
    terms: list[ChiSqTerm] = []

    for modality in config.modalities:
        if modality.name == "pdf":
            terms.append(_build_pdf_term(modality))
        elif modality.name == "sq":
            terms.append(_build_sq_term(modality, config.init.cell))
        else:
            raise ValueError(
                f"Unknown modality {modality.name!r}; implemented: 'pdf', 'sq'"
            )

    # Both constraints share one ramp shape -- see ConstraintConfig.
    ramp = dict(
        softening_window=config.constraints.softening_window,
        penalty_at_window=config.constraints.penalty_at_window,
    )
    terms.append(make_min_dist_term("min_dist", min_dist, weight=1.0, **ramp))

    max_dist = MaxDistTable.from_config(config.constraints)
    if not max_dist.is_unconstrained():
        terms.append(make_max_dist_term("max_dist", max_dist, weight=1.0, **ramp))

    # Same "don't pay for what you aren't using" rule as the max-dist
    # term above: an empty window table is every rigid-molecule run, and
    # would make this a constant 0.
    intra_windows = IntraWindowTable.from_config(config.constraints)
    if not intra_windows.is_unconstrained():
        terms.append(
            make_intramolecular_term(
                "intramolecular", intra_windows, weight=1.0, **ramp
            )
        )

    if config.energetics.enabled:
        if config.energetics.source == "file":
            from raymc.energetics.metatomic import load_model_file

            model = load_model_file(config.energetics.path)
        elif config.energetics.source == "upet":
            from raymc.energetics.metatomic import load_upet_model

            model = load_upet_model(
                model=config.energetics.model,
                size=config.energetics.size,
                version=config.energetics.version,
            )
        elif config.energetics.source == "checkpoint":
            from raymc.energetics.metatomic import load_upet_model

            model = load_upet_model(checkpoint_path=config.energetics.checkpoint_path)
        else:  # pragma: no cover - guarded by the Literal on the config
            raise ValueError(f"Unknown energetics source: {config.energetics.source!r}")
        terms.append(
            make_metatomic_term(
                "energetics",
                model,
                temperature=config.energetics.temperature,
                weight=config.energetics.weight,
            )
        )

    return ChiSqAggregator(terms)


def build_run(
    config: RunConfig, log_every: int | None = None
) -> tuple[AtomicSystem, MoveSet, ChiSqAggregator, ReverseMonteCarlo]:
    """Build everything a config describes: the initial system, the
    move set, the chi-squared aggregator, and the engine itself -- ready
    to call :meth:`~raymc.engine.rmc.ReverseMonteCarlo.run` on the
    returned system.
    """
    min_dist = MinDistTable.from_config(config.constraints)
    system = build_initial_system(config.init, min_dist=min_dist, log_every=log_every)
    move_set = MoveSet.from_config(config.moves, config.constraints)
    chi_sq = build_chi_sq(config, min_dist)
    if config.init.seed is None:
        raise ValueError(
            "init.seed must be set -- ReverseMonteCarlo requires an "
            "explicit seed, there is no default generator anymore."
        )
    engine = ReverseMonteCarlo(
        move_set=move_set,
        chi_sq_agg=chi_sq,
        n_conv_steps=config.engine.n_conv_steps,
        seed=config.init.seed,
    )
    return system, move_set, chi_sq, engine
