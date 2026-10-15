"""Tests for ChiSqTerm/ChiSqAggregator: residual vs. direct modes, weighted
combination, and that swapping a term's calculator for one with no
incremental path doesn't change a single accept/reject decision -- only
how much work it took to get there.
"""

from __future__ import annotations

import pytest
import torch

from raymc.engine.rmc import ReverseMonteCarlo
from raymc.initialize.pack import pack_on_lattice
from raymc.modalities.base import Calculator
from raymc.modalities.pdf.calculator import PDFCalculator
from raymc.moves.group_based import InsertGroup, RemoveGroup, RotateGroup, TranslateGroup
from raymc.moves.moveset import MoveSet
from raymc.residuals.aggregator import (
    ChiSqAggregator,
    ChiSqTerm,
    make_max_dist_term,
    make_min_dist_term,
    make_partial_pdf_term,
)
from raymc.utils import MaxDistTable, MinDistTable
from tests.conftest import uniform_r_grid

R_MIN, R_MAX, N_POINTS = 1.0, 6.0, 200


class _ConstantCalculator(Calculator):
    def __init__(self, value: float, shape: tuple = ()):
        self._value = torch.full(shape, value, dtype=torch.float64)

    def compute(self, system) -> torch.Tensor:
        return self._value.clone()


class _FullRecomputeOnlyPartialPDF(Calculator):
    """Wraps a fresh self-partial PDFCalculator's compute() every call --
    deliberately gives up the incremental fast path, to prove the engine
    trajectory doesn't depend on which path a term's calculator takes."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def compute(self, system) -> torch.Tensor:
        return PDFCalculator(**self._kwargs).compute(system)


# -- ChiSqTerm: residual vs. direct mode --------------------------------------


def test_residual_mode_computes_chi_square_against_target(packed_water):
    target = torch.zeros(5, dtype=torch.float64)
    term = ChiSqTerm(
        name="a",
        calculator=_ConstantCalculator(2.0, shape=(5,)),
        target=target,
        sigma=torch.ones(5, dtype=torch.float64),
    )
    value = term.init(packed_water)
    # sum, not mean: 5 points each ((2-0)/1)^2 = 4 -> sum = 20
    assert torch.allclose(value, torch.tensor(20.0, dtype=torch.float64))


def test_direct_mode_passes_the_raw_value_through_unsquared(packed_water):
    term = ChiSqTerm(name="e", calculator=_ConstantCalculator(-3.0), target=None)
    value = term.init(packed_water)
    assert value.item() == -3.0


def test_direct_mode_rejects_a_sigma():
    with pytest.raises(ValueError, match="sigma has no meaning"):
        ChiSqTerm(name="e", calculator=_ConstantCalculator(1.0), target=None, sigma=1.0)


def test_residual_term_without_sigma_raises(packed_water):
    target = torch.tensor([1.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="requires an explicit sigma"):
        ChiSqTerm(name="a", calculator=_ConstantCalculator(3.0, shape=(2,)), target=target)


def test_sigma_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match target shape"):
        ChiSqTerm(
            name="a",
            calculator=_ConstantCalculator(1.0, shape=(3,)),
            target=torch.zeros(3, dtype=torch.float64),
            sigma=torch.zeros(2, dtype=torch.float64),
        )


def test_accept_true_commits_and_updates_current_value(packed_water):
    term = ChiSqTerm(name="a", calculator=_ConstantCalculator(1.0), target=None)
    term.init(packed_water)
    term.evaluate(packed_water, delta=None)
    term.accept(True)
    assert term.current_value.item() == 1.0


def test_accept_false_discards_and_leaves_current_value_unchanged(packed_water):
    term = ChiSqTerm(name="a", calculator=_ConstantCalculator(1.0), target=None)
    term.init(packed_water)
    term.evaluate(packed_water, delta=None)
    term.accept(False)
    assert term.current_value.item() == 1.0  # unchanged from init


def test_chi_sq_per_nr_matches_hand_computed_value(packed_water):
    """chi^2/n_r for a known configuration against a known target must
    match a hand-computed value -- exercises sigma, sum-reduction (not
    mean), and the reported chi^2/n_r headline metric together."""
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    target = torch.tensor([1.5, 1.5, 1.5, 1.5], dtype=torch.float64)
    sigma = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

    term = ChiSqTerm(
        name="a", calculator=_ConstantCalculator(0.0), target=target, sigma=sigma
    )
    # by hand: ((1-1.5)/.5)^2 + ((2-1.5)/.5)^2 + ((3-1.5)/.5)^2 + ((4-1.5)/.5)^2
    #        = 1 + 1 + 9 + 25 = 36 ; n_r = 4 ; chi^2/n_r = 9.0
    report = term.report_from_raw(raw)
    assert report["chi2"] == pytest.approx(36.0)
    assert report["chi2_per_nr"] == pytest.approx(9.0)
    assert report["n_r"] == 4

    delta = raw - target
    expected_rw = float(torch.sqrt((delta**2).sum() / (target**2).sum()))
    assert report["Rw"] == pytest.approx(expected_rw)


def test_acceptance_uses_sum_not_mean_reduction(packed_water):
    target = torch.zeros(4, dtype=torch.float64)
    term = ChiSqTerm(
        name="a",
        calculator=_ConstantCalculator(1.0, shape=(4,)),
        target=target,
        sigma=torch.ones(4, dtype=torch.float64),
    )
    value = term.init(packed_water)
    # sum of 4 points each ((1-0)/1)^2 = 1 -> 4, NOT mean-reduced to 1
    assert value.item() == pytest.approx(4.0)


def test_acceptance_value_matches_reported_chi_square(packed_water):
    """Per McGreevy & Pusztai (1988), Eq. 8/11/12, the same sigma-scaled
    chi-squared drives both acceptance and the reported fit quality --
    there is no separate acceptance-width parameter."""
    target = torch.zeros(2, dtype=torch.float64)
    term = ChiSqTerm(
        name="a",
        calculator=_ConstantCalculator(2.0, shape=(2,)),
        target=target,
        sigma=torch.ones(2, dtype=torch.float64),
    )
    accept_value = term.init(packed_water)
    # sum of 2 * ((2-0)/1)^2 = 8
    assert accept_value.item() == pytest.approx(8.0)
    report = term.report(packed_water)
    assert report["chi2"] == pytest.approx(8.0)


def test_engine_raises_a_clear_error_when_initial_state_violates_a_constraint(
    water_template,
):
    """A +inf initial score must raise immediately, not silently reject
    every move forever via inf - inf = NaN.

    Still reachable now that the constraint scores a smooth penalty rather
    than a step: two molecules placed exactly on top of each other are 3.0
    A inside a 3.0 A minimum distance, i.e. 30 softening windows, which
    overflows float64 to a literal +inf (see raymc.constraints.penalty)."""
    cell = torch.eye(3) * 15.0
    # Two molecules deliberately placed on top of each other -- guaranteed
    # min-dist violation regardless of seed.
    from raymc.system import AtomicSystem, set_rigid_groups

    positions = torch.cat([water_template.positions, water_template.positions], dim=0)
    atomic_numbers = torch.cat(
        [water_template.atomic_numbers, water_template.atomic_numbers]
    )
    system = AtomicSystem(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell.unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    set_rigid_groups(
        system,
        torch.cat(
            [
                torch.zeros(water_template.num_atoms, dtype=torch.long),
                torch.ones(water_template.num_atoms, dtype=torch.long),
            ]
        ),
    )
    min_dist = MinDistTable(default=3.0)
    engine = ReverseMonteCarlo(
        move_set=MoveSet(moves=[TranslateGroup(max_displacement=0.3)], weights=[1.0]),
        chi_sq_agg=ChiSqAggregator([make_min_dist_term("min_dist", min_dist)]),
        n_conv_steps=5,
        seed=1,
    )
    with pytest.raises(RuntimeError, match="not finite"):
        engine.run(system)


# -- ChiSqAggregator -----------------------------------------------------------


def test_single_term_matches_its_own_weighted_value(packed_water):
    agg = ChiSqAggregator(
        [ChiSqTerm(name="a", calculator=_ConstantCalculator(3.0), target=None, weight=2.0)]
    )
    result = agg.init(packed_water)
    assert result.item() == 6.0


def test_multiple_terms_sum_their_weighted_contributions(packed_water):
    agg = ChiSqAggregator(
        [
            ChiSqTerm(name="a", calculator=_ConstantCalculator(1.0), target=None, weight=2.0),
            ChiSqTerm(name="b", calculator=_ConstantCalculator(10.0), target=None, weight=0.5),
        ]
    )
    result = agg.init(packed_water)
    assert result.item() == pytest.approx(7.0)  # 2*1.0 + 0.5*10.0


def test_breakdown_reports_unweighted_per_term_values(packed_water):
    agg = ChiSqAggregator(
        [
            ChiSqTerm(name="a", calculator=_ConstantCalculator(1.0), target=None, weight=100.0),
            ChiSqTerm(name="b", calculator=_ConstantCalculator(2.0), target=None, weight=100.0),
        ]
    )
    breakdown = agg.breakdown(packed_water)
    assert breakdown["a"].item() == 1.0
    assert breakdown["b"].item() == 2.0


def test_requires_at_least_one_term():
    with pytest.raises(ValueError, match="at least one"):
        ChiSqAggregator([])


# -- factories ------------------------------------------------------------------


def _flat_reference_curve():
    from raymc.modalities.pdf.reference import ReferenceCurve

    r = uniform_r_grid(R_MIN, R_MAX, N_POINTS)
    return ReferenceCurve(x=r, y=torch.ones_like(r), error=None)


def test_make_partial_pdf_term_produces_a_working_term(packed_water):
    term = make_partial_pdf_term(
        "pdf_oo", _flat_reference_curve(), atomic_number_a=8,
        sigma=0.1,
    )
    result = term.init(packed_water)
    assert result.shape == ()
    assert torch.isfinite(result)


def test_make_partial_pdf_term_gets_the_incremental_path_for_a_self_partial(packed_water):
    term = make_partial_pdf_term(
        "pdf_oo", _flat_reference_curve(), atomic_number_a=8,
        sigma=0.1,
    )
    assert term.calculator.is_self_partial


def test_make_partial_pdf_term_adopts_the_reference_grid_verbatim():
    """The decisive grid test, inverted from what it used to assert.

    The reference file now OWNS the grid: the histogram's bin centers are
    the reference's own r values and the target is the reference's own y
    values, untouched. Previously the reference was interpolated onto a
    grid the model chose, which made fit quality depend on how the file
    had been digitized and made an unevenly-sampled reference unusable.
    """
    from raymc.modalities.pdf.reference import ReferenceCurve

    # deliberately uneven: dense across [2, 3], sparse either side -- the
    # exact shape of a real total-scattering file, and impossible to
    # express at all under the old fixed-linspace arrangement.
    r = torch.cat([
        torch.arange(1.0, 2.0, 0.10, dtype=torch.float64),
        torch.arange(2.0, 3.0, 0.02, dtype=torch.float64),
        torch.arange(3.0, 6.0, 0.10, dtype=torch.float64),
    ])
    y = torch.linspace(0.0, 1.0, r.numel(), dtype=torch.float64)  # a ramp, so
    # any resampling at all would show up as changed values
    term = make_partial_pdf_term(
        "pdf_oo", ReferenceCurve(x=r, y=y, error=None), atomic_number_a=8, sigma=0.1,
    )

    assert torch.equal(term.calculator.r_grid, r)
    assert torch.equal(term.target, y)
    # and the bins really are unevenly wide, not quietly regularized
    dr = term.calculator.dr
    assert float(dr.max()) > 4 * float(dr.min())


def test_r_min_and_r_max_select_reference_points_without_resampling():
    from raymc.modalities.pdf.reference import ReferenceCurve

    r = uniform_r_grid(1.0, 6.0, 100)
    ref = ReferenceCurve(x=r, y=torch.ones_like(r), error=None)
    term = make_partial_pdf_term(
        "pdf_oo", ref, atomic_number_a=8, r_min=2.0, r_max=4.0, sigma=0.1,
    )
    kept = r[(r >= 2.0) & (r <= 4.0)]
    assert torch.equal(term.calculator.r_grid, kept)
    assert term.target.numel() == kept.numel()


def test_constraint_terms_use_direct_mode_so_the_penalty_is_not_squared(water_template):
    """The constraints' smooth penalty is already on the chi-squared scale
    (1000 for a pair one softening window past its threshold), so the term
    must pass it through as-is. Squaring it against a target of 0 -- which
    is what these terms used to do -- would silently turn that 1000 into
    1e6."""
    for term in (
        make_min_dist_term("min_dist", MinDistTable(pair_distances={(8, 8): 2.4})),
        make_max_dist_term("max_dist", MaxDistTable(pair_distances={(8, 8): 6.0})),
    ):
        assert term.target is None
        assert term.sigma is None
        raw = torch.tensor(1000.0, dtype=torch.float64)
        assert term._reduce(raw).item() == 1000.0


# -- incremental-vs-full-recompute equivalence, driven through the engine ------


def _move_set(water_template):
    return MoveSet(
        moves=[
            TranslateGroup(max_displacement=0.7),
            RotateGroup(max_angle=0.7),
            InsertGroup(template=water_template),
            RemoveGroup(),
        ],
        weights=[1.0, 1.0, 0.5, 0.5],
    )


def _target_curve():
    r_grid = torch.linspace(R_MIN, R_MAX, N_POINTS, dtype=torch.float64)
    target_g = torch.ones_like(r_grid) + 0.3 * torch.exp(
        -0.5 * ((r_grid - 2.8) / 0.2) ** 2
    )
    return target_g, torch.full_like(r_grid, 0.1)


def test_engine_run_is_identical_with_incremental_vs_full_recompute_pdf_term(
    water_template,
):
    """The decisive test: swapping the self-partial PDFCalculator's real
    incremental path for a deliberately-full-recompute-only calculator
    must not change a single accept/reject decision or final atom
    position -- only how much work it took to get there.
    """
    target_g, target_err = _target_curve()
    cell = torch.eye(3) * 15.0

    def run(calculator_factory):
        initial_state = pack_on_lattice(
            water_template,
            cell=cell,
            n_particles=8,
            min_dist=MinDistTable(default=3.0),
            seed=100,
        )
        term = ChiSqTerm(
            name="pdf_oo",
            calculator=calculator_factory(),
            target=target_g,
            sigma=target_err,
        )
        engine = ReverseMonteCarlo(
            move_set=_move_set(water_template),
            chi_sq_agg=ChiSqAggregator([term]),
            n_conv_steps=40,
            seed=7,
        )
        return engine.run(initial_state)

    kwargs = dict(atomic_number_a=8, r_grid=uniform_r_grid(R_MIN, R_MAX, N_POINTS))
    incremental = run(lambda: PDFCalculator(**kwargs))
    full = run(lambda: _FullRecomputeOnlyPartialPDF(**kwargs))
    incremental_state, incremental_history = incremental.final_state, incremental.history
    full_state, full_history = full.final_state, full.history

    assert full_history.accepted == incremental_history.accepted
    assert full_history.move_names == incremental_history.move_names
    for full_chi_sq, inc_chi_sq in zip(full_history.chi_sqs, incremental_history.chi_sqs):
        assert full_chi_sq == inc_chi_sq or abs(full_chi_sq - inc_chi_sq) < 1e-6

    assert full_state.num_nodes == incremental_state.num_nodes
    assert torch.allclose(full_state.positions, incremental_state.positions, atol=1e-6)


# -- RMCA Section 9.6, as a declared constant rather than a per-step fit --


class _FixedCalculator(Calculator):
    """Returns a preset observable, ignoring the system entirely."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def compute(self, system):  # noqa: D102
        return self.value


def _reference_and_model():
    """A reference carrying a linear normalization error, and the model
    values that would fit it perfectly once corrected."""
    raw_data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    model = 2.5 * raw_data + 0.7
    return raw_data, model


def _term(data_scale=1.0, data_offset=0.0):
    raw_data, model = _reference_and_model()
    return ChiSqTerm(
        "sq",
        _FixedCalculator(model),
        target=data_scale * raw_data + data_offset,
        sigma=torch.full((4,), 0.1, dtype=torch.float64),
        data_scale=data_scale,
        data_offset=data_offset,
    ), model


def test_declared_scale_and_offset_absorb_a_linear_data_error():
    term, model = _term(data_scale=2.5, data_offset=0.7)
    assert float(term._reduce(model)) == pytest.approx(0.0, abs=1e-16)


def test_each_correction_alone_only_absorbs_its_own_part():
    plain, model = _term()
    scaled, _ = _term(data_scale=2.5)
    shifted, _ = _term(data_offset=0.7)
    assert float(scaled._reduce(model)) < float(plain._reduce(model))
    assert float(shifted._reduce(model)) < float(plain._reduce(model))
    assert float(scaled._reduce(model)) > 0.0


def test_identity_by_default_so_the_target_is_the_data():
    raw_data, _ = _reference_and_model()
    term, _ = _term()
    assert torch.allclose(term.target, raw_data)


def test_the_applied_correction_is_reported_for_inspection():
    term, model = _term(data_scale=2.5, data_offset=0.7)
    report = term.report_from_raw(model)
    assert report["data_scale"] == pytest.approx(2.5)
    assert report["data_offset"] == pytest.approx(0.7)


def test_an_uncorrected_term_does_not_clutter_its_report():
    term, model = _term()
    assert "data_scale" not in term.report_from_raw(model)


def test_the_target_is_stationary_across_evaluations(packed_water):
    """The property the per-step least-squares fit destroyed: chi-squared
    must be a fixed objective, so that a Metropolis delta compares two
    states of the same function rather than two different functions.
    """
    term, model = _term(data_scale=2.5, data_offset=0.7)
    before = term.target.clone()
    for _ in range(5):
        term._reduce(model * 1.3)
        term._reduce(model * 0.4)
    assert torch.equal(term.target, before)
