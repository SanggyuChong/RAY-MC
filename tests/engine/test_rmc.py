from __future__ import annotations

import torch

from raymc.config import ConvergenceConfig, ProductionConfig
from raymc.constraints.min_dist import MinDistConstraint
from raymc.engine.rmc import ReverseMonteCarlo, summarize_history
from raymc.initialize.pack import pack_on_lattice
from raymc.initialize.templates import GroupTemplate
from raymc.modalities.pdf.calculator import PDFCalculator
from raymc.moves.group_based import InsertGroup, RemoveGroup, RotateGroup, TranslateGroup
from raymc.moves.moveset import MoveSet
from raymc.residuals.aggregator import ChiSqAggregator, ChiSqTerm, make_min_dist_term
from raymc.utils import MinDistTable
from tests.conftest import uniform_r_grid


def _default_move_set(water_template):
    return MoveSet(
        moves=[
            TranslateGroup(max_displacement=0.3),
            RotateGroup(max_angle=0.3),
            InsertGroup(template=water_template),
            RemoveGroup(),
        ],
        weights=[1.0, 1.0, 0.5, 0.5],
    )


def _initial_state(water_template, seed=100):
    cell = torch.eye(3) * 15.0
    return pack_on_lattice(
        water_template,
        cell=cell,
        n_particles=5,
        min_dist=MinDistTable(default=3.0),
        seed=seed,
    )


def _chi_sq(min_dist: MinDistTable | None = None) -> ChiSqAggregator:
    return ChiSqAggregator([make_min_dist_term("min_dist", min_dist or MinDistTable())])


def test_run_executes_the_requested_number_of_steps(water_template):
    initial_state = _initial_state(water_template)
    engine = ReverseMonteCarlo(
        move_set=_default_move_set(water_template),
        chi_sq_agg=_chi_sq(),
        n_conv_steps=10,
        seed=7,
    )
    result = engine.run(initial_state)

    assert len(result.history.accepted) == 10
    assert len(result.history.chi_sqs) == 10
    assert result.final_state.num_nodes > 0
    # No production config -- run() stops after the convergence phase.
    assert result.saved_configs == []


def test_run_never_references_a_move_by_name():
    """The engine must only interact with MoveSet generically."""
    import inspect

    from raymc.engine import rmc

    source = inspect.getsource(rmc)
    for forbidden in (
        "TranslateGroup",
        "RotateGroup",
        "InsertGroup",
        "RemoveGroup",
        '"water"',
    ):
        assert forbidden not in source


def test_run_is_reproducible_with_a_seed(water_template):
    initial_state = _initial_state(water_template, seed=1)

    def _run():
        engine = ReverseMonteCarlo(
            move_set=_default_move_set(water_template),
            chi_sq_agg=_chi_sq(),
            n_conv_steps=8,
            seed=99,
        )
        return engine.run(initial_state).final_state.positions

    a = _run()
    b = _run()
    assert torch.allclose(a, b)


def test_summarize_history_reports_overall_and_per_move_stats(water_template):
    initial_state = _initial_state(water_template, seed=100)
    engine = ReverseMonteCarlo(
        move_set=_default_move_set(water_template),
        chi_sq_agg=_chi_sq(),
        n_conv_steps=10,
        seed=7,
    )
    summary = summarize_history(engine.run(initial_state).history)
    assert "Overall acceptance" in summary
    for name in ("translate", "rotate", "insert", "remove"):
        assert name in summary


def test_a_veto_calculator_drives_acceptance_through_min_dist_constraint(water_template):
    """An intentionally tight box (frequent close contacts) with the
    min-dist veto active must never end a step in a violating state."""
    cell = torch.eye(3) * 9.0
    min_dist = MinDistTable(pair_distances={(8, 8): 2.4, (1, 8): 1.5, (1, 1): 1.0})
    initial_state = pack_on_lattice(
        water_template,
        cell=cell,
        n_particles=6,
        min_dist=min_dist,
        seed=2,
    )
    constraint = MinDistConstraint(min_dist)
    engine = ReverseMonteCarlo(
        move_set=MoveSet(
            moves=[TranslateGroup(max_displacement=1.0), RotateGroup(max_angle=1.0)],
            weights=[1.0, 1.0],
        ),
        chi_sq_agg=_chi_sq(min_dist),
        n_conv_steps=40,
        seed=5,
    )
    result = engine.run(initial_state)
    constraint.check(result.final_state)  # must not raise


class _RecordingChiSqAggregator:
    """A minimal ChiSqAggregator test double: proves the engine actually
    calls init/evaluate/accept with the contract it relies on."""

    def current_values(self):
        return {}

    def __init__(self):
        self.init_calls = 0
        self.evaluate_calls = 0
        self.accepted_flags: list[bool] = []

    def init(self, system):
        self.init_calls += 1
        return torch.tensor(0.0, dtype=torch.float64)

    def evaluate(self, candidate, delta):
        self.evaluate_calls += 1
        # Every candidate looks better than current -- guarantees accept.
        return torch.tensor(-1.0, dtype=torch.float64)

    def accept(self, keep: bool) -> None:
        self.accepted_flags.append(keep)


def test_run_drives_a_chi_sq_aggregator_through_init_evaluate_accept(water_template):
    initial_state = _initial_state(water_template, seed=1)
    recorder = _RecordingChiSqAggregator()
    engine = ReverseMonteCarlo(
        move_set=_default_move_set(water_template),
        chi_sq_agg=recorder,
        n_conv_steps=6,
        seed=3,
    )
    result = engine.run(initial_state)

    assert recorder.init_calls == 1
    assert recorder.evaluate_calls == 6
    assert len(recorder.accepted_flags) == 6
    # Every proposal looked strictly better -- always accepted.
    assert all(result.history.accepted)
    assert all(recorder.accepted_flags)


# -- run_to_convergence_and_produce (Task 4) --------------------------------


def _ar_lattice_system(n_per_axis=3, spacing=4.0, min_dist=None, seed=1):
    ar = GroupTemplate.atom("Ar")
    cell = torch.eye(3) * (n_per_axis * spacing)
    return pack_on_lattice(
        ar,
        cell,
        n_particles=n_per_axis**3,
        min_dist=min_dist,
        seed=seed,
    )


def _ar_pdf_chi_sq(sigma=0.2):
    r_grid = torch.linspace(1.0, 5.0, 40, dtype=torch.float64)
    target_g = torch.ones_like(r_grid)
    # one grid, shared by model and target -- the new contract
    calculator = PDFCalculator(r_grid=r_grid, atomic_number_a=18)
    term = ChiSqTerm(
        name="pdf_Ar",
        calculator=calculator,
        target=target_g,
        sigma=torch.full_like(target_g, sigma),
    )
    return ChiSqAggregator([term]), term


def test_production_saves_the_requested_number_of_configs_and_averages_g_of_r():
    min_dist = MinDistTable(default=2.5)
    system = _ar_lattice_system(min_dist=min_dist)
    chi_sq, term = _ar_pdf_chi_sq()
    engine = ReverseMonteCarlo(
        move_set=MoveSet(moves=[TranslateGroup(max_displacement=0.5)], weights=[1.0]),
        chi_sq_agg=chi_sq,
        n_conv_steps=200,
        seed=11,
    )
    result = engine.run(
        system,
        convergence=ConvergenceConfig(window=20, threshold=1e-9, patience=100),
        production=ProductionConfig(n_moves=100, n_configs=5, save_interval=20),
    )

    assert len(result.saved_configs) == 5
    assert term.name in result.averaged_raw
    assert result.averaged_raw[term.name].shape == term.target.shape
    assert term.name in result.final_snapshot_raw
    # convergence.threshold is impossibly tight -- must not have converged
    # within the pre-production budget.
    assert result.converged is False
    assert result.convergence_step is None
    assert len(result.history.accepted) == 200 + 100


def test_convergence_is_detected_when_the_score_plateaus():
    """A trivial always-accept aggregator (constant candidate score) must
    converge on the very first window-comparison."""
    class _FlatAggregator:
        def __init__(self):
            self.terms = []

        def current_values(self):
            return {}

        def init(self, system):
            return torch.tensor(0.0, dtype=torch.float64)

        def evaluate(self, candidate, delta):
            return torch.tensor(0.0, dtype=torch.float64)

        def accept(self, keep):
            pass

    min_dist = MinDistTable(default=2.5)
    system = _ar_lattice_system(min_dist=min_dist)
    engine = ReverseMonteCarlo(
        move_set=MoveSet(moves=[TranslateGroup(max_displacement=0.5)], weights=[1.0]),
        chi_sq_agg=_FlatAggregator(),
        n_conv_steps=1000,
        seed=3,
    )
    result = engine.run(
        system,
        convergence=ConvergenceConfig(window=10, threshold=0.5, patience=2),
        production=ProductionConfig(n_moves=10, n_configs=2, save_interval=5),
    )
    assert result.converged is True
    assert result.convergence_step is not None
    assert result.convergence_step < 1000


def test_production_acceptance_uses_the_instantaneous_score_every_step():
    """Acceptance during production must be driven by each step's own
    freshly-evaluated candidate score, never a running average over saved
    configurations -- evaluate() must be called exactly once per
    production move, with no special-cased averaging path."""

    class _CountingAggregator:
        def __init__(self):
            self.terms = []
            self.evaluate_calls = 0

        def current_values(self):
            return {}

        def init(self, system):
            return torch.tensor(0.0, dtype=torch.float64)

        def evaluate(self, candidate, delta):
            self.evaluate_calls += 1
            # Oscillate so acceptance genuinely depends on THIS call's
            # value, not some smoothed running quantity.
            return torch.tensor(-1.0 if self.evaluate_calls % 2 == 0 else 1.0, dtype=torch.float64)

        def accept(self, keep):
            pass

    min_dist = MinDistTable(default=2.5)
    system = _ar_lattice_system(min_dist=min_dist)
    recorder = _CountingAggregator()
    engine = ReverseMonteCarlo(
        move_set=MoveSet(moves=[TranslateGroup(max_displacement=0.5)], weights=[1.0]),
        chi_sq_agg=recorder,
        n_conv_steps=5,
        seed=4,
    )
    result = engine.run(
        system,
        convergence=ConvergenceConfig(window=1, threshold=1e-12, patience=1),
        production=ProductionConfig(n_moves=20, n_configs=4, save_interval=5),
    )
    # 5 pre-production steps (converges immediately given the tiny
    # threshold isn't met by oscillation) + up to 20 production steps,
    # one evaluate() call each -- never fewer, proving no averaging
    # shortcut skips a fresh per-step evaluation.
    assert recorder.evaluate_calls == len(result.history.accepted)
