"""Tests for the RunConfig -> ChiSqAggregator/engine orchestration layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from raymc.config import (
    ConstraintConfig,
    InitConfig,
    MinDistPairConfig,
    ModalityConfig,
    MoveConfig,
    RunConfig,
)
from raymc.constraints.max_dist import MaxDistConstraint
from raymc.constraints.min_dist import MinDistConstraint
from raymc.engine.rmc import ReverseMonteCarlo
from raymc.modalities.pdf.calculator import PDFCalculator
from raymc.orchestrate import build_chi_sq, build_run
from raymc.utils import MinDistTable


def _minimal_config(**overrides) -> RunConfig:
    defaults = dict(
        name="test",
        init=InitConfig(
            strategy="pack_on_lattice",
            molecule="water",
            n_particles=10,
            cell=(15.0, 15.0, 15.0),
            seed=1,
        ),
        moves=[MoveConfig(type="translate_group", params={"max_displacement": 0.3})],
        engine={"n_conv_steps": 5},
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_build_chi_sq_always_includes_the_min_dist_constraint():
    config = _minimal_config()
    min_dist = MinDistTable()
    chi_sq = build_chi_sq(config, min_dist)
    assert len(chi_sq.terms) == 1
    assert isinstance(chi_sq.terms[0].calculator, MinDistConstraint)


def test_build_chi_sq_omits_the_max_dist_term_when_nothing_is_constrained():
    """The default (infinite) maximum separation must cost nothing at all:
    not a cheap term that returns 0 every step, but no term."""
    chi_sq = build_chi_sq(_minimal_config(), MinDistTable())
    assert not any(isinstance(t.calculator, MaxDistConstraint) for t in chi_sq.terms)


def test_build_chi_sq_adds_the_max_dist_term_once_a_pair_is_constrained():
    config = _minimal_config(
        constraints=ConstraintConfig(
            max_pairs=[MinDistPairConfig(elements=("O", "O"), distance=6.0)]
        )
    )
    chi_sq = build_chi_sq(config, MinDistTable())
    max_terms = [t for t in chi_sq.terms if isinstance(t.calculator, MaxDistConstraint)]
    assert len(max_terms) == 1
    assert max_terms[0].calculator.max_dist.lookup(8, 8).item() == 6.0


def test_build_chi_sq_threads_the_ramp_parameters_into_both_constraints():
    config = _minimal_config(
        constraints=ConstraintConfig(
            softening_window=0.4,
            penalty_at_window=25.0,
            max_pairs=[MinDistPairConfig(elements=("O", "O"), distance=6.0)],
        )
    )
    chi_sq = build_chi_sq(config, MinDistTable())
    constraints = [
        t.calculator
        for t in chi_sq.terms
        if isinstance(t.calculator, (MinDistConstraint, MaxDistConstraint))
    ]
    assert len(constraints) == 2  # both of them, not just whichever came first
    assert {c.softening_window for c in constraints} == {0.4}
    assert {c.penalty_at_window for c in constraints} == {25.0}


def test_build_chi_sq_adds_a_pdf_term_for_each_pdf_modality():
    import torch

    from raymc.modalities.pdf.reference import ReferenceCurve

    r = torch.linspace(1.0, 6.0, 100, dtype=torch.float64)
    ref_curve = ReferenceCurve(x=r, y=torch.ones_like(r), error=None)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "ref.txt"
        target_file.write_text("\n".join(f"{ri.item()} {gi.item()}" for ri, gi in zip(r, ref_curve.y)))

        config = _minimal_config(
            modalities=[
                ModalityConfig(
                    name="pdf",
                    target_file=str(target_file),
                    element_a="O",
                    r_min=1.0,
                    r_max=6.0,
                    sigma=0.1,
                )
            ]
        )
        min_dist = MinDistTable()
        chi_sq = build_chi_sq(config, min_dist)

        assert len(chi_sq.terms) == 2
        pdf_term = next(t for t in chi_sq.terms if isinstance(t.calculator, PDFCalculator))
        assert pdf_term.calculator.atomic_number_a == 8
        assert pdf_term.calculator.is_self_partial


def test_build_chi_sq_rejects_an_unimplemented_modality_name():
    config = _minimal_config(
        modalities=[ModalityConfig(name="nmr", target_file="foo.txt", element_a="O")]
    )
    with pytest.raises(ValueError, match="Unknown modality"):
        build_chi_sq(config, MinDistTable())


def test_build_chi_sq_skips_energetics_when_disabled():
    config = _minimal_config(energetics={"enabled": False})
    chi_sq = build_chi_sq(config, MinDistTable())
    # Only the always-on min-dist veto -- no energetics term, and no
    # attempt to load a model (which would fail without the optional
    # extras installed).
    assert len(chi_sq.terms) == 1


def test_build_run_produces_a_working_engine():
    config = _minimal_config()
    system, move_set, chi_sq, engine = build_run(config)
    assert system.num_nodes > 0
    assert isinstance(engine, ReverseMonteCarlo)
    result = engine.run(system)
    assert len(result.history.accepted) == 5
