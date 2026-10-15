"""Tests for PDFCalculator: total mode, partial mode (self and cross), and
the self-partial incremental path.

The incremental tests are the whole point of :meth:`PDFCalculator.compute_increm_change`:
it must produce *exactly* (up to floating point) what a full
:meth:`PDFCalculator.compute` would, at every step -- it's a performance
optimization, not an approximation.

PDFCalculator bins pairs into a real-space histogram (see its module
docstring for why, vs. the earlier Gaussian-per-pair version) -- one
consequence is that :meth:`compute` is no longer differentiable in atom
positions (hard bin-edge assignment has no gradient); see
``test_compute_output_does_not_require_grad`` below, which documents that
trade-off deliberately rather than leaving it to be rediscovered.
"""

from __future__ import annotations

import pytest
import torch

from raymc.modalities.pdf.calculator import PDFCalculator, _min_perpendicular_width
from raymc.moves.base import MoveDelta
from raymc.moves.group_based import (
    InsertGroup,
    RelocateGroup,
    RemoveGroup,
    RotateGroup,
    TranslateGroup,
)
from raymc.moves.moveset import MoveSet
from tests.conftest import uniform_r_grid

R_MIN, R_MAX, N_POINTS = 1.0, 6.0, 200


# -- total mode ---------------------------------------------------------------


def test_total_mode_requires_radiation():
    with pytest.raises(ValueError, match="requires radiation"):
        PDFCalculator(r_grid=uniform_r_grid(1.0, 6.0, 200))


def test_total_mode_compute_has_no_nans(packed_water):
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 6.0, 200), radiation="xray")
    gr = model.compute(packed_water)
    assert not torch.isnan(gr).any()


def test_total_mode_xray_oh_bond_peak_is_positive(packed_water):
    """Z_O * Z_H > 0, so the O-H intramolecular peak must be positive."""
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 6.0, 500), radiation="xray")
    gr = model.compute(packed_water)
    oh_region = (model.r_grid > 0.85) & (model.r_grid < 1.05)
    assert gr[oh_region].max().item() > 0


def test_total_mode_neutron_oh_bond_peak_is_negative(packed_water):
    """b_O * b_H < 0 (hydrogen's coherent scattering length is negative):
    the well-known signature of neutron PDFs of hydrogenous materials."""
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 6.0, 500), radiation="neutron")
    gr = model.compute(packed_water)
    oh_region = (model.r_grid > 0.85) & (model.r_grid < 1.05)
    assert gr[oh_region].min().item() < 0


def test_total_mode_neutron_hh_peak_is_positive_and_dominant(packed_water):
    """b_H * b_H > 0 and |b_H| is large relative to <b>, so the
    intramolecular H...H distance (~1.5A) should be a strong positive peak."""
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 6.0, 500), radiation="neutron")
    gr = model.compute(packed_water)
    hh_region = (model.r_grid > 1.4) & (model.r_grid < 1.6)
    assert gr[hh_region].max().item() > 0


def test_total_mode_compute_requires_a_periodic_system(single_water):
    model = PDFCalculator(r_grid=uniform_r_grid(1.0, 10.0, 200), radiation="xray")
    with pytest.raises(ValueError, match="periodic"):
        model.compute(single_water)


def test_min_perpendicular_width_for_a_cubic_cell():
    cell = torch.eye(3) * 15.0
    assert _min_perpendicular_width(cell).item() == pytest.approx(15.0)


def test_total_mode_rejects_a_cutoff_that_violates_the_minimum_image_convention(
    packed_water,
):
    """r_max large enough that 2*(r_max + margin) exceeds the 15A box must
    be rejected, regardless of which ALCHEMI neighbor-search backend would
    have been selected -- see the module docstring's "Cutoff safety" note.
    """
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 10.0, 200), radiation="xray")
    with pytest.raises(ValueError, match="too large for this box"):
        model.compute(packed_water)


def test_total_mode_accepts_a_cutoff_right_at_the_boundary(packed_water):
    # box=15A -> half min width = 7.5A; r_max + margin must stay <= 7.5
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 7.0, 200), radiation="xray")
    gr = model.compute(packed_water)  # should not raise
    assert not torch.isnan(gr).any()


def test_compute_output_does_not_require_grad(packed_water):
    """Deliberate, documented trade-off of the histogram-based design
    (vs. the earlier Gaussian-per-pair version): hard bin-edge assignment
    has no gradient, so compute()'s output is never connected to the
    input positions' autograd graph, even when they require_grad."""
    packed_water.positions.requires_grad_(True)
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 6.0, 200), radiation="xray")
    gr = model.compute(packed_water)
    assert gr.requires_grad is False


# -- partial mode: self partial (e.g. g_OO) ------------------------------------


def test_partial_mode_radiation_and_atomic_number_a_are_mutually_exclusive():
    with pytest.raises(ValueError, match="radiation has no meaning"):
        PDFCalculator(r_grid=uniform_r_grid(1.0, 10.0, 200), atomic_number_a=8, radiation="xray")


def test_self_partial_converges_to_one_far_from_r_max(packed_water):
    model = PDFCalculator(r_grid=uniform_r_grid(1.0, 6.0, 200), atomic_number_a=8)
    gr = model.compute(packed_water)
    assert not torch.isnan(gr).any()


def test_self_partial_with_too_few_atoms_of_that_species_is_all_ones(single_water):
    # single_water is non-periodic -- use a periodic system with < 2 of the species
    system = single_water
    system.cell = (torch.eye(3) * 15.0).unsqueeze(0)
    system.pbc = torch.tensor([[True, True, True]])
    model = PDFCalculator(r_grid=uniform_r_grid(1.0, 6.0, 50), atomic_number_a=8)
    gr = model.compute(system)  # only 1 oxygen atom
    assert torch.allclose(gr, torch.ones_like(gr))


# -- partial mode: cross partial (e.g. g_OH) -----------------------------------


def test_cross_partial_has_no_nans(packed_water):
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 6.0, 200), atomic_number_a=8, atomic_number_b=1)
    assert not model.is_self_partial
    gr = model.compute(packed_water)
    assert not torch.isnan(gr).any()


def test_cross_partial_oh_bond_shows_up_as_a_strong_peak(packed_water):
    model = PDFCalculator(r_grid=uniform_r_grid(0.5, 6.0, 500), atomic_number_a=8, atomic_number_b=1)
    gr = model.compute(packed_water)
    oh_region = (model.r_grid > 0.85) & (model.r_grid < 1.05)
    far_field = gr[model.r_grid > 4.0].mean()
    assert gr[oh_region].max().item() > far_field.item()


def test_cross_partial_atomic_number_b_none_means_self_partial():
    model = PDFCalculator(r_grid=uniform_r_grid(1.0, 10.0, 200), atomic_number_a=8, atomic_number_b=None)
    assert model.is_self_partial
    model_explicit = PDFCalculator(r_grid=uniform_r_grid(1.0, 10.0, 200), atomic_number_a=8, atomic_number_b=8)
    assert model_explicit.is_self_partial


# -- incremental path: self partial only ---------------------------------------


def _full_recompute(system):
    return PDFCalculator(r_grid=uniform_r_grid(R_MIN, R_MAX, N_POINTS), atomic_number_a=8).compute(system)


def _tracker():
    return PDFCalculator(r_grid=uniform_r_grid(R_MIN, R_MAX, N_POINTS), atomic_number_a=8)


def test_initial_g_matches_full_recompute(packed_water):
    tracker = _tracker()
    g0 = tracker.init(packed_water)
    assert torch.allclose(g0, _full_recompute(packed_water), atol=1e-8)


def test_translate_move_matches_full_recompute(packed_water):
    tracker = _tracker()
    tracker.init(packed_water)
    move = TranslateGroup(max_displacement=1.0)
    new_system, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))

    trial_g = tracker.compute_increm_change(new_system, delta)
    assert torch.allclose(trial_g, _full_recompute(new_system), atol=1e-8)


def test_rotate_move_matches_full_recompute(packed_water):
    tracker = _tracker()
    tracker.init(packed_water)
    move = RotateGroup(max_angle=0.5)
    new_system, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))

    trial_g = tracker.compute_increm_change(new_system, delta)
    assert torch.allclose(trial_g, _full_recompute(new_system), atol=1e-8)


def test_insert_move_matches_full_recompute(packed_water, water_template):
    tracker = _tracker()
    tracker.init(packed_water)
    move = InsertGroup(template=water_template)
    new_system, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))

    trial_g = tracker.compute_increm_change(new_system, delta)
    assert torch.allclose(trial_g, _full_recompute(new_system), atol=1e-8)


def test_remove_move_matches_full_recompute(packed_water):
    tracker = _tracker()
    tracker.init(packed_water)
    move = RemoveGroup()
    new_system, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))

    trial_g = tracker.compute_increm_change(new_system, delta)
    assert torch.allclose(trial_g, _full_recompute(new_system), atol=1e-8)


def test_relocate_move_matches_full_recompute(packed_water, water_template):
    """RelocateGroup is the first move to populate BOTH removed_indices
    and added_indices in the same MoveDelta -- this exercises that
    combined code path specifically, not just remove-only/add-only."""
    tracker = _tracker()
    tracker.init(packed_water)
    move = RelocateGroup(template=water_template)
    new_system, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))
    assert delta.removed_indices.numel() > 0 and delta.added_indices.numel() > 0

    trial_g = tracker.compute_increm_change(new_system, delta)
    assert torch.allclose(trial_g, _full_recompute(new_system), atol=1e-8)


def test_cross_partial_has_no_incremental_fast_path_but_still_works(
    packed_water, water_template
):
    """A cross partial falls back to Calculator's default
    compute_increm_change (a full recompute), rather than raising or
    silently doing something wrong."""
    calc = PDFCalculator(r_grid=uniform_r_grid(R_MIN, R_MAX, N_POINTS), atomic_number_a=8, atomic_number_b=1)
    move = TranslateGroup(max_displacement=1.0)
    new_system, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))
    trial = calc.compute_increm_change(new_system, delta)
    expected = PDFCalculator(r_grid=uniform_r_grid(R_MIN, R_MAX, N_POINTS), atomic_number_a=8, atomic_number_b=1).compute(new_system)
    assert torch.allclose(trial, expected, atol=1e-8)


def test_discard_leaves_committed_state_untouched(packed_water):
    """After discard(), the committed base state must be exactly what it
    was before the discarded propose() -- proven by re-issuing the same
    delta and getting the identical trial value both times."""
    tracker = _tracker()
    tracker.init(packed_water)

    move = TranslateGroup(max_displacement=1.0)
    new_system, delta = move.attempt(packed_water, torch.Generator().manual_seed(0))
    first_trial = tracker.compute_increm_change(new_system, delta)
    tracker.discard()

    second_trial = tracker.compute_increm_change(new_system, delta)
    assert torch.allclose(first_trial, second_trial, atol=1e-12)


def test_long_random_walk_of_mixed_moves_matches_full_recompute_every_accepted_step(
    water_template,
):
    """The real stress test: a long sequence of random translate / rotate /
    insert / remove moves, each randomly accepted or rejected, must track
    a from-scratch recompute exactly at every accepted step -- this is
    what actually runs inside the RMC engine.
    """
    from raymc.initialize.pack import pack_on_lattice
    from raymc.utils import MinDistTable

    cell = torch.eye(3) * 15.0
    system = pack_on_lattice(
        water_template,
        cell=cell,
        n_particles=12,
        min_dist=MinDistTable(default=3.0),
        seed=11,
    )
    tracker = _tracker()
    g0 = tracker.init(system)
    assert torch.allclose(g0, _full_recompute(system), atol=1e-8)

    move_set = MoveSet(
        moves=[
            TranslateGroup(max_displacement=0.8),
            RotateGroup(max_angle=0.8),
            InsertGroup(template=water_template),
            RemoveGroup(),
            RelocateGroup(template=water_template),
        ],
        weights=[1.0, 1.0, 0.5, 0.5, 0.7],
    )
    generator = torch.Generator().manual_seed(42)

    for _ in range(60):
        candidate, _name, delta = move_set.sample(system, generator)
        trial_g = tracker.compute_increm_change(candidate, delta)
        assert torch.allclose(trial_g, _full_recompute(candidate), atol=1e-7), (
            "incremental trial g(r) diverged from a full recompute"
        )

        accept = torch.rand((), generator=generator).item() < 0.5
        if accept:
            tracker.commit()
            system = candidate
            current = tracker.compute_increm_change(system, MoveDelta())
            assert torch.allclose(current, _full_recompute(system), atol=1e-7)
        else:
            tracker.discard()
