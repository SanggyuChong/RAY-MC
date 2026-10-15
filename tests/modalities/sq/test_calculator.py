"""Tests for the reciprocal-space S(Q) modality.

The load-bearing checks here are the three that pin the normalization
down independently of the implementation:

- a perfectly uniform g(r) = 1 must transform to S(Q) = 1 exactly;
- the binned and unbinned forward models must agree, since they are the
  same expression differing only in binning;
- an independent sine transform of :class:`PDFCalculator`'s own g(r)
  must reproduce S(Q), which ties the two modalities' conventions
  together end to end.
"""

import math
import warnings

import pytest
import torch

from raymc.modalities.pdf.calculator import PDFCalculator
from raymc.modalities.sq.calculator import SQCalculator
from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem
from tests.conftest import uniform_r_grid

L = 30.0
N = 500


def _random_system(seed: int = 0, n: int = N, side: float = L, z: int = 18):
    torch.manual_seed(seed)
    return AtomicSystem(
        positions=torch.rand(n, 3, dtype=torch.float64) * side,
        atomic_numbers=torch.full((n,), z, dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * side,
        pbc=torch.tensor([[True, True, True]]),
    )


def _two_species_system(seed: int = 0, n: int = 300, side: float = 24.0):
    torch.manual_seed(seed)
    z = torch.where(torch.arange(n) % 3 == 0, 8, 1).long()
    return AtomicSystem(
        positions=torch.rand(n, 3, dtype=torch.float64) * side,
        atomic_numbers=z,
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * side,
        pbc=torch.tensor([[True, True, True]]),
    )


def _q(n: int = 60, lo: float = 0.5, hi: float = 12.0):
    return torch.linspace(lo, hi, n, dtype=torch.float64)


def _calc(**kwargs):
    """SQCalculator with a grid fine enough not to trip the dr warning."""
    params = dict(q_grid=_q(), r_max=14.0, n_r=700, radiation="neutron")
    params.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SQCalculator(**params)


# -- normalization ---------------------------------------------------


def test_uniform_pair_density_transforms_to_exactly_one():
    """g(r) = 1 everywhere must give S(Q) = 1 at every Q.

    This is the sharpest available check that the pair term and the
    uniform background share one quadrature: any mismatch between them
    shows up here immediately, and nowhere else so cleanly.
    """
    calc = _calc(window="none")
    ideal_pairs = calc._ideal_pair_counts((N,))[0]
    shell = 4 * math.pi * calc.r_grid**2 * calc.dr
    counts = ideal_pairs * shell / (L**3)

    s = calc._assemble(
        calc._pair_term_from_counts(counts.unsqueeze(0)),
        (18,),
        (N,),
        torch.tensor(L**3, dtype=torch.float64),
    )
    assert torch.allclose(s, torch.ones_like(s), atol=1e-10)


def test_uniform_density_still_transforms_to_one_under_a_window():
    """The window multiplies pair term and background alike, so damping
    must not shift the S(Q) = 1 baseline -- only suppress ripple."""
    for window in ("lorch", "hann"):
        calc = _calc(window=window)
        ideal_pairs = calc._ideal_pair_counts((N,))[0]
        shell = 4 * math.pi * calc.r_grid**2 * calc.dr
        counts = ideal_pairs * shell / (L**3)
        s = calc._assemble(
            calc._pair_term_from_counts(counts.unsqueeze(0)),
            (18,),
            (N,),
            torch.tensor(L**3, dtype=torch.float64),
        )
        assert torch.allclose(s, torch.ones_like(s), atol=1e-10), window


def test_random_configuration_hovers_about_one():
    s = _calc(window="none").compute(_random_system())
    assert s.mean().item() == pytest.approx(1.0, abs=0.05)


def test_matches_an_independent_transform_of_the_pdf_modality_g_of_r():
    """Tie the two modalities together: transform PDFCalculator's own
    g(r) by hand and require SQCalculator to reproduce it.

    This validates the whole chain -- shell-volume normalization, the
    N_a(N_a-1) pair convention, the 4*pi*rho/Q prefactor -- against code
    that shares none of it.
    """
    system = _random_system()
    r_max, n_r = 14.0, 700
    q = _q()

    pdf = PDFCalculator(r_grid=uniform_r_grid(1e-9, r_max, n_r), atomic_number_a=18)
    g = pdf.compute(system)
    dr = pdf.dr
    rho = N / L**3

    expected = 1.0 + (4 * math.pi * rho / q) * torch.stack(
        [(pdf.r_grid * (g - 1.0) * torch.sin(qi * pdf.r_grid) * dr).sum() for qi in q]
    )
    got = _calc(q_grid=q, r_max=r_max, n_r=n_r, window="none").compute(system)
    assert torch.allclose(got, expected, atol=1e-8)


# -- the two forward models ------------------------------------------


def test_histogram_and_debye_paths_agree():
    system = _random_system()
    binned = _calc(window="none").compute(system)
    direct = _calc(window="none", forward_model="debye").compute(system)
    assert torch.allclose(binned, direct, atol=5e-3)


def test_finer_binning_converges_towards_the_unbinned_path():
    """The only difference between the paths is binning, so refining the
    r grid must shrink it."""
    system = _random_system()
    direct = _calc(n_r=700, window="none", forward_model="debye").compute(system)
    coarse = (_calc(n_r=350, window="none").compute(system) - direct).abs().max()
    fine = (_calc(n_r=2800, window="none").compute(system) - direct).abs().max()
    assert fine < coarse


def test_unknown_forward_model_raises():
    with pytest.raises(ValueError, match="Unknown forward_model"):
        _calc(forward_model="fft")


# -- conventions and modes -------------------------------------------


def test_f_convention_is_s_minus_one():
    system = _random_system()
    s = _calc(convention="S").compute(system)
    f = _calc(convention="F").compute(system)
    assert torch.allclose(s - 1.0, f, atol=1e-12)


def test_partial_mode_returns_the_selected_pair():
    system = _two_species_system()
    q = _q(30, 0.6, 10.0)
    oo = _calc(q_grid=q, r_max=11.0, n_r=550, radiation=None, atomic_number_a=8)
    oh = _calc(
        q_grid=q, r_max=11.0, n_r=550, radiation=None, atomic_number_a=8,
        atomic_number_b=1,
    )
    assert not torch.allclose(oo.compute(system), oh.compute(system))


def test_partial_mode_rejects_weighting_arguments():
    with pytest.raises(ValueError, match="radiation has no meaning"):
        _calc(atomic_number_a=8)
    with pytest.raises(ValueError, match="species_weights has no meaning"):
        SQCalculator(
            q_grid=_q(), r_max=14.0, n_r=700,
            atomic_number_a=8, species_weights={"O": 1.0},
        )


def test_partial_for_an_absent_element_raises():
    calc = _calc(radiation=None, atomic_number_a=8)
    with pytest.raises(ValueError, match="does not contain"):
        calc.compute(_random_system())


def test_total_mode_requires_radiation():
    with pytest.raises(ValueError, match="requires radiation"):
        SQCalculator(q_grid=_q(), r_max=14.0, n_r=700)


def test_species_weight_override_changes_the_total():
    system = _two_species_system()
    q = _q(30, 0.6, 10.0)
    plain = _calc(q_grid=q, r_max=11.0, n_r=550).compute(system)
    heavy = _calc(
        q_grid=q, r_max=11.0, n_r=550, species_weights={"H": 6.671}
    ).compute(system)
    assert not torch.allclose(plain, heavy)


# -- guards ------------------------------------------------------------


def test_zero_or_negative_q_is_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        _calc(q_grid=torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_r_max_beyond_half_the_box_raises():
    calc = _calc(r_max=16.0, n_r=800)
    with pytest.raises(ValueError, match="too large for this box"):
        calc.compute(_random_system())


def test_non_periodic_system_raises():
    system = AtomicSystem(
        positions=torch.rand(10, 3, dtype=torch.float64),
        atomic_numbers=torch.full((10,), 18, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="periodic system"):
        _calc().compute(system)


def test_coarse_r_grid_warns_per_rmca_section_9_4():
    with pytest.warns(UserWarning, match="coarser than"):
        SQCalculator(q_grid=_q(), r_max=14.0, n_r=20, radiation="neutron")


def test_q_below_two_pi_over_l_warns_per_rmca_section_9_4():
    calc = _calc(q_grid=torch.tensor([0.05, 5.0], dtype=torch.float64))
    with pytest.warns(UserWarning, match="Q_min = 2\\*pi/L"):
        calc.compute(_random_system())


# -- incremental path --------------------------------------------------


def test_incremental_matches_full_recompute_across_accept_and_reject():
    system = _two_species_system()
    q = _q(40, 0.6, 10.0)
    kw = dict(q_grid=q, r_max=11.0, n_r=550)
    calc, reference = _calc(**kw), _calc(**kw)

    state = system
    calc.init(state)
    generator = torch.Generator().manual_seed(11)

    for step in range(8):
        moved = torch.randint(0, state.num_nodes, (2,), generator=generator)
        positions = state.positions.clone()
        positions[moved] += (
            torch.rand(2, 3, generator=generator, dtype=torch.float64) - 0.5
        ) * 1.5
        candidate = AtomicSystem(
            positions=positions,
            atomic_numbers=state.atomic_numbers,
            cell=state.cell,
            pbc=state.pbc,
        )
        got = calc.compute_increm_change(candidate, MoveDelta(moved_indices=moved))
        assert torch.allclose(got, reference.compute(candidate), atol=1e-10)

        if step % 3:
            calc.commit()
            state = candidate
        else:
            calc.discard()

    assert torch.allclose(calc._from_counts(), reference.compute(state), atol=1e-10)


def test_init_returns_what_compute_would():
    system = _random_system()
    calc, reference = _calc(), _calc()
    assert torch.allclose(calc.init(system), reference.compute(system), atol=1e-10)


def test_a_move_that_changes_atom_count_falls_back_to_full_compute():
    """Insert/remove shifts every concentration and prefactor at once, so
    the calculator must recompute rather than take a delta."""
    system = _random_system(n=200, side=L)
    calc = _calc()
    calc.init(system)

    smaller = AtomicSystem(
        positions=system.positions[:-1],
        atomic_numbers=system.atomic_numbers[:-1],
        cell=system.cell,
        pbc=system.pbc,
    )
    got = calc.compute_increm_change(
        smaller, MoveDelta(removed_indices=torch.tensor([199]))
    )
    assert torch.allclose(got, _calc().compute(smaller), atol=1e-10)


def test_windowed_paths_also_agree_and_damping_suppresses_low_q_ripple():
    """The window has to reach the unbinned path identically, and it has
    to do what it is for: quieten the truncation ripple."""
    system = _random_system()
    binned = _calc(window="lorch").compute(system)
    direct = _calc(window="lorch", forward_model="debye").compute(system)
    assert torch.allclose(binned, direct, atol=5e-3)

    low_q = slice(0, 10)
    undamped = _calc(window="none").compute(system)[low_q]
    damped = _calc(window="hann").compute(system)[low_q]
    assert (damped - 1.0).abs().max() < (undamped - 1.0).abs().max()


def test_weighted_total_equals_the_faber_ziman_combination_of_its_partials():
    """S(Q) - 1 must equal SUM_ab w_ab(Q) [A_ab(Q) - 1] built from the
    partials computed independently.

    This is the check that pins the weighting layer specifically, and it
    bites hardest for X-ray, whose w_ab(Q) vary with Q: a mis-ordered
    pair row or a dropped (2 - delta_ab) multiplicity would still leave
    the total looking plausible, but not equal to this sum.
    """
    from raymc.modalities.sq.scattering import (
        faber_ziman_weights,
        species_form_factors,
    )

    system = _two_species_system()
    q = _q(40, 0.6, 10.0)
    kw = dict(q_grid=q, r_max=11.0, n_r=550)

    n_o = int((system.atomic_numbers == 8).sum())
    n_h = int((system.atomic_numbers == 1).sum())
    total_atoms = n_o + n_h
    concentrations = torch.tensor(
        [n_o / total_atoms, n_h / total_atoms], dtype=torch.float64
    )

    for radiation in ("xray", "neutron"):
        partials = torch.stack(
            [
                _calc(**kw, radiation=None, atomic_number_a=a, atomic_number_b=b)
                .compute(system)
                - 1.0
                for a, b in ((8, 8), (8, 1), (1, 1))
            ]
        )
        weights = faber_ziman_weights(
            concentrations,
            species_form_factors([8, 1], q, radiation),
        )
        expected = 1.0 + (weights * partials).sum(dim=0)
        got = _calc(**kw, radiation=radiation).compute(system)
        assert torch.allclose(got, expected, atol=1e-10), radiation
