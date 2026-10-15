import periodictable as pt
import pytest
import torch

from raymc.modalities.sq.scattering import (
    faber_ziman_weights,
    species_form_factors,
    unordered_species_pairs,
)

Q = torch.linspace(0.5, 12.0, 25, dtype=torch.float64)


def test_neutron_factors_are_constant_in_q():
    f = species_form_factors([8, 1], Q, "neutron")
    assert f.shape == (2, Q.numel())
    assert torch.allclose(f[0], torch.full_like(Q, pt.elements[8].neutron.b_c))
    assert torch.allclose(f[1], torch.full_like(Q, pt.elements[1].neutron.b_c))


def test_xray_form_factor_is_q_dependent_and_equals_z_at_zero():
    f = species_form_factors([8], torch.tensor([1e-9], dtype=torch.float64), "xray")
    assert f[0, 0].item() == pytest.approx(8.0, abs=1e-3)
    decaying = species_form_factors([8], Q, "xray")[0]
    assert torch.all(decaying[1:] < decaying[:-1])


def test_override_replaces_the_tabulated_value_at_every_q():
    f = species_form_factors([1], Q, "neutron", overrides={"H": 6.671})
    assert torch.allclose(f[0], torch.full_like(Q, 6.671))


def test_override_for_an_absent_element_raises_rather_than_being_ignored():
    with pytest.raises(ValueError, match="not present in this system"):
        species_form_factors([8], Q, "neutron", overrides={"Na": 3.63})


def test_xray_beyond_the_tabulated_range_raises():
    with pytest.raises(ValueError, match="past the"):
        species_form_factors([8], torch.tensor([80.0], dtype=torch.float64), "xray")


def test_unknown_radiation_raises():
    with pytest.raises(ValueError, match="Unknown radiation"):
        species_form_factors([8], Q, "electron")


def test_pair_ordering_follows_rmca_section_9_7():
    assert unordered_species_pairs(3) == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_faber_ziman_weights_sum_to_one_at_every_q():
    concentrations = torch.tensor([1 / 3, 2 / 3], dtype=torch.float64)
    f = species_form_factors([8, 1], Q, "xray")
    w = faber_ziman_weights(concentrations, f)
    assert w.shape == (3, Q.numel())
    assert torch.allclose(w.sum(dim=0), torch.ones_like(Q))


def test_single_species_weight_is_identically_one():
    f = species_form_factors([18], Q, "neutron")
    w = faber_ziman_weights(torch.tensor([1.0], dtype=torch.float64), f)
    assert torch.allclose(w, torch.ones(1, Q.numel(), dtype=torch.float64))
