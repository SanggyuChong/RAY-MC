from __future__ import annotations

import pytest
import torch

from raymc.modalities.pdf.scattering_weights import atomic_scattering_weights


def test_xray_weight_is_atomic_number():
    z = torch.tensor([8, 1, 6])
    weights = atomic_scattering_weights(z, "xray")
    assert weights.tolist() == [8.0, 1.0, 6.0]


def test_neutron_weight_hydrogen_is_negative():
    z = torch.tensor([8, 1])
    weights = atomic_scattering_weights(z, "neutron")
    assert weights[0].item() > 0  # oxygen: positive coherent scattering length
    assert weights[1].item() < 0  # hydrogen: negative coherent scattering length


def test_unknown_radiation_raises():
    with pytest.raises(ValueError, match="Unknown radiation"):
        atomic_scattering_weights(torch.tensor([8]), "electron")
