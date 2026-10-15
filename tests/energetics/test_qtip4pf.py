"""Validation of the exported q-TIP4P/F metatomic model.

The model is a build artifact, not source: ``scripts/build_qtip4pf.py``
produces it and these tests check that what came out is actually the
q-TIP4P/F potential rather than something merely plausible. Skipped when
the artifact isn't present, since building it needs ``torch-pme``, which
RAY-MC deliberately does not depend on.

Strategy: every term is pinned against an *independent* calculation, not
against a previously recorded output of this same model.

- Bond and angle are compared to the closed-form expressions evaluated
  here, on isolated molecules where nothing else contributes.
- The intermolecular part (M-site electrostatics + Lennard-Jones) is
  compared to a direct minimum-image Coulomb sum over hand-constructed
  M sites. This is the check that matters most: the M-site construction
  and the removal of the intramolecular self-energy are the subtlest
  part of the model, and a sign or factor error there would still give
  smooth, plausible-looking energies.
- Forces are compared to finite differences of the model's own energy,
  which catches an inconsistency between energy and its gradient (the
  M-site displacement has to differentiate back onto the real atoms).
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("metatomic.torch")

from raymc.system import AtomicSystem  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "qtip4pf-mta.pt"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=(
        f"{MODEL_PATH} not built -- run `python scripts/build_qtip4pf.py` "
        f"(needs the torch-pme build-time dependency)"
    ),
)

# q-TIP4P/F parameters, repeated here on purpose: a test that imported
# them from the builder would pass even if the builder's numbers were
# wrong. These are Habershon, Markland & Manolopoulos, JCP 131, 024501
# (2009), Table I.
R0 = 0.9419  # Angstrom
KR = 2 * 116.09 * 2.287**2  # kcal/mol/Angstrom^2
ALPHA = 2.287  # 1/Angstrom
THETA0 = 107.4 * math.pi / 180  # radians
KTHETA = 87.85  # kcal/mol/rad^2
SIGMA = 3.1589  # Angstrom
EPSILON = 0.1852  # kcal/mol
CUTOFF = 7.0  # Angstrom
M_GAMMA = 0.73612
M_CHARGE = 1.1128  # e

COULOMB_KCAL_A = 332.0637129  # kcal*Angstrom/mol/e^2 (torchpme kcalmol_A)
EV_PER_KCAL_MOL = 1.0 / 23.060548

# Box edge for "isolated" tests. Big enough that a molecule's interaction
# with its own periodic images is ~5e-3 kcal/mol (measured, and shown to
# decay with box size), which sets the tolerances below.
BOX = 30.0
IMAGE_TOLERANCE_KCAL = 0.02


@pytest.fixture(scope="module")
def energetics():
    from raymc.energetics import MetatomicEnergetics, load_model_file

    return MetatomicEnergetics(model=load_model_file(MODEL_PATH))


def _system(positions, atomic_numbers, box=BOX):
    return AtomicSystem(
        positions=torch.tensor(positions, dtype=torch.float64),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        cell=(torch.eye(3, dtype=torch.float64) * box).unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )


def _kcal(energetics, positions, atomic_numbers, box=BOX):
    """Model energy for a configuration, converted back to the model's
    own kcal/mol so it can be compared with the formulas above (the
    adapter asks metatomic for eV)."""
    value = energetics.compute(_system(positions, atomic_numbers, box))
    return float(value) / EV_PER_KCAL_MOL


def _water(origin, r1=R0, r2=R0, theta=THETA0):
    """O at *origin*, both H in the xy plane, bisector along +x."""
    ox, oy, oz = origin
    half = theta / 2
    return [
        [ox, oy, oz],
        [ox + r1 * math.cos(half), oy + r1 * math.sin(half), oz],
        [ox + r2 * math.cos(half), oy - r2 * math.sin(half), oz],
    ]


def _bond(r):
    dr = r - R0
    a = dr * ALPHA
    return 0.5 * KR * dr**2 * (1 - a + 7 / 12 * a**2)


def _bend(theta):
    return 0.5 * KTHETA * (theta - THETA0) ** 2


def _lj(r):
    c6 = (SIGMA / r) ** 6
    offset6 = (SIGMA / CUTOFF) ** 6
    return 4 * EPSILON * (c6**2 - c6) - 4 * EPSILON * offset6 * (offset6 - 1)


def _charge_sites(molecule):
    """[(position, charge)] for the M site and the two H of one molecule."""
    o, h1, h2 = (torch.tensor(p, dtype=torch.float64) for p in molecule)
    m = o + (1 - M_GAMMA) * 0.5 * ((h1 - o) + (h2 - o))
    return [(m, -M_CHARGE), (h1, 0.5 * M_CHARGE), (h2, 0.5 * M_CHARGE)]


def _direct_coulomb(mol_a, mol_b):
    """Direct Coulomb between two molecules' charge sites, kcal/mol."""
    return sum(
        float(qa * qb * COULOMB_KCAL_A / torch.linalg.vector_norm(pa - pb))
        for pa, qa in _charge_sites(mol_a)
        for pb, qb in _charge_sites(mol_b)
    )


def test_equilibrium_monomer_has_zero_internal_energy(energetics):
    """At the equilibrium geometry every intramolecular term vanishes by
    construction, so a lone molecule's energy is only its (small)
    interaction with its own periodic images."""
    assert _kcal(energetics, _water((0, 0, 0)), [8, 1, 1]) == pytest.approx(
        0.0, abs=IMAGE_TOLERANCE_KCAL
    )


@pytest.mark.parametrize(
    "r1,r2,theta_deg",
    [
        (1.05, R0, 107.4),
        (0.85, R0, 107.4),
        (1.05, 0.90, 107.4),
        (R0, R0, 100.0),
        (R0, R0, 115.0),
        (1.02, 0.92, 104.0),
    ],
)
def test_bond_and_angle_terms_match_closed_form(energetics, r1, r2, theta_deg):
    theta = theta_deg * math.pi / 180
    got = _kcal(energetics, _water((0, 0, 0), r1, r2, theta), [8, 1, 1])
    expected = _bond(r1) + _bond(r2) + _bend(theta)
    assert got == pytest.approx(expected, abs=IMAGE_TOLERANCE_KCAL)


@pytest.mark.parametrize("separation", [3.0, 4.0, 5.0, 6.0])
def test_intermolecular_matches_lj_plus_direct_coulomb(energetics, separation):
    """The M-site electrostatics must agree with a direct Coulomb sum over
    hand-built M sites. Both molecules sit at their equilibrium internal
    geometry, so bond and angle contribute exactly nothing and the whole
    energy is LJ + intermolecular Coulomb."""
    mol_a = _water((0, 0, 0))
    # second molecule rotated 180 degrees about z, so it faces the first
    mol_b = [[2 * separation - x, -y, z] for x, y, z in _water((separation, 0, 0))]

    got = _kcal(energetics, mol_a + mol_b, [8, 1, 1, 8, 1, 1])
    expected = _lj(separation) + _direct_coulomb(mol_a, mol_b)
    assert got == pytest.approx(expected, abs=IMAGE_TOLERANCE_KCAL)


def test_periodic_image_residual_shrinks_with_box_size(energetics):
    """Pins down that the small disagreement above really is the periodic
    image interaction, and not a constant error in the potential: it must
    decay as the box grows."""
    mol_a = _water((0, 0, 0))
    mol_b = [[8.0 - x, -y, z] for x, y, z in _water((4.0, 0, 0))]
    reference = _lj(4.0) + _direct_coulomb(mol_a, mol_b)

    residuals = [
        abs(_kcal(energetics, mol_a + mol_b, [8, 1, 1, 8, 1, 1], box=box) - reference)
        for box in (20.0, 30.0, 60.0)
    ]
    assert residuals[0] > residuals[1] > residuals[2]
    assert residuals[-1] < 2e-3


def test_forces_match_finite_differences(energetics):
    """Catches an energy/gradient inconsistency -- in particular the
    M-site displacement failing to differentiate back onto O and H."""
    mol_a = _water((0, 0, 0))
    mol_b = [[6.0 - x, -y, z] for x, y, z in _water((3.0, 0, 0))]
    positions = mol_a + mol_b
    numbers = [8, 1, 1, 8, 1, 1]

    _, forces = energetics.energy_forces(_system(positions, numbers))
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3), atol=1e-3)

    delta = 1e-3
    for atom, component in [(0, 0), (1, 1), (3, 0), (4, 2), (5, 1)]:
        plus = [row[:] for row in positions]
        plus[atom][component] += delta
        minus = [row[:] for row in positions]
        minus[atom][component] -= delta
        e_plus = _kcal(energetics, plus, numbers) * EV_PER_KCAL_MOL
        e_minus = _kcal(energetics, minus, numbers) * EV_PER_KCAL_MOL
        numeric = -(e_plus - e_minus) / (2 * delta)
        assert float(forces[atom, component]) == pytest.approx(numeric, abs=0.01)


def test_energy_is_invariant_under_translation_rotation_and_reordering(energetics):
    """Atom reordering matters here specifically because the model
    identifies O-H bonds by a shortest-distance heuristic rather than by
    atom order -- RMC must be able to hand it atoms in any order."""
    mol_a = _water((0, 0, 0))
    mol_b = [[6.0 - x, -y, z] for x, y, z in _water((3.0, 0, 0))]
    positions = mol_a + mol_b
    numbers = [8, 1, 1, 8, 1, 1]
    reference = _kcal(energetics, positions, numbers)

    shifted = [[x + 7.3, y - 2.1, z + 0.9] for x, y, z in positions]
    assert _kcal(energetics, shifted, numbers) == pytest.approx(reference, abs=5e-3)

    cos, sin = math.cos(0.7), math.sin(0.7)
    rotated = [[x * cos - y * sin, x * sin + y * cos, z] for x, y, z in positions]
    assert _kcal(energetics, rotated, numbers) == pytest.approx(reference, abs=5e-3)

    interleaved = [positions[0], positions[3], positions[1], positions[2],
                   positions[4], positions[5]]
    assert _kcal(energetics, interleaved, [8, 8, 1, 1, 1, 1]) == pytest.approx(
        reference, abs=5e-3
    )


def test_hydrogen_bonded_dimer_is_physical(energetics):
    """End-to-end physics check: a TIP4P-family model must bind the water
    dimer by roughly 5-7 kcal/mol at an O-O distance near 2.8 A."""
    numbers = [8, 1, 1, 8, 1, 1]
    half = THETA0 / 2

    def dimer(separation):
        donor = [
            [0.0, 0.0, 0.0],
            [R0, 0.0, 0.0],  # the donated H, pointing at the acceptor
            [R0 * math.cos(THETA0), R0 * math.sin(THETA0), 0.0],
        ]
        acceptor = [
            [separation, 0.0, 0.0],
            [separation + R0 * math.cos(half), R0 * math.sin(half), 0.0],
            [separation + R0 * math.cos(half), -R0 * math.sin(half), 0.0],
        ]
        return donor + acceptor

    curve = {d: _kcal(energetics, dimer(d), numbers) for d in
             (2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2, 3.6, 5.0)}
    best_d = min(curve, key=curve.get)

    assert 2.6 <= best_d <= 3.0, f"dimer minimum at {best_d} A: {curve}"
    assert -7.0 < curve[best_d] < -5.0, f"binding energy {curve[best_d]} kcal/mol"
    # and it must be a genuine minimum, not a monotone slide
    assert curve[2.5] > curve[best_d] < curve[5.0]


def test_model_runs_without_torch_pme_installed():
    """The reason RAY-MC can ship this model without depending on
    torch-pme: AtomisticModel.save TorchScript-compiles torch-pme's
    pure-Python source into the serialized graph, and torch-pme has no
    native extensions to bundle. If this ever breaks, torch-pme silently
    becomes a runtime dependency of RAY-MC."""
    script = f"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torchpme":
            raise ImportError("torchpme blocked for this test")
sys.meta_path.insert(0, Block())

import torch
from raymc.energetics import MetatomicEnergetics, load_model_file
from raymc.system import AtomicSystem

calc = MetatomicEnergetics(model=load_model_file({str(MODEL_PATH)!r}))
system = AtomicSystem(
    positions=torch.tensor({_water((0, 0, 0))!r}, dtype=torch.float64),
    atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
    cell=(torch.eye(3, dtype=torch.float64) * {BOX}).unsqueeze(0),
    pbc=torch.ones(1, 3, dtype=torch.bool),
)
energy = float(calc.compute(system))
assert "torchpme" not in sys.modules, "torchpme got imported after all"
assert abs(energy) < 1e-3, energy
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
