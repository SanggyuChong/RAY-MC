"""Build and export the q-TIP4P/F water model as a metatomic ``.pt``.

This script is NOT part of the ``raymc`` package and is never imported by
it. It exists so the exported model is reproducible and reviewable rather
than a binary that appeared from someone's scratch directory.

Why it lives outside the package: the model definition needs `torch-pme
<https://lab-cosmo.github.io/torch-pme>`_ for its P3M electrostatics, but
``AtomisticModel.save()`` TorchScript-compiles torch-pme's (pure-Python)
source into the serialized graph, and torch-pme ships no native
extensions. The exported ``.pt`` therefore runs with only
``metatomic-torch``/``vesin`` installed -- verified by loading and
evaluating it in a process with ``torchpme`` blocked from importing. So
RAY-MC gains no new dependency; only *building* the artifact needs one.

The model is a port of the atomistic-cookbook recipe
(https://atomistic-cookbook.org/examples/water-model/water-model.html) by
Loche, Langer and Ceriotti, which implements the flexible four-point
q-TIP4P/F potential of Habershon, Markland & Manolopoulos, JCP 131,
024501 (2009), https://doi.org/10.1063/1.3167790.

Units throughout are the model's own: Angstrom, kcal/mol, radians. The
exported model declares those in its ``ModelCapabilities``, and metatomic
converts to whatever a caller asks for -- RAY-MC asks for eV.

Usage::

    python scripts/build_qtip4pf.py --output data/models/qtip4pf-mta.pt

Caveat: a TorchScript archive is tied to the metatensor/metatomic ABI it
was built against. Rebuild it when those move (the ``--check`` self-test
at the end of this script is the fastest way to find out that you must).
"""

# NOTE: deliberately no `from __future__ import annotations` here --
# metatomic introspects forward()'s real annotation objects, and PEP 563
# string annotations make that check fail with a confusing error.

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchpme
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

# q-TIP4P/F parameters (Habershon et al. 2009, Table I), in the model's
# own units. oh_bond_k is the harmonic coefficient of the quartic-
# truncated Morse expansion, k_r = 2 D_r alpha_r^2.
QTIP4PF_PARAMETERS = dict(
    cutoff=7.0,  # Angstrom
    lj_sigma=3.1589,  # Angstrom
    lj_epsilon=0.1852,  # kcal/mol
    m_gamma=0.73612,  # M-site interpolation (1.0 -> three-point model)
    m_charge=1.1128,  # |q_M|, in e; q_H = +m_charge/2
    oh_bond_eq=0.9419,  # Angstrom
    oh_bond_k=2 * 116.09 * 2.287**2,  # kcal/mol/Angstrom^2
    oh_bond_alpha=2.287,  # 1/Angstrom
    hoh_angle_eq=107.4 * torch.pi / 180,  # radians
    hoh_angle_k=87.85,  # kcal/mol/rad^2
)

# P3M smearing / mesh, giving ~1e-4 relative accuracy on forces. Same
# shape as what torchpme.tuning.tune_p3m returns, so --tune can drop in.
DEFAULT_P3M_OPTIONS = (1.4, {"interpolation_nodes": 5, "mesh_spacing": 1.33}, 0)


def bond_energy(
    distances: torch.Tensor,
    coefficient_k: torch.Tensor,
    coefficient_alpha: torch.Tensor,
    coefficient_r0: torch.Tensor,
) -> torch.Tensor:
    """Quartic-truncated Morse potential for the O-H bond.

    Reproduces a Morse potential near the minimum but cannot dissociate,
    because the expansion is truncated at the quartic term.
    """
    dr = distances - coefficient_r0
    alpha_dr = dr * coefficient_alpha
    return 0.5 * coefficient_k * dr**2 * (1 - alpha_dr + alpha_dr**2 * 7 / 12)


def bend_energy(
    angles: torch.Tensor, coefficient: torch.Tensor, equilibrium_angle: torch.Tensor
) -> torch.Tensor:
    """Harmonic potential for the H-O-H angle (radians)."""
    return 0.5 * coefficient * (angles - equilibrium_angle) ** 2


def lennard_jones_pair(
    distances: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cutoff: torch.Tensor,
) -> torch.Tensor:
    """Lennard-Jones, shifted so it is exactly zero at *cutoff*."""
    c6 = (sigma / distances) ** 6
    c12 = c6**2
    lj = 4 * epsilon * (c12 - c6)

    sigma_cutoff_6 = (sigma / cutoff) ** 6
    offset = 4 * epsilon * sigma_cutoff_6 * (sigma_cutoff_6 - 1)
    return lj - offset


def get_molecular_geometry(
    system: System,
    nlo: NeighborListOptions,
    m_gamma: torch.Tensor,
    m_charge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, System, torch.Tensor, torch.Tensor]:
    """Bond lengths, H-O-H angles, and the M-site system for each molecule.

    Covalent O-H bonds are identified heuristically, as the shortest
    ``2 * n_oxygen`` O-H distances in the neighbor list, because an
    external engine may reorder atoms internally. The heuristic fails if
    molecules approach each other too closely or the structure is very
    hot -- both cases raise rather than silently mis-assigning.

    The M sites are virtual charge sites at
    ``M = O + (1 - gamma) * 0.5 * (OH_1 + OH_2)``. Forces on them are
    split back onto O and H consistently by autodifferentiation, since
    the displacement is built from the real positions.
    """
    neighbors = system.get_neighbor_list(nlo)
    species = system.types

    neigh_ij = torch.stack(
        (
            neighbors.samples.column("first_atom"),
            neighbors.samples.column("second_atom"),
        ),
        dim=1,
    )
    # reshape, not squeeze: a system with a single pair would otherwise
    # collapse to shape [3] instead of [1, 3].
    neigh_dij = neighbors.values.reshape(-1, 3)

    # all O-H distances, sorted shortest first
    oh_mask = species[neigh_ij[:, 0]] != species[neigh_ij[:, 1]]
    oh_ij = neigh_ij[oh_mask]
    oh_dij = neigh_dij[oh_mask]
    oh_dist = torch.linalg.vector_norm(oh_dij, dim=1).reshape(-1)
    oh_sort = torch.argsort(oh_dist)

    # oxygen index of each O-H pair, ordered by distance
    oh_oidx = torch.where(species[oh_ij[:, 0]] == 8, oh_ij[:, 0], oh_ij[:, 1])[oh_sort]
    # orient every bond vector O->H
    oh_dij = oh_dij * torch.where(species[oh_ij[:, 0]] == 8, 1.0, -1.0).reshape(-1, 1)

    # assume the 2*n_oxygen shortest O-H pairs are exactly the covalent bonds
    o_idx = torch.nonzero(species == 8).reshape(-1)
    n_oh = len(o_idx) * 2
    oh_oidx_sort = torch.argsort(oh_oidx[:n_oh])

    oh_dij_oidx = oh_oidx[oh_oidx_sort]
    # each O index must appear exactly twice, so consecutive pairs cancel
    twoo_chk = oh_dij_oidx[::2] - oh_dij_oidx[1::2]
    if (twoo_chk != 0).any():
        raise RuntimeError("Cannot assign OH bonds to water molecules univocally.")

    oh_dij_sort = oh_dij[oh_sort[:n_oh]][oh_oidx_sort]
    doh_1 = oh_dij_sort[::2]
    doh_2 = oh_dij_sort[1::2]

    oh_dist = torch.concatenate(
        [
            torch.linalg.vector_norm(doh_1, dim=1),
            torch.linalg.vector_norm(doh_2, dim=1),
        ]
    )

    if oh_dist.max() > 2.0:
        raise ValueError(
            "Unphysical O-H bond length. Check that the molecules are entire, and "
            "atoms are listed in the expected OHH order."
        )

    n_mol = len(doh_1)
    hoh_angles = torch.arccos(
        torch.sum(doh_1 * doh_2, dim=1) / (oh_dist[:n_mol] * oh_dist[n_mol:])
    )

    # scatter the O->H vectors back onto per-atom order, so all atoms can
    # be displaced to their M sites in one shot
    oh1_vecs = torch.zeros_like(system.positions)
    oh1_vecs[oh_dij_oidx[::2]] = doh_1
    oh2_vecs = torch.zeros_like(system.positions)
    oh2_vecs[oh_dij_oidx[1::2]] = doh_2

    om_displacements = (1 - m_gamma) * 0.5 * (oh1_vecs + oh2_vecs)

    m_system = System(
        types=system.types,
        positions=system.positions + om_displacements,
        cell=system.cell,
        pbc=system.pbc,
    )

    # Shift the neighbor list so it refers to M sites rather than O atoms.
    # This assumes no pair crosses the cutoff as a result, which is only
    # approximately true -- hence the deliberately generous 7 A cutoff and
    # strict=False.
    nl = system.get_neighbor_list(nlo)
    i_idx = nl.samples.column("first_atom")
    j_idx = nl.samples.column("second_atom")
    m_nl = TensorBlock(
        nl.values + (om_displacements[j_idx] - om_displacements[i_idx]).reshape(-1, 3, 1),
        nl.samples,
        nl.components,
        nl.properties,
    )
    m_system.add_neighbor_list(nlo, m_nl)

    charges = (
        torch.where(species == 8, -m_charge, 0.5 * m_charge)
        .reshape(-1, 1)
        .to(dtype=system.positions.dtype, device=system.positions.device)
    )
    samples = Labels(
        "atom", torch.arange(len(system), device=charges.device).reshape(-1, 1)
    )
    properties = Labels(
        "charge", torch.zeros(1, 1, device=charges.device, dtype=torch.int32)
    )
    data = TensorBlock(
        values=charges, samples=samples, components=[], properties=properties
    )
    tensor = TensorMap(
        keys=Labels("_", torch.zeros(1, 1, device=charges.device, dtype=torch.int32)),
        blocks=[data],
    )
    m_system.add_data(name="charge", tensor=tensor)

    # intramolecular M-H and H-H distances, for removing the self-energy
    hh_dist = torch.sqrt(((doh_1 - doh_2) ** 2).sum(dim=1))
    dmh_1 = doh_1 - om_displacements[oh_dij_oidx[::2]]
    dmh_2 = doh_2 - om_displacements[oh_dij_oidx[1::2]]
    mh_dist = torch.concatenate(
        [
            torch.linalg.vector_norm(dmh_1, dim=1),
            torch.linalg.vector_norm(dmh_2, dim=1),
        ]
    )

    return oh_dist, hoh_angles, m_system, mh_dist, hh_dist


class WaterModel(torch.nn.Module):
    """Flexible four-point water model in the metatomic interface.

    Not hard-coded to q-TIP4P/F: classical TIP4P works too, and setting
    ``m_gamma=1.0`` collapses the M site onto the oxygen, giving a
    three-point model such as SPC/Fw.
    """

    def __init__(
        self,
        cutoff: float,
        lj_sigma: float,
        lj_epsilon: float,
        m_gamma: float,
        m_charge: float,
        oh_bond_eq: float,
        oh_bond_k: float,
        oh_bond_alpha: float,
        hoh_angle_eq: float,
        hoh_angle_k: float,
        p3m_options: Optional[tuple] = None,
    ):
        super().__init__()

        if p3m_options is None:
            p3m_options = DEFAULT_P3M_OPTIONS
        p3m_smearing, p3m_parameters, _ = p3m_options

        self.p3m_calculator = torchpme.metatensor.P3MCalculator(
            potential=torchpme.CoulombPotential(
                p3m_smearing, prefactor=torchpme.prefactors.kcalmol_A
            ),
            **p3m_parameters,
        )
        # bare 1/r, used only to subtract the intramolecular self-energy
        self.coulomb = torchpme.CoulombPotential()

        # Half list, and strict=False so pairs beyond the cutoff are
        # allowed: harmless for PME, cheaper to build, and it gives the
        # M-site shift above some slack.
        self.nlo = NeighborListOptions(cutoff=cutoff, full_list=False, strict=False)

        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.register_buffer("lj_sigma", torch.tensor(lj_sigma))
        self.register_buffer("lj_epsilon", torch.tensor(lj_epsilon))
        self.register_buffer("m_gamma", torch.tensor(m_gamma))
        self.register_buffer("m_charge", torch.tensor(m_charge))
        self.register_buffer("oh_bond_eq", torch.tensor(oh_bond_eq))
        self.register_buffer("oh_bond_k", torch.tensor(oh_bond_k))
        self.register_buffer("oh_bond_alpha", torch.tensor(oh_bond_alpha))
        self.register_buffer("hoh_angle_eq", torch.tensor(hoh_angle_eq))
        self.register_buffer("hoh_angle_k", torch.tensor(hoh_angle_k))

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.nlo]

    def _setup_systems(
        self,
        systems: List[System],
        selected_atoms: Optional[Labels] = None,
    ) -> Tuple[System, TensorBlock]:
        if len(systems) > 1:
            raise ValueError(f"only one system supported, got {len(systems)}")

        system_i = 0
        system = systems[system_i]

        # Discard "ghost" atoms, which engines doing domain decomposition
        # (LAMMPS) include in the neighbor list they hand over.
        if selected_atoms is not None:
            current_system_mask = selected_atoms.column("system") == system_i
            current_atoms = selected_atoms.column("atom")
            current_atoms = current_atoms[current_system_mask].to(torch.long)

            types = system.types[current_atoms]
            positions = system.positions[current_atoms]
            system_clean = System(types, positions, system.cell, system.pbc)
            for nlo in system.known_neighbor_lists():
                system_clean.add_neighbor_list(nlo, system.get_neighbor_list(nlo))
        else:
            system_clean = system
        return system_clean, system.get_neighbor_list(self.nlo)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if list(outputs.keys()) != ["energy"]:
            raise ValueError(
                f"`outputs` keys ({', '.join(outputs.keys())}) contain unsupported "
                "keys. Only 'energy' is supported."
            )
        if outputs["energy"].sample_kind == "atom":
            raise NotImplementedError("per-atom energies are not supported.")

        system, neighbors = self._setup_systems(systems, selected_atoms)

        neighbor_indices = torch.stack(
            (
                neighbors.samples.column("first_atom"),
                neighbors.samples.column("second_atom"),
            ),
            dim=1,
        )
        species = system.types
        oo_mask = (species[neighbor_indices[:, 0]] == 8) & (
            species[neighbor_indices[:, 1]] == 8
        )
        oo_distances = torch.linalg.vector_norm(neighbors.values[oo_mask], dim=1)
        energy_lj = lennard_jones_pair(
            oo_distances, self.lj_sigma, self.lj_epsilon, self.cutoff
        ).sum()

        d_oh, a_hoh, m_system, mh_dist, hh_dist = get_molecular_geometry(
            system, self.nlo, self.m_gamma, self.m_charge
        )

        e_bond = bond_energy(
            d_oh, self.oh_bond_k, self.oh_bond_alpha, self.oh_bond_eq
        ).sum()
        e_bend = bend_energy(a_hoh, self.hoh_angle_k, self.hoh_angle_eq).sum()

        # long-range electrostatics, over the M-site system
        m_neighbors = m_system.get_neighbor_list(self.nlo)
        potentials = self.p3m_calculator(m_system, m_neighbors).block(0).values
        charges = m_system.get_data("charge").block().values
        energy_coulomb = (potentials * charges).sum()

        # Coulomb interactions within a molecule are already accounted for
        # by the bond and angle terms, so remove them.
        energy_self = (
            (
                self.coulomb.from_dist(mh_dist).sum() * (-self.m_charge)
                + self.coulomb.from_dist(hh_dist).sum() * 0.5 * self.m_charge
            )
            * 0.5
            * self.m_charge
            * torchpme.prefactors.kcalmol_A
        )

        energy_tot = e_bond + e_bend + energy_lj + energy_coulomb - energy_self

        samples = Labels(
            ["system"],
            torch.arange(len(systems), device=energy_tot.device).reshape(-1, 1),
        )
        properties = Labels(["energy"], torch.tensor([[0]], device=energy_tot.device))
        block = TensorBlock(
            values=torch.sum(energy_tot).reshape(-1, 1),
            samples=samples,
            components=torch.jit.annotate(List[Labels], []),
            properties=properties,
        )
        return {
            "energy": TensorMap(
                Labels("_", torch.tensor([[0]], device=energy_tot.device)), [block]
            ),
        }


def build_model(p3m_options: Optional[tuple] = None) -> AtomisticModel:
    """Wrap a q-TIP4P/F :class:`WaterModel` as a metatomic ``AtomisticModel``.

    ``interaction_range`` is infinite: the long-range electrostatic term
    means forces cannot be assumed to decay beyond the cutoff.
    """
    model = WaterModel(**QTIP4PF_PARAMETERS, p3m_options=p3m_options)
    outputs = {"energy": ModelOutput(unit="kcal/mol", sample_kind="system")}
    capabilities = ModelCapabilities(
        outputs=outputs,
        atomic_types=[1, 8],
        interaction_range=torch.inf,
        length_unit="angstrom",
        supported_devices=["cuda", "cpu"],
        dtype="float32",
    )
    metadata = ModelMetadata(
        name="q-TIP4P/F",
        description=(
            "Flexible four-point water model of Habershon, Markland & "
            "Manolopoulos, JCP 131, 024501 (2009), ported from the "
            "atomistic-cookbook metatomic recipe."
        ),
        references={
            "model": ["https://doi.org/10.1063/1.3167790"],
            "implementation": [
                "https://atomistic-cookbook.org/examples/water-model/water-model.html"
            ],
        },
    )
    return AtomisticModel(model.eval(), metadata, capabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/qtip4pf-mta.pt"),
        help="Where to write the exported model.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    # collect_extensions bundles any native Torch extensions next to the
    # model for non-Python engines. torch-pme contributes none, which is
    # exactly why RAY-MC can load this without torch-pme installed.
    model.save(str(args.output), collect_extensions=str(args.output.parent / "extensions"))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
