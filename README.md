# RAY-MC

RAY-MC is an open-source package for ML-driven reverse Monte Carlo (RMC)
and hybrid RMC simulations for atomic-scale structural resolution from
experimental data, developed and maintained by Sanggyu "Raymond" Chong.
Core of the code was developed as part of his Eric and Wendy Schmidt AI in
Science Fellowship at Oxford in the summer of 2026 hosted by Prof. Andrew
Goodwin.

RAY-MC is kept as modular as possible so that various modality
combinations, often uniquely enabled by ML forward models (hence
ML-driven!), can be readily supported in the RMC simulations. Currently
only integrates [`metatomic`](https://metatensor.org) models for HRMC and
other ML-dependent modalities.

## Installation

Requires Python 3.11–3.13.

```
pip install .
```

| Dependency | Needed for |
|---|---|
| `torch` | Tensors and autograd throughout |
| `nvalchemi-toolkit` | Atomic data structures and neighbour lists |
| `ase` | Reading and writing structures |
| `numpy`, `periodictable`, `pyyaml` | Numerics, element data, run configs |

Optional extras, installed with `pip install ".[extra1,extra2]"`:

| Extra | Adds | Needed for |
|---|---|---|
| `plot` | `matplotlib` | The fit plot written at the end of a run |
| `metatomic` | `metatomic-torch`, `vesin` | Hybrid RMC with any metatomic model |
| `upet` | `upet` | Hybrid RMC with PET-MAD and other UPET potentials |
| `dev` | `pytest`, `ruff` | Running the tests and linting |

For example, everything needed to run both examples below:

```
pip install ".[plot,metatomic,upet]"
```

## Running

```
raymc CONFIG.yaml --out-dir OUT
```

Two ready-to-run inputs are in `examples/`, with their data in
`examples/data/`. Run them from the repository root:

```
raymc examples/ice_sq_combined_flexible.yaml
```

| Input | System | Method |
|---|---|---|
| `ice_sq_combined_flexible.yaml` | Low-density amorphous ice, 500 flexible water molecules | RMC fitted to the X-ray and neutron S(Q) simultaneously |
| `asi_hrmc_from_melt.yaml` | Amorphous silicon, 512 atoms, from a hard-sphere melt | Hybrid RMC with PET-MAD, fitted to the X-ray S(Q) |

The amorphous ice data are the D2O total structure factors of Bizid et al.,
J. Chem. Phys. 87, 2225 (1987),
[doi:10.1063/1.453149](https://doi.org/10.1063/1.453149) (X-ray), and
Bellissent-Funel et al., J. Chem. Phys. 87, 2231 (1987),
[doi:10.1063/1.453150](https://doi.org/10.1063/1.453150) (neutron), as
fitted by RMC in Pusztai, Phys. Rev. B 61, 28 (2000). The amorphous silicon
data are the X-ray S(Q) of Laaziri et al., Phys. Rev. Lett. 82, 3460
(1999), [doi:10.1103/PhysRevLett.82.3460](https://doi.org/10.1103/PhysRevLett.82.3460).

Each run writes `final_structure.xyz`, `production_trajectory.xyz` and
`results.pt` (chi-squared history) to `<run-name>_output/`, or to
`--out-dir`. The hybrid RMC example downloads the PET-MAD model from
Hugging Face on first use, so it needs the `metatomic` and `upet` extras
and network access. `raymc.config.RunConfig` documents every configuration
field.

## Planned features

These features are currently under development:

- **Electronic structure constraints:** band gap/DOS@Fermi, predicted
  on-the-fly ML DOS model, as constraints in the fit.
- **Network components:** compatibility for materials with well-defined
  building blocks (e.g. Prussian blue analogues), with their automatic
  detection and tailored moves.
- **Single-crystal diffuse scattering:** self-explanatory, ambitiously under
  development.

## Licence

MIT -- see `LICENSE`.
