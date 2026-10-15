"""Generate the initial system for an RMC run based on inputs."""

from __future__ import annotations

import warnings

import torch

from raymc.config import InitConfig
from raymc.initialize.load import load_structure
from raymc.initialize.pack import (
    _lattice_sites_per_axis,
    bragg_peak_max_sk,
    equilibrate_hard_spheres,
    pack_on_lattice,
)
from raymc.initialize.templates import load_group_template
from raymc.system import AtomicSystem, check_no_atomic_overlap
from raymc.utils import MinDistTable

_PACK_STRATEGIES = {
    "pack_on_lattice": pack_on_lattice,
}


def build_initial_system(
    config: InitConfig,
    min_dist: MinDistTable | None = None,
    log_every: int | None = None,
) -> AtomicSystem:
    """Construct the starting :class:`~raymc.system.AtomicSystem` for RMC.

    Dispatches on :attr:`InitConfig.strategy`:

    - ``"pack_on_lattice"``: :func:`raymc.initialize.pack.pack_on_lattice`.
    - ``"load_structure"``: :func:`raymc.initialize.load.load_structure`
      (doesn't take *min_dist* -- see that function's docstring for why).

    Parameters
    ----------
    config : InitConfig
        The run's ``init`` section, e.g. from :meth:`raymc.config.RunConfig.from_yaml`.
    min_dist : MinDistTable | None
        The run's minimum-distance table (built from
        :class:`raymc.config.ConstraintConfig`), threaded through to a
        packing strategy's overlap validation. Defaults to a bare
        :class:`~raymc.utils.MinDistTable` when not supplied. Unused for
        ``"load_structure"``.
    log_every : int | None
        If set, print progress every this many steps during the optional
        ``equilibrate`` stage -- see
        :func:`raymc.initialize.pack.equilibrate_hard_spheres`.

    Returns
    -------
    AtomicSystem
    """
    if config.strategy in _PACK_STRATEGIES:
        template = load_group_template(config.molecule)
        cell = torch.diag(torch.tensor(config.cell, dtype=torch.float64))
        pbc = torch.tensor(config.pbc)
        pack_fn = _PACK_STRATEGIES[config.strategy]
        if min_dist is None:
            min_dist = MinDistTable()

        # A separate hard-sphere exclusion used ONLY to produce a
        # structurally realistic starting configuration -- see
        # EquilibrateConfig's docstring for the physical reasoning. Falls
        # back to the RMC table when the config asks for nothing of its
        # own (EquilibrateConfig.is_explicit).
        equilibrate_min_dist = min_dist
        if config.equilibrate is not None:
            equilibrate_min_dist = config.equilibrate.min_dist_table() or min_dist

        # Hold the packing to whatever the NEXT stage will enforce: the
        # equilibration table when a melt follows, the RMC table when the
        # lattice goes straight to the engine. Checking a raw lattice
        # against a constraint three stages downstream is the strictest
        # and least meaningful point available -- see pack_on_lattice's
        # min_dist docstring.
        system = pack_fn(
            template,
            cell=cell,
            n_particles=config.n_particles,
            pbc=pbc,
            min_dist=equilibrate_min_dist,
            seed=config.seed,
        )
        if config.equilibrate is not None:
            # config.n_particles (what pack_on_lattice was actually asked
            # for) is the count bragg_peak_max_sk needs to score -- its own
            # round(N ** (1/3)) fallback guess only happens to be right
            # when n_particles is a perfect cube.
            n_per_axis = _lattice_sites_per_axis(cell, config.n_particles)
            sk_before = bragg_peak_max_sk(system, n_per_axis=n_per_axis)
            print(f"[raymc] max Bragg S(k) before equilibration: {sk_before:.1f}")
            system = equilibrate_hard_spheres(
                system,
                min_dist=equilibrate_min_dist,
                n_steps=config.equilibrate.n_steps,
                max_displacement=config.equilibrate.max_displacement,
                seed=config.seed,
                log_every=log_every,
            )
            sk_after = bragg_peak_max_sk(system, n_per_axis=n_per_axis)
            print(f"[raymc] max Bragg S(k) after equilibration: {sk_after:.1f}")

            # Report, but do not refuse: equilibrating at a looser
            # exclusion than RMC enforces is a legitimate choice (see
            # EquilibrateConfig.min_dist), and the smooth distance penalty
            # prices a shallow starting violation as a large finite number
            # the walker can relax out of. Worth saying out loud, though --
            # a structure starting deep inside its own constraint spends
            # its first thousands of moves climbing out, and if it starts
            # deeper than ~26.6 * softening_window the penalty overflows to
            # +inf and the engine stops at step 0.
            try:
                check_no_atomic_overlap(system, min_dist)
            except ValueError as exc:
                warnings.warn(
                    f"the equilibrated structure starts inside the run's own "
                    f"min-dist constraint: {exc} RMC will begin with a "
                    f"non-zero min_dist penalty and relax out of it.",
                    stacklevel=2,
                )
        return system
    if config.strategy == "load_structure":
        return load_structure(config.structure_file)
    raise ValueError(f"Unknown initialization strategy: {config.strategy!r}")
