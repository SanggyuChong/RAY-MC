"""Packing of atoms/molecules into a periodic simulation box.

Single entry point for RAY-MC's packing strategy:
- :func:`pack_on_lattice`: places one atom/molecule per site on a
regular grid. To achive target count exactly, excess may be randomly culled.
- :func:`equilibrate_hard_spheres`: a min_dist only MC is performed to
  bring system to a randomized configuration at the same density.

TODO: multi-species packing.
"""

from __future__ import annotations

import math
import warnings

import torch

from raymc.initialize.templates import GroupTemplate
from raymc.system import (
    AtomicSystem,
    check_no_atomic_overlap,
    rigid_groups,
    set_rigid_groups,
)
from raymc.utils import MinDistTable, random_rotation_matrices


def _lattice_sites_per_axis(cell: torch.Tensor, n_particles: int) -> list[int]:
    """Sites per axis of the simple-cubic-spacing lattice
    :func:`pack_on_lattice` builds for *n_particles* particles in *cell*.

    Shared by :func:`pack_on_lattice` (to actually build the grid) and
    :func:`bragg_peak_max_sk`'s callers (to score it against the *real*
    lattice spacing) -- duplicating this arithmetic let the two drift
    apart whenever ``n_particles`` isn't a perfect cube (e.g. 526), since
    :func:`bragg_peak_max_sk` was guessing ``round(n_particles ** (1/3))``
    instead.
    """
    box_lengths = torch.diagonal(cell.to(torch.float64))
    volume = box_lengths.prod()
    spacing = (float(volume) / n_particles) ** (1 / 3)
    return [
        max(1, math.ceil(round(float(box_lengths[d]) / spacing, 6))) for d in range(3)
    ]


def pack_on_lattice(
    template: GroupTemplate,
    cell: torch.Tensor,
    n_particles: int,
    pbc: torch.Tensor | None = None,
    min_dist: MinDistTable | None = None,
    seed: int | None = None,
) -> AtomicSystem:
    """Place *n_particles* copies of *template* on a simple cubic-spacing
    lattice of sites, one per site.

    Only supports a diagonal (orthorhombic) *cell*, which need not be
    cubic. A single target spacing is derived from the average
    per-molecule volume (``volume / n_particles``), then each axis
    independently gets enough sites (``ceil(axis_length / spacing)``) to
    hold that spacing -- so a non-cubic box (e.g. a 2x2x1 supercell) gets
    uniform spacing on every axis rather than the short axis being
    squeezed by reusing a site count sized for the long axes. This
    oversized grid is then randomly culled down to *n_particles* -- the
    rest of the sites are simply not placed.

    Parameters
    ----------
    template : GroupTemplate
        Rigid group to place (molecule or lone atom), from
        :func:`raymc.initialize.templates.load_molecule` or
        :meth:`~raymc.initialize.templates.GroupTemplate.atom`.
    cell : torch.Tensor
        [3, 3] diagonal cell vectors (Angstrom).
    n_particles : int
        Number of molecules/atoms to place. The resulting number density
        is ``n_particles / volume`` -- compute *n_particles* from
        whatever target density you want beforehand (``round(density *
        volume)``).
    pbc : torch.Tensor | None
        [3] bool periodicity per direction. Defaults to fully periodic.
    min_dist : MinDistTable | None
        Per-element-pair minimum allowed distance, used only for the
        final :func:`~raymc.system.check_no_atomic_overlap` check, which
        **warns rather than raises** (site spacing itself comes from
        *n_particles*, not *min_dist*). Defaults to a bare
        :class:`~raymc.utils.MinDistTable`.

        Pass the exclusion the NEXT stage will actually enforce -- the
        equilibration table when there is a melt to follow, the RMC one
        when the lattice is handed straight to the engine. That is what
        :func:`raymc.initialize.build.build_initial_system` does.

        A raw lattice is a deliberately artificial arrangement that a melt
        exists to destroy, and its worst-case contact is set by geometry
        rather than by anything physical: water's outermost atom sits
        0.783 A from its centroid, so two adjacent sites at 3.15 A
        spacing reach ~1.58 A in the worst orientation. Holding that
        arrangement to a real physical minimum would veto perfectly good
        runs, and which pair trips first moves from seed to seed.
    seed : int | None
        Optional seed for reproducible site selection and orientations.

    Returns
    -------
    AtomicSystem
        Positions, atomic numbers, cell/pbc, and validated rigid groups
        (one group per placed molecule).

    Raises
    ------
    ValueError
        If *cell* isn't diagonal, or if the random orientations at this
        density happen to produce an atomic overlap (try a different seed
        or a smaller *n_particles*) -- see
        :func:`~raymc.system.check_no_atomic_overlap`.
    """
    cell = cell.to(torch.float64)
    off_diagonal = cell - torch.diag(torch.diagonal(cell))
    if off_diagonal.abs().max() > 1e-8:
        raise ValueError("pack_on_lattice only supports an orthorhombic cell")
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")

    if pbc is None:
        pbc = torch.tensor([True, True, True])
    if min_dist is None:
        min_dist = MinDistTable()
    generator = (
        torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    )

    box_lengths = torch.diagonal(cell)
    volume = box_lengths.prod()
    n_molecules = n_particles

    # round() before ceil(): box_length / spacing is exactly an integer
    # whenever n_particles divides the box into a perfect cube of sites
    # (e.g. 1000 molecules in a cubic cell), but floating-point error can push
    # that ratio a hair above the integer (e.g. 10.000000000000002) --
    # left unrounded, ceil() would silently add a whole extra layer of
    # sites per axis, collapsing the realized spacing well below the
    # intended value (confirmed: 1000 sites/axis=10 became 11, spacing
    # 3.61 A became 3.28 A -- enough to matter for a min-dist constraint
    # calibrated against the intended spacing).
    n_per_axis = _lattice_sites_per_axis(cell, n_particles)
    n_sites = n_per_axis[0] * n_per_axis[1] * n_per_axis[2]
    n_molecules = min(n_molecules, n_sites)

    coords_1d = [
        (torch.arange(n_per_axis[d], dtype=torch.float64) + 0.5)
        * (box_lengths[d] / n_per_axis[d])
        for d in range(3)
    ]

    grid = torch.stack(torch.meshgrid(*coords_1d, indexing="ij"), dim=-1).reshape(-1, 3)
    keep = torch.randperm(n_sites, generator=generator)[:n_molecules]
    grid = grid[keep]

    rotations = random_rotation_matrices(n_molecules, generator)
    template_positions = template.positions.to(torch.float64)
    rotated = torch.einsum("mij,aj->mai", rotations, template_positions)
    positions = (rotated + grid.unsqueeze(1)).reshape(-1, 3)

    atomic_numbers = template.atomic_numbers.repeat(n_molecules)
    groups = torch.arange(n_molecules).repeat_interleave(template.num_atoms)

    system = AtomicSystem(
        positions=positions.to(torch.float32),
        atomic_numbers=atomic_numbers,
        cell=cell.unsqueeze(0).to(torch.float32),
        pbc=pbc.unsqueeze(0),
    )
    set_rigid_groups(system, groups)
    try:
        check_no_atomic_overlap(system, min_dist)
    except ValueError as exc:
        warnings.warn(
            f"the packed lattice has a contact below the exclusion it was "
            f"checked against: {exc} This is usually the lattice's own "
            f"worst-case geometry rather than a problem -- the melt that "
            f"follows should clear it. Check the structure if there is no "
            f"equilibration stage.",
            stacklevel=2,
        )
    return system


def equilibrate_hard_spheres(
    system: AtomicSystem,
    min_dist: MinDistTable,
    n_steps: int,
    max_displacement: float,
    seed: int | None = None,
    log_every: int | None = None,
) -> AtomicSystem:
    """Melt a packed structure (typically from :func:`pack_on_lattice`) by
    proposing random single-group translations and keeping any that do not
    make *min_dist* worse. No energy term, no annealing -- purely a
    greedy relaxation on the minimum-approach penalty.

    **Accepts a starting structure that already violates min_dist**, and
    relaxes out of it. The acceptance rule is "the touched pairs' penalty
    must not increase", which from a violation-free start is exactly the
    old strict rule (a zero penalty can only stay zero), and from a
    violating one descends toward zero instead of refusing to begin.

    That matters because a raw lattice frequently cannot satisfy a
    physical exclusion: water's outermost atom sits 0.783 A from its
    centroid, so two adjacent sites at 3.15 A spacing reach ~1.58 A in
    the worst orientation. Demanding a clean start made any H-H minimum
    above ~1.56 A impossible to ask for, at any density, no matter how
    many steps were available to fix it.

    A melt that enforces only *some* of the run's pairs is worse than
    useless for the others: melting 500 water molecules at an O-O
    exclusion alone, with everything else at the 1.0 fallback, takes the
    lattice from 153 H-H contacts under 2.0 A to 455 of them. Pass the
    exclusion you actually want the starting structure to satisfy, for
    every pair you care about.

    Useful when :func:`pack_on_lattice`'s regular grid is too structured
    (see :func:`bragg_peak_max_sk`) to hand to RMC as-is -- this stage only
    ever accepts moves that keep the structure valid, melting an
    already-valid lattice into a liquid-like one at the same density.

    The move-size/melt relationship is non-monotonic -- measured on an 8x8x8
    lattice (512 atoms) at a 3.3 A cutoff, 150k moves:

    ==================  ==========  ===========
    max_displacement     acceptance   Bragg S(k)
    ==================  ==========  ===========
    0.2 A                57%         34.9
    0.4 A                35%         5.3
    0.8 A                13.5%       2.6
    1.5 A                2.0%        117.8 (fails to melt -- too large a
                                       step jams instead of exploring)
    ==================  ==========  ===========

    0.4-0.8 A is the safe range; *max_displacement* well outside it can
    silently fail to melt the lattice even at large *n_steps* -- always
    check :func:`bragg_peak_max_sk` on the result (should land under ~5;
    a perfect lattice of this size scores ~512).

    Parameters
    ----------
    system : AtomicSystem
        Starting structure -- typically a lattice packing at the full
        target density.
    min_dist : MinDistTable
        The SAME minimum-approach-distance table the RMC stage's
        :class:`~raymc.constraints.min_dist.MinDistConstraint` will use --
        equilibrating at a different cutoff than RMC enforces leaves RMC
        starting in violation (see the NaN-acceptance bug this whole
        pipeline exists to avoid).
    n_steps : int
        Number of Metropolis attempts.
    max_displacement : float
        Max per-move Cartesian displacement (Angstrom), each axis drawn
        uniformly in ``[-max_displacement, max_displacement]``.
    seed : int | None
        Optional seed for reproducibility.
    log_every : int | None
        If set, print progress (step count, max Bragg S(k) -- see
        :func:`bragg_peak_max_sk`) every this many steps.

    Returns
    -------
    AtomicSystem
        The equilibrated structure (rigid groups preserved).
    """
    from raymc.moves.group_based import TranslateGroup

    n_groups = int(rigid_groups(system).max()) + 1 if system.num_nodes > 0 else 0
    n_per_axis = _lattice_sites_per_axis(system.cell.squeeze(0), n_groups)

    generator = (
        torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    )
    move = TranslateGroup(max_displacement=max_displacement)

    from raymc.constraints.min_dist import MinDistConstraint

    constraint = MinDistConstraint(min_dist)
    state = system
    initial = float(constraint.compute(state))
    if initial > 0:
        warnings.warn(
            f"equilibrate_hard_spheres: the starting structure already "
            f"violates min_dist (penalty {initial:.3g}). Relaxing out of it "
            f"rather than refusing -- see this function's docstring.",
            stacklevel=2,
        )

    # A run that STARTS clean can never leave zero under this rule, so the
    # pre-move score is known to be 0 without computing it and the loop
    # keeps the single-score-per-step cost it has always had. A run that
    # starts violating pays two incremental scores per step throughout --
    # decided once, here, rather than re-tested during the loop, because
    # asking "is the whole box clean yet" costs a global O(N) compute and
    # would dwarf the incremental work it was meant to save.
    relaxing = initial > 0

    for step in range(n_steps):
        candidate, delta = move.attempt(state, generator)
        after = constraint.compute_increm_change(candidate, delta)
        if relaxing:
            # Compare the SAME pairs before and after. The move leaves
            # every untouched-untouched pair exactly where it was, so the
            # difference between these two is the difference in the WHOLE
            # system's penalty -- even when that penalty is not zero.
            before = constraint.compute_increm_change(state, delta)
            keep = bool(after <= before)
        else:
            keep = bool(after == 0)
        if keep:
            state = candidate

        if log_every and (step + 1) % log_every == 0:
            print(
                f"[raymc] equilibrate_hard_spheres step {step + 1:>7}/{n_steps}  "
                f"max Bragg S(k) = {bragg_peak_max_sk(state, n_per_axis=n_per_axis):.1f}"
            )

    final = float(constraint.compute(state))
    if final > 0:
        warnings.warn(
            f"equilibrate_hard_spheres: finished with min_dist still violated "
            f"(penalty {final:.3g}, from {initial:.3g}). More n_steps may "
            f"clear it; a penalty that will not budge means the exclusion is "
            f"not satisfiable at this density.",
            stacklevel=2,
        )
    return state


def bragg_peak_max_sk(
    system: AtomicSystem,
    n_per_axis: int | list[int] | None = None,
    shell: int = 1,
) -> float:
    """Max static structure factor S(k) over the low-order Bragg points of
    the simple-cubic lattice *system* started from -- a liquid-melt
    diagnostic (see :func:`equilibrate_hard_spheres`).

    ``S(k) = (1/N) |sum_j exp(i k . r_j)|^2`` at reciprocal-lattice
    vectors ``k = (2*pi/a) * (h, k, l)``, ``a`` the lattice spacing,
    ``(h, k, l)`` all small nonzero integer triples within *shell* of the
    origin (``max(|h|,|k|,|l|) <= shell``, excluding the origin itself).
    A perfect lattice scores S(k) ~ N at these vectors; a well-melted
    liquid scores ~1-3.

    Parameters
    ----------
    system : AtomicSystem
        Structure to score -- typically the pre-/post-equilibration
        structure, assumed cubic, N a perfect cube (as
        :func:`pack_on_lattice` produces for a cubic cell/count).
    n_per_axis : int | list[int] | None
        Sites per axis of the original lattice -- a single int for a
        uniform grid, or one int per axis (see
        :func:`_lattice_sites_per_axis`, which callers building on a
        lattice should pass rather than leaving this to guess). Defaults
        to ``round(N ** (1/3))``, which is only exact when *N* happens to
        be a perfect cube.
    shell : int
        How many low-order (h, k, l) shells to check (default: 1, i.e.
        the 6 permutations of (+-1, 0, 0)).

    Returns
    -------
    float
        The maximum S(k) found over the checked Bragg points.
    """
    positions = system.positions.to(torch.float64)
    n = positions.shape[0]
    cell = system.cell.squeeze(0).to(torch.float64)
    box_lengths = torch.diagonal(cell)

    if n_per_axis is None:
        n_per_axis = round(n ** (1 / 3))
    if isinstance(n_per_axis, int):
        n_per_axis = [n_per_axis, n_per_axis, n_per_axis]

    hkl = torch.arange(-shell, shell + 1)
    grid = torch.stack(torch.meshgrid(hkl, hkl, hkl, indexing="ij"), dim=-1).reshape(-1, 3)
    grid = grid[(grid.abs().sum(dim=-1) > 0)]  # drop the origin

    spacing = box_lengths / torch.tensor(n_per_axis, dtype=torch.float64)  # [3]
    k_vectors = 2 * math.pi * grid.to(torch.float64) / spacing  # [n_k, 3]

    phase = positions @ k_vectors.T  # [n, n_k]
    real = torch.cos(phase).sum(dim=0)
    imag = torch.sin(phase).sum(dim=0)
    sk = (real**2 + imag**2) / n
    return float(sk.max())
