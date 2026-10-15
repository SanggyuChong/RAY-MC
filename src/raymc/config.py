"""RAY-MC's YAML settings for configuring the structural resolution workflow.

Declares which modalities are active, which forward model and MLIP back each
of them, which engine to run, and how the initial structure is built. Only
:class:`InitConfig` is wired to real code as of the Tier-1 prototype;
the rest of :class:`RunConfig` is the schema the remaining architecture
layers (modality plugins, MLIP adapter, residual aggregator, engine) will read
from as they're implemented, so run configs don't need to change shape
later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MinDistPairConfig(BaseModel):
    """One user-supplied approach distance for an element pair.

    Used for ``constraints.pairs`` (a *minimum* approach distance),
    ``constraints.max_pairs`` (a *maximum* separation), and
    ``init.equilibrate.pairs`` (a minimum approach distance enforced
    during pre-equilibration only) -- the shape is identical, only which
    table it feeds differs.
    """

    elements: tuple[str, str]
    """Element symbols, e.g. ``("O", "O")`` -- order doesn't matter."""
    distance: float = Field(gt=0)
    """Allowed distance (Angstrom) between this pair, for atoms in
    different rigid groups -- a lower bound in ``constraints.pairs`` and
    ``init.equilibrate.pairs``, an upper bound in
    ``constraints.max_pairs``."""


class EquilibrateConfig(BaseModel):
    """Hard-sphere Metropolis equilibration stage (see
    :func:`raymc.initialize.pack.equilibrate_hard_spheres`) -- melts a
    lattice-packed starting structure at the same minimum-approach
    distance the RMC stage will enforce, so RMC never starts in
    constraint violation.
    """

    n_steps: int = Field(default=150_000, gt=0)
    max_displacement: float = Field(default=0.5, gt=0)
    """Per-move max displacement (Angstrom). 0.4-0.8 A melts a lattice
    without jamming (see the module docstring's measured acceptance/S(k)
    table) -- both much smaller and much larger step sizes stall."""
    min_dist: float | None = None
    """Optional hard-sphere diameter (Angstrom) used ONLY for this
    pre-RMC equilibration stage, decoupled from the RMC run's own
    minimum-approach constraint (``constraints.pairs`` /
    :class:`ConstraintConfig`).

    These are two different physical questions: the RMC constraint's
    min-dist must match where the target data actually leaves zero (a
    property of the experimental g(r)), while the *pre*-equilibration
    diameter only needs to produce a good starting configuration -- e.g.
    an effective hard-sphere diameter from liquid-state theory (Ashcroft-
    Lekner) that reproduces the real fluid's packing structure, which can
    differ from the data's own onset distance.

    When ``None`` (default), equilibration falls back to the run's actual
    min-dist constraint table.

    A *looser* equilibration exclusion than the RMC constraint is allowed,
    and is a legitimate thing to want: it hands RMC a structure that
    starts slightly inside its own constraint and lets the fit relax out
    of it. This was rejected while the distance constraints were a
    ``0``/``+inf`` step, because starting in violation then meant
    chi-squared was literally infinite at step 0 and the run could never
    begin. The smooth penalty (:mod:`raymc.constraints.penalty`) prices a
    shallow violation as a large finite number instead, so the walker can
    climb out.

    It is still possible to start too deep to recover: the penalty
    overflows float64 to a true ``+inf`` past roughly
    ``26.6 * softening_window`` of violation -- 2.7 A at the default
    window, but only 0.27 A if you tighten it to 0.01. The engine catches
    that at step 0 with a clear error
    (:func:`raymc.engine.rmc._raise_if_not_finite`) rather than a config
    validator guessing in advance.

    Shorthand for :attr:`default`, and kept because it reads better for
    the common one-species case. Setting both is an error rather than a
    silent precedence rule."""

    default: float = Field(default=1.0, gt=0)
    """Fallback pre-equilibration hard-sphere diameter (Angstrom) for any
    element pair not listed in :attr:`pairs` -- the per-pair spelling of
    :attr:`min_dist`, mirroring :attr:`ConstraintConfig.default`."""

    pairs: list[MinDistPairConfig] = Field(default_factory=list)
    """Per-element-pair pre-equilibration hard-sphere diameters, mirroring
    :attr:`ConstraintConfig.pairs`.

    This exists because a scalar cannot express the case that motivates
    the whole decoupling. Enforcing an O-O minimum approach during the
    melt means setting it for O-O *only*: as a scalar it would also apply
    to intermolecular O-H, forbidding hydrogen-bond geometry outright and
    jamming the melt near close packing. Per-pair, the melt is shaped
    exactly where the physics asks for it and left alone everywhere
    else."""

    @model_validator(mode="after")
    def _check_min_dist_and_default_agree(self) -> EquilibrateConfig:
        if self.min_dist is not None and "default" in self.model_fields_set:
            raise ValueError(
                "init.equilibrate: set either 'min_dist' or 'default', not "
                "both -- 'min_dist' is shorthand for 'default' and they "
                "would otherwise silently disagree."
            )
        return self

    def is_explicit(self) -> bool:
        """Whether the user actually asked for a distinct pre-equilibration
        exclusion, as opposed to leaving the whole thing unset.

        The distinction matters and cannot be read off the field values:
        :attr:`default` has a value (1.0) whether or not anyone set it, so
        building a table unconditionally would silently switch every
        existing config -- which relies on falling back to the RMC
        constraint table -- onto a 1.0 A exclusion it never asked for."""
        if self.min_dist is not None:
            return True
        # `min_dist: null` is this field's own documented spelling of
        # "not set" (fall back to the RMC table), so an explicit null must
        # not count as asking for a table of its own.
        return bool({"default", "pairs"} & self.model_fields_set)

    def min_dist_table(self):
        """This stage's own :class:`~raymc.utils.MinDistTable`, or ``None``
        to fall back to the run's RMC constraint table (today's behaviour
        when nothing here is set -- see :meth:`is_explicit`)."""
        if not self.is_explicit():
            return None
        import periodictable as pt

        from raymc.utils import MinDistTable

        default = self.min_dist if self.min_dist is not None else self.default
        return MinDistTable(
            pair_distances={
                (
                    pt.elements.symbol(pair.elements[0]).number,
                    pt.elements.symbol(pair.elements[1]).number,
                ): pair.distance
                for pair in self.pairs
            },
            default=default,
        )


class InitConfig(BaseModel):
    """How the initial AtomicSystem is constructed.

    Two strategies:

    - ``"pack_on_lattice"`` (Tier-1 scope): places one copy per site on a
      regular grid, reaching any density deterministically, via
      :func:`raymc.initialize.pack.pack_on_lattice`.
    - ``"load_structure"``: read a structure built externally (e.g. a
      crystalline system from a symmetry group / topology generator) from
      a file, via :func:`raymc.initialize.load.load_structure`. RAY-MC
      does not build such structures itself, only validates and reads them.

    Both strategies run the same atomic-overlap validation on the result
    (see :func:`raymc.system.check_no_atomic_overlap`).
    """

    strategy: Literal["pack_on_lattice", "load_structure"] = "pack_on_lattice"

    # -- pack_on_lattice fields --
    molecule: str | None = None
    """A name in RAY-MC's molecule library, a path to a user-supplied XYZ
    file, or a bare element symbol for a lone-atom species (e.g. "Na") --
    resolved by :func:`raymc.initialize.templates.load_group_template`."""
    n_particles: int | None = Field(default=None, gt=0)
    """Number of molecules/atoms to place. The resulting number density is
    ``n_particles / volume`` -- compute this from whatever target density
    you want beforehand (``round(density * volume)``)."""
    cell: tuple[float, float, float] | None = None
    """Cubic box edge lengths (Angstrom): (a, b, c) along x, y, z."""
    pbc: tuple[bool, bool, bool] = (True, True, True)
    seed: int | None = None

    # -- load_structure fields --
    structure_file: str | None = None
    """Path to a structure file for ``"load_structure"``."""

    equilibrate: EquilibrateConfig | None = None
    """Optional hard-sphere Metropolis equilibration stage run on the
    lattice-packed structure before RMC starts -- see
    :func:`raymc.initialize.pack.equilibrate_hard_spheres`. Needed
    whenever the lattice's regular grid is too structured to hand to RMC
    as-is (see that function's docstring)."""

    @model_validator(mode="after")
    def _check_required_fields_for_strategy(self) -> InitConfig:
        if self.strategy == "pack_on_lattice":
            missing = [
                name
                for name in ("molecule", "n_particles", "cell")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"init.strategy {self.strategy!r} requires {missing}")
        elif self.strategy == "load_structure":
            if self.structure_file is None:
                raise ValueError(
                    "init.strategy 'load_structure' requires 'structure_file'"
                )
        return self


class ModalityConfig(BaseModel):
    """One active experimental-observable modality.

    Unknown keys are rejected rather than ignored. Pydantic's default is
    to drop them silently, which meant a config could carry a field that
    had been renamed or removed -- ``renormalize: true`` on a build that
    no longer has it, say -- and run to completion doing something other
    than what it says, with nothing in the logs to show for it.

    Two are wired up (see :func:`raymc.orchestrate.build_chi_sq`):

    - ``name: "pdf"`` -- real-space g(r)/G(r), backed by
      :class:`raymc.modalities.pdf.calculator.PDFCalculator`.
    - ``name: "sq"`` -- reciprocal-space structure factor S(Q), backed by
      :class:`raymc.modalities.sq.calculator.SQCalculator`.

    Fields are grouped by which modality reads them. ``element_a`` /
    ``element_b`` select a partial in either modality; ``r_max`` and
    ``n_points`` describe the real-space histogram in both (for "sq"
    that histogram is the transform's integration grid, which always
    starts at r = 0, so ``r_min`` is meaningless there and setting it is
    an error rather than a silent no-op).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Modality identifier -- "pdf" or "sq"."""
    label: str | None = None
    """Name this modality's chi-squared term is reported under. Defaults
    to ``name``, which is fine for a single data set -- but RMCA's whole
    multiple-data-set case (Section 4: fitting neutron and X-ray, or
    several isotopic substitutions, simultaneously) means two modalities
    can share a ``name``. Their terms would then collide in every
    per-term report dict, one silently overwriting the other, so labels
    are required to be unique across a run."""
    target_file: str
    """Path to the experimental reference data for this modality."""
    forward_model: str | None = None
    """Which Calculator implementation to use. Unused by "pdf" (which has
    only one). For "sq" it selects ``"histogram_ft"`` (default -- RMCA's
    own approach, fast) or ``"debye"`` (unbinned reference path, much
    slower); see :mod:`raymc.modalities.sq.calculator`."""
    weight: float = 1.0
    """Contribution of this modality's residual to the total loss."""

    # -- "pdf" modality fields --
    element_a: str | None = None
    """Element symbol, e.g. "O" -- the partial PDF's first species."""
    element_b: str | None = None
    """Element symbol for a cross partial (e.g. "H" for g_OH); omitted or
    equal to ``element_a`` means the self partial (e.g. g_OO)."""
    r_min: float | None = Field(default=None, gt=0)
    """Lower bound selecting a sub-range of the reference file's OWN r
    points. It never resamples: the histogram's bins are the reference
    grid itself (see
    :func:`raymc.residuals.aggregator.make_partial_pdf_term`). Omit to
    use every point in the file."""
    r_max: float | None = None
    """For "pdf", the upper bound of the same selection. For "sq" this is
    a different quantity entirely -- the sine transform's r integration
    cutoff, which must not exceed half the box's narrowest width and
    defaults to exactly that. The two meanings share a name for
    historical reasons; read :func:`raymc.orchestrate._build_sq_term`
    before setting it on an "sq" modality."""
    sigma: float | None = Field(default=None, gt=0)
    """Constant per-point experimental error (see
    :class:`raymc.residuals.aggregator.ChiSqTerm`), used for both the
    reported chi-squared and the Metropolis acceptance decision (McGreevy
    & Pusztai (1988), Eq. 8/11/12 -- a single sigma plays both roles). If
    omitted, the reference data file's own error column is used instead --
    required to be present in that case (no default of 1)."""

    # -- "sq" modality fields --
    radiation: Literal["neutron", "xray"] | None = None
    """Which radiation the total S(Q) was measured with, fixing the
    Faber-Ziman weights. Required for a total S(Q); must be omitted for a
    partial (``element_a`` given), which is unweighted by construction."""
    species_weights: dict[str, float] | None = None
    """Per-element scattering-factor overrides, e.g. ``{H: 6.671}`` for
    deuterium -- the isotopic-substitution knob, following fullrmc's
    ``elementsWeight``. See
    :func:`raymc.modalities.sq.scattering.species_form_factors`."""
    q_min: float | None = Field(default=None, gt=0)
    q_max: float | None = Field(default=None, gt=0)
    """Restrict the fit to experimental Q points within ``[q_min, q_max]``
    -- RMCA's nx1/nx2 index window, expressed as physical values. The
    model is evaluated on the data's own Q points, so these select rather
    than resample. Q = 0 is always excluded (S(Q) is singular there)."""
    n_r: int = Field(default=500, gt=0)
    """Number of bins on the sine transform's INTERNAL r integration grid
    over ``[0, r_max)``. An implementation detail of the forward model:
    unrelated to the data's Q grid (which comes from the file) and to the
    "pdf" modality's r bins (which also come from the file). Only the
    transform's own accuracy depends on it -- ``dr = r_max / n_r`` should
    stay well under ``2*pi/(5*q_max)``, which the calculator checks."""
    window: Literal["none", "lorch", "hann"] = "lorch"
    """Truncation window damping the model's g(r) - 1 before transforming
    -- see :mod:`raymc.modalities.sq.window`."""
    convention: Literal["S", "F"] = "S"
    """``"S"``: the target tends to 1 at high Q. ``"F"``: it tends to 0
    (RMCA Eq. 15's own convention). Must match the data file."""
    data_scale: float = 1.0
    data_offset: float = 0.0
    """Declared normalization correction applied ONCE to the reference
    values as ``data_scale * y + data_offset``, before the run starts.

    These replace the earlier ``renormalize``/``offset`` flags, which
    solved for the same two numbers by least squares at every proposed
    move. That was wrong twice over: a target that moves between
    proposals means the Metropolis criterion is comparing two different
    objectives, and the solve is ill-conditioned precisely when it looks
    most useful -- against a model of the wrong sign it returns a large
    negative scale and swamps the structural signal with normalization
    noise.

    RMCA Section 9.6 exists because measured structure factors really do
    carry normalization errors. Correcting them is still supported; it
    just has to be a decision you record here rather than one the fit
    makes silently each step."""

    @model_validator(mode="after")
    def _check_modality_fields(self) -> ModalityConfig:
        if self.name == "pdf":
            if self.element_a is None:
                raise ValueError("modality 'pdf' requires 'element_a'")
            return self

        if self.name != "sq":
            return self

        if self.element_a is None and self.radiation is None:
            raise ValueError(
                "modality 'sq' requires 'radiation' for a total S(Q), or "
                "'element_a' to fit an unweighted partial A_ab(Q)"
            )
        if self.element_a is not None and self.radiation is not None:
            raise ValueError(
                "modality 'sq': 'radiation' has no meaning for a partial "
                "A_ab(Q) (set by 'element_a'), which is unweighted"
            )
        if self.forward_model is not None and self.forward_model not in (
            "histogram_ft",
            "debye",
        ):
            raise ValueError(
                f"modality 'sq': forward_model must be 'histogram_ft' or "
                f"'debye', got {self.forward_model!r}"
            )
        if "r_min" in self.model_fields_set:
            raise ValueError(
                "modality 'sq': 'r_min' is meaningless -- the sine transform "
                "must integrate from r = 0 (below contact g(r) = 0 still "
                "contributes through the background term), so the histogram "
                "always spans [0, r_max). Remove it."
            )
        if (
            self.q_min is not None
            and self.q_max is not None
            and self.q_max <= self.q_min
        ):
            raise ValueError(
                f"modality 'sq': q_max ({self.q_max}) must exceed q_min "
                f"({self.q_min})"
            )
        return self


class MoveConfig(BaseModel):
    """One active Monte Carlo move: its type, selection weight, and params.

    ``type`` is resolved against a name -> constructor registry at
    :meth:`raymc.moves.moveset.MoveSet.from_config` time (built-ins:
    ``"translate_group"``, ``"rotate_group"``, ``"insert_group"``,
    ``"remove_group"``, ``"relocate_group"`` -- see
    :mod:`raymc.moves.group_based`; ``"translate_atom"`` -- see
    :mod:`raymc.moves.atomic`). The engine
    never references a move type directly; it only ever holds the
    resulting :class:`~raymc.moves.moveset.MoveSet`.
    """

    type: str
    weight: float = Field(default=1.0, gt=0)
    """Relative selection probability among the run's active moves."""
    params: dict = Field(default_factory=dict)
    """Move-specific keyword arguments, e.g. ``max_displacement`` for
    ``"translate_group"`` or ``molecule`` for ``"insert_group"``."""


class EnergeticsConfig(BaseModel):
    """The interatomic potential backing the hybrid RMC energetics.

    Built on metatomic (see :class:`raymc.energetics.metatomic.MetatomicEnergetics`)
    so any metatomic-compatible model works, not just checkpoints from one
    provider. ``source`` picks how the model is loaded:

    - ``"file"`` loads an exported metatomic ``.pt`` from ``path`` via
      :func:`raymc.energetics.metatomic.load_model_file` -- for an
      empirical potential built offline, e.g. the q-TIP4P/F water model
      from ``scripts/build_qtip4pf.py``. Needs only the "metatomic"
      extra, whatever the model was built on.
    - ``"upet"`` loads a UPET checkpoint
      (https://lab-cosmo.github.io/upet/latest/, weights on
      https://huggingface.co/lab-cosmo/upet) via
      :func:`raymc.energetics.metatomic.load_upet_model`, using
      ``model``/``size``/``version``. Needs the "upet" extra
      (``pip install raymc[upet]``) as well.
    - ``"checkpoint"`` loads a UPET checkpoint already on disk, from
      ``checkpoint_path`` -- the only option where huggingface.co is
      unreachable, and the better one for a published fit either way: a
      file you can hash beats a version string resolved remotely.
    """

    enabled: bool = False
    """Off by default: requires the optional "metatomic" extra (plus
    "upet" for the "upet" source), which most environments won't have
    installed. :func:`raymc.orchestrate.build_chi_sq` skips this term
    entirely when False, rather than trying (and failing) to load a model."""
    source: Literal["file", "upet", "checkpoint"] = "upet"
    path: Path | None = None
    """Exported metatomic ``.pt``. Required by (and only meaningful for)
    ``source="file"``."""
    checkpoint_path: str | None = None
    """Path to a local UPET checkpoint, required by (and only meaningful
    for) ``source="checkpoint"``. Keep the UPET filename convention --
    ``pet-mad-xs-v1.5.0.ckpt`` -- since that is what the loader reads the
    model/size/version off."""
    model: str = "pet-mad"
    size: str = "s"
    version: str = "1.5.0"
    temperature: float = Field(default=300.0, gt=0)
    """Temperature (K) setting how strongly the energy competes with the
    data residuals.

    This is not a cosmetic knob -- it is what makes the energetics term
    dimensionally meaningful. The engine accepts with
    ``min(1, exp(-(chi'^2 - chi^2) / 2))`` (see
    :class:`raymc.engine.rmc.ReverseMonteCarlo`), while hybrid RMC
    requires the energy to enter the exponent as ``ΔE / (k_B T)``. A
    "direct"-mode term contributes ``w * E`` to chi-squared and so
    ``-w * ΔE / 2`` to the exponent, which forces ``w = 2 / (k_B T)``.
    :func:`raymc.residuals.aggregator.make_metatomic_term` derives that
    factor from this temperature, so ``weight`` below can stay a plain
    dimensionless number.

    Concretely: at 300 K the term's chi-squared weight is ~77.4 eV^-1. A
    bare ``weight=1`` with no temperature at all -- what this config used
    to mean -- corresponds to ~23,000 K, at which the energy is
    effectively ignored."""
    weight: float = 1.0
    """Dimensionless coupling of the energetics term, on top of the
    ``2 / (k_B T)`` factor derived from ``temperature``. Leave at 1.0 for
    textbook hybrid RMC; lower it to let the data dominate, raise it to
    lean on the potential."""

    @model_validator(mode="after")
    def _check_source_fields(self) -> "EnergeticsConfig":
        if not self.enabled:
            return self
        if self.source == "file" and self.path is None:
            raise ValueError("energetics.source='file' requires energetics.path")
        if self.source != "file" and self.path is not None:
            raise ValueError(
                f"energetics.path is only meaningful for source='file', "
                f"not source={self.source!r}"
            )
        if self.source == "checkpoint" and self.checkpoint_path is None:
            raise ValueError(
                "energetics.source='checkpoint' requires energetics.checkpoint_path"
            )
        if self.source != "checkpoint" and self.checkpoint_path is not None:
            raise ValueError(
                f"energetics.checkpoint_path is only meaningful for "
                f"source='checkpoint', not source={self.source!r}"
            )
        return self


class IntramolecularWindowConfig(BaseModel):
    """One allowed bonded ``[min, max]`` distance window for an element
    pair *within* a rigid group.

    Distinct from :class:`MinDistPairConfig`, which carries a single
    distance and describes atoms in *different* groups. A bonded distance
    needs both bounds together -- it is one window, not two independent
    thresholds that happen to share a pair.
    """

    elements: tuple[str, str]
    """Element symbols, e.g. ``("O", "H")`` -- order doesn't matter."""
    min: float = Field(ge=0)
    """Smallest allowed separation (Angstrom) for this pair inside one
    group."""
    max: float = Field(gt=0)
    """Largest allowed separation (Angstrom) for this pair inside one
    group."""

    @model_validator(mode="after")
    def _check_window_is_nonempty(self) -> IntramolecularWindowConfig:
        if self.max <= self.min:
            raise ValueError(
                f"constraints.intramolecular {list(self.elements)}: max "
                f"({self.max} A) must exceed min ({self.min} A) -- no "
                "distance could satisfy an empty window, so every "
                "configuration would be penalized."
            )
        return self


class ConstraintConfig(BaseModel):
    """Structural distance constraints applied on top of the
    residual-driven acceptance criterion.

    Two of them, mirror images of each other:

    - **minimum distance** (``default``/``pairs``, backing
      :class:`raymc.constraints.min_dist.MinDistConstraint` and
      :class:`raymc.utils.MinDistTable`): the close-contact veto. Following
      RMCProfile's convention, the user supplies a minimum approach
      distance directly, per element pair -- not a radius derived from any
      covalent/van-der-Waals table. Any element pair the run actually
      encounters but isn't listed in ``pairs`` falls back to ``default``,
      with a warning printed the first time that happens. The same table
      also backs the initializer's own overlap check
      (:func:`raymc.system.check_no_atomic_overlap`), so a run's initial
      structure and its RMC moves are held to the same standard.
    - **maximum distance** (``max_default``/``max_pairs``, backing
      :class:`raymc.constraints.max_dist.MaxDistConstraint` and
      :class:`raymc.utils.MaxDistTable`): opt-in, defaulting to ``inf`` --
      no constraint, and no runtime cost at all (see that constraint's
      module docstring). Read its "note what every pair means" paragraph
      before setting one: a finite entry requires *every* such pair in the
      box to stay within that distance, not just the nearest neighbour.

    - **intramolecular window** (``intramolecular``, backing
      :class:`raymc.constraints.intramolecular.IntramolecularConstraint`
      and :class:`raymc.utils.IntraWindowTable`): the complement of the
      other two, and the only one describing a pair of atoms *inside* one
      rigid group. Also opt-in, also free when unused.

    None is a hard 0/+inf veto any more: all three score a smooth,
    differentiable penalty in how far a pair is past its threshold, ramping
    from 0 at the threshold to ``penalty_at_window`` one ``softening_window``
    beyond it and diverging past that -- see
    :mod:`raymc.constraints.penalty`.
    """

    default: float = Field(default=1.0, gt=0)
    """Fallback minimum distance (Angstrom) for any element pair not
    listed in ``pairs``."""
    pairs: list[MinDistPairConfig] = Field(default_factory=list)

    max_default: float = Field(default=float("inf"), gt=0)
    """Fallback maximum distance (Angstrom) for any element pair not
    listed in ``max_pairs``. Defaults to ``inf``: no maximum-distance
    constraint at all, which is also the case that costs nothing at
    runtime. In YAML, write a finite value plainly (``max_default: 6.0``)
    or ``.inf`` for the default."""
    max_pairs: list[MinDistPairConfig] = Field(default_factory=list)

    intramolecular: list[IntramolecularWindowConfig] = Field(default_factory=list)
    """Allowed bonded ``[min, max]`` distance windows for pairs of atoms
    *within* one rigid group, backing
    :class:`raymc.constraints.intramolecular.IntramolecularConstraint` and
    :class:`raymc.utils.IntraWindowTable`.

    The exact complement of ``pairs``/``max_pairs`` above, which say
    nothing about a same-group pair. Opt-in and empty by default, which
    costs nothing at runtime: with every move rigid, an intramolecular
    distance is constant by construction and there is nothing to police.
    It becomes load-bearing the moment a move can displace a single atom
    (``translate_atom``) -- without a window, a molecule's own bonds are
    scored by nothing at all and it dissolves silently. See
    :meth:`raymc.moves.moveset.MoveSet.from_config`, which warns about
    exactly that combination."""

    softening_window: float = Field(default=0.1, gt=0)
    """Width (Angstrom) of the window over which BOTH constraints ramp
    from a penalty of 0 (exactly at the configured distance) to
    ``penalty_at_window`` that far past it, diverging beyond -- see
    :mod:`raymc.constraints.penalty`. Smaller means a harder, more
    veto-like wall; larger gives the Metropolis walker a gentler gradient
    to be steered by."""

    penalty_at_window: float = Field(default=1000.0, gt=0)
    """What one ``softening_window`` of violation costs, for BOTH
    constraints -- the height of the ramp whose extent is
    ``softening_window``.

    Sets how hard the constraints are relative to the run's other
    chi-squared terms, so the number that matters is its size next to the
    per-step chi-squared changes those terms produce. The default of 1000
    dwarfs a typical move's change, making a full window of violation
    effectively a veto. Raise it if a modality's residual is large enough
    to buy its way through; lower it to let the data genuinely trade
    against the constraint."""

    @model_validator(mode="after")
    def _check_max_dist_is_above_min_dist(self) -> ConstraintConfig:
        """Reject a pair whose maximum separation is below its own minimum
        approach distance -- no configuration could ever satisfy both, so
        the run would be penalized no matter what it did."""
        import periodictable as pt

        def _z(pair: MinDistPairConfig) -> tuple[int, int]:
            z1 = pt.elements.symbol(pair.elements[0]).number
            z2 = pt.elements.symbol(pair.elements[1]).number
            return (min(z1, z2), max(z1, z2))

        min_by_pair = {_z(pair): pair.distance for pair in self.pairs}
        max_by_pair = {_z(pair): pair.distance for pair in self.max_pairs}
        checks = [(None, self.default, self.max_default)]
        for key in set(min_by_pair) | set(max_by_pair):
            checks.append(
                (
                    key,
                    min_by_pair.get(key, self.default),
                    max_by_pair.get(key, self.max_default),
                )
            )
        for key, r_min, r_max in checks:
            if r_max <= r_min:
                where = (
                    "constraints.default/max_default"
                    if key is None
                    else f"element pair (Z={key[0]}, Z={key[1]})"
                )
                raise ValueError(
                    f"{where}: maximum distance ({r_max} A) is not above the "
                    f"minimum distance ({r_min} A) -- no structure can satisfy "
                    "both, so every configuration would be penalized."
                )
        return self


class ConvergenceConfig(BaseModel):
    """Simple moving-average convergence detection for the RMC stage (see
    :mod:`raymc.engine.rmc`): chi-squared is judged converged once the
    fractional change between two consecutive moving-average windows of
    the (sigma-scaled) acceptance chi-squared drops below
    *threshold*, sustained for *patience* consecutive checks (guards
    against a single lucky quiet window)."""

    window: int = Field(default=1000, gt=0)
    """Number of steps averaged into one moving-average sample."""
    threshold: float = Field(default=0.02, gt=0)
    """Fractional change between consecutive windows below which the run
    is considered converged."""
    patience: int = Field(default=3, gt=0)
    """Consecutive windows the fractional change must stay below
    *threshold* before convergence is declared."""


class ProductionConfig(BaseModel):
    """Post-convergence production phase (see module docstring of
    :mod:`raymc.engine.rmc`): after RMC reaches convergence, continue
    running and periodically save configurations, then report g(r)
    averaged over the saved set -- McGreevy & Pusztai (1988) step 7."""

    n_moves: int = Field(default=150_000, gt=0)
    n_configs: int = Field(default=50, gt=0)
    save_interval: int | None = None
    """Moves between saved configurations. Defaults to
    ``n_moves // n_configs`` when omitted."""
    trajectory_file: str = "production_trajectory.xyz"
    """Where the saved configurations are written, as an extended XYZ
    trajectory (relative to the run's ``--out-dir``)."""
    restart_from: Literal["final", "best"] = "final"
    """Which convergence-phase state production continues from: "final"
    (the convergence phase's actual trajectory endpoint) or "best" (the
    lowest-chi-squared state visited during convergence). Either way the
    chi-squared aggregator is fully re-seeded (a full recompute, not an
    incremental carry-over) before production starts, so production
    always continues from an internally consistent point -- see
    :meth:`raymc.engine.rmc.ReverseMonteCarlo.run`."""


class EngineConfig(BaseModel):
    """Which structure-resolution engine to run, and its settings."""

    kind: Literal["reverse_monte_carlo", "generative"] = "reverse_monte_carlo"
    n_conv_steps: int = Field(default=1000, gt=0)
    """Number of Metropolis steps to run BEFORE convergence detection
    would otherwise stop the run -- an upper bound on the convergence
    stage when ``convergence`` is set, or the plain step count when it
    isn't."""
    convergence: ConvergenceConfig | None = None
    """Optional convergence detection -- if set, the convergence stage may
    stop (and hand off to ``production``, if configured) before
    ``n_conv_steps`` is reached. See :class:`ConvergenceConfig`."""
    production: ProductionConfig | None = None
    """Optional post-convergence production/averaging phase -- see
    :class:`ProductionConfig`."""


class RunConfig(BaseModel):
    """Top-level RAY-MC run configuration."""

    name: str
    init: InitConfig
    moves: list[MoveConfig] = Field(default_factory=list)
    modalities: list[ModalityConfig] = Field(default_factory=list)
    energetics: EnergeticsConfig = Field(default_factory=EnergeticsConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)

    @model_validator(mode="after")
    def _check_modality_labels_are_unique(self) -> RunConfig:
        """Two modalities must not report under the same term name --
        see :attr:`ModalityConfig.label`."""
        seen: dict[str, int] = {}
        for modality in self.modalities:
            key = modality.label or modality.name
            seen[key] = seen.get(key, 0) + 1
        duplicated = sorted(k for k, n in seen.items() if n > 1)
        if duplicated:
            raise ValueError(
                f"modality term name(s) {duplicated} used more than once. "
                "Fitting several data sets at once (RMCA Section 4) needs "
                "each to be distinguishable -- give them distinct 'label' "
                "values, e.g. label: sq_xray and label: sq_neutron."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load and validate a :class:`RunConfig` from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
