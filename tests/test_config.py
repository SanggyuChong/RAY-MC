from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from raymc.config import (
    ConstraintConfig,
    EnergeticsConfig,
    EquilibrateConfig,
    InitConfig,
    MinDistPairConfig,
    ModalityConfig,
    RunConfig,
)

_REPO = Path(__file__).parents[1]
_EXAMPLES = sorted((_REPO / "examples").glob("*.yaml"))


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.stem)
def test_every_example_config_loads_and_its_data_exist(path):
    config = RunConfig.from_yaml(path)
    for modality in config.modalities:
        assert (_REPO / modality.target_file).is_file()


def test_run_config_loads_the_combined_ice_example():
    path = _REPO / "examples" / "ice_sq_combined_flexible.yaml"
    config = RunConfig.from_yaml(path)
    assert config.init.strategy == "pack_on_lattice"
    assert config.init.molecule == "water"
    assert [m.type for m in config.moves] == ["translate_atom"]
    assert [m.label for m in config.modalities] == ["sq_xray", "sq_neutron"]
    assert [m.radiation for m in config.modalities] == ["xray", "neutron"]
    assert config.modalities[1].data_scale == 3.0
    assert config.constraints.pairs[0] == MinDistPairConfig(
        elements=("O", "O"), distance=2.4
    )
    assert config.engine.kind == "reverse_monte_carlo"
    assert config.energetics.enabled is False


def test_modality_config_pdf_requires_element_a():
    with pytest.raises(ValidationError, match="requires 'element_a'"):
        ModalityConfig(name="pdf", target_file="foo.txt")


def test_modality_config_pdf_accepts_element_a():
    config = ModalityConfig(name="pdf", target_file="foo.txt", element_a="O")
    assert config.element_a == "O"
    assert config.element_b is None
    # the reference file owns the grid now: no r_min/r_max/n_points
    # defaults that could silently disagree with the data
    assert config.r_min is None
    assert config.r_max is None
    assert config.data_scale == 1.0
    assert config.data_offset == 0.0


def test_modality_config_non_pdf_name_does_not_require_element_a():
    # Only "pdf" is implemented, but the schema itself doesn't hardcode
    # that -- a future modality name isn't forced through the pdf check.
    config = ModalityConfig(name="nmr", target_file="foo.txt")
    assert config.element_a is None


def test_energetics_config_defaults_to_disabled():
    config = EnergeticsConfig()
    assert config.enabled is False
    assert config.source == "upet"
    assert config.checkpoint_path is None


def test_energetics_checkpoint_source_accepts_a_local_path():
    config = EnergeticsConfig(
        enabled=True, source="checkpoint", checkpoint_path="models/pet-mad-xs-v1.5.0.ckpt"
    )
    assert config.checkpoint_path == "models/pet-mad-xs-v1.5.0.ckpt"


def test_energetics_checkpoint_source_requires_a_path():
    with pytest.raises(ValidationError, match="requires energetics.checkpoint_path"):
        EnergeticsConfig(enabled=True, source="checkpoint")


def test_energetics_rejects_a_checkpoint_path_the_source_would_ignore():
    """A path set alongside source 'upet' would be silently dropped and the
    run would quietly fetch a different model than the config names."""
    with pytest.raises(ValidationError, match="only meaningful for"):
        EnergeticsConfig(enabled=True, source="upet", checkpoint_path="some.ckpt")


def test_constraint_config_defaults_to_one_angstrom_and_no_pairs():
    config = ConstraintConfig()
    assert config.default == 1.0
    assert config.pairs == []


def test_constraint_config_defaults_to_no_maximum_distance_at_all():
    """The maximum-distance constraint is opt-in: its default must be
    infinite, so an existing config that says nothing about it keeps
    behaving exactly as before (and pays nothing for it)."""
    config = ConstraintConfig()
    assert config.max_default == float("inf")
    assert config.max_pairs == []


def test_constraint_config_defaults_the_softening_window_to_a_tenth_of_an_angstrom():
    assert ConstraintConfig().softening_window == 0.1


def test_constraint_config_defaults_the_penalty_at_window_to_a_thousand():
    assert ConstraintConfig().penalty_at_window == 1000.0


def test_constraint_config_accepts_a_custom_penalty_at_window():
    assert ConstraintConfig(penalty_at_window=25.0).penalty_at_window == 25.0


def test_constraint_config_rejects_a_nonpositive_penalty_at_window():
    """A zero or negative height would make a violation free or actively
    rewarded."""
    for bad in (0.0, -1000.0):
        with pytest.raises(ValidationError):
            ConstraintConfig(penalty_at_window=bad)


def test_constraint_config_reads_an_infinite_max_default_from_yaml(tmp_path):
    """YAML's own spelling of infinity, so a config can state the default
    explicitly."""
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: x\n"
        "init:\n"
        "  strategy: pack_on_lattice\n"
        "  molecule: water\n"
        "  n_particles: 4\n"
        "  cell: [10.0, 10.0, 10.0]\n"
        "constraints:\n"
        "  max_default: .inf\n"
    )
    assert RunConfig.from_yaml(path).constraints.max_default == float("inf")


def test_constraint_config_accepts_a_maximum_above_the_minimum():
    config = ConstraintConfig(
        default=1.0,
        pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.4)],
        max_pairs=[MinDistPairConfig(elements=("O", "O"), distance=3.5)],
    )
    assert config.max_pairs[0].distance == 3.5


def test_constraint_config_rejects_a_maximum_below_the_minimum_for_a_pair():
    """No structure could satisfy both, so every configuration would be
    penalized forever -- catch it at config load, not 10^6 steps in."""
    with pytest.raises(ValidationError, match="Z=8, Z=8"):
        ConstraintConfig(
            pairs=[MinDistPairConfig(elements=("O", "O"), distance=3.5)],
            max_pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.4)],
        )


def test_constraint_config_rejects_a_max_default_below_the_min_default():
    with pytest.raises(ValidationError, match="default/max_default"):
        ConstraintConfig(default=3.0, max_default=2.0)


def test_constraint_config_checks_a_pair_against_the_other_tables_default():
    """An explicit maximum for O-O is still checked against the *minimum*
    default when O-O has no explicit minimum of its own."""
    with pytest.raises(ValidationError, match="Z=8, Z=8"):
        ConstraintConfig(
            default=3.0,
            max_pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.5)],
        )


def test_run_config_defaults_constraints_when_omitted():
    config = RunConfig(
        name="x",
        init={
            "strategy": "pack_on_lattice",
            "molecule": "water",
            "n_particles": 5,
            "cell": [10.0, 10.0, 10.0],
        },
    )
    assert config.constraints.default == 1.0
    assert config.constraints.pairs == []


def test_run_config_defaults_engine_when_omitted():
    config = RunConfig(
        name="x",
        init={
            "strategy": "pack_on_lattice",
            "molecule": "water",
            "n_particles": 5,
            "cell": [10.0, 10.0, 10.0],
        },
    )
    assert config.engine.kind == "reverse_monte_carlo"
    assert config.engine.n_conv_steps == 1000


def test_init_config_pack_on_lattice_requires_its_fields():
    with pytest.raises(ValidationError, match="requires"):
        InitConfig(strategy="pack_on_lattice", molecule="water")


def test_init_config_load_structure_requires_structure_file():
    with pytest.raises(ValidationError, match="requires 'structure_file'"):
        InitConfig(strategy="load_structure")


def test_init_config_load_structure_accepts_a_bare_path():
    config = InitConfig(strategy="load_structure", structure_file="foo.xyz")
    assert config.structure_file == "foo.xyz"


def test_init_config_defaults_to_pack_on_lattice():
    config = InitConfig(
        molecule="water", n_particles=5, cell=(10.0, 10.0, 10.0)
    )
    assert config.strategy == "pack_on_lattice"


def _run_config_kwargs(equilibrate_min_dist: float | None, constraint_distance: float):
    return dict(
        name="x",
        init=InitConfig(
            strategy="pack_on_lattice",
            molecule="Ar",
            n_particles=50,
            cell=(15.0, 15.0, 15.0),
            equilibrate=EquilibrateConfig(min_dist=equilibrate_min_dist),
        ),
        constraints=ConstraintConfig(
            pairs=[MinDistPairConfig(elements=("Ar", "Ar"), distance=constraint_distance)]
        ),
    )


def test_run_config_accepts_an_equilibrate_min_dist_override_larger_than_constraint():
    config = RunConfig(**_run_config_kwargs(3.4, 3.2))
    assert config.init.equilibrate.min_dist == 3.4


def test_run_config_allows_an_equilibrate_min_dist_smaller_than_the_constraint():
    """No longer rejected -- see EquilibrateConfig.min_dist. The smooth
    penalty makes a shallow starting violation recoverable, so this is a
    choice the config is allowed to express."""
    config = RunConfig(**_run_config_kwargs(3.0, 3.2))
    assert config.init.equilibrate.min_dist == 3.0
    assert config.constraints.pairs[0].distance == 3.2


def test_run_config_allows_omitted_equilibrate_min_dist():
    config = RunConfig(**_run_config_kwargs(None, 3.2))
    assert config.init.equilibrate.min_dist is None


# -- per-pair pre-equilibration min-dist table ------------------------------
#
# The case these fields exist for: an O-O minimum approach enforced during
# the melt ONLY, with no O-O entry in the RMC constraint table at all. A
# scalar `min_dist` cannot express it -- it would apply the same distance
# to intermolecular O-H, forbidding hydrogen-bond geometry.


def _water_run_kwargs(equilibrate: EquilibrateConfig, constraints: ConstraintConfig):
    return dict(
        name="x",
        init=InitConfig(
            strategy="pack_on_lattice",
            molecule="water",
            n_particles=8,
            cell=(15.0, 15.0, 15.0),
            equilibrate=equilibrate,
        ),
        constraints=constraints,
    )


def test_equilibrate_pairs_may_restrain_a_pair_the_rmc_table_leaves_open():
    config = RunConfig(
        **_water_run_kwargs(
            EquilibrateConfig(
                default=1.0,
                pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.25)],
            ),
            ConstraintConfig(default=1.0),
        )
    )
    table = config.init.equilibrate.min_dist_table()
    assert table.lookup(8, 8).item() == 2.25
    assert table.lookup(8, 1).item() == 1.0  # O-H left free for H-bonds


def test_equilibrate_may_be_looser_than_the_rmc_constraint_per_pair():
    """A looser equilibration exclusion is allowed, not rejected.

    It hands RMC a structure starting slightly inside its own constraint
    and lets the fit relax out. That was a hard error while the distance
    constraints were a 0/+inf step -- chi-squared was literally infinite
    at step 0 -- but the smooth penalty prices a shallow violation as a
    large finite number, so the walker can climb out. Nothing at config
    time should be second-guessing that.
    """
    config = RunConfig(
        **_water_run_kwargs(
            EquilibrateConfig(
                default=1.0,
                pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.25)],
            ),
            ConstraintConfig(
                default=1.0,
                pairs=[MinDistPairConfig(elements=("O", "O"), distance=2.4)],
            ),
        )
    )
    assert config.init.equilibrate.min_dist_table().lookup(8, 8).item() == 2.25
    assert config.constraints.pairs[0].distance == 2.4


def test_equilibrate_may_be_looser_than_the_rmc_default():
    config = RunConfig(
        **_water_run_kwargs(
            EquilibrateConfig(default=0.8), ConstraintConfig(default=1.2)
        )
    )
    assert config.init.equilibrate.default == 0.8
    assert config.constraints.default == 1.2


def test_min_dist_is_shorthand_for_default():
    equilibrate = EquilibrateConfig(min_dist=2.5)
    assert equilibrate.is_explicit()
    assert equilibrate.min_dist_table().lookup(8, 8).item() == 2.5


def test_setting_both_min_dist_and_default_is_rejected():
    with pytest.raises(ValidationError, match="not.*both"):
        EquilibrateConfig(min_dist=2.5, default=2.5)


def test_an_unset_equilibrate_falls_back_to_the_rmc_table():
    """Today's behaviour, which must not change: `default` has a value
    whether or not anyone set it, so only `model_fields_set` can tell."""
    assert not EquilibrateConfig(n_steps=10).is_explicit()
    assert EquilibrateConfig(n_steps=10).min_dist_table() is None
    # An explicit null is this field's documented spelling of "omitted".
    assert not EquilibrateConfig(min_dist=None).is_explicit()
