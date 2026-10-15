"""Tests for the `raymc` CLI entry point."""

from __future__ import annotations

import os

import yaml

from raymc.cli import main


def _write_tiny_config(tmp_path, filename: str = "config.yaml", **overrides) -> str:
    config = {
        "name": "cli_test",
        "init": {
            "strategy": "pack_on_lattice",
            "molecule": "water",
            "n_particles": 10,
            "cell": [15.0, 15.0, 15.0],
            "seed": 1,
        },
        "moves": [
            {"type": "translate_group", "params": {"max_displacement": 0.3}},
        ],
        "engine": {"n_conv_steps": 5},
    }
    config.update(overrides)
    path = tmp_path / filename
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


def test_cli_runs_end_to_end_and_writes_outputs(tmp_path):
    config_path = _write_tiny_config(tmp_path)
    out_dir = tmp_path / "out"

    exit_code = main([config_path, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot"])

    assert exit_code == 0
    assert (out_dir / "final_structure.xyz").exists()
    assert (out_dir / "results.pt").exists()
    assert not (out_dir / "fit_and_history.png").exists()  # --no-plot


def test_cli_default_out_dir_uses_run_name(tmp_path, monkeypatch):
    config_path = _write_tiny_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    main([config_path, "--log-every", "0", "--no-plot"])

    assert os.path.isdir(tmp_path / "cli_test_output")
    assert (tmp_path / "cli_test_output" / "final_structure.xyz").exists()


def test_cli_writes_a_plot_when_a_pdf_modality_is_configured(tmp_path):
    import torch

    r = torch.linspace(1.0, 6.0, 50, dtype=torch.float64)
    ref_path = tmp_path / "ref.txt"
    ref_path.write_text("\n".join(f"{ri.item()} 1.0" for ri in r))

    config_path = _write_tiny_config(
        tmp_path,
        modalities=[
            {
                "name": "pdf",
                "target_file": str(ref_path),
                "element_a": "O",
                "r_min": 1.0,
                "r_max": 6.0,
                "sigma": 0.1,
            }
        ],
    )
    out_dir = tmp_path / "out"

    main([config_path, "--out-dir", str(out_dir), "--log-every", "0"])

    assert (out_dir / "fit_pdf.png").exists()
    assert (out_dir / "chi_history.png").exists()
    assert not (out_dir / "fit_and_history.png").exists()


def test_cli_writes_a_production_trajectory_when_convergence_and_production_are_configured(
    tmp_path,
):
    import torch

    r = torch.linspace(1.0, 6.0, 50, dtype=torch.float64)
    ref_path = tmp_path / "ref.txt"
    ref_path.write_text("\n".join(f"{ri.item()} 1.0" for ri in r))

    config_path = _write_tiny_config(
        tmp_path,
        modalities=[
            {
                "name": "pdf",
                "target_file": str(ref_path),
                "element_a": "O",
                "r_min": 1.0,
                "r_max": 6.0,
                "sigma": 0.1,
            }
        ],
        engine={
            "n_conv_steps": 20,
            "convergence": {"window": 5, "threshold": 1e-9, "patience": 100},
            "production": {"n_moves": 15, "n_configs": 3, "save_interval": 5},
        },
    )
    out_dir = tmp_path / "out"

    exit_code = main([config_path, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot"])

    assert exit_code == 0
    assert (out_dir / "production_trajectory.xyz").exists()


# --------------------------------------------------------------------
# Checkpoint / resume.
# --------------------------------------------------------------------


def test_checkpointing_writes_a_structure_and_a_state(tmp_path):
    config_path = _write_tiny_config(tmp_path, engine={"n_conv_steps": 10})
    out_dir = tmp_path / "out"

    main([config_path, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
          "--checkpoint-every", "4"])

    assert (out_dir / "checkpoint.xyz").exists()
    assert (out_dir / "checkpoint.pt").exists()
    # No .tmp files left behind: the write is rename-into-place precisely
    # so a process dying mid-write cannot leave a corrupt checkpoint.
    assert not list(out_dir.glob("*.tmp"))


def test_checkpoint_records_how_far_the_chain_got(tmp_path):
    import torch

    config_path = _write_tiny_config(tmp_path, engine={"n_conv_steps": 10})
    out_dir = tmp_path / "out"
    main([config_path, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
          "--checkpoint-every", "4"])

    saved = torch.load(out_dir / "checkpoint.pt", weights_only=False)
    assert saved["steps_done"] == 8  # the last multiple of 4 at or below 10
    assert len(saved["history_chi_sqs"]) == 8


def test_resume_runs_only_the_remaining_steps_and_keeps_the_history(tmp_path):
    """The whole point: a resumed run covers the same total chain, not the
    same total again."""
    import torch

    config_path = _write_tiny_config(
        tmp_path, "full.yaml", engine={"n_conv_steps": 20}
    )
    out_dir = tmp_path / "out"

    # A short first leg, checkpointed, then stopped by pretending the
    # budget was smaller than it is. Its own file: two configs sharing one
    # path silently makes the second leg re-read the first leg's budget,
    # which looks like a passing test for the wrong reason.
    short_config = _write_tiny_config(
        tmp_path, "short.yaml", engine={"n_conv_steps": 8}
    )
    main([short_config, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
          "--checkpoint-every", "4"])
    first = torch.load(out_dir / "checkpoint.pt", weights_only=False)
    assert first["steps_done"] == 8

    main([config_path, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
          "--resume"])

    saved = torch.load(out_dir / "results.pt", weights_only=False)
    # 8 carried over + the 12 that were left = the configured 20.
    assert len(saved["history_chi_sqs"]) == 20
    for trace in saved["history_term_chi_sqs"].values():
        assert len(trace) == 20


def test_resume_starts_from_the_checkpointed_structure(tmp_path):
    """Not from the config's initializer, which for a packing strategy
    would silently throw away every move the first leg made."""
    from ase.io import read

    import numpy as np

    config_path = _write_tiny_config(
        tmp_path, "full.yaml", engine={"n_conv_steps": 200}
    )
    out_dir = tmp_path / "out"
    short = _write_tiny_config(tmp_path, "short.yaml", engine={"n_conv_steps": 20})
    main([short, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
          "--checkpoint-every", "10"])
    first_leg_end = read(out_dir / "checkpoint.xyz").get_positions()

    main([config_path, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
          "--resume", "--checkpoint-every", "10"])
    resumed_end = read(out_dir / "final_structure.xyz").get_positions()

    # It continued from the first leg's endpoint rather than re-packing a
    # fresh lattice: the structures are close (180 more moves of at most
    # 0.3 A) but not identical, and above all the resumed run did not
    # throw the first leg away.
    assert resumed_end.shape == first_leg_end.shape
    assert not np.allclose(resumed_end, first_leg_end)
    displacement = np.linalg.norm(resumed_end - first_leg_end, axis=1).max()
    assert displacement < 20.0  # nothing like a re-pack of the whole box


def test_resume_without_a_checkpoint_says_so(tmp_path):
    import pytest

    config_path = _write_tiny_config(tmp_path)
    out_dir = tmp_path / "empty"
    out_dir.mkdir()
    with pytest.raises(SystemExit, match="no checkpoint"):
        main([config_path, "--out-dir", str(out_dir), "--log-every", "0",
              "--no-plot", "--resume"])


def test_a_checkpoint_written_after_resume_describes_the_whole_chain(tmp_path):
    """steps_done and the history must agree, or the next interruption
    silently discards everything before the last resume.

    The failure this pins was real: a resumed run reported the true total
    in steps_done while saving only its own leg's history, so a checkpoint
    looked complete and was not.
    """
    import torch

    first = _write_tiny_config(tmp_path, "first.yaml", engine={"n_conv_steps": 20})
    second = _write_tiny_config(tmp_path, "second.yaml", engine={"n_conv_steps": 60})
    out_dir = tmp_path / "out"

    main([first, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
          "--checkpoint-every", "10"])
    main([second, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
          "--resume", "--checkpoint-every", "10"])

    saved = torch.load(out_dir / "checkpoint.pt", weights_only=False)
    assert saved["steps_done"] == len(saved["history_chi_sqs"])
    assert saved["steps_done"] == 60
    for trace in saved["history_term_chi_sqs"].values():
        assert len(trace) == 60


def test_resuming_twice_keeps_every_leg(tmp_path):
    """Three legs, and the chain still has to add up -- a run interrupted
    repeatedly is exactly the case checkpointing exists for."""
    import torch

    out_dir = tmp_path / "out"
    for index, total in enumerate((20, 50, 90)):
        config = _write_tiny_config(
            tmp_path, f"leg{index}.yaml", engine={"n_conv_steps": total}
        )
        argv = [config, "--out-dir", str(out_dir), "--log-every", "0", "--no-plot",
                "--checkpoint-every", "10"]
        if index:
            argv.append("--resume")
        main(argv)
        saved = torch.load(out_dir / "checkpoint.pt", weights_only=False)
        assert saved["steps_done"] == len(saved["history_chi_sqs"]) == total
