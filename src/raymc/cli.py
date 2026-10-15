""" Command Line Interface for RAY-MC.

Runs an RMC job straight from a :class:`~raymc.config.RunConfig` YAML file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from ase import Atoms
from ase.io import write as ase_write

from raymc.config import RunConfig
from raymc.engine.rmc import summarize_history
from raymc.initialize.pack import bragg_peak_max_sk
from raymc.orchestrate import build_run


def _to_ase(system) -> Atoms:
    return Atoms(
        numbers=system.atomic_numbers.numpy(),
        positions=system.positions.detach().numpy(),
        cell=system.cell.squeeze(0).numpy() if system.cell is not None else None,
        pbc=system.cell is not None,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="raymc", description="Run a RAY-MC job from a RunConfig YAML file."
    )
    parser.add_argument("config", help="Path to a RunConfig YAML file")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write outputs to (default: '<run-name>_output')",
    )
    parser.add_argument(
        "--log-every", type=int, default=1000, help="Print progress every N steps"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip writing the PDF/chi-squared plot"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save the current structure and history every N convergence "
        "steps, so an interrupted run can be resumed with --resume instead "
        "of restarted. Costs one small file write per checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint in --out-dir: start from the "
        "saved structure, run only the convergence steps that were left, "
        "and keep the saved history. An RMC chain is memoryless given its "
        "configuration, so this is a genuine continuation -- but the "
        "generator stream is not restored, so it is not bit-for-bit "
        "identical to an uninterrupted run.",
    )
    return parser.parse_args(argv)


CHECKPOINT_STRUCTURE = "checkpoint.xyz"
CHECKPOINT_STATE = "checkpoint.pt"


def _write_checkpoint(
    out_dir: str, steps_done: int, state, history, carried: dict | None = None
) -> None:
    """Persist enough to continue this run from here.

    Written to a temporary name and renamed into place, because the whole
    point of a checkpoint is to survive the process dying -- and a process
    that dies halfway through writing its own recovery file is worse than
    one that never wrote it.

    *carried* is the history a ``--resume`` inherited, prepended here so
    that every checkpoint describes the WHOLE chain and not just the leg
    that happens to be running. Without it a resumed run overwrites the
    checkpoint with its own short history while still reporting the true
    total in ``steps_done``, so the two disagree -- and the next
    interruption discards everything before the last resume. A checkpoint
    that silently drops history is worse than no checkpoint, because it
    looks like one.
    """
    structure_path = os.path.join(out_dir, CHECKPOINT_STRUCTURE)
    ase_write(structure_path + ".tmp", _to_ase(state), format="extxyz")
    os.replace(structure_path + ".tmp", structure_path)

    carried = carried or {}
    chi_sqs = list(carried.get("history_chi_sqs", [])) + list(history.chi_sqs)
    accepted = list(carried.get("history_accepted", [])) + list(history.accepted)
    move_names = list(carried.get("history_move_names", [])) + list(history.move_names)
    term_traces = {}
    for name in set(carried.get("history_term_chi_sqs", {})) | set(
        history.term_chi_sqs
    ):
        term_traces[name] = list(
            carried.get("history_term_chi_sqs", {}).get(name, [])
        ) + list(history.term_chi_sqs.get(name, []))

    state_path = os.path.join(out_dir, CHECKPOINT_STATE)
    torch.save(
        {
            # Invariant every reader relies on: steps_done == len(history).
            "steps_done": steps_done,
            "history_chi_sqs": chi_sqs,
            "history_accepted": accepted,
            "history_move_names": move_names,
            "history_term_chi_sqs": term_traces,
        },
        state_path + ".tmp",
    )
    os.replace(state_path + ".tmp", state_path)


def _load_checkpoint(out_dir: str):
    """The saved structure and history, or raise if there is none."""
    from raymc.initialize.load import load_structure

    state_path = os.path.join(out_dir, CHECKPOINT_STATE)
    structure_path = os.path.join(out_dir, CHECKPOINT_STRUCTURE)
    if not (os.path.isfile(state_path) and os.path.isfile(structure_path)):
        raise SystemExit(
            f"[raymc] --resume: no checkpoint in {out_dir!r} (expected "
            f"{CHECKPOINT_STATE} and {CHECKPOINT_STRUCTURE}). A run only "
            "writes them when given --checkpoint-every."
        )
    saved = torch.load(state_path, weights_only=False)
    return load_structure(structure_path), saved


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = RunConfig.from_yaml(args.config)
    out_dir = args.out_dir or f"{config.name}_output"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[raymc] building '{config.name}' from {args.config}...")
    system, move_set, chi_sq, engine = build_run(config, log_every=args.log_every or None)
    print(
        f"[raymc] {system.num_nodes} atoms, {len(chi_sq.terms)} chi-squared term(s), "
        f"{config.engine.n_conv_steps} max convergence steps"
        + (
            f" + up to {config.engine.production.n_moves} production steps"
            if config.engine.production is not None
            else ""
        )
    )

    resumed = None
    if args.resume:
        system, resumed = _load_checkpoint(out_dir)
        steps_done = int(resumed["steps_done"])
        engine.n_conv_steps = max(config.engine.n_conv_steps - steps_done, 0)
        print(
            f"[raymc] resuming from {steps_done} completed convergence steps; "
            f"{engine.n_conv_steps} to go"
        )

    checkpoint = None
    if args.checkpoint_every:
        # Count from where the chain actually is, not from where this
        # process started, so a run resumed twice still knows how many
        # convergence steps it has really done.
        already_done = 0 if resumed is None else int(resumed["steps_done"])

        def checkpoint(steps_done, state, history):
            _write_checkpoint(
                out_dir, already_done + steps_done, state, history, resumed
            )

    t0 = time.time()
    result = engine.run(
        system,
        convergence=config.engine.convergence,
        production=config.engine.production,
        log_every=args.log_every or None,
        checkpoint=checkpoint,
        checkpoint_every=args.checkpoint_every,
    )
    final_state, history = result.final_state, result.history
    if resumed is not None:
        # Stitch the pre-interruption history back on, so the saved traces
        # and the plot cover the whole chain rather than only its tail.
        history.chi_sqs = list(resumed["history_chi_sqs"]) + history.chi_sqs
        history.accepted = list(resumed["history_accepted"]) + history.accepted
        history.move_names = list(resumed["history_move_names"]) + history.move_names
        for name, trace in resumed["history_term_chi_sqs"].items():
            tail = history.term_chi_sqs.get(name, [])
            history.term_chi_sqs[name] = list(trace) + tail
    elapsed = time.time() - t0
    n_total_moves = len(history.accepted)
    print(f"[raymc] done in {elapsed:.1f}s ({elapsed / max(n_total_moves, 1) * 1000:.2f} ms/step)")
    print()
    print(summarize_history(history))
    print(f"\n[raymc] max Bragg S(k) at end of run: {bragg_peak_max_sk(final_state):.1f}")

    breakdown = chi_sq.breakdown(final_state)
    report = chi_sq.report(final_state)
    print(_format_breakdown(chi_sq, breakdown, report))

    if config.engine.convergence is not None:
        print(
            f"\n[raymc] convergence: "
            f"{'reached at move ' + str(result.convergence_step) if result.converged else 'NOT reached'}"
        )

    if result.saved_configs:
        print(f"[raymc] production: {len(result.saved_configs)} configs saved")
        for term in chi_sq.terms:
            if term.name not in result.averaged_raw:
                continue
            avg_report = term.report_from_raw(result.averaged_raw[term.name])
            snap_report = term.report_from_raw(result.final_snapshot_raw[term.name])
            print(
                f"[raymc] '{term.name}' averaged-over-{len(result.saved_configs)}-configs "
                f"chi^2/n_r={avg_report['chi2_per_nr']:.4f} Rw={avg_report['Rw']:.4f}  "
                f"(final snapshot alone: chi^2/n_r={snap_report['chi2_per_nr']:.4f} "
                f"Rw={snap_report['Rw']:.4f})"
            )

        traj_path = os.path.join(out_dir, config.engine.production.trajectory_file)
        images = [_to_ase(s) for s in result.saved_configs]
        ase_write(traj_path, images)
        print(f"[raymc] wrote {len(images)}-config production trajectory to {traj_path}")

    xyz_path = os.path.join(out_dir, "final_structure.xyz")
    ase_write(xyz_path, _to_ase(final_state))
    print(f"[raymc] wrote final structure to {xyz_path}")

    results_path = os.path.join(out_dir, "results.pt")
    torch.save(
        {
            "history_chi_sqs": torch.tensor(history.chi_sqs),
            "history_accepted": torch.tensor(history.accepted),
            # Per-term traces, not just the total. The total alone cannot
            # tell which term is improving and which is paying for it --
            # and for a term whose penalty is an invertible function of a
            # physical quantity this trace IS that quantity's trajectory,
            # recorded for free. Saved as plain lists rather
            # than tensors so a term that appeared partway through keeps
            # its NaN padding intact.
            "history_term_chi_sqs": dict(history.term_chi_sqs),
            "breakdown": breakdown,
        },
        results_path,
    )
    print(f"[raymc] wrote chi-squared history to {results_path}")

    if not args.no_plot:
        _write_plot(out_dir, chi_sq, final_state, history, config, result)

    return 0


def _panel_axes(calculator):
    """Per-modality plot axes: grid, labels, and x limits."""
    from raymc.modalities.sq.calculator import SQCalculator

    if isinstance(calculator, SQCalculator):
        q = calculator.q_grid
        ylabel = "S(Q)" if calculator.convention == "S" else "F(Q)"
        return q, "Q (1/A)", ylabel, (float(q.min()), float(q.max()))

    # PDF: 1 A margin below r_min into the excluded-volume region,
    # matching McGreevy & Pusztai Fig. 1 -- g(r) is identically zero
    # below r_min by construction (hard min-dist constraint), so this
    # shows the excluded-volume onset rather than just cropping to the
    # computed grid.
    return (
        calculator.r_grid,
        "r (A)",
        "g(r)",
        (calculator.r_min - 1.0, calculator.r_max),
    )


def _format_breakdown(chi_sq, breakdown, report) -> str:
    """Per-term chi-squared as a table, with each term's realized share.

    A raw dict of tensors is readable for one term and noise for four,
    and the share column is what makes a modality quietly dominating a
    combined fit visible at a glance rather than by arithmetic.
    """
    rows = []
    weighted = {
        term.name: term.weight * float(breakdown[term.name]) for term in chi_sq.terms
    }
    total = sum(weighted.values()) or 1.0
    for term in chi_sq.terms:
        raw = float(breakdown[term.name])
        per_term = report.get(term.name, {})
        rows.append(
            f"  {term.name:<18}{term.weight:>7.2f}{raw:>14.4g}"
            f"{weighted[term.name]:>14.4g}{100 * weighted[term.name] / total:>7.1f}%"
            f"{per_term.get('chi2_per_nr', float('nan')):>12.4f}"
            f"{per_term.get('Rw', float('nan')):>10.4f}"
        )
    head = (
        f"\nFinal chi-squared breakdown:\n"
        f"  {'term':<18}{'weight':>7}{'chi2':>14}{'weighted':>14}{'share':>8}"
        f"{'chi2/n_r':>12}{'Rw':>10}\n"
        f"  {'-' * 73}"
    )
    tail = f"  {'-' * 73}\n  {'TOTAL':<18}{'':>7}{'':>14}{total:>14.4g}"
    return "\n".join([head, *rows, tail])


def _safe_name(label: str) -> str:
    """A term label reduced to something safe to put in a filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label)


def _write_plot(
    out_dir: str, chi_sq, final_state, history, config: RunConfig, result=None
) -> None:
    """Write one figure per experimental modality, plus one for the
    chi-squared history.

    Separate files rather than stacked panels: a combined X-ray + neutron
    run has as many fit panels as data sets, and a single tall figure gets
    unreadable at exactly the point it starts being interesting.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[raymc] matplotlib not installed -- skipping plot (pass --no-plot to silence this)")
        return

    from raymc.modalities.pdf.calculator import PDFCalculator
    from raymc.modalities.sq.calculator import SQCalculator

    data_terms = [
        term
        for term in chi_sq.terms
        if isinstance(term.calculator, (PDFCalculator, SQCalculator))
        and term.target is not None
    ]
    written: list[str] = []

    for term in data_terms:
        grid, xlabel, ylabel, xlim = _panel_axes(term.calculator)
        fig, ax = plt.subplots(figsize=(8, 4.5))

        model_final = term.calculator.compute(final_state)
        # term.target already carries the modality's declared data_scale /
        # data_offset, applied once at load -- so this is exactly the
        # curve the fit is driven against, with nothing left to reconcile.
        ax.plot(grid, term.target, label="target", color="black", lw=1.2)

        averaged = result.averaged_raw.get(term.name) if result is not None else None
        if averaged is not None:
            ax.plot(
                grid,
                averaged,
                label=f"model (averaged over {len(result.saved_configs)} saved configs)",
                color="tab:blue",
                lw=2.0,
            )
            ax.plot(
                grid,
                model_final,
                label="final snapshot (reference)",
                color="tab:orange",
                lw=0.9,
                ls="--",
                alpha=0.7,
            )
        else:
            ax.plot(
                grid,
                model_final,
                label=f"final ({len(history.chi_sqs)} moves)",
                color="tab:blue",
                lw=1.4,
            )

        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        report = term.report_from_raw(model_final)
        ax.set_title(f"{term.name}  (Rw = {report['Rw']:.4f}, weight {term.weight:g})")
        ax.legend(fontsize=8)
        fig.tight_layout()

        path = os.path.join(out_dir, f"fit_{_safe_name(term.name)}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(history.chi_sqs, color="black", lw=1.0, label="total (weighted)")
    # Per-term traces alongside the total: in a combined fit the total can
    # fall while one data set climbs, and only the split shows it.
    for name, trace in history.term_chi_sqs.items():
        if any(v > 0 for v in trace):
            ax.semilogy(trace, lw=0.7, alpha=0.85, label=f"{name} (unweighted)")
    ax.set_xlabel("step")
    ax.set_ylabel("chi-squared (log scale)")
    ax.set_title(f"Chi-squared history (acceptance: {history.acceptance_rate:.1%})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(out_dir, "chi_history.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    print(f"[raymc] wrote {len(written)} plots:")
    for path in written:
        print(f"          {path}")


if __name__ == "__main__":
    sys.exit(main())
