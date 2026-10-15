"""Main reverse Monte Carlo (RMC) engine.

Runs the RMC simulation using the Metropolis algorithm:
draw an RMC "move", score the new state with
:class:`~raymc.residuals.aggregator.ChiSqAggregator`, then decide
to accept or reject.

Also comes with an optional convergence detection feature. Engine
enforces a production phase, from which multiple states are sampled
and the average statistics are reported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

from raymc.moves.moveset import MoveSet
from raymc.residuals.aggregator import ChiSqAggregator
from raymc.system import AtomicSystem

if TYPE_CHECKING:
    from raymc.config import ConvergenceConfig, ProductionConfig


@dataclass
class RunHistory:
    """Per-step bookkeeping for one Metropolis run."""

    accepted: list[bool] = field(default_factory=list)
    move_names: list[str] = field(default_factory=list)
    chi_sqs: list[float] = field(default_factory=list)
    """Weighted total driving the Metropolis decision, one entry per step."""

    term_chi_sqs: dict[str, list[float]] = field(default_factory=dict)
    """Per-term unweighted chi-squared, same length and alignment as
    :attr:`chi_sqs`. The total alone cannot tell a combined X-ray +
    neutron fit apart from a single-modality one: it falls either way.
    These traces show which data set is actually improving, and whether
    the other is paying for it."""

    def record_terms(self, values: dict[str, float]) -> None:
        """Append one step's per-term values, keeping every trace the
        same length even if a term appears partway through."""
        for name, value in values.items():
            trace = self.term_chi_sqs.setdefault(name, [])
            # pad a term first seen mid-run so index i means step i everywhere
            while len(trace) < len(self.chi_sqs) - 1:
                trace.append(float("nan"))
            trace.append(value)

    @property
    def acceptance_rate(self) -> float:
        if not self.accepted:
            return 0.0
        return sum(self.accepted) / len(self.accepted)


def summarize_history(history: RunHistory) -> str:
    """Print a summary of the RMC run: overall acceptance,
    per-move-type attempt & acceptance rate."""
    total_steps = len(history.accepted)
    total_accepted = sum(history.accepted)
    overall_rate = total_accepted / total_steps if total_steps else 0.0

    attempted: Counter[str] = Counter()
    accepted_by_move: Counter[str] = Counter()
    for name, acc in zip(history.move_names, history.accepted):
        attempted[name] += 1
        if acc:
            accepted_by_move[name] += 1

    lines = [
        f"Overall acceptance: {total_accepted}/{total_steps} ({overall_rate:.1%})",
        "Per-move breakdown:",
    ]
    # Width from the actual names rather than a fixed 12, so a longer one
    # ("translate_atom") doesn't push every column on its row out of line.
    width = max((len(name) for name in attempted), default=0)
    for name, count in sorted(attempted.items(), key=lambda kv: -kv[1]):
        share = count / total_steps if total_steps else 0.0
        move_rate = accepted_by_move[name] / count if count else 0.0
        lines.append(
            f"  {name:<{width}} attempted {count:>6} ({share:>5.1%} of all steps), "
            f"accepted {accepted_by_move[name]:>6} ({move_rate:.1%})"
        )
    return "\n".join(lines)


@dataclass
class ProductionResult:
    """Output of :meth:`ReverseMonteCarlo.run`."""

    final_state: AtomicSystem
    history: RunHistory
    converged: bool
    convergence_step: int | None
    saved_configs: list[AtomicSystem]
    averaged_raw: dict[str, torch.Tensor]
    final_snapshot_raw: dict[str, torch.Tensor]


def _raise_if_not_finite(chi_sq: torch.Tensor) -> None:
    if not torch.isfinite(chi_sq):
        raise RuntimeError(
            "Chi-squared is not finite most likely due to a hard constraint" \
            "kicking in at step 0. Fix the initial structure and rerun the " \
            "simulation, if possible."
        )


class ReverseMonteCarlo:
    """Main RMC engine class.

    Parameters
    ----------
    move_set : MoveSet
        Where candidate moves come from -- see
        :meth:`~raymc.moves.moveset.MoveSet.from_config`.
    chi_sq : ChiSqAggregator
        Aggregator handling the chi-squared computation. -- see
        :class:`raymc.residuals.aggregator.ChiSqAggregator`. Lower is
        better. Its terms' values are already scaled by their own
        ``sigma`` (see :class:`raymc.residuals.aggregator.ChiSqTerm`),
        so the Metropolis criterion here needs no extra temperature
        knob -- see McGreevy & Pusztai (1988), Mol. Sim. 1, 359: acceptance
        ``= min(1, exp(-(chi'^2 - chi^2) / 2))``.
    n_steps : int
        Number of RMC steps to run.
    seed : int | None
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        move_set: MoveSet,
        chi_sq_agg: ChiSqAggregator,
        n_conv_steps: int,
        seed: int,
    ) -> None:
        self.move_set = move_set
        self.chi_sq_agg = chi_sq_agg
        self.n_conv_steps = n_conv_steps
        self.generator = torch.Generator().manual_seed(seed)

    def run_convergence(
        self,
        initial_state: AtomicSystem,
        convergence: "ConvergenceConfig | None" = None,
        log_every: int | None = None,
        checkpoint: "Callable[[int, AtomicSystem, RunHistory], None] | None" = None,
        checkpoint_every: int | None = None,
    ) -> tuple[AtomicSystem, AtomicSystem, RunHistory, bool, int | None]:
        """Run the convergence phase of RMC simulation, up to
        :attr:`n_conv_steps`. If *convergence* is given, may stop early --
        chi-squared is averaged into consecutive non-overlapping windows of
        ``convergence.window`` steps, and convergence is declared once the
        fractional change between consecutive window averages stays below
        ``convergence.threshold`` for ``convergence.patience`` windows in a
        row.

        Parameters
        ----------
        initial_state : AtomicSystem
            Starting atomic system.
        convergence : ConvergenceConfig | None
            Optional early-stop detection -- see above.
        log_every : int | None
            If set, print progress: current step, current chi-squared,
            current acceptance ratio.
        checkpoint : callable | None
            Called with ``(steps_done, state, history)`` every
            *checkpoint_every* steps. The engine says nothing about what a
            checkpoint IS -- persisting it is the caller's business (see
            :mod:`raymc.cli`) -- it only provides the hook, because this
            loop is the only place that knows when a step has finished
            cleanly.

            Worth having for any run long enough that losing it hurts. An
            RMC chain is memoryless given its current configuration, so a
            saved configuration is a valid place to continue from: a
            restart is a genuine continuation of the chain, not an
            approximation to one. What it does NOT preserve is the
            generator stream, so a resumed run is not bit-for-bit
            reproducible against an uninterrupted one -- statistically
            equivalent, not identical.
        checkpoint_every : int | None
            Steps between *checkpoint* calls. Ignored without *checkpoint*.

        Returns
        -------
        tuple[AtomicSystem, AtomicSystem, RunHistory, bool, int | None]
            Final atomic system (the actual trajectory endpoint), the
            lowest-chi-squared atomic system visited along the way, run
            history, whether convergence was declared, and the step it was
            declared at (``None`` if never declared).
        """
        state = initial_state
        best_state = initial_state
        history = RunHistory()

        current_chi_sq = self.chi_sq_agg.init(state)
        _raise_if_not_finite(current_chi_sq)
        best_chi_sq = current_chi_sq

        window_chi_sqs: list[float] = []
        last_window_mean: float | None = None
        streak = 0
        convergence_step: int | None = None
        converged = False

        for step in range(self.n_conv_steps):
            state, current_chi_sq, move_name, accepted = self._step(
                state, current_chi_sq
            )

            if current_chi_sq < best_chi_sq:
                best_chi_sq = current_chi_sq
                best_state = state

            history.accepted.append(accepted)
            history.move_names.append(move_name)
            history.chi_sqs.append(float(current_chi_sq))
            history.record_terms(self.chi_sq_agg.current_values())

            if log_every and (step + 1) % log_every == 0:
                running_acc = sum(history.accepted) / len(history.accepted)
                print(
                    f"[step {step + 1:>6}/{self.n_conv_steps}] "
                    f"current chi sq. = {float(current_chi_sq):.4f}  "
                    f"running acceptance = {running_acc:.1%}"
                )

            if checkpoint is not None and checkpoint_every:
                if (step + 1) % checkpoint_every == 0:
                    checkpoint(step + 1, state, history)

            if convergence is not None:
                window_chi_sqs.append(float(current_chi_sq))
                if len(window_chi_sqs) == convergence.window:
                    window_mean = sum(window_chi_sqs) / len(window_chi_sqs)
                    window_chi_sqs = []
                    if last_window_mean is not None:
                        denom = abs(last_window_mean) if last_window_mean != 0 else 1.0
                        frac_change = abs(window_mean - last_window_mean) / denom
                        if frac_change < convergence.threshold:
                            streak += 1
                            if streak >= convergence.patience:
                                convergence_step = step + 1
                                converged = True
                                print(
                                    f"[raymc] chi-squared converged at move "
                                    f"{convergence_step} (fractional change "
                                    f"< {convergence.threshold} for "
                                    f"{convergence.patience} consecutive "
                                    f"{convergence.window}-step windows)"
                                )
                                break
                        else:
                            streak = 0
                    last_window_mean = window_mean

        return state, best_state, history, converged, convergence_step

    def _step(
        self, state: AtomicSystem, current_chi_sq: torch.Tensor
    ) -> tuple[AtomicSystem, torch.Tensor, str, bool]:
        """One MC step: draw a candidate, score it, accept/reject."""

        candidate, move_name, delta = self.move_set.sample(state, self.generator)
        candidate_chi_sq = self.chi_sq_agg.evaluate(candidate, delta)

        del_chi_sq = candidate_chi_sq - current_chi_sq
        accept_prob = torch.clamp(torch.exp(-del_chi_sq / 2.0), max=1.0)
        accepted = bool(torch.rand((), generator=self.generator) < accept_prob)
        self.chi_sq_agg.accept(accepted)

        if accepted:
            return candidate, candidate_chi_sq, move_name, True
        return state, current_chi_sq, move_name, False

    def run(
        self,
        initial_state: AtomicSystem,
        convergence: "ConvergenceConfig | None" = None,
        production: "ProductionConfig | None" = None,
        log_every: int | None = None,
        checkpoint: "Callable[[int, AtomicSystem, RunHistory], None] | None" = None,
        checkpoint_every: int | None = None,
    ) -> ProductionResult:
        """Convergence phase then production phase -- see
        :meth:`run_convergence` for *checkpoint*/*checkpoint_every*, which
        are passed through to it. The production phase is not
        checkpointed: it is the short tail of a run, and it saves its own
        configurations as it goes."""
        (
            final_state,
            best_state,
            history,
            converged,
            convergence_step,
        ) = self.run_convergence(
            initial_state, convergence, log_every, checkpoint, checkpoint_every
        )

        if production is None:
            return ProductionResult(
                final_state=final_state,
                history=history,
                converged=converged,
                convergence_step=convergence_step,
                saved_configs=[],
                averaged_raw={},
                final_snapshot_raw={},
            )

        state = best_state if production.restart_from == "best" else final_state
        if production.restart_from == "best":
            print(
                f"[raymc] production restarting from the best convergence-phase "
                f"state, not its final one"
            )
        # Always a full recompute (never the incremental path) -- whichever
        # state production continues from must be "wholly saved" and
        # internally consistent, since the aggregator's own incremental
        # caches are still synced to the convergence phase's actual last
        # step, not necessarily to `state` here.
        current_chi_sq = self.chi_sq_agg.init(state)

        data_terms = [t for t in self.chi_sq_agg.terms if t.target is not None]
        save_interval = production.save_interval or max(
            1, production.n_moves // production.n_configs
        )
        saved_configs: list[AtomicSystem] = []
        accum_raw = {t.name: torch.zeros_like(t.target) for t in data_terms}

        prod_step = 0
        while prod_step < production.n_moves and len(saved_configs) < production.n_configs:
            state, current_chi_sq, move_name, accepted = self._step(
                state, current_chi_sq
            )
            prod_step += 1
            history.accepted.append(accepted)
            history.move_names.append(move_name)
            history.chi_sqs.append(float(current_chi_sq))
            history.record_terms(self.chi_sq_agg.current_values())

            if prod_step % save_interval == 0:
                saved_configs.append(state)
                for term in data_terms:
                    accum_raw[term.name] = accum_raw[term.name] + term.calculator.compute(state)

            if log_every and prod_step % log_every == 0:
                print(
                    f"[production step {prod_step:>6}/{production.n_moves}] "
                    f"saved {len(saved_configs)}/{production.n_configs} configs"
                )

        n_saved = max(len(saved_configs), 1)
        averaged_raw = {name: accum_raw[name] / n_saved for name in accum_raw}
        final_snapshot_raw = {term.name: term.calculator.compute(state) for term in data_terms}

        return ProductionResult(
            final_state=state,
            history=history,
            converged=converged,
            convergence_step=convergence_step,
            saved_configs=saved_configs,
            averaged_raw=averaged_raw,
            final_snapshot_raw=final_snapshot_raw,
        )
