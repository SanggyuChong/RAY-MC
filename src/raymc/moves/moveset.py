"""The collection of active moves the engine samples at each RMC step.

This is the object the engine actually interacts with. The engine calls
:meth:`MoveSet.sample`, which handles which move to draw and attempt.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from raymc.moves.base import Move, MoveDelta
from raymc.system import AtomicSystem

if TYPE_CHECKING:
    from raymc.config import ConstraintConfig, MoveConfig

# Name -> constructor, resolving a move config's ``params`` dict into a
# concrete Move instance. New move categories (species-swap, torsion)
# register here too when they're implemented; nothing outside this file
# needs to change to add one.
_MOVE_REGISTRY: dict[str, "type"] = {}


@dataclass(frozen=True)
class MoveBuildContext:
    """Everything a move might need at construction time that isn't in its
    own ``params``: run-level state declared once in the config rather
    than repeated in every move's params, where two moves could disagree
    about it. Empty for the built-in moves, none of which need any; it is
    the hook a registered factory takes as its second argument.
    """


def register_move(name: str, factory) -> None:
    """Register a move-config factory under *name* for
    :meth:`MoveSet.from_config` to look up.

    Parameters
    ----------
    name : str
        The ``type`` string used in YAML run configs.
    factory : Callable[[dict], Move] | Callable[[dict, MoveBuildContext], Move]
        Builds a :class:`Move` instance from a config's ``params`` dict,
        and optionally from the run-level :class:`MoveBuildContext`. Both
        arities are accepted: a move that needs nothing beyond its own
        params says so by taking one argument.
    """
    _MOVE_REGISTRY[name] = factory


def _build(factory, params: dict, context: MoveBuildContext) -> Move:
    """Call a registered factory with whichever arity it declared."""
    try:
        n_params = len(inspect.signature(factory).parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins, C callables
        n_params = 1
    return factory(params, context) if n_params >= 2 else factory(params)


class MoveSet:
    """A weighted collection of :class:`Move` instances.

    Parameters
    ----------
    moves : list[Move]
    weights : list[float]
        Relative selection probability for each move (need not sum to 1).
    """

    def __init__(self, moves: list[Move], weights: list[float]):
        if len(moves) != len(weights):
            raise ValueError("moves and weights must be the same length")
        if not moves:
            raise ValueError("MoveSet requires at least one move")
        self.moves = moves
        self.weights = torch.tensor(weights, dtype=torch.float64)

    def sample(
        self, system: AtomicSystem, generator: torch.Generator
    ) -> tuple[AtomicSystem, str, MoveDelta]:
        """Weighted-randomly pick a currently-applicable move and apply it.

        Rather than drawing from every registered move and retrying on a
        transiently-inapplicable one (e.g. "remove" on a system with no
        rigid groups), this first filters to the moves whose
        :meth:`~raymc.moves.base.Move.is_applicable` says yes for
        *system*, then draws (weighted) from just that subset -- so the
        result is deterministic given the system's state and RNG draw,
        with no retry loop and no silent cap on how many draws get tried.

        Parameters
        ----------
        system : AtomicSystem
            Current state to draw a candidate from.
        generator : torch.Generator

        Returns
        -------
        tuple[AtomicSystem, str, MoveDelta]
            The candidate system, the name of the move that produced it,
            and which atoms it touched (see :class:`MoveDelta`).

        Raises
        ------
        RuntimeError
            If no registered move is currently applicable to *system*.
        """
        applicable = [
            (move, weight)
            for move, weight in zip(self.moves, self.weights.tolist())
            if move.is_applicable(system)
        ]
        if not applicable:
            raise RuntimeError(
                f"No applicable move for the current system state "
                f"(registered moves: {[m.name for m in self.moves]})"
            )

        moves, weights = zip(*applicable)
        idx = int(
            torch.multinomial(
                torch.tensor(weights, dtype=torch.float64), 1, generator=generator
            ).item()
        )
        move = moves[idx]
        candidate, delta = move.attempt(system, generator)
        return candidate, move.name, delta

    @classmethod
    def from_config(
        cls,
        move_configs: list["MoveConfig"],
        constraints: "ConstraintConfig | None" = None,
    ) -> MoveSet:
        """Build a :class:`MoveSet` from a run config's ``moves`` section.

        Parameters
        ----------
        move_configs : list[MoveConfig]
            E.g. ``RunConfig.moves`` from a loaded YAML file.
        constraints : ConstraintConfig | None
            The run's ``constraints`` section, used only to validate that
            the chosen moves are actually safe to run against it -- see
            :func:`_warn_about_unrestrained_atomic_moves`. Optional so
            that a caller assembling a move set by hand (tests, scripts)
            need not have one; the check is simply skipped then.

        Returns
        -------
        MoveSet

        Raises
        ------
        ValueError
            If a config references an unregistered move ``type``.
        """
        _ensure_builtin_moves_registered()
        context = MoveBuildContext()

        moves, weights = [], []
        for cfg in move_configs:
            if cfg.type not in _MOVE_REGISTRY:
                raise ValueError(
                    f"Unknown move type {cfg.type!r}; registered types: "
                    f"{sorted(_MOVE_REGISTRY)}"
                )
            moves.append(_build(_MOVE_REGISTRY[cfg.type], cfg.params, context))
            weights.append(cfg.weight)

        if constraints is not None:
            _warn_about_unrestrained_atomic_moves(moves, constraints)
        return cls(moves, weights)


def _ensure_builtin_moves_registered() -> None:
    """Register RAY-MC's built-in group-based moves, once, lazily.

    Deferred import avoids a hard import-time dependency from this
    generic module onto a specific move category.
    """
    if "translate_group" in _MOVE_REGISTRY:
        return

    from raymc.initialize.templates import load_group_template
    from raymc.moves.atomic import TranslateAtom
    from raymc.moves.group_based import (
        InsertGroup,
        RelocateGroup,
        RemoveGroup,
        RotateGroup,
        TranslateGroup,
    )

    register_move(
        "translate_group",
        lambda p: TranslateGroup(max_displacement=p["max_displacement"]),
    )
    register_move(
        "translate_atom",
        lambda p: TranslateAtom(max_displacement=p["max_displacement"]),
    )
    register_move("rotate_group", lambda p: RotateGroup(max_angle=p["max_angle"]))
    register_move(
        "insert_group",
        lambda p: InsertGroup(template=load_group_template(p["molecule"])),
    )
    register_move("remove_group", lambda p: RemoveGroup())
    register_move(
        "relocate_group",
        lambda p: RelocateGroup(template=load_group_template(p["molecule"])),
    )


def _warn_about_unrestrained_atomic_moves(
    moves: list[Move], constraints: "ConstraintConfig"
) -> None:
    """Warn when a move that displaces atoms individually is active but
    nothing restrains a molecule's own internal geometry.

    This combination is silently destructive rather than loudly broken.
    The min- and max-distance constraints skip same-group pairs by
    construction (see :mod:`raymc.constraints.pairwise`), so with no
    intramolecular window an O-H bond is scored by nothing at all: it can
    collapse to zero or stretch across the box, molecules dissolve into
    loose atoms, and the only symptom is a strange g(r) at the end of a
    run that otherwise looks healthy.

    A warning rather than an error: it is perfectly legitimate on a
    monatomic system, where every group is one atom and there is no
    internal geometry to hold together.

    """
    from raymc.moves.atomic import TranslateAtom
    from raymc.utils import IntraWindowTable

    atomic_moves = [
        m.name for m in moves if isinstance(m, TranslateAtom)
    ]
    if not atomic_moves:
        return
    if not IntraWindowTable.from_config(constraints).is_unconstrained():
        return

    warnings.warn(
        "\n"
        "    ====================================================================\n"
        "    single-atom moves are active with NO intramolecular constraint\n"
        "    ====================================================================\n"
        f"    active atomic move(s): {sorted(set(atomic_moves))}\n"
        "    constraints.intramolecular: empty\n"
        "\n"
        "    Nothing is scoring the distances inside a molecule. The min- and\n"
        "    max-distance constraints skip same-group pairs by construction, so\n"
        "    a bond displaced by these moves is restrained by nothing and is\n"
        "    free to collapse or stretch without bound. Molecules will dissolve\n"
        "    into loose atoms, and the only symptom will be a strange g(r).\n"
        "\n"
        "    If the system is monatomic this is fine and expected -- a one-atom\n"
        "    group has no internal geometry to hold together. Otherwise add\n"
        "    windows under constraints.intramolecular, e.g. for water:\n"
        "\n"
        "        constraints:\n"
        "          intramolecular:\n"
        "            - elements: [O, H]\n"
        "              min: 0.9\n"
        "              max: 1.1\n"
        "            - elements: [H, H]\n"
        "              min: 1.5\n"
        "              max: 1.65\n"
        "    ====================================================================",
        UserWarning,
        stacklevel=3,
    )
