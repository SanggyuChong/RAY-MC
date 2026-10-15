"""Abstract move interface shared by every move category.

:mod:`raymc.moves.group_based` implements this today; future move
categories -- species-swap, torsional rotation -- implement it the same
way, as sibling modules. A solver never imports a concrete move: it only
holds a :class:`~raymc.moves.moveset.MoveSet` and calls it generically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from raymc.system import AtomicSystem

_EMPTY_INDICES = torch.empty(0, dtype=torch.long)


class MoveNotApplicable(Exception):
    """Raised by :meth:`Move.attempt` when the move cannot currently be
    applied to the given system (e.g. removing a molecule from a system
    that has none)."""


@dataclass(frozen=True)
class MoveDelta:
    """An effective "delta" of which atoms an RMC move has actually affected."""

    added_indices: torch.Tensor = field(default_factory=lambda: _EMPTY_INDICES)
    """Indices into the NEW system: atoms that did not exist before."""

    removed_indices: torch.Tensor = field(default_factory=lambda: _EMPTY_INDICES)
    """Indices into the OLD system: atoms that no longer exist."""

    moved_indices: torch.Tensor = field(default_factory=lambda: _EMPTY_INDICES)
    """Indices valid in BOTH systems (same index, same atom identity,
    since a move never reorders or resizes around these atoms) whose
    position changed."""


class Move(ABC):
    """One Monte Carlo move type.

    Attributes
    ----------
    name : str
        Identifier used in YAML run configs and move-history logging.
    """

    name: str

    @abstractmethod
    def attempt(
        self, system: AtomicSystem, generator: torch.Generator
    ) -> tuple[AtomicSystem, MoveDelta]:
        """Return a new candidate system with this move applied, and a
        :class:`MoveDelta` describing exactly which atoms changed.

        Must not mutate *system*, since the move can be rejected.

        Raises
        ------
        MoveNotApplicable
            If this move cannot be applied to *system* right now (should
            not happen if :meth:`is_applicable` was checked first).
        """
        raise NotImplementedError

    def is_applicable(self, system: AtomicSystem) -> bool:
        """Cheap, side-effect-free check of whether this move could
        currently apply to *system*, without actually attempting one.

        :meth:`~raymc.moves.moveset.MoveSet.sample` uses this to filter
        the weighted draw down to only the moves that could possibly
        succeed, rather than drawing and retrying. Defaults to always
        applicable; override for a move that can be transiently
        unavailable (e.g. ``RemoveGroup`` on an empty system).
        """
        return True
