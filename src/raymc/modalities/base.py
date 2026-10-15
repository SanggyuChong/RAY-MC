"""Blueprint for the calculators for each RMC data modality.

A :class:`Calculator` provides:

- ``compute(system)``: given an AtomicSystem, its associated observable
  for this modality/constraint/energetics term (a curve, a scalar).
  Required, and must be side-effect-free -- callers (e.g.
  :meth:`raymc.residuals.aggregator.ChiSqAggregator.breakdown`) may call
  it for inspection at any time without disturbing a running engine.
- ``compute_increm_change(system, delta)``: optional fast path -- the
  same kind of value as ``compute``, updated from a
  :class:`~raymc.moves.base.MoveDelta` against whatever state was last
  seeded/committed, instead of recomputed from the whole system. Default
  implementation just calls ``compute(system)``, so a calculator with no
  incremental path needs to override nothing.
- ``init``/``commit``/``discard``: the small state lifecycle
  ``compute_increm_change`` needs when it actually keeps a running cache
  (see :class:`~raymc.modalities.pdf.calculator.PDFCalculator`'s
  self-partial case) -- seed the cache, keep a proposed change, or throw
  it away. All three default to a no-op (or, for ``init``, to
  ``compute``), so a stateless calculator overrides none of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from raymc.moves.base import MoveDelta
from raymc.system import AtomicSystem


class Calculator(ABC):
    """Turns an :class:`~raymc.system.AtomicSystem` into a raw observable
    value for one chi-squared term."""

    @abstractmethod
    def compute(self, system: AtomicSystem) -> torch.Tensor:
        """Full, stateless computation of the observable for *system*."""
        raise NotImplementedError

    def init(self, system: AtomicSystem) -> torch.Tensor:
        """Seed any running state :meth:`compute_increm_change` needs, and
        return the same value :meth:`compute` would. Default: just
        ``compute(system)`` -- override only if there's real state to seed."""
        return self.compute(system)

    def compute_increm_change(
        self, system: AtomicSystem, delta: MoveDelta
    ) -> torch.Tensor:
        """Optional fast path: the candidate value after applying *delta*
        to whatever state was last seeded (:meth:`init`) or kept
        (:meth:`commit`), without recomputing from the whole system.

        Default falls back to a full :meth:`compute` -- override for a
        real incremental update.
        """
        return self.compute(system)

    def commit(self) -> None:
        """Keep the last :meth:`compute_increm_change` trial as the new
        committed state. No-op default for a calculator with no running
        state to keep."""

    def discard(self) -> None:
        """Throw away the last :meth:`compute_increm_change` trial. No-op
        default, see :meth:`commit`."""
