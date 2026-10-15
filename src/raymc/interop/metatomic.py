"""Shared metatomic interop: turning an :class:`AtomicSystem` into the
``System`` objects a metatomic model expects, with its neighbor lists
attached.

Metatomic-backed subsystems such as :mod:`raymc.energetics.metatomic` go
through here, so a system is converted the same way whichever model is
about to see it.

Requires the "metatomic" extra: ``pip install raymc[metatomic]``.
"""

from __future__ import annotations

import torch

from raymc.system import AtomicSystem

try:
    import metatomic.torch as mta
    from vesin.metatomic import neighbor_lists_for_model
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "raymc.interop.metatomic requires the 'metatomic' extra: "
        "pip install raymc[metatomic]"
    ) from exc


def to_metatomic_system(
    system: AtomicSystem, dtype: torch.dtype = torch.float32
) -> "mta.System":
    """Convert an :class:`AtomicSystem` to a metatomic ``System``.

    A metatomic model wants floating-point positions/cell (*dtype*,
    float32 by default) and ``int32`` types; a non-periodic system (no
    cell/pbc) is represented as an all-zero cell with no periodic
    directions, matching metatomic's own convention.
    """
    positions = system.positions.to(dtype)
    types = system.atomic_numbers.to(torch.int32)
    if system.cell is not None:
        cell = system.cell.squeeze(0).to(dtype)
    else:
        cell = torch.zeros((3, 3), dtype=dtype)
    if system.pbc is not None:
        pbc = system.pbc.squeeze(0)
    else:
        pbc = torch.zeros(3, dtype=torch.bool)

    # TODO: pass a precomputed neighbor list through instead of always
    # recomputing one via add_neighbor_list -- needs checking whether
    # metatomic/vesin actually support an incremental/reused neighbor list
    # across calls with slightly moved positions; not verified, so left as
    # a real recompute for now.
    return mta.System(types=types, positions=positions, cell=cell, pbc=pbc)


def neighbor_list_calculators(model: "mta.AtomisticModel") -> list:
    """Neighbor-list calculators satisfying *model*'s declared requirements.

    Depends only on the model (its cutoffs), not on any particular
    system, so a caller builds these once and reuses them for every
    evaluation.
    """
    return neighbor_lists_for_model("angstrom", model, model_length_unit="angstrom")


def attach_neighbor_lists(calculators: list, systems: list["mta.System"]) -> None:
    """Attach every calculator's neighbor list to every system, in place."""
    for calculator in calculators:
        calculator.add_neighbor_list(systems)
