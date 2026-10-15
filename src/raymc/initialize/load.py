"""Loading a user-supplied structure file as the initial system."""

from __future__ import annotations

from pathlib import Path

from ase.io import read

from raymc.system import AtomicSystem, check_no_unphysical_clash


def load_structure(path: str | Path) -> AtomicSystem:
    """Load a structure file (any ASE compatible format, e.g. XYZ, CIF) as
    the initial :class:`~raymc.system.AtomicSystem` for a run.

    No rigid grouping is inferred yet (see the TODO below), so
    :func:`~raymc.system.check_no_atomic_overlap`'s same-rigid-group
    exclusion can't help here -- every atom is trivially its own group,
    which would make even an ordinary bond look like an overlap under a
    min-dist table. This validates with
    :func:`~raymc.system.check_no_unphysical_clash` instead, which uses
    covalent radii and needs no grouping information to tell a real bond
    from unphysical overlap.

    TODO: infer rigid groups (molecule detection) from the loaded
    structure, so this can validate with the same user-configured
    :class:`~raymc.utils.MinDistTable` the initializer's other strategies
    and the RMC engine use, instead of the covalent-radius fallback.

    Parameters
    ----------
    path : str | Path
        Path to the structure file.

    Returns
    -------
    AtomicSystem

    Raises
    ------
    ValueError
        If the loaded structure contains an unphysical atomic clash --
        see :func:`~raymc.system.check_no_unphysical_clash`.
    """
    atoms = read(str(path))
    system = AtomicSystem.from_atoms(atoms)
    check_no_unphysical_clash(system)
    return system
