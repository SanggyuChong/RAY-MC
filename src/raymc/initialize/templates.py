"""Rigid group templates: what :mod:`raymc.initialize.pack` places and
:mod:`raymc.moves.group_based` inserts/removes as one unit.

A :class:`GroupTemplate` covers both cases uniformly -- a multi-atom rigid
molecule loaded from an XYZ file, or a single free atom -- so packing and
move code never has to branch on "is this an atom or a molecule."
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Union

import periodictable as pt
import torch

_DATABASE_PACKAGE = "raymc.database.molecules"


@dataclass(frozen=True)
class GroupTemplate:
    """A rigid group's reference geometry, centered on its own centroid.

    Represents either a multi-atom molecule (loaded from an XYZ file) or a
    single free atom (``num_atoms == 1``, ``positions == [[0, 0, 0]]``) --
    :mod:`raymc.initialize.pack` and :mod:`raymc.moves.group_based` treat
    both the same way, as "one rigid group to place/insert/remove."

    Attributes
    ----------
    name : str
        Identifying name (a database name for a molecule, or the element
        symbol for a bare atom).
    symbols : tuple[str, ...]
        Element symbols, one per atom, in file order.
    atomic_numbers : torch.Tensor
        Integer atomic numbers, shape [n_atoms].
    positions : torch.Tensor
        Cartesian coordinates (Angstrom), shape [n_atoms, 3], translated so
        the centroid sits at the origin.
    """

    name: str
    symbols: tuple[str, ...]
    atomic_numbers: torch.Tensor
    positions: torch.Tensor

    @property
    def num_atoms(self) -> int:
        return len(self.symbols)

    @classmethod
    def atom(cls, symbol: str) -> "GroupTemplate":
        """Build a single-atom template for element *symbol*, e.g. ``"Na"``."""
        return cls(
            name=symbol,
            symbols=(symbol,),
            atomic_numbers=torch.tensor(
                [pt.elements.symbol(symbol).number], dtype=torch.long
            ),
            positions=torch.zeros((1, 3), dtype=torch.float64),
        )


def _parse_xyz_text(text: str, name: str) -> GroupTemplate:
    lines = text.strip().splitlines()
    n_atoms = int(lines[0].strip())
    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(
            f"Malformed XYZ file for '{name}': header declares {n_atoms} "
            f"atoms but found {len(atom_lines)} atom lines."
        )

    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in atom_lines:
        symbol, x, y, z = line.split()[:4]
        symbols.append(symbol)
        coords.append([float(x), float(y), float(z)])

    positions = torch.tensor(coords, dtype=torch.float64)
    positions = positions - positions.mean(dim=0)
    atomic_numbers = torch.tensor(
        [pt.elements.symbol(s).number for s in symbols], dtype=torch.long
    )

    return GroupTemplate(
        name=name,
        symbols=tuple(symbols),
        atomic_numbers=atomic_numbers,
        positions=positions,
    )


def load_molecule(name_or_path: Union[str, Path]) -> GroupTemplate:
    """Load a rigid molecule template, by database name or by file path.

    Parameters
    ----------
    name_or_path : str | Path
        Either a name in RAY-MC's bundled molecule database (e.g.
        ``"water"``, looked up under ``raymc/database/molecules/``), or a
        path to a user-supplied ``.xyz`` file.

    Returns
    -------
    GroupTemplate

    Raises
    ------
    FileNotFoundError
        If *name_or_path* isn't an existing file and doesn't match a
        database name either.
    """
    path = Path(name_or_path)
    if path.suffix == ".xyz" and path.is_file():
        return _parse_xyz_text(path.read_text(), name=path.stem)

    resource = resources.files(_DATABASE_PACKAGE) / f"{name_or_path}.xyz"
    if resource.is_file():
        return _parse_xyz_text(resource.read_text(), name=str(name_or_path))

    raise FileNotFoundError(
        f"'{name_or_path}' is neither a path to an existing .xyz file nor "
        f"a molecule name in the RAY-MC database (looked for {resource})."
    )


def load_group_template(spec: Union[str, Path]) -> GroupTemplate:
    """Resolve a config value to a :class:`GroupTemplate`: a molecule (by
    database name or file path, via :func:`load_molecule`), or, failing
    that, a bare element symbol (via :meth:`GroupTemplate.atom`).

    This is what :mod:`raymc.initialize.build` calls for
    ``InitConfig.molecule`` -- the one field covers both "place water
    molecules" (``"water"``) and "place sodium atoms" (``"Na"``).
    """
    try:
        return load_molecule(spec)
    except FileNotFoundError:
        pass

    try:
        return GroupTemplate.atom(str(spec))
    except Exception as exc:
        raise FileNotFoundError(
            f"'{spec}' is not a database molecule, an existing .xyz file, "
            f"or a recognized element symbol."
        ) from exc
