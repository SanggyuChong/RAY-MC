"""Monte Carlo moves that act on a single atom.

The counterpart to :mod:`raymc.moves.group_based`, whose every move
displaces a whole rigid group at once and therefore leaves each
molecule's internal geometry exactly as the template built it. A water's
O-H sits at 0.9584 A forever under those moves, making g_OH(r) a true
delta function -- an undamped ripple in A_OH(Q) that never converges, and
one that neutron weighting on natural-H water amplifies catastrophically
(w_OH = -30.84).

Displacing one atom at a time is what lets a molecule flex. It is also
the move lineage RMCA and RMC++ use for molecular systems: molecules are
held together by explicit constraints rather than by rigid-body moves, so
a single atomic move type can serve a molecular liquid and a monatomic
one alike.

That makes the pairing with
:class:`raymc.constraints.intramolecular.IntramolecularConstraint` load-
bearing rather than optional. Nothing else scores an intramolecular
distance: the min- and max-distance constraints skip same-group pairs by
construction, so without a window an O-H bond is free to collapse to zero
or stretch across the box, and the only symptom is a strange g(r). See
:meth:`raymc.moves.moveset.MoveSet.from_config`, which warns when this
move is active and no such window is configured.
"""

from __future__ import annotations

import torch

from raymc.moves.base import Move, MoveDelta, MoveNotApplicable
from raymc.system import AtomicSystem, carry_node_labels


class TranslateAtom(Move):
    """Displace ONE randomly-chosen atom by a random vector.

    The displacement direction is uniform on the sphere and its magnitude
    is uniform in ``[0, max_displacement]`` -- the same distribution
    :class:`~raymc.moves.group_based.TranslateGroup` uses, applied to a
    single atom rather than to a group's centroid.

    RMCA-family codes run constrained molecular systems at roughly 0.05 A
    here. The right value is whatever lands acceptance in the usual
    30-50% band, which is a property of the run, not of the move.
    """

    name = "translate_atom"

    def __init__(self, max_displacement: float):
        self.max_displacement = max_displacement

    def is_applicable(self, system: AtomicSystem) -> bool:
        return bool(system.num_nodes > 0)

    def attempt(
        self, system: AtomicSystem, generator: torch.Generator
    ) -> tuple[AtomicSystem, MoveDelta]:
        n = system.num_nodes
        if n == 0:
            raise MoveNotApplicable("system has no atoms to translate")

        index = int(torch.randint(0, n, (1,), generator=generator).item())

        direction = torch.randn(3, generator=generator, dtype=torch.float64)
        direction = direction / direction.norm()
        magnitude = (
            torch.rand(1, generator=generator, dtype=torch.float64)
            * self.max_displacement
        )
        moved = system.positions[index].to(torch.float64) + direction * magnitude

        # Wrapped back into the box like every other move, so positions
        # never drift arbitrarily far out over a long run. This can leave
        # a molecule straddling a periodic edge, which is harmless here:
        # every consumer of an intra-group distance goes through the
        # minimum-image convention (`unwrap_group_positions`,
        # `minimum_image_delta`), and a molecule is far smaller than half
        # the box.
        if system.cell is not None and system.pbc is not None:
            cell = system.cell.squeeze(0).to(torch.float64)
            pbc = system.pbc.squeeze(0)
            frac = moved @ torch.linalg.inv(cell)
            shift = torch.floor(frac) * pbc.to(frac.dtype)
            moved = moved - shift @ cell

        positions = system.positions.clone()
        positions[index] = moved.to(positions.dtype)

        new_system = AtomicSystem(
            positions=positions,
            atomic_numbers=system.atomic_numbers,
            atomic_masses=system.atomic_masses,
            cell=system.cell,
            pbc=system.pbc,
        )
        # Group membership is unchanged -- an atom keeps its molecule even
        # when it moves alone. Carried without re-checking for the same
        # reason the group moves do it: re-running the geometric
        # connectivity check every step would be pure overhead, and here it
        # would be actively wrong, since that check infers bonds from summed
        # covalent radii and a legitimately stretched bond stops being
        # perceived.
        carry_node_labels(system, new_system)
        return new_system, MoveDelta(
            moved_indices=torch.tensor([index], dtype=torch.long)
        )
