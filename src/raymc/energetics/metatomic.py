"""Metatomic-based energetics adapter for hybrid RMC.

This module integrates `metatomic
<https://docs.metatensor.org/metatomic/>`_: models capable of
predicting the energy and forces for atomistic system. Any object satisfying
metatomic's ``AtomisticModel`` interface works here, ranging from MLIP
checkpoints to empirical potentials wrapped as metatomic models.

Requires the "metatomic" extra: ``pip install raymc[metatomic]``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from raymc.modalities.base import Calculator
from raymc.system import AtomicSystem

try:
    import metatomic.torch as mta

    from raymc.interop.metatomic import (
        attach_neighbor_lists,
        neighbor_list_calculators,
        to_metatomic_system,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "MetatomicEnergetics requires the 'metatomic' extra: "
        "pip install raymc[metatomic]"
    ) from exc


class MetatomicEnergetics(Calculator):
    """A metatomic model, wrapped as an energy scorer.

    Implements :class:`raymc.modalities.base.Calculator` via
    :meth:`compute` (single system, "direct" energetics term -- see
    :func:`raymc.residuals.aggregator.make_metatomic_term`). No
    incremental ``compute_increm_change`` override exists yet: every call
    recomputes the neighbor list and full energy from scratch (see the
    TODO in :func:`raymc.interop.metatomic.to_metatomic_system` and below
    in :meth:`energy` about
    caching a cutoff radius to eventually only touch atoms within range of
    a move -- both deferred until we've verified what metatomic's own API
    actually supports).

    Parameters
    ----------
    model : metatomic.torch.AtomisticModel
        Any already-loaded metatomic model -- a UPET checkpoint (see
        :func:`load_upet_model`), or any other model built on the
        metatomic interface (e.g. an empirical potential).
    device : str
        Torch device to run on.
    dtype : torch.dtype
        Floating-point precision for positions/cell handed to the model.
    """

    def __init__(
        self,
        model: "mta.AtomisticModel",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self._model = model.to(self.device)
        # Neighbor-list calculators depend only on the model's own
        # requirements (cutoff, etc.), not on any particular system, so
        # they're computed once here and reused for every call.
        self._neighbor_calculators = neighbor_list_calculators(self._model)
        # TODO: cache each calculator's own cutoff here (r_cut) so a
        # future incremental energy update can restrict itself to atoms
        # within range of a move, instead of always scoring the whole
        # system -- deferred: needs the calculators' cutoff attribute
        # name verified against the installed vesin/metatomic version.

        # eV regardless of what the model itself works in: metatomic
        # converts between the unit declared in the model's own
        # ModelCapabilities and the unit requested here (q-TIP4P/F, for
        # one, is defined in kcal/mol). ``quantity=``/``per_atom=`` are
        # both deprecated in metatomic >=0.1.16; ``sample_kind`` replaces
        # the latter.
        self._energy_output = {
            "energy": mta.ModelOutput(unit="eV", sample_kind="system")
        }
        self._eval_options = mta.ModelEvaluationOptions(
            length_unit="angstrom", outputs=self._energy_output
        )

    def compute(self, system: AtomicSystem) -> torch.Tensor:
        """Potential energy (eV) for one system -- satisfies
        :class:`raymc.modalities.base.Calculator`."""
        return self.energy([system])[0]

    def energy(self, systems: list[AtomicSystem]) -> torch.Tensor:
        """Batched potential energy, one call for every system.

        Parameters
        ----------
        systems : list[AtomicSystem]

        Returns
        -------
        torch.Tensor
            [len(systems)] energies in eV.
        """
        mta_systems = [to_metatomic_system(s, self.dtype) for s in systems]
        attach_neighbor_lists(self._neighbor_calculators, mta_systems)
        outputs = self._model(mta_systems, self._eval_options, check_consistency=False)
        return outputs["energy"].block().values.squeeze(-1)

    def energy_forces(self, system: AtomicSystem) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and forces for one system, via autograd on positions.

        Not needed by the Metropolis RMC loop itself (which only ever
        scores a scalar), but kept available for local relaxation or the
        eventual gradient-based generative solver, per the project's
        design goal that forward models/energetics serve both solvers
        unmodified.

        Parameters
        ----------
        system : AtomicSystem

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Energy (scalar, eV) and forces ([n_atoms, 3], eV/Angstrom).
        """
        positions = system.positions.to(self.dtype).clone().requires_grad_(True)
        working = AtomicSystem(
            positions=positions,
            atomic_numbers=system.atomic_numbers,
            cell=system.cell,
            pbc=system.pbc,
        )
        mta_system = to_metatomic_system(working, self.dtype)
        attach_neighbor_lists(self._neighbor_calculators, [mta_system])
        outputs = self._model([mta_system], self._eval_options, check_consistency=False)
        energy = outputs["energy"].block().values.squeeze()
        (gradient,) = torch.autograd.grad(energy, positions)
        forces = -gradient
        return energy.detach(), forces.detach()


def load_model_file(path: "str | Path") -> "mta.AtomisticModel":
    """Load an exported metatomic model from a TorchScript ``.pt`` file.

    The general escape hatch for any model that isn't fetched from a
    checkpoint hub -- in particular an empirical potential built and
    exported offline, such as the q-TIP4P/F water model produced by
    ``scripts/build_qtip4pf.py``.

    Loading needs nothing beyond the "metatomic" extra even when the
    model was *built* on top of other libraries: ``AtomisticModel.save``
    TorchScript-compiles the model's pure-Python dependencies into the
    serialized graph, and bundles any native extensions alongside. The
    q-TIP4P/F model's P3M electrostatics come from ``torch-pme``, which
    RAY-MC does not depend on and does not need installed to run it.

    Note that a TorchScript archive is tied to the metatensor/metatomic
    ABI it was built against, so a ``.pt`` may need rebuilding after those
    are upgraded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No metatomic model at {path}. Build one first, e.g. "
            f"`python scripts/build_qtip4pf.py --output {path}`."
        )
    return mta.load_atomistic_model(str(path))


def load_upet_model(
    model: str = "pet-mad",
    size: str = "s",
    version: str = "1.5.0",
    checkpoint_path: str | None = None,
) -> "mta.AtomisticModel":
    """Load a UPET checkpoint as a metatomic model.

    UPET (https://lab-cosmo.github.io/upet/latest/, weights on
    https://huggingface.co/lab-cosmo/upet) is one source of metatomic
    models among possibly several (PET-MAD, PET-OAM, PET-OMat, ...) --
    the result is a plain metatomic ``AtomisticModel`` like any other,
    ready to hand to :class:`MetatomicEnergetics`.

    Parameters
    ----------
    model, size, version
        Name the checkpoint to fetch from the UPET weights repository.
        Ignored when *checkpoint_path* is given.
    checkpoint_path : str | None
        Path to a checkpoint file already on disk, loaded instead of
        fetching anything. This is not merely an offline convenience: an
        environment whose network policy denies huggingface.co cannot use
        the download path at all, and pinning a run to a specific local checkpoint file is
        better provenance for a published fit than a version string
        resolved against a remote repository that can move under it.

        UPET reads the model/size/version off the *filename* when it
        follows the repository's own convention -- ``pet-mad-xs-v1.5.0.ckpt``
        -- so keep that spelling rather than renaming the file.

    Requires the "upet" extra: ``pip install raymc[upet]``.
    """
    try:
        from upet import get_upet
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "load_upet_model requires the 'upet' extra: pip install raymc[upet]"
        ) from exc
    if checkpoint_path is not None:
        from pathlib import Path

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"energetics checkpoint not found: {checkpoint_path!r}. "
                "Checkpoints are not tracked in this repository; download "
                "one from https://huggingface.co/lab-cosmo/upet."
            )
        return get_upet(checkpoint_path=checkpoint_path)
    return get_upet(model=model, size=size, version=version)
