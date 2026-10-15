"""Adapters between RAY-MC's own objects and third-party atomistic APIs.

Nothing in :mod:`raymc`'s core imports this package, and nothing here is
imported eagerly by it: every module under ``interop`` is allowed to
require an optional extra, and the subsystems that need one
(:mod:`raymc.energetics`) import it themselves.
"""
