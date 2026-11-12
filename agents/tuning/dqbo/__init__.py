"""DQBO: the existing quality-constrained multiobjective Bayesian optimizer."""
from importlib import import_module

__all__ = ["LegacyEquivalentQualityBO", "OptimizerConfig", "FixedSearchSpace"]


def __getattr__(name):
    modules = {"LegacyEquivalentQualityBO": "legacy_equivalent",
               "OptimizerConfig": "optimizer", "FixedSearchSpace": "space"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.src.services.engine.{modules[name]}"), name)
