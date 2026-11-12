"""Lazy public native BO and local Repair services."""
from importlib import import_module

__all__ = ["audit_candidate_result", "execute_bo", "execute_bo_round", "prepare_bo_context",
           "propose_next_bo_candidate", "repair_bo_candidate", "repair_candidate_known_constraints"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".workflow", __name__), name)
    globals()[name] = value
    return value
