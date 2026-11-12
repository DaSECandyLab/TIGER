"""Complete repair pipeline, including proposal review and reflection."""
from importlib import import_module

__all__ = ["RepairCoordinator", "RepairScope", "reflect_episode"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = "repair_reflection" if name == "reflect_episode" else "repair_bo"
    return getattr(import_module(f"{__name__}.src.services.{module}"), name)
