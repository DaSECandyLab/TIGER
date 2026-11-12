"""Resumable native BO loop with run-local Repair and reflection."""
from __future__ import annotations

from typing import Any

from agents.tuning.dqbo.src.services.engine.history import result_from_history
from .workflow import _load_object, _validate_frozen_context, execute_bo_round


def execute_native_bo(context_path: str, *, structured_inference_client: Any | None = None) -> dict[str, Any]:
    context = _load_object(context_path)
    _validate_frozen_context(context)
    config = context["optimizer_config"]
    budget = int(config["rounds"])
    if budget < 1:
        raise ValueError("rounds maximum budget must be positive")
    warmup = int(config["repair_cold_start_evaluations"]) if config.get("repair_bo_enabled") else None
    while True:
        before = _load_object(context["history_path"])
        count = len(before.get("trials", []))
        if count >= budget:
            break
        execute_bo_round(context_path, warmup_complete_observations=warmup,
                         structured_inference_client=structured_inference_client)
        after = _load_object(context["history_path"])
        if len(after.get("trials", [])) <= count:
            raise RuntimeError("BO round completed without persisting a new Trial")
    history = _load_object(context["history_path"])
    return {"status": "executed", "context_path": context_path,
            "result_path": context["result_path"], "history_path": context["history_path"],
            "rounds_completed": len(history.get("trials", [])), "rounds_budget": budget,
            "termination": {"reason": "maximum_budget_reached", "rounds_budget": budget},
            "dataset_counts": result_from_history(history)["dataset_counts"]}
