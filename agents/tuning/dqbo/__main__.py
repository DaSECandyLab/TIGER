"""Run the existing DQBO and full repair pipeline with fixed workload ranges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .ranges import DEFAULT_CONFIG, materialize_ranges


def prepare_config(config: dict, *, ranges_config=DEFAULT_CONFIG) -> dict:
    config = dict(config)
    workload, model = config.pop("workload"), config.pop("model")
    screening = config.pop("screening_result", None)
    if "domain_spec" in config or "parameter_results" in config:
        raise ValueError("The DQBO entry uses workload ranges; remove domain_spec/parameter_results")
    config.update(materialize_ranges(
        workload=workload, model=model, baseline_config=config["baseline_config"],
        output_dir=Path(config["output_dir"]) / "fixed_range_inputs",
        config=ranges_config, screening_result=screening,
    ))
    options = dict(config.get("optimizer_options") or {})
    options.setdefault("repair_bo_enabled", True)
    options.setdefault("repair_reflect_enabled", True)
    config["optimizer_options"] = options
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ranges-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true", help="Write frozen inputs; no LLM calls or trials")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    config = prepare_config(json.loads(args.config.read_text()), ranges_config=args.ranges_config)
    path = Path(config["output_dir"]) / "fixed_range_inputs/candidate_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"config": str(path), **{k: config[k] for k in ("domain_spec", "parameter_results")}}))
        return
    from agents.run_candidate_optimizer import run_candidate_optimizer
    result = run_candidate_optimizer(config, python_executable=args.python)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
