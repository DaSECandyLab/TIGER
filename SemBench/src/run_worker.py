"""
Subprocess worker for running a single system in an isolated environment.

This script is invoked by run.py when per-system virtual environments are
detected. It runs a single system's queries and saves results to disk.
The main run.py process then reads those results for evaluation.

Usage (called automatically by run.py):
    .venvs/flockmtl/bin/python src/run_worker.py \
        --system flockmtl --use-case movie --queries 1 3 \
        --model Qwen2-7B-Instruct --scale-factor 40
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    from run import SUPPORTED_USE_CASES

    parser = argparse.ArgumentParser(
        description="Run a single system's queries in an isolated environment"
    )
    parser.add_argument(
        "--system", required=True, choices=["flockmtl"], help="Experimental system"
    )
    parser.add_argument(
        "--use-case", required=True, choices=SUPPORTED_USE_CASES, help="Experimental workload"
    )
    parser.add_argument(
        "--queries", nargs="+", default=None, help="Query IDs to run"
    )
    parser.add_argument(
        "--model", default="Qwen2-7B-Instruct", help="Served model name"
    )
    parser.add_argument(
        "--scale-factor", type=int, default=None, help="Dataset scale factor"
    )
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip data setup phase"
    )
    parser.add_argument(
        "--cacheblend-mode",
        choices=["auto", "on", "off"],
        default=os.environ.get("SEMBENCH_CACHEBLEND_MODE", "auto"),
        help="Control FlockMTL SQL-side cacheblend markers.",
    )

    args = parser.parse_args()

    # Import runner infrastructure
    from run import get_runner_class, parse_query_ids
    from flockmtl_cacheblend import CACHEBLEND_MODE_ENV, normalize_cacheblend_mode

    args.cacheblend_mode = normalize_cacheblend_mode(args.cacheblend_mode)
    os.environ[CACHEBLEND_MODE_ENV] = args.cacheblend_mode

    # Parse query IDs
    query_ids = None
    if args.queries:
        query_ids = parse_query_ids(args.queries)

    # Get runner class
    runner_class = get_runner_class(args.system, args.use_case)
    if not runner_class:
        print(f"Failed to import runner for {args.system}")
        sys.exit(1)

    # Initialize and run
    try:
        runner = runner_class(
            use_case=args.use_case,
            scale_factor=args.scale_factor,
            skip_setup=args.skip_setup,
            model_name=args.model,
        )
        try:
            metrics = runner.run_all_queries(queries=query_ids)
        finally:
            close_runner = getattr(runner, "close", None)
            if callable(close_runner):
                close_runner()

        # Write a completion marker with summary for the parent process
        summary = {}
        for query_id, metric in metrics.items():
            summary[f"Q{query_id}"] = metric.to_dict()

        # Print summary as JSON to stdout for parent process
        print(f"\n__WORKER_RESULT__{json.dumps(summary)}__END_WORKER_RESULT__")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\n__WORKER_ERROR__{str(e)}__END_WORKER_ERROR__")
        sys.exit(1)


if __name__ == "__main__":
    main()
