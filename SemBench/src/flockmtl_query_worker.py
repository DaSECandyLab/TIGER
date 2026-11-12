#!/usr/bin/env python3
"""Run a single FlockMTL query in an isolated temporary DuckDB database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
from dotenv import load_dotenv

# Add src directory to Python path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import get_runner_class
from flockmtl_cacheblend import CACHEBLEND_MODE_ENV, normalize_cacheblend_mode
from runner.generic_runner import GenericQueryMetric


def write_artifacts(
    metric: GenericQueryMetric,
    metric_output: Path,
    result_output: Path,
) -> None:
    metric_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.parent.mkdir(parents=True, exist_ok=True)

    metric.results.to_pickle(result_output)
    with metric_output.open("w", encoding="utf-8") as metric_file:
        json.dump(metric.to_dict(), metric_file, indent=2)


def write_query_stagger_completion_marker() -> None:
    coord_dir = os.environ.get("FLOCKMTL_QUERY_STAGGER_COORD_DIR")
    raw_order_index = os.environ.get("FLOCKMTL_QUERY_ORDER_INDEX")
    if not coord_dir or raw_order_index is None:
        return

    try:
        order_index = int(raw_order_index)
    except (TypeError, ValueError):
        return

    done_marker = Path(coord_dir) / f"query_{order_index}.done"
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text("done\n", encoding="utf-8")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    parser = argparse.ArgumentParser(
        description="Run one FlockMTL query in an isolated worker"
    )
    parser.add_argument("--system", default="flockmtl")
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--scale-factor", type=int, default=None)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--metric-output", required=True)
    parser.add_argument("--result-output", required=True)
    parser.add_argument(
        "--cacheblend-mode",
        choices=["auto", "on", "off"],
        default=os.environ.get(CACHEBLEND_MODE_ENV, "auto"),
        help="Control FlockMTL SQL-side cacheblend markers.",
    )
    args = parser.parse_args()
    if args.query.isdigit():
        args.query = int(args.query)
    args.cacheblend_mode = normalize_cacheblend_mode(args.cacheblend_mode)
    os.environ[CACHEBLEND_MODE_ENV] = args.cacheblend_mode

    metric_output = Path(args.metric_output)
    result_output = Path(args.result_output)
    os.environ["FLOCKMTL_QUERY_CONCURRENCY"] = "1"

    try:
        runner_class = get_runner_class(args.system, args.use_case)
        if not runner_class:
            raise RuntimeError(
                f"Failed to import runner for {args.system}/{args.use_case}"
            )

        with tempfile.TemporaryDirectory(
            prefix=f"flockmtl_q{args.query}_"
        ) as temp_dir:
            home_dir = Path(temp_dir)
            (home_dir / ".duckdb" / "flock_storage").mkdir(
                parents=True, exist_ok=True
            )
            temp_db = home_dir / f"{args.use_case}_q{args.query}.duckdb"
            os.environ["HOME"] = str(home_dir)
            os.environ["FLOCKMTL_DB_FILE"] = str(temp_db)

            runner = runner_class(
                use_case=args.use_case,
                scale_factor=args.scale_factor,
                skip_setup=args.skip_setup,
                model_name=args.model,
            )
            try:
                metric = runner.execute_query(args.query)
                write_artifacts(metric, metric_output, result_output)
            finally:
                close_runner = getattr(runner, "close", None)
                if callable(close_runner):
                    close_runner()

    except Exception as exc:
        failed_metric = GenericQueryMetric(
            query_id=args.query,
            status="failed",
            execution_time=0.0,
            query_wall_time=0.0,
            results=pd.DataFrame(),
            error=str(exc),
        )
        write_artifacts(failed_metric, metric_output, result_output)
        raise
    finally:
        write_query_stagger_completion_marker()


if __name__ == "__main__":
    main()
