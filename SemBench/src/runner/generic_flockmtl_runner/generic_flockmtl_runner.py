"""
DuckDB FlockMTL runner implementation.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Dict, List

import pandas as pd
from jinja2 import Environment
from overrides import override

from flockmtl_cacheblend import apply_cacheblend_mode_to_sql, current_cacheblend_mode
from runner.generic_runner import GenericQueryMetric, GenericRunner

jinja_env = Environment(variable_start_string="<<", variable_end_string=">>")


class GenericFlockMTLRunner(GenericRunner):
    """Runner for FlockMTL."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize DuckDB FlockMTL runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        self.flockmtl_conn = None
        self.query_concurrency = self._parse_query_concurrency()
        self.query_start_stagger_s = self._parse_query_start_stagger_s()
        self.skip_setup = skip_setup
        self.query_worker_script = self.base_path / "src" / "flockmtl_query_worker.py"

    @override
    def get_system_name(self) -> str:
        return "flockmtl"

    def _parse_query_concurrency(self) -> int:
        raw_value = os.environ.get("FLOCKMTL_QUERY_CONCURRENCY", "1")
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            print(
                "Invalid FLOCKMTL_QUERY_CONCURRENCY="
                f"{raw_value!r}; falling back to 1"
            )
            return 1

    def _parse_query_start_stagger_s(self) -> float:
        raw_value = os.environ.get("FLOCKMTL_QUERY_START_STAGGER_MS", "0")
        try:
            stagger_ms = max(0.0, float(raw_value))
        except (TypeError, ValueError):
            print(
                "Invalid FLOCKMTL_QUERY_START_STAGGER_MS="
                f"{raw_value!r}; falling back to 0"
            )
            return 0.0
        return stagger_ms / 1000.0

    def _get_query_text(self, query_id: int) -> str:
        if self.scenario_handler is None:
            query_text = self._discover_query_text(query_id)
        else:
            query_text = self.scenario_handler.get_query_text(
                query_id, self.get_system_name()
            )
        return apply_cacheblend_mode_to_sql(query_text)

    def _make_metric(
        self,
        query_id: int,
        status: str,
        results: pd.DataFrame | None = None,
        execution_time: float | None = None,
        query_wall_time: float | None = None,
        time_to_first_vllm_request: float | None = None,
        first_vllm_request_path: str | None = None,
        observed_vllm_request_count: int | None = None,
        error: str | None = None,
    ) -> GenericQueryMetric:
        metric = GenericQueryMetric(
            query_id=query_id,
            status=status,
            execution_time=execution_time,
            query_wall_time=query_wall_time,
            time_to_first_vllm_request=time_to_first_vllm_request,
            first_vllm_request_path=first_vllm_request_path,
            observed_vllm_request_count=observed_vllm_request_count,
            results=results if results is not None else pd.DataFrame(),
            error=error,
        )
        if status != "failed":
            # FlockMTL does not currently expose per-query token accounting in
            # the SemBench runner path, so keep the existing placeholder values.
            metric.token_usage = 0
            metric.money_cost = 0.0
        return metric

    @override
    def execute_query(self, query_id: int) -> GenericQueryMetric:
        if self.flockmtl_conn is None:
            raise RuntimeError("FlockMTL connection is not initialized")

        query_text = self._get_query_text(query_id)
        cacheblend_mode = current_cacheblend_mode()
        if cacheblend_mode != "auto":
            print(f"FlockMTL SQL cacheblend mode: {cacheblend_mode}")
        templated_query = jinja_env.from_string(query_text).render(
            model_name=self.model_name
        )
        print(templated_query)

        start_time = time.perf_counter()
        try:
            query_job = self.flockmtl_conn.execute(templated_query)
            df = query_job.fetchdf()
            completed_at = time.perf_counter()
            timing = self._build_query_timing(start_time, completed_at)
            return self._make_metric(
                query_id=query_id,
                status="success",
                results=df,
                execution_time=timing["execution_time"],
                query_wall_time=timing["query_wall_time"],
                time_to_first_vllm_request=timing["time_to_first_vllm_request"],
                first_vllm_request_path=timing["first_vllm_request_path"],
                observed_vllm_request_count=timing[
                    "observed_vllm_request_count"
                ],
            )
        except Exception as exc:
            completed_at = time.perf_counter()
            timing = self._build_query_timing(start_time, completed_at)
            print(
                f"  Error executing query {query_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            return self._make_metric(
                query_id=query_id,
                status="failed",
                execution_time=timing["execution_time"],
                query_wall_time=timing["query_wall_time"],
                time_to_first_vllm_request=timing["time_to_first_vllm_request"],
                first_vllm_request_path=timing["first_vllm_request_path"],
                observed_vllm_request_count=timing[
                    "observed_vllm_request_count"
                ],
                error=str(exc),
            )

    def _worker_artifact_dir(self, root_dir: Path, query_id: int) -> Path:
        artifact_dir = root_dir / f"Q{query_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _stagger_coord_dir(self, root_dir: Path) -> Path:
        coord_dir = root_dir / "query_request_stagger"
        coord_dir.mkdir(parents=True, exist_ok=True)
        return coord_dir

    def _build_worker_command(
        self,
        query_id: int,
        metric_path: Path,
        result_path: Path,
    ) -> list[str]:
        cmd = [
            sys.executable,
            str(self.query_worker_script),
            "--system",
            self.get_system_name(),
            "--use-case",
            self.use_case,
            "--query",
            str(query_id),
            "--model",
            self.model_name,
            "--metric-output",
            str(metric_path),
            "--result-output",
            str(result_path),
        ]
        if self.scale_factor is not None:
            cmd += ["--scale-factor", str(self.scale_factor)]
        if self.skip_setup:
            cmd.append("--skip-setup")
        return cmd

    def _load_worker_metric(
        self,
        query_id: int,
        metric_path: Path,
        result_path: Path,
    ) -> GenericQueryMetric:
        with metric_path.open("r", encoding="utf-8") as metric_file:
            metric_data = json.load(metric_file)

        results = pd.read_pickle(result_path) if result_path.exists() else pd.DataFrame()
        return self._make_metric(
            query_id=query_id,
            status=metric_data.get("status", "failed"),
            results=results,
            execution_time=metric_data.get("execution_time"),
            query_wall_time=metric_data.get("query_wall_time"),
            time_to_first_vllm_request=metric_data.get(
                "time_to_first_vllm_request"
            ),
            first_vllm_request_path=metric_data.get("first_vllm_request_path"),
            observed_vllm_request_count=metric_data.get(
                "observed_vllm_request_count"
            ),
            error=metric_data.get("error"),
        )

    def _execute_query_worker(
        self,
        query_id: int,
        query_order_index: int,
        artifact_root: Path,
        stagger_coord_dir: Path | None = None,
    ) -> GenericQueryMetric:
        artifact_dir = self._worker_artifact_dir(artifact_root, query_id)
        metric_path = artifact_dir / "metric.json"
        result_path = artifact_dir / "result.pkl"
        log_path = artifact_dir / "worker.log"

        env = os.environ.copy()
        env["FLOCKMTL_QUERY_CONCURRENCY"] = "1"
        env.pop("FLOCKMTL_DB_FILE", None)
        env.pop("FLOCKMTL_QUERY_REQUEST_STAGGER_APPLIED", None)
        if stagger_coord_dir is not None and self.query_start_stagger_s > 0:
            env["FLOCKMTL_QUERY_ORDER_INDEX"] = str(query_order_index)
            env["FLOCKMTL_QUERY_STAGGER_COORD_DIR"] = str(stagger_coord_dir)
        else:
            env.pop("FLOCKMTL_QUERY_ORDER_INDEX", None)
            env.pop("FLOCKMTL_QUERY_STAGGER_COORD_DIR", None)

        cmd = self._build_worker_command(query_id, metric_path, result_path)
        proc = subprocess.run(
            cmd,
            cwd=str(self.base_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        log_path.write_text(proc.stdout or "", encoding="utf-8")
        if proc.stdout:
            for line in proc.stdout.splitlines():
                print(f"  [flockmtl Q{query_id}] {line}")

        if metric_path.exists():
            metric = self._load_worker_metric(query_id, metric_path, result_path)
            if proc.returncode != 0 and metric.status != "failed":
                metric.status = "failed"
                metric.error = (
                    f"worker exited with code {proc.returncode} without "
                    "reporting a failed query"
                )
            return metric

        error = (
            f"worker exited with code {proc.returncode}; "
            f"missing metric artifact {metric_path}"
        )
        return self._make_metric(
            query_id=query_id,
            status="failed",
            error=error,
        )

    @override
    def execute_queries(
        self, query_ids: List[int]
    ) -> Dict[int, GenericQueryMetric]:
        ordered_query_ids = list(query_ids)
        if self.query_concurrency <= 1 or len(ordered_query_ids) <= 1:
            return {
                query_id: self.execute_query(query_id)
                for query_id in ordered_query_ids
            }

        max_workers = min(self.query_concurrency, len(ordered_query_ids))
        print(
            f"Running {len(ordered_query_ids)} flockmtl queries with "
            f"query_concurrency={max_workers}"
        )
        if self.query_start_stagger_s > 0:
            print(
                "Staggering first flockmtl LLM request emission by "
                f"{self.query_start_stagger_s * 1000.0:.1f} ms"
            )

        results: Dict[int, GenericQueryMetric] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            with tempfile.TemporaryDirectory(
                prefix="flockmtl_query_workers_"
            ) as temp_dir:
                artifact_root = Path(temp_dir)
                stagger_coord_dir = (
                    self._stagger_coord_dir(artifact_root)
                    if self.query_start_stagger_s > 0
                    else None
                )
                for index, query_id in enumerate(ordered_query_ids):
                    futures[
                        executor.submit(
                            self._execute_query_worker,
                            query_id,
                            index,
                            artifact_root,
                            stagger_coord_dir,
                        )
                    ] = query_id

                for future in as_completed(futures):
                    query_id = futures[future]
                    try:
                        results[query_id] = future.result()
                    except Exception as exc:
                        results[query_id] = self._make_metric(
                            query_id=query_id,
                            status="failed",
                            error=(
                                "query worker orchestration failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )

        return {query_id: results[query_id] for query_id in ordered_query_ids}

    def close(self) -> None:
        return None

    def _build_query_timing(
        self,
        start_time: float,
        completed_at: float,
    ) -> dict[str, float | int | str | None]:
        query_wall_time = max(0.0, completed_at - start_time)
        return {
            "execution_time": query_wall_time,
            "query_wall_time": query_wall_time,
            "time_to_first_vllm_request": None,
            "first_vllm_request_path": None,
            "observed_vllm_request_count": None,
        }
