import os

import duckdb
import pandas as pd

from cacheblend_hotpotqa import dataset_dir, evaluate_answers, load_dataset, partition_query_sql
from evaluator.generic_evaluator import GenericEvaluator, SingleAccuracyScore


class CacheblendRagEvaluator(GenericEvaluator):
    def __init__(self, use_case: str, scale_factor: int = 1) -> None:
        super().__init__(use_case, scale_factor)

    def _load_domain_data(self) -> None:
        return None

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:
        sql_path = dataset_dir() / "query" / "gold_sql" / f"Q{query_id}.sql"
        if not sql_path.exists():
            raise FileNotFoundError(f"Gold SQL file not found: {sql_path}")

        conn = duckdb.connect()
        try:
            load_dataset(conn, dataset_dir())
            sql = sql_path.read_text(encoding="utf-8")
            sql = partition_query_sql(sql, terminal_id=getattr(self, "terminal_id", None))
            result = conn.execute(sql).fetchdf()
        finally:
            conn.close()

        gt_root = self._results_path / "ground_truth"
        if os.environ.get("CACHEBLEND_RAG_PARTITION_COUNT"):
            terminal_id = getattr(self, "terminal_id", None)
            if terminal_id is None:
                terminal_id = os.environ["CACHEBLEND_RAG_TERMINAL_ID"]
            gt_root = gt_root / f"terminal_{int(terminal_id)}"
        gt_path = gt_root / f"Q{query_id}.csv"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(gt_path, index=False)
        return result

    def _evaluate_single_query(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> SingleAccuracyScore:
        scores = self._answer_metrics(system_results, ground_truth)
        return SingleAccuracyScore(accuracy=scores["exact_match"], metric_type="hotpotqa_answer_em")

    @staticmethod
    def _answer_metrics(system_results, ground_truth):
        for frame in (system_results, ground_truth):
            if not {"row_id", "answer"}.issubset(frame.columns):
                raise ValueError("HotpotQA results must contain row_id and answer")
            if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
                raise ValueError("HotpotQA row IDs must be non-null and unique")
        return evaluate_answers(
            dict(zip(system_results.row_id, system_results.answer)),
            dict(zip(ground_truth.row_id, ground_truth.answer)),
        )

    def _additional_quality_metrics(self, query_id, system_results, ground_truth):
        return self._answer_metrics(system_results, ground_truth)
