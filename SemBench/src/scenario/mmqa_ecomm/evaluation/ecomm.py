from pathlib import Path
import pandas as pd
import duckdb
from evaluator.tabular_quality import (business_rows, decision_samples, logical_quality,
                                       score_rows, normalize_id, normalize_label)

from evaluator.generic_evaluator import (
    GenericEvaluator,
    QueryMetricRetrieval,
    QueryMetricAggregation,
    SingleAccuracyScore,
)

class EcommEvaluator(GenericEvaluator):
    def __init__(self, use_case: str, scale_factor: int = None) -> None:
        super().__init__(use_case, scale_factor)
        self.data_dir = self._root / "data"

    def _load_domain_data(self) -> None:
        # The mixed workload supplies an already prepared data directory.
        pass

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:
        if int(query_id) not in {1, 3, 5, 7}:
            raise ValueError(f"Query is outside the prepared Ecomm sample: {query_id}")
        sql_path = self._root / "query" / "gold_sql" / f"Q{query_id}.sql"
        prepared = Path(self.data_dir) / "query" / "gold_sql" / f"Q{query_id}.sql"
        if prepared.is_file():
            sql_path = prepared
        with duckdb.connect() as conn:
            conn.execute("SET file_search_path = ?", [str(self.data_dir)])
            return conn.execute(sql_path.read_text()).fetchdf()

    def _evaluate_single_query(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> "QueryMetricRetrieval | QueryMetricAggregation | SingleAccuracyScore":
        if int(query_id) not in {1, 3, 5, 7}:
            raise ValueError(f"Query is outside the prepared Ecomm sample: {query_id}")
        columns = ["id", "category"] if int(query_id) in {3, 5} else ["id"]
        return score_rows(self, business_rows(system_results), ground_truth, columns,
                          normalizers={"id": normalize_id, "category": normalize_label})

    def _additional_quality_metrics(self, query_id, system_results, ground_truth):
        if getattr(self, "system_name", "flockmtl") != "flockmtl":
            return {}
        q = int(query_id)
        if q in {3, 5}:
            frame = business_rows(system_results)
            samples = [{"id": row.id, "prediction": normalize_label(row.category)}
                       for row in frame.itertuples(index=False)]
            expected = [(row.id, normalize_label(row.category))
                        for row in ground_truth.itertuples(index=False)]
            return logical_quality(samples, expected)
        if q in {1, 7}:
            path = str(Path(self.data_dir) / "styles_details.parquet")
            with duckdb.connect() as conn:
                prepared_candidates = Path(self.data_dir) / "query" / "gold_sql" / f"Q{q}_candidates.sql"
                if prepared_candidates.is_file():
                    conn.execute("SET file_search_path = ?", [str(self.data_dir)])
                    candidates = conn.execute(prepared_candidates.read_text()).fetchall()
                    universe = ([f"{normalize_id(a)}-{normalize_id(b)}" for a, b, _ in candidates]
                                if q == 7 else [normalize_id(row[0]) for row in candidates])
                elif q == 7:
                    conn.execute("SET file_search_path = ?", [str(self.data_dir)])
                    pair_sql = self._root / "query" / "gold_sql" / "Q7_candidates.sql"
                    pairs = conn.execute(pair_sql.read_text()).fetchall()
                    universe = [f"{normalize_id(a)}-{normalize_id(b)}" for a, b, _ in pairs]
                else:
                    ids = conn.execute("SELECT id FROM read_parquet(?)", [path]).fetchall()
                    universe = [normalize_id(row[0]) for row in ids]
            positives = set(ground_truth.id.map(normalize_id))
            return logical_quality(decision_samples(system_results),
                                   ((key, key in positives) for key in universe), binary=True)
        return {}
