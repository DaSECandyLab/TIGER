"""Score the text MMQA queries used by the mixed workload (Q3 and Q6)."""
import json
import shutil
from collections import Counter
from typing import Union

import pandas as pd
from evaluator.tabular_quality import business_rows, decision_samples, logical_quality
from evaluator.generic_evaluator import GenericEvaluator, QueryMetricRetrieval

def compute_metrics(results: list, ground_truth: Union[set, list]):
    tp = sum((Counter(results) & Counter(ground_truth)).values())
    fp = len(results) - tp

    assert tp + fp == len(
        results
    ), "True Positives and False Positives do not match the results length."
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(ground_truth) if len(ground_truth) > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return QueryMetricRetrieval(precision, recall, f1_score)


class MMQAEvaluator(GenericEvaluator):
    def __init__(self, use_case: str, scale_factor: int) -> None:
        super().__init__(use_case, scale_factor)

        self.use_case = use_case

    def _load_domain_data(self) -> None:
        pass

    def _get_ground_truth(self, query_id: int) -> str:
        src = self._root / "query" / "natural_language" / f"q{query_id}.json"

        # Copy ground truth JSON to raw_results/ground_truth/
        gt_dir = self._results_path / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, gt_dir / f"Q{query_id}.json")

        return src

    def _evaluate_single_query(
        self, query_id: int, system_results: pd.DataFrame, ground_truth: str
    ) -> QueryMetricRetrieval:
        """Evaluate a single query based on its type."""
        try:
            query_id = int(query_id)
        except ValueError:
            query_id = int(query_id[:-1])  # "3a" -> "3"
        evaluate_fn = self._discover_evaluate_impl(query_id)
        return evaluate_fn(business_rows(system_results), ground_truth)

    def _evaluate_q3(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = system_results["title"].tolist()

        with open(ground_truth_filepath, "r") as f:
            ground_truth = set(json.load(f).get("ground_truth"))

        return compute_metrics(results, ground_truth)

    def _evaluate_q6(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = system_results["Airlines"].tolist()

        with open(ground_truth_filepath, "r") as f:
            ground_truth = set(json.load(f).get("ground_truth"))

        return compute_metrics(results, ground_truth)

    def _additional_quality_metrics(self, query_id, system_results, ground_truth):
        query = str(query_id).lower().removeprefix("q")
        if not (query.startswith("3") or query.startswith("6")):
            raise ValueError(f"Query is outside the prepared MMQA sample: {query_id}")
        with open(ground_truth, encoding="utf-8") as file:
            gold = json.load(file)["ground_truth"]
        filename, column = (("lizzy_caplan_text_data.csv", "title") if query.startswith("3")
                            else ("tampa_international_airport.csv", "Airlines"))
        universe = pd.read_csv(self._root / "data" / f"sf_{self.scale_factor}" / filename)[column].tolist()
        positives = set(gold)
        return logical_quality(decision_samples(system_results),
                               [(key, key in positives) for key in universe], binary=True)
