"""Dispatch each mixed query to its original evaluator and frozen data."""
import json

from evaluator.generic_evaluator import GenericEvaluator
from .mmqa import MMQAEvaluator
from .ecomm import EcommEvaluator
from scenario.mmqa_ecomm.mmqa_ecomm_scenario import files_root


class MMQAEcommEvaluator(GenericEvaluator):
    def _load_domain_data(self):
        pass

    def __init__(self, use_case, scale_factor):
        super().__init__(use_case, scale_factor)
        self._root = files_root()
        self._results_path = self._root / "raw_results"
        self._metrics_path = self._root / "metrics"
        self.mapping = json.loads((self._root / "manifest.json").read_text())["queries"]
        self.delegates = {"mmqa": MMQAEvaluator("mmqa", scale_factor),
                          "ecomm": EcommEvaluator("ecomm", 1000)}
        for name, evaluator in self.delegates.items():
            evaluator._root = self._root / "sources" / name
            evaluator._results_path = self._results_path / name
            evaluator.system_name = "flockmtl"
        self.delegates["ecomm"].data_dir = self._root / "sources" / "ecomm" / "data"

    def _delegate(self, query_id):
        query = self.mapping[str(query_id)]
        return self.delegates[query["source"]], query["source_query_id"]

    def _get_ground_truth(self, query_id):
        evaluator, original = self._delegate(query_id)
        return evaluator._get_ground_truth(original)

    def _evaluate_single_query(self, query_id, system_results, ground_truth):
        evaluator, original = self._delegate(query_id)
        return evaluator._evaluate_single_query(original, system_results, ground_truth)

    def _additional_quality_metrics(self, query_id, system_results, ground_truth):
        evaluator, original = self._delegate(query_id)
        return evaluator._additional_quality_metrics(original, system_results, ground_truth)
