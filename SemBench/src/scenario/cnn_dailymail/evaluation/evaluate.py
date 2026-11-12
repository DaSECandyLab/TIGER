"""Reference-based summary quality. Scores are macro averages on [0, 1]."""
import pandas as pd
from rouge_score import rouge_scorer

from evaluator.generic_evaluator import GenericEvaluator, SingleAccuracyScore
from scenario.cnn_dailymail.cnn_dailymail_scenario import ROOT, CNNDailyMailScenario


class CNNDailyMailEvaluator(GenericEvaluator):
    def _load_domain_data(self):
        self.scenario_handler = CNNDailyMailScenario(self.scale_factor)
        self.references = pd.read_csv(self.scenario_handler.root / "data" / "references.csv", keep_default_na=False)

    def _get_ground_truth(self, query_id):
        query_id = int(str(query_id).removeprefix("Q").removeprefix("q"))
        return self.references.loc[
            (self.references.query_id == query_id)
            & (self.references.query_row <= self.scenario_handler.scale_factor),
            ["row_id", "summary"],
        ].copy()

    @staticmethod
    def score(system_results, ground_truth):
        if not {"row_id", "summary"}.issubset(system_results.columns):
            raise ValueError("Expected row_id and summary columns")
        if system_results.row_id.isna().any() or system_results.row_id.duplicated().any():
            raise ValueError("Prediction IDs must be non-null and unique")
        if not set(system_results.row_id).issubset(set(ground_truth.row_id)):
            raise ValueError("Unexpected prediction IDs")
        if ground_truth.empty:
            raise ValueError("No reference rows")
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals = dict.fromkeys(("rouge1", "rouge2", "rougeL", "exact_match", "nonempty_rate"), 0.0)
        predictions = dict(zip(system_results.row_id, system_results.summary))
        for row in ground_truth.itertuples():
            text = predictions.get(row.row_id, "")
            text = text if isinstance(text, str) else ""
            for key, value in scorer.score(row.summary, text).items():
                totals[key] += value.fmeasure
            totals["exact_match"] += float(text == row.summary)
            totals["nonempty_rate"] += float(bool(text.strip()))
        return {key: value / len(ground_truth) for key, value in totals.items()}

    def _evaluate_single_query(self, query_id, system_results, ground_truth):
        return SingleAccuracyScore(
            accuracy=self.score(system_results, ground_truth)["rougeL"],
            metric_type="cnn_dailymail_rougeL",
        )

    def _additional_quality_metrics(self, query_id, system_results, ground_truth):
        return self.score(system_results, ground_truth)
