"""
Created on July 27, 2025

@author: Jiale Lao

MovieEvaluator Implementation based on generic_evaluator
Uses DuckDB with ground truth SQL queries to generate reference results
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict
import sys
import pandas as pd
import numpy as np
import duckdb
from scipy.stats import spearmanr, kendalltau

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluator.generic_evaluator import GenericEvaluator, QueryMetricRetrieval, QueryMetricAggregation, QueryMetricRank

class MovieEvaluator(GenericEvaluator):
    """Evaluator for the movie benchmark using the reusable framework."""

    _EMBEDDED_CLASSIFICATION_COLUMNS = frozenset(
        {
            "classification_sample_count",
            "classification_true_positive_count",
            "classification_true_negative_count",
            "classification_false_positive_count",
            "classification_false_negative_count",
            "classification_invalid_prediction_count",
        }
    )

    def __init__(self, use_case: str, scale_factor: int = None) -> None:
        super().__init__(use_case, scale_factor)

    def _load_domain_data(self) -> None:
        """Load the movie data CSV files."""
        data_path = self._root / "data" / f"sf_{self.scale_factor}"
        self.movies_df = pd.read_csv(data_path / "Movies.csv")
        self.reviews_df = pd.read_csv(data_path / "Reviews.csv")

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:  
        """Generate ground truth using DuckDB and gold SQL files."""
        sql_path = self._root / "query" / "gold_sql" / f"Q{query_id}.sql"
        
        if not sql_path.exists():
            raise FileNotFoundError(f"Gold SQL file not found: {sql_path}")
        
        # Read the SQL query
        with open(sql_path, 'r') as f:
            sql_query = f.read().strip()
        
        # Create DuckDB connection and load data
        conn = duckdb.connect()
        
        # Register DataFrames as tables
        conn.register('Movies', self.movies_df)
        conn.register('Reviews', self.reviews_df)
        
        # Execute the query and get results
        result = conn.execute(sql_query).fetchdf()
        
        # Save ground truth
        gt_path = self._results_path / "ground_truth" / f"Q{query_id}.csv"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(gt_path, index=False)
        
        conn.close()
        return result

    def _evaluate_single_query(self, query_id: int, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> "QueryMetricRetrieval | QueryMetricAggregation | QueryMetricRank":
        """Evaluate a single query based on its type."""
        evaluate_fn = self._discover_evaluate_impl(query_id)
        return evaluate_fn(system_results, ground_truth)

    @staticmethod
    def _binary_classification_quality(
        predicted_positive,
        actual_positive,
        universe,
        *,
        invalid_prediction_count: int = 0,
    ) -> Dict[str, float]:
        """Compute accuracy on the logical input universe, not output rows."""

        predicted = Counter(predicted_positive) & Counter(universe)
        actual = Counter(actual_positive) & Counter(universe)
        true_positives = sum((predicted & actual).values())
        false_positives = sum((predicted - actual).values())
        false_negatives = sum((actual - predicted).values())
        sample_count = sum(Counter(universe).values())
        true_negatives = (
            sample_count
            - true_positives
            - false_positives
            - false_negatives
            - invalid_prediction_count
        )
        accuracy = (
            (true_positives + true_negatives) / sample_count
            if sample_count
            else 1.0
        )
        positive_count = sum(actual.values())
        negative_count = sample_count - positive_count
        positive_recall = true_positives / positive_count if positive_count else 1.0
        negative_recall = true_negatives / negative_count if negative_count else 1.0
        return {
            "accuracy": float(accuracy),
            "balanced_accuracy": float((positive_recall + negative_recall) / 2),
            "classification_sample_count": float(sample_count),
            "classification_true_positive_count": float(true_positives),
            "classification_true_negative_count": float(true_negatives),
            "classification_false_positive_count": float(false_positives),
            "classification_false_negative_count": float(false_negatives),
            "classification_invalid_prediction_count": float(
                invalid_prediction_count
            ),
        }

    @staticmethod
    def _normalized_pair(left, right) -> tuple[str, str]:
        return tuple(sorted((str(left), str(right))))

    def _review_classification_quality(
        self, query_id: int, system_results: pd.DataFrame
    ) -> Dict[str, float]:
        reviews = self.reviews_df
        if query_id == 2:
            reviews = reviews[reviews["id"] == "taken_3"]
        universe = Counter(reviews["reviewId"].dropna().astype(str))
        actual = Counter(
            reviews.loc[
                reviews["scoreSentiment"].astype(str).str.upper() == "POSITIVE",
                "reviewId",
            ]
            .dropna()
            .astype(str)
        )
        predicted = (
            Counter(system_results.iloc[:, 0].dropna().astype(str))
            if len(system_results.columns) >= 1
            else Counter()
        )
        return self._binary_classification_quality(predicted, actual, universe)

    def _sampled_pair_truth(self) -> Dict[int, Dict[str, set[tuple[str, str]]]]:
        cached = getattr(self, "_sampled_pair_truth_cache", None)
        if cached is not None:
            return cached
        connection = duckdb.connect()
        try:
            connection.register("movie_accuracy_reviews", self.reviews_df)
            sampled = connection.execute(
                """
                WITH unique_reviews AS (
                  SELECT reviewId, MIN(scoreSentiment) AS scoreSentiment
                  FROM movie_accuracy_reviews
                  WHERE id = 'ant_man_and_the_wasp_quantumania'
                  GROUP BY reviewId
                )
                SELECT
                  r1.reviewId AS reviewId1,
                  r2.reviewId AS reviewId2,
                  r1.scoreSentiment AS sentiment1,
                  r2.scoreSentiment AS sentiment2
                FROM unique_reviews AS r1
                JOIN unique_reviews AS r2 ON r1.reviewId < r2.reviewId
                ORDER BY hash(r1.reviewId, r2.reviewId), r1.reviewId, r2.reviewId
                LIMIT 2000
                """
            ).fetchdf()
        finally:
            connection.close()
        universe: set[tuple[str, str]] = set()
        same: set[tuple[str, str]] = set()
        opposite: set[tuple[str, str]] = set()
        for row in sampled.itertuples(index=False):
            pair = self._normalized_pair(row.reviewId1, row.reviewId2)
            universe.add(pair)
            (same if row.sentiment1 == row.sentiment2 else opposite).add(pair)
        cached = {
            5: {"universe": universe, "positive": same},
            6: {"universe": universe, "positive": opposite},
            7: {"universe": universe, "positive": opposite},
        }
        self._sampled_pair_truth_cache = cached
        return cached

    def _pair_classification_quality(
        self, query_id: int, system_results: pd.DataFrame
    ) -> Dict[str, float]:
        truth = self._sampled_pair_truth()[query_id]
        predicted = {
            self._normalized_pair(row[1], row[2])
            for row in system_results.itertuples(index=False, name=None)
            if len(row) >= 3 and pd.notna(row[1]) and pd.notna(row[2])
        }
        return self._binary_classification_quality(
            Counter(predicted),
            Counter(truth["positive"]),
            Counter(truth["universe"]),
        )

    @staticmethod
    def _embedded_classification_quality(
        system_results: pd.DataFrame,
    ) -> Dict[str, float]:
        def first_number(column: str):
            if column not in system_results.columns:
                return None
            numeric = pd.to_numeric(system_results[column], errors="coerce").dropna()
            return float(numeric.iloc[0]) if not numeric.empty else None

        sample_count = first_number("classification_sample_count")
        true_positive = first_number("classification_true_positive_count")
        true_negative = first_number("classification_true_negative_count")
        false_positive = first_number("classification_false_positive_count")
        false_negative = first_number("classification_false_negative_count")
        invalid = first_number("classification_invalid_prediction_count")
        values = (
            sample_count,
            true_positive,
            true_negative,
            false_positive,
            false_negative,
            invalid,
        )
        if any(value is None for value in values) or not sample_count:
            raise ValueError(
                "aggregation result lacks embedded classification confusion counts"
            )
        positive_count = true_positive + false_negative
        negative_count = true_negative + false_positive
        positive_recall = true_positive / positive_count if positive_count else 1.0
        negative_recall = true_negative / negative_count if negative_count else 1.0
        return {
            "accuracy": float((true_positive + true_negative) / sample_count),
            "balanced_accuracy": float((positive_recall + negative_recall) / 2),
            "classification_sample_count": float(sample_count),
            "classification_true_positive_count": float(true_positive),
            "classification_true_negative_count": float(true_negative),
            "classification_false_positive_count": float(false_positive),
            "classification_false_negative_count": float(false_negative),
            "classification_invalid_prediction_count": float(invalid),
        }

    @classmethod
    def _has_embedded_classification_quality(
        cls, system_results: pd.DataFrame
    ) -> bool:
        """Return whether a packed query emitted its confusion-count contract."""

        return cls._EMBEDDED_CLASSIFICATION_COLUMNS.issubset(system_results.columns)

    @classmethod
    def _embedded_retrieval_evaluation(
        cls, system_results: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """Derive legacy retrieval diagnostics from packed confusion counts."""

        quality = cls._embedded_classification_quality(system_results)
        true_positive = quality["classification_true_positive_count"]
        false_positive = quality["classification_false_positive_count"]
        false_negative = quality["classification_false_negative_count"]
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 1.0 if not recall_denominator else 0.0
        )
        recall = (
            true_positive / recall_denominator if recall_denominator else 1.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return QueryMetricRetrieval(
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
        )

    @staticmethod
    def _q9_within1_accuracy(
        system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> float:
        predicted = system_results.groupby("reviewId")["reviewScore"].mean()
        actual = ground_truth.groupby("reviewId")["reviewScore"].mean()
        common_ids = predicted.index.intersection(actual.index)
        if common_ids.empty:
            return 0.0
        absolute_errors = (predicted.loc[common_ids] - actual.loc[common_ids]).abs()
        return float((absolute_errors <= 1.0).mean())

    def _additional_quality_metrics(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> Dict[str, float]:
        """Expose the scorecleanv3 Q1-Q8 logical-sample accuracy contract."""

        if query_id in {1, 2, 5, 6, 7} and self._has_embedded_classification_quality(
            system_results
        ):
            return self._embedded_classification_quality(system_results)
        if query_id in {1, 2}:
            return self._review_classification_quality(query_id, system_results)
        if query_id in {5, 6, 7}:
            return self._pair_classification_quality(query_id, system_results)
        if query_id in {3, 4, 8}:
            return self._embedded_classification_quality(system_results)
        if query_id == 9:
            if "repeat_id" in system_results.columns:
                repeat_values = [
                    self._q9_within1_accuracy(group, ground_truth)
                    for _, group in system_results.groupby("repeat_id", dropna=False)
                ]
                within1_accuracy = (
                    float(np.mean(repeat_values)) if repeat_values else 0.0
                )
            else:
                within1_accuracy = self._q9_within1_accuracy(
                    system_results, ground_truth
                )
            return {"within1_accuracy": within1_accuracy}
        return {}

    def _evaluate_q1(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q1: All clearly positive reviews (any movie)."""
        if self._has_embedded_classification_quality(system_results):
            return self._embedded_retrieval_evaluation(system_results)
        return self._generic_retrieval_evaluation(system_results, ground_truth)
    
    def _evaluate_q2(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q2: All positive reviews for movie "taken_3"."""
        if self._has_embedded_classification_quality(system_results):
            return self._embedded_retrieval_evaluation(system_results)
        return self._generic_retrieval_evaluation(system_results, ground_truth)
    
    def _evaluate_q3(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricAggregation:
        """Q3: Count of positive reviews for movie "taken_3" - aggregation query."""
        return self._generic_aggregation_evaluation(system_results, ground_truth)
    
    def _evaluate_q4(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricAggregation:
        """Q4: Positivity ratio of reviews for movie "taken_3" - aggregation query."""
        return self._generic_aggregation_evaluation(system_results, ground_truth)
    
    def _evaluate_q5(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q5: All review pairs that express the same sentiment."""
        return self._evaluate_review_pairs(system_results, ground_truth)
    
    def _evaluate_q6(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q6: All review pairs that express opposite sentiments."""
        if self._has_embedded_classification_quality(system_results):
            return self._embedded_retrieval_evaluation(system_results)
        return self._evaluate_review_pairs(system_results, ground_truth)
    
    def _evaluate_q7(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q7: All Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania' - retrieval query without limit."""
        return self._evaluate_review_pairs(system_results, ground_truth)
    
    def _evaluate_q8(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricAggregation:
        """Q8: Calculate the number of positive and negative reviews for movie "taken_3" - aggregation query."""
        return self._evaluate_sentiment_counts(system_results, ground_truth)
    
    def _evaluate_q9(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRank:
        """Q9: Score from 1 to 5 how much did the reviewer like the movie - ranking query."""
        if "repeat_id" in system_results.columns:
            repeat_metrics = [
                self._generic_ranking_evaluation(group, ground_truth)
                for _, group in system_results.groupby("repeat_id", dropna=False)
            ]
            if not repeat_metrics:
                return QueryMetricRank()
            return QueryMetricRank(
                spearman_correlation=float(
                    np.mean([metric.spearman_correlation for metric in repeat_metrics])
                ),
                kendall_tau=float(
                    np.mean([metric.kendall_tau for metric in repeat_metrics])
                ),
                ranking_item_precision=float(
                    np.mean([metric.ranking_item_precision for metric in repeat_metrics])
                ),
                ranking_item_recall=float(
                    np.mean([metric.ranking_item_recall for metric in repeat_metrics])
                ),
                ranking_item_f1_score=float(
                    np.mean([metric.ranking_item_f1_score for metric in repeat_metrics])
                ),
            )
        return self._generic_ranking_evaluation(system_results, ground_truth)
    
    def _evaluate_q10(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRank:
        """Q10: Rank the movies based on movie reviews - ranking query."""
        return self._generic_ranking_evaluation(system_results, ground_truth)

    def _evaluate_review_pairs(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """
        Evaluate queries that return pairs of reviews (Q5, Q6).
        
        Uses column positions instead of names: assumes first 3 columns are id, reviewId1, reviewId2.
        """
        if len(system_results) == 0:
            return QueryMetricRetrieval(precision=1.0 if len(ground_truth) == 0 else 0.0)
        if len(ground_truth) == 0:
            return QueryMetricRetrieval()

        # Ensure we have at least 3 columns (id, reviewId1, reviewId2)
        if len(system_results.columns) < 3 or len(ground_truth.columns) < 3:
            return QueryMetricRetrieval()

        def create_pair_tuple(row, columns):
            """Create a normalized tuple for pair comparison using column positions."""
            if len(columns) >= 3:
                movie_id = row[columns[0]]  # First column: movie id
                val1 = row[columns[1]]     # Second column: reviewId1  
                val2 = row[columns[2]]     # Third column: reviewId2
                
                if pd.notna(val1) and pd.notna(val2) and pd.notna(movie_id):
                    base_tuple = tuple(sorted([val1, val2]))
                    return (movie_id, base_tuple)
            return None

        # Create sets of normalized pairs using column positions
        sys_cols = list(system_results.columns)
        gt_cols = list(ground_truth.columns)
        
        sys_pairs = {t for t in system_results.apply(lambda r: create_pair_tuple(r, sys_cols), axis=1) if t is not None}
        gt_pairs = {t for t in ground_truth.apply(lambda r: create_pair_tuple(r, gt_cols), axis=1) if t is not None}
        
        # Calculate metrics
        correct = sys_pairs & gt_pairs
        precision = len(correct) / len(sys_pairs) if sys_pairs else 0.0
        recall = len(correct) / len(gt_pairs) if gt_pairs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        
        return QueryMetricRetrieval(precision, recall, f1)

    def _evaluate_sentiment_counts(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricAggregation:
        """
        Evaluate Q7 sentiment count query (GROUP BY scoreSentiment with COUNT).
        
        Compares counts for each sentiment type and calculates aggregation errors.
        """
        if len(system_results) == 0 or len(ground_truth) == 0:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0
            )

        # Use column positions: first column is sentiment, second is count
        if len(system_results.columns) < 2 or len(ground_truth.columns) < 2:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0
            )
        
        sys_sentiment_col = system_results.columns[0]
        sys_count_col = system_results.columns[1]
        gt_sentiment_col = ground_truth.columns[0]
        gt_count_col = ground_truth.columns[1]
        
        # Create dictionaries for easy lookup
        sys_counts = {}
        for _, row in system_results.iterrows():
            sentiment = row[sys_sentiment_col]
            count = row[sys_count_col]
            if pd.notna(sentiment) and pd.notna(count):
                try:
                    sys_counts[str(sentiment).strip().upper()] = float(count)
                except (ValueError, TypeError):
                    continue
        
        gt_counts = {}
        for _, row in ground_truth.iterrows():
            sentiment = row[gt_sentiment_col]
            count = row[gt_count_col]
            if pd.notna(sentiment) and pd.notna(count):
                try:
                    gt_counts[str(sentiment).strip().upper()] = float(count)
                except (ValueError, TypeError):
                    continue
        
        if not sys_counts or not gt_counts:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0
            )
        
        # Calculate errors for each sentiment type
        total_absolute_error = 0.0
        total_relative_error = 0.0
        valid_comparisons = 0
        
        # Get all sentiment types from both results
        all_sentiments = set(sys_counts.keys()) | set(gt_counts.keys())
        
        for sentiment in all_sentiments:
            sys_count = sys_counts.get(sentiment, 0.0)
            gt_count = gt_counts.get(sentiment, 0.0)
            
            # Calculate absolute error
            abs_error = abs(sys_count - gt_count)
            total_absolute_error += abs_error
            
            # Calculate relative error (avoid division by zero)
            if gt_count != 0:
                rel_error = abs_error / abs(gt_count)
                total_relative_error += rel_error
                valid_comparisons += 1
            elif sys_count != 0:
                # Ground truth is 0 but system predicted non-zero
                total_relative_error += 1.0
                valid_comparisons += 1
            # If both are 0, no error to add
        
        # Calculate final metrics
        result = QueryMetricAggregation()
        result.absolute_error = total_absolute_error
        result.aggregation_reference_scale = max(
            1.0,
            sum(abs(value) for value in gt_counts.values()),
        )
        
        if valid_comparisons > 0:
            result.relative_error = total_relative_error / valid_comparisons
            result.mean_absolute_percentage_error = result.relative_error * 100
        else:
            result.relative_error = 0.0
            result.mean_absolute_percentage_error = 0.0
        
        return result
