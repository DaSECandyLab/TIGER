"""Canonical tabular scoring through the same retrieval routine as Movie."""

from numbers import Integral, Real
from collections import Counter, defaultdict
import json

import pandas as pd


def normalize_id(value):
    if pd.isna(value):
        return None
    if isinstance(value, Integral) or (isinstance(value, Real) and value == int(value)):
        return str(int(value))
    return str(value).strip()


def normalize_label(value):
    if pd.isna(value):
        return None
    return " ".join(str(value).strip().strip('"').casefold().split()) or None


def score_rows(evaluator, predictions, gold, columns, *, normalizers=None):
    """Match rows one-to-one; surplus duplicates are false positives.

    Explicit columns prevent the generic compatibility path from accidentally
    matching only a common ID while ignoring a missing answer column. Null
    predictions count as incorrect rows; null gold values fail closed.
    """
    normalizers = normalizers or {}
    frames = []
    for is_gold, frame in ((False, predictions), (True, gold)):
        if frame.empty and len(frame.columns) == 0:
            frame = pd.DataFrame(columns=columns)
        if not frame.columns.is_unique or not set(columns).issubset(frame.columns):
            raise ValueError(f"Expected unique result columns {columns}, got {list(frame.columns)}")
        frame = frame.loc[:, columns].copy()
        for col in columns:
            if col in normalizers:
                frame[col] = frame[col].map(normalizers[col])
        if is_gold and frame.isna().to_numpy().any():
            raise ValueError("Ground truth contains missing keys or labels")
        # A private hashable sentinel cannot collide with any real answer.
        invalid = object()
        for col in columns:
            frame[col] = frame[col].astype(object).map(lambda v: invalid if pd.isna(v) else v)
        frames.append(frame)
    return evaluator._generic_retrieval_evaluation(*frames)


def business_rows(frame):
    if "__quality_samples" not in frame:
        return frame
    return frame.loc[frame["__quality_samples"].isna()].drop(columns="__quality_samples").reset_index(drop=True)


def decision_samples(frame):
    if "__quality_samples" not in frame:
        raise ValueError("Missing per-request diagnostics; rerun the instrumented Flock SQL")
    payloads = frame["__quality_samples"].dropna()
    if len(payloads) != 1:
        raise ValueError("Expected exactly one per-request diagnostic payload")
    rows = json.loads(payloads.iloc[0])
    if not isinstance(rows, list) or any(set(r) != {"id", "prediction"} for r in rows):
        raise ValueError("Malformed per-request diagnostics")
    return rows


def logical_quality(samples, expected, *, binary=False):
    """Score logical input samples, not returned rows or HTTP batches.

    expected is a sequence of (ID, gold label). Repeated IDs are allowed only
    with identical gold labels (e.g. multiple symptom rows for one patient).
    Missing/invalid predictions are always incorrect. Unknown/excess IDs fail.
    """
    truth, counts = {}, Counter()
    for key, label in expected:
        key = normalize_id(key)
        if key is None or label is None:
            raise ValueError("Missing logical-sample ground truth")
        if key in truth and truth[key] != label:
            raise ValueError(f"Conflicting labels for {key}")
        truth[key] = label
        counts[key] += 1
    predictions = defaultdict(list)
    for row in samples:
        key = normalize_id(row["id"])
        if key not in counts or len(predictions[key]) >= counts[key]:
            raise ValueError(f"Unknown or excess logical sample: {key}")
        predictions[key].append(row["prediction"])
    result = Counter()
    for key, count in counts.items():
        for prediction in predictions[key] + [None] * (count - len(predictions[key])):
            if binary:
                value = str(prediction).lower() if prediction is not None else None
                prediction = {"true": True, "false": False}.get(value)
            if prediction is None:
                result["invalid"] += 1
            else:
                correct = prediction == truth[key]
                result["correct"] += int(correct)
                if binary:
                    result[("true_" if correct else "false_") + ("positive" if prediction else "negative")] += 1
    total = sum(counts.values())
    metrics = {"accuracy": result["correct"] / total if total else 1.0,
               "classification_sample_count": float(total),
               "classification_correct_count": float(result["correct"]),
               "classification_invalid_prediction_count": float(result["invalid"])}
    if binary:
        for name in ("true_positive", "true_negative", "false_positive", "false_negative"):
            metrics[f"classification_{name}_count"] = float(result[name])
        positive_count = sum(counts[k] for k in truth if truth[k])
        negative_count = total - positive_count
        metrics["balanced_accuracy"] = (
            (result["true_positive"] / positive_count if positive_count else 1.0) +
            (result["true_negative"] / negative_count if negative_count else 1.0)) / 2
    return metrics
