"""用于 trial 的 Prometheus、运行时采样和 SemBench 指标辅助工具。"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import threading
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .vllm import http_get


METRICS_PARSER_VERSION = 2
QUALITY_METRIC_NAMES = (
    "precision",
    "recall",
    "f1_score",
    "relative_error",
    "absolute_error",
    "mean_absolute_percentage_error",
    "spearman_correlation",
    "kendall_tau",
    "ranking_item_precision",
    "ranking_item_recall",
    "ranking_item_f1_score",
    "accuracy",
)
RETRIEVAL_QUALITY_METRICS = frozenset({"precision", "recall", "f1_score"})
AGGREGATION_QUALITY_METRICS = frozenset(
    {"absolute_error", "relative_error", "mean_absolute_percentage_error"}
)
RANKING_QUALITY_METRICS = frozenset(
    {
        "spearman_correlation",
        "kendall_tau",
        "ranking_item_precision",
        "ranking_item_recall",
        "ranking_item_f1_score",
    }
)


@dataclass
class PromSample:
    name: str
    labels: dict[str, str]
    value: float


class RuntimeFeatureSampler:
    """trial 执行期间的轻量运行时采样器。

    主工作流只需要 start/stop/summary 三步：
    1. workload 开始前启动；
    2. workload 跑完后停止；
    3. 把 GPU、显存、vLLM scheduler/KV gauge 汇总成 mean/p90。

    这里故意不用复杂依赖，优先调用机器上稳定存在的 `nvidia-smi` 和 vLLM
    `/metrics`，采不到就留下 None，让 workload 特征分组里明确标注缺口。
    """

    def __init__(self, metrics_url: str, interval_s: float = 0.2) -> None:
        self.metrics_url = metrics_url
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_util_samples: list[float] = []
        self._hbm_usage_samples: list[float] = []
        self._gpu_clock_samples: list[float] = []
        self._gpu_memory_clock_samples: list[float] = []
        self._gpu_power_samples: list[float] = []
        self._gpu_temperature_samples: list[float] = []
        self._cpu_steal_ratio_samples: list[float] = []
        self._previous_cpu_times: tuple[float, float] | None = None
        self._kv_cache_usage_samples: list[float] = []
        self._running_request_samples: list[float] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2.0))
        return self.summary()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_gpu()
            self._sample_cpu_steal()
            self._sample_vllm_metrics()
            self._stop_event.wait(self.interval_s)

    def _sample_gpu(self) -> None:
        # nvidia-smi 输出每张卡一行；多卡时这里取均值，表达本 trial 的整体 GPU 压力。
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,"
                    "clocks.current.graphics,clocks.current.memory,power.draw,"
                    "temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            return
        if proc.returncode != 0:
            return
        util_values: list[float] = []
        hbm_values: list[float] = []
        gpu_clock_values: list[float] = []
        memory_clock_values: list[float] = []
        power_values: list[float] = []
        temperature_values: list[float] = []
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 7:
                continue
            try:
                util = float(parts[0])
                used = float(parts[1])
                total = float(parts[2])
            except ValueError:
                continue
            util_values.append(util)
            if total > 0:
                hbm_values.append(used / total)
            for values, raw in (
                (gpu_clock_values, parts[3]),
                (memory_clock_values, parts[4]),
                (power_values, parts[5]),
                (temperature_values, parts[6]),
            ):
                try:
                    values.append(float(raw))
                except ValueError:
                    # 某些 GPU/驱动可能返回 N/A；其他指标仍应正常采集。
                    pass
        if util_values:
            self._gpu_util_samples.append(float(sum(util_values) / len(util_values)))
        if hbm_values:
            self._hbm_usage_samples.append(float(sum(hbm_values) / len(hbm_values)))
        for samples, values in (
            (self._gpu_clock_samples, gpu_clock_values),
            (self._gpu_memory_clock_samples, memory_clock_values),
            (self._gpu_power_samples, power_values),
            (self._gpu_temperature_samples, temperature_values),
        ):
            if values:
                samples.append(float(sum(values) / len(values)))

    def _sample_cpu_steal(self) -> None:
        """记录虚拟机被宿主机抢占的 CPU 比例，定位跨重启性能漂移。"""
        try:
            first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
            fields = [float(value) for value in first_line.split()[1:9]]
        except (OSError, IndexError, ValueError):
            return
        if len(fields) < 8:
            return
        current = (sum(fields), fields[7])
        previous = self._previous_cpu_times
        self._previous_cpu_times = current
        if previous is None:
            return
        total_delta = current[0] - previous[0]
        steal_delta = current[1] - previous[1]
        if total_delta > 0 and steal_delta >= 0:
            self._cpu_steal_ratio_samples.append(steal_delta / total_delta)

    def _sample_vllm_metrics(self) -> None:
        try:
            samples = parse_prometheus(http_get(self.metrics_url, timeout=3))
        except (urllib.error.URLError, TimeoutError, OSError):
            return
        kv_values = metric_values_any(
            samples,
            ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        )
        running_values = metric_values_any(
            samples,
            ["vllm:num_requests_running"],
        )
        # KV usage 是每个 engine 的比例，跨 DP engine 应取均值；running requests
        # 是每个 engine 的请求数，跨 engine 应求和。零值也必须保留，否则会把
        # 空闲采样静默丢掉并系统性高估运行时均值。
        if kv_values is not None:
            self._kv_cache_usage_samples.append(
                float(sum(kv_values) / len(kv_values))
            )
        if running_values is not None:
            self._running_request_samples.append(float(sum(running_values)))

    def summary(self) -> dict[str, float | None]:
        return {
            "gpu_utilization_mean": _mean(self._gpu_util_samples),
            "gpu_utilization_p90": _percentile(self._gpu_util_samples, 90),
            "hbm_usage_mean": _mean(self._hbm_usage_samples),
            "hbm_usage_p90": _percentile(self._hbm_usage_samples, 90),
            "gpu_graphics_clock_mhz_mean": _mean(self._gpu_clock_samples),
            "gpu_graphics_clock_mhz_p90": _percentile(self._gpu_clock_samples, 90),
            "gpu_memory_clock_mhz_mean": _mean(self._gpu_memory_clock_samples),
            "gpu_power_watts_mean": _mean(self._gpu_power_samples),
            "gpu_power_watts_p90": _percentile(self._gpu_power_samples, 90),
            "gpu_temperature_c_mean": _mean(self._gpu_temperature_samples),
            "cpu_steal_ratio_mean": _mean(self._cpu_steal_ratio_samples),
            "cpu_steal_ratio_p90": _percentile(self._cpu_steal_ratio_samples, 90),
            "kv_cache_usage_mean": _mean(self._kv_cache_usage_samples),
            "kv_cache_usage_p90": _percentile(self._kv_cache_usage_samples, 90),
            "actual_batch_size_mean": _mean(self._running_request_samples),
            "actual_batch_size_p95": _percentile(self._running_request_samples, 95),
        }


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


_METRIC_NAME_RE = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _parse_labels(text: str) -> dict[str, str] | None:
    """Parse a Prometheus/OpenMetrics label set, including escaped values."""
    labels: dict[str, str] = {}
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        name_match = _LABEL_NAME_RE.match(text, index)
        if name_match is None:
            return None
        name = name_match.group(0)
        index = name_match.end()
        while index < length and text[index].isspace():
            index += 1
        if index >= length or text[index] != "=":
            return None
        index += 1
        while index < length and text[index].isspace():
            index += 1
        if index >= length or text[index] != '"':
            return None
        index += 1
        value_chars: list[str] = []
        while index < length:
            char = text[index]
            if char == '"':
                index += 1
                break
            if char == "\\":
                index += 1
                if index >= length:
                    return None
                escaped = text[index]
                value_chars.append(
                    "\n"
                    if escaped == "n"
                    else escaped
                    if escaped in ('"', "\\")
                    else f"\\{escaped}"
                )
                index += 1
                continue
            value_chars.append(char)
            index += 1
        else:
            return None
        labels[name] = "".join(value_chars)
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        if text[index] != ",":
            return None
        index += 1
    return labels


def _parse_sample_line(line: str) -> PromSample | None:
    """Parse one text exposition sample.

    Supports both Prometheus and OpenMetrics forms, including optional
    timestamps and exemplars. Only the first token after the metric/labels is
    the sample value; timestamps must never be mistaken for the value.
    """
    name_match = _METRIC_NAME_RE.match(line)
    if name_match is None:
        return None
    name = name_match.group(0)
    index = name_match.end()
    labels: dict[str, str] = {}
    if index < len(line) and line[index] == "{":
        label_start = index + 1
        index = label_start
        quoted = False
        escaped = False
        while index < len(line):
            char = line[index]
            if escaped:
                escaped = False
            elif char == "\\" and quoted:
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif char == "}" and not quoted:
                break
            index += 1
        if index >= len(line) or line[index] != "}":
            return None
        parsed_labels = _parse_labels(line[label_start:index])
        if parsed_labels is None:
            return None
        labels = parsed_labels
        index += 1
    if index >= len(line) or not line[index].isspace():
        return None
    remainder = line[index:].lstrip()
    if not remainder:
        return None
    value_text = remainder.split(None, 1)[0]
    try:
        value = float(value_text)
    except ValueError:
        return None
    return PromSample(name=name, labels=labels, value=value)


def parse_prometheus(text: str) -> list[PromSample]:
    samples: list[PromSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sample = _parse_sample_line(line)
        if sample is not None:
            samples.append(sample)
    return samples


def metric_values(
    samples: Iterable[PromSample],
    name: str,
    labels: dict[str, str] | None = None,
) -> list[float]:
    labels = labels or {}
    return [
        sample.value
        for sample in samples
        if sample.name == name
        and math.isfinite(sample.value)
        and all(sample.labels.get(key) == value for key, value in labels.items())
    ]


def metric_values_any(
    samples: Iterable[PromSample],
    names: Iterable[str],
    labels: dict[str, str] | None = None,
) -> list[float] | None:
    sample_list = list(samples)
    for name in names:
        values = metric_values(sample_list, name, labels)
        if values:
            return values
    return None


def metric_exists(
    samples: Iterable[PromSample],
    name: str,
    labels: dict[str, str] | None = None,
) -> bool:
    return bool(metric_values(samples, name, labels))


def metric_sum(
    samples: Iterable[PromSample], name: str, labels: dict[str, str] | None = None
) -> float:
    return float(sum(metric_values(samples, name, labels)))


def _series_values(
    samples: Iterable[PromSample],
    name: str,
    labels: dict[str, str] | None = None,
) -> dict[tuple[tuple[str, str], ...], float]:
    """Return finite sample values keyed by their complete label set."""
    labels = labels or {}
    series: dict[tuple[tuple[str, str], ...], float] = {}
    for sample in samples:
        if sample.name != name or not math.isfinite(sample.value):
            continue
        if not all(sample.labels.get(key) == value for key, value in labels.items()):
            continue
        key = tuple(sorted(sample.labels.items()))
        series[key] = series.get(key, 0.0) + sample.value
    return series


def delta_metric(
    before: Iterable[PromSample],
    after: Iterable[PromSample],
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    """Compute a counter delta per label series and tolerate counter resets.

    Subtracting two already-aggregated totals is wrong when one engine resets
    while another keeps increasing. Series that disappear are ignored; newly
    appearing or reset series contribute their current non-negative value.
    """
    before_series = _series_values(before, name, labels)
    after_series = _series_values(after, name, labels)
    total = 0.0
    for key, after_value in after_series.items():
        before_value = before_series.get(key)
        if before_value is None or after_value < before_value:
            total += max(0.0, after_value)
        else:
            total += after_value - before_value
    return float(total)


def delta_metric_any(
    before: Iterable[PromSample],
    after: Iterable[PromSample],
    names: Iterable[str],
    labels: dict[str, str] | None = None,
) -> float:
    """兼容不同 vLLM/LMCache 版本的指标名。

    vLLM 版本之间可能切换 `_total` 后缀。按第一个实际存在的名称取值；
    一个合法的零增量不能触发到另一个别名的错误回退。
    """
    before_list = list(before)
    after_list = list(after)
    for name in names:
        if metric_exists(before_list, name, labels) or metric_exists(
            after_list, name, labels
        ):
            return delta_metric(before_list, after_list, name, labels)
    return 0.0


def parse_lmcache_hit_summary(log_path: Path) -> dict[str, float | int | None]:
    """从 LMCache 日志里提取 token 级命中率。

    LMCache 当前会打印形如：
    `Total tokens 5785, ... LMCache hit tokens: 5785`。
    这比只看“请求是否命中”更适合建模，因为长 prompt 的部分命中也很重要。
    """
    if not log_path.exists():
        return {
            "lmcache_request_count": 0,
            "lmcache_total_tokens": 0,
            "lmcache_hit_tokens": 0,
            "lmcache_cache_hit_rate": None,
        }
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"Total tokens\s+(\d+).*?LMCache hit tokens:\s+(\d+)",
        text,
    )
    total_tokens = sum(int(total) for total, _ in matches)
    hit_tokens = sum(int(hit) for _, hit in matches)
    return {
        "lmcache_request_count": len(matches),
        "lmcache_total_tokens": total_tokens,
        "lmcache_hit_tokens": hit_tokens,
        "lmcache_cache_hit_rate": (
            float(hit_tokens / total_tokens) if total_tokens > 0 else None
        ),
    }


def histogram_quantile_delta(
    before: Iterable[PromSample], after: Iterable[PromSample], name: str, quantile: float
) -> float | None:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    before_list = list(before)
    after_list = list(after)
    bucket_name = f"{name}_bucket"
    bounds = {
        sample.labels["le"]
        for sample in after_list
        if sample.name == bucket_name and "le" in sample.labels
    }
    infinite_bounds = {
        bound
        for bound in bounds
        if bound.lower() in {"+inf", "inf"}
    }
    if not infinite_bounds:
        return None
    total = sum(
        delta_metric(
            before_list,
            after_list,
            bucket_name,
            {"le": bound},
        )
        for bound in infinite_bounds
    )
    if total <= 0:
        return None
    target = total * quantile
    numeric_buckets: list[tuple[float, float]] = []
    for le_text in bounds:
        if le_text in infinite_bounds:
            continue
        try:
            le = float(le_text)
        except ValueError:
            continue
        if not math.isfinite(le):
            continue
        cumulative_delta = delta_metric(
            before_list,
            after_list,
            bucket_name,
            {"le": le_text},
        )
        numeric_buckets.append((le, cumulative_delta))
    numeric_buckets.sort(key=lambda item: item[0])
    previous_le = 0.0
    previous_cumulative = 0.0
    for le, cumulative_delta in numeric_buckets:
        # Counter resets or a scrape racing an update can make adjacent bucket
        # deltas slightly non-monotonic. A cumulative histogram cannot decrease.
        cumulative_delta = max(previous_cumulative, cumulative_delta)
        if cumulative_delta >= target:
            bucket_count = cumulative_delta - previous_cumulative
            if bucket_count <= 0 or le <= previous_le:
                return le
            bucket_fraction = (target - previous_cumulative) / bucket_count
            bucket_fraction = min(1.0, max(0.0, bucket_fraction))
            return previous_le + (le - previous_le) * bucket_fraction
        previous_le = le
        previous_cumulative = cumulative_delta
    # Observations in +Inf cannot be interpolated. PromQL returns the highest
    # finite upper bound in this case, which is more useful than dropping P95.
    return numeric_buckets[-1][0] if numeric_buckets else None


def histogram_quantile_delta_any(
    before: Iterable[PromSample],
    after: Iterable[PromSample],
    names: Iterable[str],
    quantile: float,
) -> float | None:
    """按候选 histogram 名称提取一个分位数。

    不同 vLLM 版本 TPOT 相关指标名略有差异；主流程只关心“能否拿到可用分位数”，
    因此这里按优先级返回第一个存在的 histogram。
    """
    for name in names:
        value = histogram_quantile_delta(before, after, name, quantile)
        if value is not None:
            return value
    return None


def histogram_mean_delta_any(
    before: Iterable[PromSample],
    after: Iterable[PromSample],
    names: Iterable[str],
) -> float | None:
    """按候选 histogram 名称用 sum/count 提取均值。"""
    before_list = list(before)
    after_list = list(after)
    for name in names:
        count_name = f"{name}_count"
        sum_name = f"{name}_sum"
        if not (
            metric_exists(before_list, count_name)
            or metric_exists(after_list, count_name)
        ):
            continue
        count = delta_metric(before_list, after_list, count_name)
        total = delta_metric(before_list, after_list, sum_name)
        if count > 0:
            return total / count
        return None
    return None


def histogram_count_delta_any(
    before: Iterable[PromSample],
    after: Iterable[PromSample],
    names: Iterable[str],
) -> float:
    """Return the observation count for the first available histogram."""
    before_list = list(before)
    after_list = list(after)
    for name in names:
        count_name = f"{name}_count"
        if metric_exists(before_list, count_name) or metric_exists(
            after_list, count_name
        ):
            return delta_metric(before_list, after_list, count_name)
    return 0.0


def summarize_vllm_latency_metrics(
    before: Iterable[PromSample],
    after: Iterable[PromSample],
    *,
    completed_request_count: float,
    generation_tokens: float,
) -> dict[str, float | int | None]:
    """Compute TTFT, TPOT and E2E metrics with consistent units and fallbacks."""
    before_list = list(before)
    after_list = list(after)
    ttft_name = "vllm:time_to_first_token_seconds"
    e2e_name = "vllm:e2e_request_latency_seconds"
    tpot_names = [
        "vllm:request_time_per_output_token_seconds",
        "vllm:inter_token_latency_seconds",
    ]

    ttft_count = delta_metric(before_list, after_list, f"{ttft_name}_count")
    ttft_sum = delta_metric(before_list, after_list, f"{ttft_name}_sum")
    e2e_count = delta_metric(before_list, after_list, f"{e2e_name}_count")
    e2e_sum = delta_metric(before_list, after_list, f"{e2e_name}_sum")
    mean_ttft_ms = (
        ttft_sum / ttft_count * 1000.0 if ttft_count > 0 else None
    )
    mean_e2e_ms = e2e_sum / e2e_count * 1000.0 if e2e_count > 0 else None

    ttft_p50 = histogram_quantile_delta(
        before_list, after_list, ttft_name, 0.50
    )
    ttft_p95 = histogram_quantile_delta(
        before_list, after_list, ttft_name, 0.95
    )
    ttft_p99 = histogram_quantile_delta(
        before_list, after_list, ttft_name, 0.99
    )
    e2e_p95 = histogram_quantile_delta(
        before_list, after_list, e2e_name, 0.95
    )
    mean_tpot_s = histogram_mean_delta_any(
        before_list, after_list, tpot_names
    )
    tpot_count = histogram_count_delta_any(
        before_list, after_list, tpot_names
    )
    tpot_p50 = histogram_quantile_delta_any(
        before_list, after_list, tpot_names, 0.50
    )
    tpot_p95 = histogram_quantile_delta_any(
        before_list, after_list, tpot_names, 0.95
    )

    # Fall back only when vLLM does not expose a TPOT histogram:
    # TPOT ~= (sum(E2E) - sum(TTFT)) / non-first generated tokens.
    request_count = (
        completed_request_count
        if completed_request_count > 0
        else e2e_count
    )
    decode_token_count = generation_tokens - request_count
    proxy_tpot_ms: float | None = None
    if request_count > 0 and decode_token_count <= 0:
        proxy_tpot_ms = 0.0
    elif decode_token_count > 0 and e2e_sum >= ttft_sum:
        proxy_tpot_ms = ((e2e_sum - ttft_sum) / decode_token_count) * 1000.0

    mean_tpot_ms = (
        mean_tpot_s * 1000.0
        if mean_tpot_s is not None
        else proxy_tpot_ms
    )
    tpot_p50_ms = (
        tpot_p50 * 1000.0 if tpot_p50 is not None else proxy_tpot_ms
    )
    tpot_p95_ms = (
        tpot_p95 * 1000.0 if tpot_p95 is not None else proxy_tpot_ms
    )
    return {
        "ttft_sample_count": int(ttft_count),
        "e2e_latency_sample_count": int(e2e_count),
        "tpot_sample_count": int(tpot_count),
        "mean_ttft_ms": mean_ttft_ms,
        "ttft_p50_ms": ttft_p50 * 1000.0 if ttft_p50 is not None else None,
        "ttft_p95_ms": ttft_p95 * 1000.0 if ttft_p95 is not None else None,
        "p99_ttft_ms": ttft_p99 * 1000.0 if ttft_p99 is not None else None,
        "mean_tpot_ms": mean_tpot_ms,
        "tpot_p50_ms": tpot_p50_ms,
        "tpot_p95_ms": tpot_p95_ms,
        "mean_e2e_latency_ms": mean_e2e_ms,
        "p95_latency_ms": e2e_p95 * 1000.0 if e2e_p95 is not None else None,
    }


def load_sembench_metrics(args: argparse.Namespace) -> dict:
    metrics_path = Path(args.sembench_dir) / "files" / args.use_case / "metrics" / "flockmtl.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as fh:
        all_metrics = json.load(fh)
    selected = {f"Q{query}": all_metrics.get(f"Q{query}") for query in args.queries}
    return {key: value for key, value in selected.items() if value is not None}


def _finite_quality_values(rows: list[dict], metric_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            values.append(numeric_value)
    return values


def _representative_quality_metric_names(row: dict) -> tuple[str, ...]:
    metric_names = set(row)
    if metric_names & RETRIEVAL_QUALITY_METRICS:
        return ("precision", "recall")
    if metric_names & AGGREGATION_QUALITY_METRICS:
        return ("absolute_error", "relative_error")
    if metric_names & RANKING_QUALITY_METRICS:
        return ("spearman_correlation", "ranking_item_recall")
    if "accuracy" in metric_names:
        return ("accuracy",)
    return ()


def _summarize_quality_metrics(
    rows: list[dict], expected_metric_count: int
) -> dict:
    metric_values: dict[str, list[float]] = {}
    for row in rows:
        for metric_name in _representative_quality_metric_names(row):
            values = _finite_quality_values([row], metric_name)
            if values:
                metric_values.setdefault(metric_name, []).extend(values)
    metric_counts = {
        metric_name: len(values)
        for metric_name, values in metric_values.items()
        if values
    }
    recognized_metric_count = sum(
        1
        for row in rows
        if any(
            _finite_quality_values([row], metric_name)
            for metric_name in _representative_quality_metric_names(row)
        )
    )
    coverage = (
        min(1.0, recognized_metric_count / expected_metric_count)
        if expected_metric_count > 0
        else 0.0
    )
    return {
        "quality_metric_counts": metric_counts,
        "quality_metric_coverage": coverage,
    }


def summarize_sembench(metrics: dict) -> dict:
    query_rows = [row for row in metrics.values() if isinstance(row, dict)]
    def is_successful(row: dict) -> bool:
        status = row.get("status")
        return status == "success" or (
            status is None
            and any(
                _finite_quality_values([row], metric_name)
                for metric_name in QUALITY_METRIC_NAMES
            )
        )

    failed = [row for row in query_rows if not is_successful(row)]
    successful_rows = [
        row for row in query_rows if is_successful(row)
    ]
    execution_times = [
        float(row.get("execution_time") or 0.0)
        for row in query_rows
        if is_successful(row)
    ]
    query_wall_times = [
        float(row.get("query_wall_time") or 0.0)
        for row in query_rows
        if is_successful(row) and row.get("query_wall_time") is not None
    ]
    return {
        "query_count": len(query_rows),
        "failed_query_count": len(failed),
        "total_execution_time": sum(execution_times),
        "max_execution_time": max(execution_times) if execution_times else None,
        "total_query_wall_time": sum(query_wall_times),
        "max_query_wall_time": (
            max(query_wall_times) if query_wall_times else None
        ),
        **_summarize_quality_metrics(
            successful_rows,
            expected_metric_count=len(query_rows),
        ),
    }


def load_tpcc_workload_metrics(workload_metrics_path: Path) -> dict:
    """读取 TPCC workload driver 输出的聚合指标。"""
    if not workload_metrics_path.exists():
        return {}
    with workload_metrics_path.open("r", encoding="utf-8") as metric_file:
        return json.load(metric_file)


def load_tpcc_sembench_metrics(workload_metrics: dict) -> dict:
    """从 TPCC workload 聚合结果中提取 query 级 evaluate 指标，复用 trial 结果结构。"""
    evaluation = workload_metrics.get("evaluation")
    if not isinstance(evaluation, dict):
        return {}
    query_metrics = evaluation.get("queries")
    if not isinstance(query_metrics, dict):
        return {}
    return query_metrics
