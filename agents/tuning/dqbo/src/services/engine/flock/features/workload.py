"""从 Flock/SemBench trial 产物中提取 workload 建模特征。

主工作流只有三步：
1. `extract_workload_features`：把 workload.json、query trace、vLLM/LMCache/GPU
   运行时指标合成一份扁平特征。
2. `categorize_workload_features`：按方案文档分成 benchmark/query/data/runtime/quality
   五组，方便人工检查哪些特征缺采样源。
3. `workload_features_for_model`：把特征转成 BO/随机森林可直接消费的数值/类别列。
"""

from __future__ import annotations

import math
import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np


# 模型输入字段清单：嵌套分布会留在 JSON 里供人工分析；进入模型时只取稳定摘要。
# 隐藏约束随机森林只能吃扁平的数值/类别列；这里列出会进入模型的
# 这些是 workload 数值特征。嵌套的分布特征会同时保留在结果 JSON 中，但给模型时
# 只取 p50/p90/p95/max 这类稳定摘要。
NUMERIC_WORKLOAD_FEATURES = (
    "terminal_count",
    "max_in_flight",
    "effective_per_terminal_max_in_flight",
    "request_rate",
    "terminal_query_slot_count",
    "terminal_unique_query_count",
    "query_instance_count",
    "total_llm_request_count",
    "mean_llm_requests_per_query",
    "p50_llm_requests_per_query",
    "p95_llm_requests_per_query",
    "max_llm_requests_per_query",
    "candidate_rows_p50",
    "candidate_rows_p90",
    "candidate_rows_max",
    "theoretical_global_in_flight_upper",
    "mean_wave_count",
    "max_wave_count",
    "mean_configured_request_cap",
    "max_configured_request_cap",
    "query_throughput_qps",
    "mean_query_execution_latency_ms",
    "p95_query_execution_latency_ms",
    "mean_ttft_ms",
    "mean_e2e_latency_ms",
    "p95_latency_ms",
    "prompt_tokens_per_request",
    "generation_tokens_per_request",
    "input_output_ratio",
    "prefix_cache_hit_rate",
    "lmcache_cache_hit_rate",
    "system_prompt_len_p50",
    "system_prompt_len_p95",
    "system_prompt_len_max",
    "data_prompt_len_p50",
    "data_prompt_len_p95",
    "data_prompt_len_max",
    "output_len_p50",
    "output_len_p95",
    "output_len_max",
    "template_reuse_template_count",
    "template_reuse_max_count",
    "prefix_reuse_ratio",
    "shared_prefix_len_p50",
    "shared_prefix_len_p90",
    "shared_prefix_len_max",
    "shared_cacheblend_len_p50",
    "shared_cacheblend_len_p90",
    "shared_cacheblend_len_max",
    "cacheblend_reuse_ratio",
    "gpu_utilization_mean",
    "gpu_utilization_p90",
    "kv_cache_usage_mean",
    "kv_cache_usage_p90",
    "hbm_usage_mean",
    "hbm_usage_p90",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "actual_batch_size_mean",
    "actual_batch_size_p95",
    "actual_global_in_flight_peak",
    "quality_gate_passed",
    "quality_metric_coverage",
)

# 类别型 workload 特征保留原始标签，后续由 BO 侧 one-hot 编码。
CATEGORICAL_WORKLOAD_FEATURES = (
    "request_control_mode",
    "request_cap_strategy",
    "request_cap_formula",
    "arrival_mode",
    "concurrent_query_mix",
)

WORKLOAD_FEATURE_NAMES = list(NUMERIC_WORKLOAD_FEATURES + CATEGORICAL_WORKLOAD_FEATURES)
MISSING_NUMERIC_VALUE = None
UNLIMITED_REQUEST_RATE_FEATURE_VALUE = 1_000_000_000.0


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """把外部配置/指标安全转成 float；inf/unlimited 保留为 inf。"""
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "inf", "infinity", "unlimited"}:
        return math.inf
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def _finite_float(value: Any, default: float = -1.0) -> float:
    # 模型输入不能出现 NaN/Inf/None；不可用值统一落到 -1。
    # 结果 JSON 里的原始特征仍保留 None，便于判断哪些指标缺采样源。
    parsed = _safe_float(value)
    if parsed is None or math.isinf(parsed):
        return default
    return float(parsed)


def _json_feature_float(value: Any, default: float | None = None) -> float | None:
    # 写入结果 JSON 的原始 workload 特征保留 None 表示不可用/不适用；
    # 后续进入模型时再由 workload_features_for_model 统一转成 -1。
    parsed = _safe_float(value, default)
    if parsed is None or math.isinf(parsed):
        return default
    return float(parsed)


def _request_rate_feature(value: Any) -> float | None:
    """把 request_rate 转成模型可用数值；inf/unlimited 映射为固定极大值。"""
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if math.isinf(parsed):
        return UNLIMITED_REQUEST_RATE_FEATURE_VALUE
    return float(parsed)


def _percentile(values: list[float], q: float) -> float | None:
    """统一计算百分位；空样本返回 None，避免把缺失误写成 0。"""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _distribution(values: list[float], percentiles: tuple[int, ...]) -> dict[str, float | None]:
    """把一组样本压缩成 pXX/max 分布摘要。"""
    return {
        **{f"p{q}": _percentile(values, q) for q in percentiles},
        "max": max(values) if values else None,
    }


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    """计算两个 token-id 序列的最长公共前缀长度。"""
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def _longest_common_prefix_text(a: str, b: str) -> str:
    """计算两个文本 prompt 的最长公共前缀。"""
    idx = 0
    max_idx = min(len(a), len(b))
    while idx < max_idx and a[idx] == b[idx]:
        idx += 1
    return a[:idx]


def _estimate_token_len(text: str, full_text: str | None, full_token_count: int | None) -> int:
    """估算子串 token 数；有完整 prompt token 数时按字符占比折算。"""
    if not text:
        return 0
    if full_text and full_token_count and len(full_text) > 0:
        # 如果 trace 里有完整 prompt 令牌数但没有分段令牌数，则用字符占比
        # 估算 system/data/prefix 子段长度；这比纯 split 更接近 tokenizer 结果。
        return int(round(full_token_count * (len(text) / len(full_text))))
    # 仅在 provider 未暴露 usage/token ids 时，才退回到按空白分词的粗估。
    return max(1, len(text.split()))


def _terminal_sequences(workload_metrics: dict[str, Any]) -> dict[int, list[int]]:
    """读取 terminal -> query 序列，用于 workload 配置和 query mix 特征。"""
    # 这里兼容 TPCC workload：每个 terminal 可有一条 query 序列；旧格式只有一条共享序列。
    # 这里统一成 terminal_id -> [query_id]，用于 concurrent_query_mix 和 slot 统计。
    sequences = workload_metrics.get("terminal_sql_sequences")
    if isinstance(sequences, dict):
        return {
            int(terminal_id): [(int(query_id) if str(query_id).isdigit() else str(query_id)) for query_id in query_ids]
            for terminal_id, query_ids in sequences.items()
            if isinstance(query_ids, list)
        }

    shared = workload_metrics.get("terminal_sql_sequence")
    if isinstance(shared, list):
        terminal_count = int(workload_metrics.get("terminal_count") or 0)
        return {
            terminal_id: [(int(query_id) if str(query_id).isdigit() else str(query_id)) for query_id in shared]
            for terminal_id in range(terminal_count)
        }
    return {}


def _query_records(workload_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """从 TPCC workload JSON 中展开 terminal/query 记录。"""
    # 在 SemBench 的 TPCC driver 中，每个 terminal 的 query 结果会嵌套在 terminals 下。
    # 展平后每个 query record 都带 terminal_id，方便后续按 terminal/step 建模。
    terminals = workload_metrics.get("terminals")
    if not isinstance(terminals, list):
        return []
    records: list[dict[str, Any]] = []
    for terminal in terminals:
        if not isinstance(terminal, dict):
            continue
        terminal_id = terminal.get("terminal_id")
        queries = terminal.get("queries")
        if isinstance(queries, list):
            for query in queries:
                if not isinstance(query, dict):
                    continue
                normalized_query = dict(query)
                if normalized_query.get("terminal_id") is None and terminal_id is not None:
                    normalized_query["terminal_id"] = terminal_id
                records.append(normalized_query)
    return records


def _query_function_type(query: dict[str, Any]) -> str:
    """推断 query 的主 LLM 函数类型，用于 operator mix。"""
    if query.get("function_type"):
        return str(query["function_type"])

    batch_functions = query.get("batch_capable_functions")
    non_batch_functions = query.get("non_batch_capable_functions")
    if isinstance(batch_functions, list) and len(batch_functions) == 1:
        return str(batch_functions[0])
    if isinstance(non_batch_functions, list) and len(non_batch_functions) == 1:
        return str(non_batch_functions[0])

    # 旧 TPCC workload.json 会把 function_type 记录在 wave/request 粒度。
    # 建模时只需要 query 的主算子类型，因此取第一个实际请求的 function_type；
    # 若一个 query 未来混合多种 LLM 函数，再扩展为按 request 逐条计数即可。
    for wave in query.get("waves") or []:
        if not isinstance(wave, dict):
            continue
        for request in wave.get("requests") or []:
            if isinstance(request, dict) and request.get("function_type"):
                return str(request["function_type"])
    return "unknown"


def _request_records(query_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 query/wave/request 三层 trace 展平成请求级记录。"""
    # 请求级 trace 是 prompt/prefix/cacheblend 特征的唯一可靠来源。
    # 这里把 query -> wave -> request 拉平成一组请求，并补齐 query 上下文。
    requests: list[dict[str, Any]] = []
    for query in query_records:
        for wave in query.get("waves") or []:
            if not isinstance(wave, dict):
                continue
            for request in wave.get("requests") or []:
                if not isinstance(request, dict):
                    continue
                normalized = dict(request)
                normalized.setdefault("terminal_id", query.get("terminal_id"))
                normalized.setdefault("query_id", query.get("query_id"))
                normalized.setdefault("step_index", query.get("step_index"))
                normalized.setdefault("function_type", _query_function_type(query))
                requests.append(normalized)
    return requests


def _request_prompt_token_ids(request: dict[str, Any]) -> list[int] | None:
    """读取请求实际发送给 vLLM 的 token ids；CacheBlend 路径优先用它。"""
    # 在 CacheBlend 的 token-id completion 中，会直接记录 prompt_token_ids；
    # 用 token id 计算共享前缀比用文本估算更可靠。
    metadata = request.get("prompt_metadata")
    if not isinstance(metadata, dict):
        return None
    token_ids = metadata.get("prompt_token_ids")
    if isinstance(token_ids, list) and all(isinstance(token, int) for token in token_ids):
        return token_ids
    return None


def _request_prompt_text(request: dict[str, Any]) -> str:
    """读取请求 prompt 文本；没有 trace payload 时返回空串。"""
    # 普通 chat/text completion 走 prompt_text；旧 trace 没有该字段时返回空串，
    # 让调用方自然降级为缺失特征而不是误造数据。
    metadata = request.get("prompt_metadata")
    if not isinstance(metadata, dict):
        return ""
    prompt_text = metadata.get("prompt_text")
    return prompt_text if isinstance(prompt_text, str) else ""


def _request_full_prompt_tokens(request: dict[str, Any]) -> int | None:
    """读取完整 prompt token 数，优先 token ids，其次 provider usage。"""
    # 优先使用 token ids，因为它和实际发给 vLLM 的序列完全一致；
    # 否则退回 OpenAI usage.prompt_tokens。
    token_ids = _request_prompt_token_ids(request)
    if token_ids is not None:
        return len(token_ids)
    prompt_tokens = request.get("prompt_tokens")
    if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
        return int(prompt_tokens)
    return None


def _prompt_len_features(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """提取 system/data/output 长度和模板复用特征。"""
    # 提取方案里的 system/data/output/template 特征都来自 request trace。
    # 对 chat 请求，system_text/data_text 可直接取；对 text/token-id 请求，
    # 通过同 query/function 内的公共前缀近似模板/system 部分。
    system_lengths: list[float] = []
    data_lengths: list[float] = []
    output_lengths: list[float] = []
    template_counter: Counter[str] = Counter()

    text_groups: dict[tuple[Any, Any], list[tuple[str, int | None]]] = defaultdict(list)
    for request in requests:
        prompt_text = _request_prompt_text(request)
        full_tokens = _request_full_prompt_tokens(request)
        text_groups[(request.get("query_id"), request.get("function_type"))].append(
            (prompt_text, full_tokens)
        )

    common_prefix_by_group: dict[tuple[Any, Any], str] = {}
    for group_key, items in text_groups.items():
        # 同一个 query_id + function_type 通常共享 SQL prompt 模板；
        # 取组内最长公共前缀，作为没有显式 system message 时的模板代理。
        texts = [text for text, _ in items if text]
        if not texts:
            common_prefix_by_group[group_key] = ""
            continue
        prefix = texts[0]
        for text in texts[1:]:
            prefix = _longest_common_prefix_text(prefix, text)
        common_prefix_by_group[group_key] = prefix

    for request in requests:
        metadata = request.get("prompt_metadata") if isinstance(request.get("prompt_metadata"), dict) else {}
        prompt_text = _request_prompt_text(request)
        full_tokens = _request_full_prompt_tokens(request)
        system_text = metadata.get("system_text") if isinstance(metadata, dict) else None
        data_text = metadata.get("data_text") if isinstance(metadata, dict) else None
        if isinstance(system_text, str) and system_text:
            system_lengths.append(float(_estimate_token_len(system_text, prompt_text, full_tokens)))
        else:
            # 对 Flock movie workload，多数请求是文本补全，没有 system role。
            # 此时用组内公共前缀刻画“模板/系统指令”长度。
            prefix = common_prefix_by_group.get((request.get("query_id"), request.get("function_type")), "")
            system_lengths.append(float(_estimate_token_len(prefix, prompt_text, full_tokens)))

        if isinstance(data_text, str) and data_text:
            data_lengths.append(float(_estimate_token_len(data_text, prompt_text, full_tokens)))
        elif full_tokens is not None:
            # 对 token-id completion，没有可读 data_text；用“总长度 - 模板长度”近似数据段。
            data_lengths.append(float(max(0, full_tokens - system_lengths[-1])))

        generation_tokens = request.get("generation_tokens")
        if isinstance(generation_tokens, (int, float)) and generation_tokens >= 0:
            output_lengths.append(float(generation_tokens))

        template_prefix = common_prefix_by_group.get((request.get("query_id"), request.get("function_type")), "")
        # 不把完整 prompt 存进 template_reuse_count，避免把 review 文本等数据内容泄进摘要；
        # 哈希只标识“同 query/function 下复用的模板前缀”。
        template_hash = hashlib.sha1(
            f"{request.get('query_id')}|{request.get('function_type')}|{template_prefix}".encode("utf-8")
        ).hexdigest()
        template_counter[template_hash] += 1

    system_dist = _distribution(system_lengths, (50, 95))
    data_dist = _distribution(data_lengths, (50, 95))
    output_dist = _distribution(output_lengths, (50, 95))
    return {
        "system_prompt_len_distribution": system_dist,
        "data_prompt_len_distribution": data_dist,
        "output_len_distribution": output_dist,
        "template_reuse_count": {
            "template_count": len(template_counter),
            "max_reuse_count": max(template_counter.values()) if template_counter else 0,
            "counts": dict(template_counter),
            "extraction_method": "system message when available; otherwise common-prefix template surrogate",
        },
    }


def _prefix_reuse_features(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """计算普通 prefix cache 理论可复用比例和共享前缀长度。"""
    # prefix_reuse_ratio 表示“理论可复用前缀长度 / prompt 总长度”。
    # 同 query/function 组内先求共享前缀，再把每条请求可复用的前缀 token 数
    # 累加作分子；实际 vLLM 命中率另由运行时 prefix_cache_hit_rate 记录。
    grouped_tokens: dict[tuple[Any, Any], list[list[int]]] = defaultdict(list)
    grouped_texts: dict[tuple[Any, Any], list[tuple[str, int | None]]] = defaultdict(list)

    for request in requests:
        group_key = (request.get("query_id"), request.get("function_type"))
        token_ids = _request_prompt_token_ids(request)
        if token_ids is not None:
            grouped_tokens[group_key].append(token_ids)
        else:
            text = _request_prompt_text(request)
            if text:
                grouped_texts[group_key].append((text, _request_full_prompt_tokens(request)))

    shared_lengths: list[float] = []
    reusable_prefix_tokens = 0.0
    total_prompt_tokens = 0.0
    for token_lists in grouped_tokens.values():
        # 对 token-id 路径，直接在令牌空间求最长公共前缀，避免文本/tokenizer 偏差。
        if len(token_lists) < 2:
            total_prompt_tokens += float(sum(len(token_ids) for token_ids in token_lists))
            continue
        prefix = token_lists[0]
        for token_ids in token_lists[1:]:
            prefix = prefix[: _longest_common_prefix_len(prefix, token_ids)]
        for token_ids in token_lists:
            total_prompt_tokens += float(len(token_ids))
            shared_lengths.append(float(len(prefix)))
            reusable_prefix_tokens += float(len(prefix))

    for text_items in grouped_texts.values():
        # 文本路径先求字符公共前缀，再按完整 prompt token 数估算 token 长度。
        if len(text_items) < 2:
            total_prompt_tokens += float(
                sum(
                    full_tokens or max(1, len(text.split()))
                    for text, full_tokens in text_items
                )
            )
            continue
        prefix = text_items[0][0]
        for text, _ in text_items[1:]:
            prefix = _longest_common_prefix_text(prefix, text)
        for text, full_tokens in text_items:
            total_prompt_tokens += float(full_tokens or max(1, len(text.split())))
            shared_len = _estimate_token_len(prefix, text, full_tokens)
            shared_lengths.append(float(shared_len))
            reusable_prefix_tokens += float(shared_len)

    return {
        "prefix_reuse_ratio": (
            float(reusable_prefix_tokens / total_prompt_tokens)
            if total_prompt_tokens > 0
            else None
        ),
        "shared_prefix_len_distribution": _distribution(shared_lengths, (50, 90)),
    }


def _cacheblend_reuse_features(requests: list[dict[str, Any]], tokenizer_path=None) -> dict[str, Any]:
    """Count repeated delimiter-separated segments across distinct requests.

    Ratio is theoretical reusable-token coverage, not a measured hit rate and
    not a cold-start-adjusted saving. Delimiters are excluded from its numerator.
    RAG count uses the Flock envelope convention: instruction, evidence..., question.
    Without token boundaries/tokenizer, report unknown rather than full-prompt reuse.
    """
    segment_counter = Counter()
    occurrences = Counter()
    segment_lengths = {}
    total_prompt_tokens = 0
    observed = 0
    missing = 0
    chunk_counts = []
    tokenizer = None
    separator_cache = {}
    for request in requests:
        metadata = request.get("prompt_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        token_ids = _request_prompt_token_ids(request)
        text = _request_prompt_text(request)
        full_tokens = _request_full_prompt_tokens(request)
        total_prompt_tokens += len(token_ids) if token_ids is not None else (full_tokens or (max(1, len(text.split())) if text else 0))
        if not metadata.get("cacheblend"):
            continue
        delimiter = metadata.get("blend_special_str") or " # # "
        if token_ids is not None:
            separator = metadata.get("blend_separator_token_ids")
            if not separator and tokenizer_path:
                if tokenizer is None:
                    from tokenizers import Tokenizer
                    from pathlib import Path
                    path = Path(tokenizer_path)
                    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json" if path.is_dir() else path))
                if delimiter not in separator_cache:
                    separator_cache[delimiter] = tokenizer.encode(delimiter, add_special_tokens=False).ids
                separator = separator_cache[delimiter]
            if not isinstance(separator, list) or not separator or any(type(t) is not int for t in separator):
                missing += 1
                continue
            segments = []
            start = i = 0
            while i <= len(token_ids)-len(separator):
                if token_ids[i:i+len(separator)] == separator:
                    segments.append(token_ids[start:i])
                    i += len(separator)
                    start = i
                else:
                    i += 1
            segments.append(token_ids[start:])
            if len(segments) < 2:
                missing += 1
                continue
            keys = [(tuple(segment), len(segment)) for segment in segments if segment]
            count = sum(bool(segment) for segment in segments[1:-1])
        else:
            if not text or delimiter not in text:
                missing += 1
                continue
            segments = text.split(delimiter)
            keys = [(hashlib.sha1(s.encode()).hexdigest(), _estimate_token_len(s, text, full_tokens))
                    for s in segments if s.strip()]
            count = sum(bool(s.strip()) for s in segments[1:-1])
        observed += 1
        chunk_counts.append(float(count))
        # A segment repeated within just one request is not cross-request reuse.
        segment_counter.update(set(key for key, _ in keys))
        for key, length in keys:
            occurrences[key] += 1
            segment_lengths[key] = length
    reusable = [float(segment_lengths[k]) for k,n in segment_counter.items() if n > 1]
    reusable_tokens = sum(segment_lengths[k]*occurrences[k] for k,n in segment_counter.items() if n > 1)
    return {
        "shared_cacheblend_len_distribution": _distribution(reusable, (50, 90)),
        "cacheblend_reuse_ratio": reusable_tokens/total_prompt_tokens if observed and not missing and total_prompt_tokens else None,
        "rag_retrieved_chunk_count_distribution": _distribution(chunk_counts, (50, 90)),
        "cacheblend_segment_observed_requests": observed,
        "cacheblend_segment_missing_requests": missing,
        "cacheblend_segment_method": "delimiter_segments; RAG count excludes instruction/question envelope",
    }


def _theoretical_global_in_flight_upper(
    workload_metrics: dict[str, Any],
    query_records: list[dict[str, Any]],
) -> int | None:
    """计算 terminal 异步推进时 workload 可暴露的全局在途请求上限。

    每个 terminal 内的 query 串行执行、不同 terminal 并行执行，因此先取每个
    terminal 任一 query 能同时暴露的最大请求数，再跨 terminal 求和。最后再受
    workload driver 的全局 ``max_in_flight`` 限制。这个静态上限描述 workload
    结构，不等同于实际运行峰值或 vLLM 内部 waiting queue。
    """

    per_terminal_max: dict[int, int] = defaultdict(int)
    for query in query_records:
        terminal_id = query.get("terminal_id")
        if terminal_id is None:
            continue
        request_count = int(
            query.get("total_llm_requests") or query.get("row_count") or 0
        )
        configured_cap = query.get("configured_request_cap")
        if type(configured_cap) in {int, float} and float(configured_cap) > 0:
            request_count = min(request_count, int(configured_cap))
        per_terminal_max[int(terminal_id)] = max(
            per_terminal_max[int(terminal_id)], request_count
        )

    if not per_terminal_max:
        return None
    structural_upper = sum(per_terminal_max.values())
    configured_max_in_flight = _safe_float(workload_metrics.get("max_in_flight"))
    if (
        configured_max_in_flight is not None
        and not math.isinf(configured_max_in_flight)
        and configured_max_in_flight > 0
    ):
        structural_upper = min(structural_upper, int(configured_max_in_flight))
    return int(structural_upper)


def extract_workload_features(
    *,
    workload_metrics: dict[str, Any],
    runtime_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """主入口：构造方案文档定义的 workload 特征。

    输入分两类：
    - workload_metrics：TPCC driver/query trace 给出的静态 workload 和画像数据；
    - runtime_metrics：trial 执行时采到的 vLLM、LMCache、GPU、质量目标数据。
    输出同时服务人工 JSON 检查和 BO 模型输入。
    """
    runtime_metrics = runtime_metrics or {}
    summary = workload_metrics.get("summary") or {}
    # 三层来源：
    # 1. workload_metrics: TPCC driver 的 terminal/query/profile/trace 结构；
    # 2. runtime_metrics: vLLM Prometheus delta 和 trial 运行耗时；
    # 3. sembench_summary: 质量评估结果。
    query_records = _query_records(workload_metrics)
    request_records = _request_records(query_records)
    terminal_sequences = _terminal_sequences(workload_metrics)

    # ``total_llm_requests`` is the logical full-query cardinality. Packed SQL
    # may span multiple DuckDB chunks, so its raw trace can describe only one
    # chunk; the terminal worker records that separately as
    # ``trace_llm_request_count``.
    llm_request_counts = [
        float(query.get("total_llm_requests") or query.get("row_count") or 0)
        for query in query_records
    ]
    candidate_rows = [
        # candidate_rows 表示进入 LLM 算子的候选行/请求规模；
        # 在 TPCC trace 中，`row_count` 是 query 输出行数，不能优先用它。
        float(query.get("total_llm_requests") or query.get("row_count") or 0)
        for query in query_records
    ]
    wave_counts = [float(query.get("wave_count") or 0) for query in query_records]
    configured_caps = [
        float(query.get("configured_request_cap") or query.get("final_effective_request_cap") or 0)
        for query in query_records
    ]

    function_request_counts: Counter[str] = Counter()
    for query in query_records:
        function_type = _query_function_type(query)
        function_request_counts[function_type] += int(
            query.get("total_llm_requests") or query.get("row_count") or 0
        )
    total_llm_requests = int(
        summary.get("total_llm_request_count")
        or sum(function_request_counts.values())
        or sum(llm_request_counts)
    )

    # operator_mix 必须按画像运行中实际触发的 LLM 请求数加权；
    # 只看 SQL 里函数出现次数会把 Q1/Q7 这类请求规模差异抹平。
    operator_mix = {
        function_type: (
            float(request_count / total_llm_requests)
            if total_llm_requests > 0
            else 0.0
        )
        for function_type, request_count in sorted(function_request_counts.items())
    }

    terminal_query_slot_count = sum(len(seq) for seq in terminal_sequences.values())
    unique_queries = {
        (int(query_id) if str(query_id).isdigit() else str(query_id))
        for seq in terminal_sequences.values()
        for query_id in seq
    }
    concurrent_query_mix = ",".join(f"Q{query_id}" for query_id in sorted(unique_queries, key=lambda q: (int(re.match(r"\d+", str(q))[0]), str(q))))
    theoretical_global_in_flight_upper = _theoretical_global_in_flight_upper(
        workload_metrics,
        query_records,
    )

    prompt_tokens = _safe_float(runtime_metrics.get("prompt_tokens"), 0.0) or 0.0
    generation_tokens = _safe_float(runtime_metrics.get("generation_tokens"), 0.0) or 0.0
    request_delta = (
        _safe_float(runtime_metrics.get("request_success_total_delta"))
        if runtime_metrics.get("request_success_total_delta") is not None
        else _safe_float(runtime_metrics.get("request_success_stop_delta"), 0.0)
    ) or 0.0
    ttft_p95_ms = runtime_metrics.get("ttft_p95_ms")
    if ttft_p95_ms is None:
        ttft_p95_ms = runtime_metrics.get("p99_ttft_ms")
    # 这里的 prompt/prefix/cacheblend 画像是 workload 静态画像，来自第一次画像运行
    # 的请求 trace；后续 BO trial 可复用这些特征。
    prompt_len_features = _prompt_len_features(request_records)
    prefix_features = _prefix_reuse_features(request_records)
    cacheblend_features = _cacheblend_reuse_features(request_records, runtime_metrics.get("tokenizer_path") or runtime_metrics.get("resolved_model_path"))
    quality_summary = runtime_metrics.get("sembench_summary") or summary
    features: dict[str, Any] = {
        # 基准配置特征：只依赖 workload/driver 参数，同一 workload 提取一次即可。
        "terminal_count": int(workload_metrics.get("terminal_count") or 0),
        "request_control_mode": str(workload_metrics.get("request_control_mode") or "unknown"),
        "max_in_flight": _finite_float(workload_metrics.get("max_in_flight")),
        "effective_per_terminal_max_in_flight": _json_feature_float(
            workload_metrics.get("effective_per_terminal_max_in_flight")
        ),
        "request_cap_strategy": str(
            workload_metrics.get("request_cap_strategy") or "unknown"
        ),
        "request_cap_formula": str(
            workload_metrics.get("request_cap_formula") or "unknown"
        ),
        "arrival_mode": str(runtime_metrics.get("flock_request_arrival") or "unknown"),
        "request_rate": _request_rate_feature(runtime_metrics.get("flock_request_rate")),
        "concurrent_query_mix": concurrent_query_mix,
        "terminal_query_slot_count": terminal_query_slot_count,
        "terminal_unique_query_count": len(unique_queries),
        # 查询/数据画像特征：依赖 query trace 和候选行规模，第一次画像后稳定。
        "query_instance_count": len(query_records),
        "total_llm_request_count": total_llm_requests,
        "mean_llm_requests_per_query": (
            float(sum(llm_request_counts) / len(llm_request_counts))
            if llm_request_counts
            else 0.0
        ),
        "p50_llm_requests_per_query": _percentile(llm_request_counts, 50) or 0.0,
        "p95_llm_requests_per_query": _percentile(llm_request_counts, 95) or 0.0,
        "max_llm_requests_per_query": max(llm_request_counts) if llm_request_counts else 0.0,
        "candidate_rows_p50": _percentile(candidate_rows, 50) or 0.0,
        "candidate_rows_p90": _percentile(candidate_rows, 90) or 0.0,
        "candidate_rows_max": max(candidate_rows) if candidate_rows else 0.0,
        "theoretical_global_in_flight_upper": theoretical_global_in_flight_upper,
        "mean_wave_count": float(sum(wave_counts) / len(wave_counts)) if wave_counts else 0.0,
        "max_wave_count": max(wave_counts) if wave_counts else 0.0,
        "mean_configured_request_cap": (
            float(sum(configured_caps) / len(configured_caps))
            if configured_caps
            else 0.0
        ),
        "max_configured_request_cap": max(configured_caps) if configured_caps else 0.0,
        "query_throughput_qps": _finite_float(runtime_metrics.get("query_throughput_qps"), 0.0),
        "mean_query_execution_latency_ms": _finite_float(
            runtime_metrics.get("mean_query_execution_latency_ms"),
            0.0,
        ),
        "p95_query_execution_latency_ms": _finite_float(
            runtime_metrics.get("p95_query_execution_latency_ms"),
            0.0,
        ),
        "mean_ttft_ms": _finite_float(runtime_metrics.get("mean_ttft_ms"), 0.0),
        "mean_e2e_latency_ms": _finite_float(runtime_metrics.get("mean_e2e_latency_ms"), 0.0),
        "p95_latency_ms": _finite_float(runtime_metrics.get("p95_latency_ms"), 0.0),
        "prompt_tokens_per_request": (
            float(prompt_tokens / request_delta) if request_delta > 0 else 0.0
        ),
        "generation_tokens_per_request": (
            float(generation_tokens / request_delta) if request_delta > 0 else 0.0
        ),
        "input_output_ratio": (
            float(prompt_tokens / generation_tokens) if generation_tokens > 0 else 0.0
        ),
        # 两类 cache 命中率分开建模：
        # - prefix_cache_hit_rate：vLLM 本地 prefix cache；
        # - lmcache_cache_hit_rate：LMCache/KV connector 的外部 KV 命中。
        "prefix_cache_hit_rate": _json_feature_float(
            runtime_metrics.get("prefix_cache_hit_rate"),
        ),
        "lmcache_cache_hit_rate": _json_feature_float(
            runtime_metrics.get("lmcache_cache_hit_rate"),
        ),
        # 这里记录 Prompt 形状特征：保留嵌套分布给人工查看，同时展开 p50/p95/max 给 BO。
        "system_prompt_len_distribution": prompt_len_features["system_prompt_len_distribution"],
        "system_prompt_len_p50": prompt_len_features["system_prompt_len_distribution"]["p50"],
        "system_prompt_len_p95": prompt_len_features["system_prompt_len_distribution"]["p95"],
        "system_prompt_len_max": prompt_len_features["system_prompt_len_distribution"]["max"],
        "data_prompt_len_distribution": prompt_len_features["data_prompt_len_distribution"],
        "data_prompt_len_p50": prompt_len_features["data_prompt_len_distribution"]["p50"],
        "data_prompt_len_p95": prompt_len_features["data_prompt_len_distribution"]["p95"],
        "data_prompt_len_max": prompt_len_features["data_prompt_len_distribution"]["max"],
        "output_len_distribution": prompt_len_features["output_len_distribution"],
        "output_len_p50": prompt_len_features["output_len_distribution"]["p50"],
        "output_len_p95": prompt_len_features["output_len_distribution"]["p95"],
        "output_len_max": prompt_len_features["output_len_distribution"]["max"],
        "template_reuse_count": prompt_len_features["template_reuse_count"],
        "template_reuse_template_count": prompt_len_features["template_reuse_count"]["template_count"],
        "template_reuse_max_count": prompt_len_features["template_reuse_count"]["max_reuse_count"],
        "prefix_reuse_ratio": prefix_features["prefix_reuse_ratio"],
        "shared_prefix_len_distribution": prefix_features["shared_prefix_len_distribution"],
        "shared_prefix_len_p50": prefix_features["shared_prefix_len_distribution"]["p50"],
        "shared_prefix_len_p90": prefix_features["shared_prefix_len_distribution"]["p90"],
        "shared_prefix_len_max": prefix_features["shared_prefix_len_distribution"]["max"],
        "shared_cacheblend_len_distribution": cacheblend_features["shared_cacheblend_len_distribution"],
        "shared_cacheblend_len_p50": cacheblend_features["shared_cacheblend_len_distribution"]["p50"],
        "shared_cacheblend_len_p90": cacheblend_features["shared_cacheblend_len_distribution"]["p90"],
        "shared_cacheblend_len_max": cacheblend_features["shared_cacheblend_len_distribution"]["max"],
        "cacheblend_reuse_ratio": cacheblend_features["cacheblend_reuse_ratio"],
        "rag_retrieved_chunk_count_distribution": cacheblend_features["rag_retrieved_chunk_count_distribution"],
        "cacheblend_segment_observed_requests": cacheblend_features["cacheblend_segment_observed_requests"],
        "cacheblend_segment_missing_requests": cacheblend_features["cacheblend_segment_missing_requests"],
        "cacheblend_segment_method": cacheblend_features["cacheblend_segment_method"],

        # 运行时资源/调度器特征：只有外部采样源存在时才会非空；
        # 函数 categorize_workload_features 会把缺失原因写进 missing_or_partial_features。
        "gpu_utilization_mean": runtime_metrics.get("gpu_utilization_mean"),
        "gpu_utilization_p90": runtime_metrics.get("gpu_utilization_p90"),
        "kv_cache_usage_mean": runtime_metrics.get("kv_cache_usage_mean"),
        "kv_cache_usage_p90": runtime_metrics.get("kv_cache_usage_p90"),
        "hbm_usage_mean": runtime_metrics.get("hbm_usage_mean"),
        "hbm_usage_p90": runtime_metrics.get("hbm_usage_p90"),
        "ttft_p50_ms": runtime_metrics.get("ttft_p50_ms"),
        "ttft_p95_ms": ttft_p95_ms,
        "tpot_p50_ms": runtime_metrics.get("tpot_p50_ms"),
        "tpot_p95_ms": runtime_metrics.get("tpot_p95_ms"),
        "actual_batch_size_mean": runtime_metrics.get("actual_batch_size_mean"),
        "actual_batch_size_p95": runtime_metrics.get("actual_batch_size_p95"),
        "actual_global_in_flight_peak": _json_feature_float(
            runtime_metrics.get(
                "actual_global_in_flight_peak",
                workload_metrics.get("global_in_flight_peak"),
            )
        ),
        # 质量特征：来自 SemBench evaluation summary，随 trial 输出质量变化。
        "quality_gate_passed": (
            float(quality_summary["quality_gate_passed"])
            if isinstance(quality_summary, dict)
            and isinstance(quality_summary.get("quality_gate_passed"), bool)
            else None
        ),
        "quality_metric_coverage": (
            quality_summary.get("quality_metric_coverage")
            if isinstance(quality_summary, dict)
            else None
        ),
        "operator_mix": operator_mix,
        "llm_request_count_per_query": {
            f"terminal_{query.get('terminal_id')}:step_{query.get('step_index')}:Q{query.get('query_id')}": int(
                query.get("total_llm_requests") or query.get("row_count") or 0
            )
            for query in query_records
        },
        "llm_request_count_source_per_query": {
            f"terminal_{query.get('terminal_id')}:step_{query.get('step_index')}:Q{query.get('query_id')}": str(
                query.get("llm_request_count_source") or "legacy_fallback"
            )
            for query in query_records
        },
    }

    for function_type, ratio in operator_mix.items():
        features[f"operator_mix_{function_type}"] = ratio
    return features


def categorize_workload_features(features: dict[str, Any]) -> dict[str, Any]:
    """把扁平特征按方案文档分组，并显式列出缺失采样源。"""
    # 结果 JSON 中保留分类视图，方便人工对照建模方案；扁平视图仍保留给 BO/随机森林使用。
    # extraction_scope 标识该组特征应多久刷新一次。
    groups = {
        "benchmark_config_features": {
            "terminal_count": features.get("terminal_count"),
            "request_control_mode": features.get("request_control_mode"),
            "max_in_flight": features.get("max_in_flight"),
            "effective_per_terminal_max_in_flight": features.get(
                "effective_per_terminal_max_in_flight"
            ),
            "request_cap_strategy": features.get("request_cap_strategy"),
            "request_cap_formula": features.get("request_cap_formula"),
            "arrival_mode": features.get("arrival_mode"),
            "request_rate": features.get("request_rate"),
            "concurrent_query_mix": features.get("concurrent_query_mix"),
            "terminal_query_slot_count": features.get("terminal_query_slot_count"),
            "terminal_unique_query_count": features.get("terminal_unique_query_count"),
            "extraction_scope": "static_once_per_workload",
        },
        "query_features": {
            "operator_mix": features.get("operator_mix"),
            "llm_request_count_per_query": features.get("llm_request_count_per_query"),
            "system_prompt_len_distribution": features.get("system_prompt_len_distribution"),
            "output_len_distribution": features.get("output_len_distribution"),
            "input_output_ratio": features.get("input_output_ratio"),
            "template_reuse_count": features.get("template_reuse_count"),
            "theoretical_global_in_flight_upper": features.get(
                "theoretical_global_in_flight_upper"
            ),
            "extraction_scope": "static_after_one_profile_run",
        },
        "data_features": {
            "candidate_rows_distribution": {
                "p50": features.get("candidate_rows_p50"),
                "p90": features.get("candidate_rows_p90"),
                "max": features.get("candidate_rows_max"),
            },
            "data_prompt_len_distribution": features.get("data_prompt_len_distribution"),
            "prefix_reuse_ratio": features.get("prefix_reuse_ratio"),
            "shared_prefix_len_distribution": features.get("shared_prefix_len_distribution"),
            "shared_cacheblend_len_distribution": features.get("shared_cacheblend_len_distribution"),
            "cacheblend_reuse_ratio": features.get("cacheblend_reuse_ratio"),
            "rag_retrieved_chunk_count_distribution": features.get("rag_retrieved_chunk_count_distribution"),
            "extraction_scope": "static_after_one_profile_run",
        },
        "runtime_features": {
            "gpu_utilization": {
                "mean": features.get("gpu_utilization_mean"),
                "p90": features.get("gpu_utilization_p90"),
            },
            "kv_cache_usage_or_hbm_usage": {
                "kv_cache_mean": features.get("kv_cache_usage_mean"),
                "kv_cache_p90": features.get("kv_cache_usage_p90"),
                "hbm_mean": features.get("hbm_usage_mean"),
                "hbm_p90": features.get("hbm_usage_p90"),
            },
            "ttft_tpot_distribution": {
                "ttft_p50_ms": features.get("ttft_p50_ms"),
                "ttft_p95_ms": features.get("ttft_p95_ms"),
                "tpot_p50_ms": features.get("tpot_p50_ms"),
                "tpot_p95_ms": features.get("tpot_p95_ms"),
                "mean_ttft_ms": features.get("mean_ttft_ms"),
                "mean_e2e_latency_ms": features.get("mean_e2e_latency_ms"),
            },
            "actual_batch_size": {
                "mean": features.get("actual_batch_size_mean"),
                "p95": features.get("actual_batch_size_p95"),
            },
            "actual_global_in_flight": {
                "peak": features.get("actual_global_in_flight_peak"),
            },
            "prefix_cache_hit_rate": {
                "prefix_cache_hit_rate": features.get("prefix_cache_hit_rate"),
            },
            "lmcache_cache_hit_rate": {
                "lmcache_cache_hit_rate": features.get("lmcache_cache_hit_rate"),
            },
            "query_throughput_qps": features.get("query_throughput_qps"),
            "mean_query_execution_latency_ms": features.get("mean_query_execution_latency_ms"),
            "p95_query_execution_latency_ms": features.get("p95_query_execution_latency_ms"),
            "extraction_scope": "per_trial_runtime",
        },
        "quality_features": {
            "gate_passed": features.get("quality_gate_passed"),
            "metric_coverage": features.get("quality_metric_coverage"),
            "extraction_scope": "per_trial_evaluation",
        },
    }
    groups["missing_or_partial_features"] = {
        # 这些特征在方案里有定义，但当前 trial 产物没有足够采样源；
        # 显式写出原因，避免静默把 None 当成“值为 0”。
        "gpu_utilization": (
            "requires nvidia-smi/DCGM sampling"
            if features.get("gpu_utilization_mean") is None
            else None
        ),
        "kv_cache_usage_or_hbm_usage": (
            "requires vLLM KV/HBM scheduler metrics or GPU memory sampling"
            if features.get("kv_cache_usage_mean") is None and features.get("hbm_usage_mean") is None
            else None
        ),
        "actual_batch_size": (
            "requires vLLM scheduler active sequence/batched token metrics"
            if features.get("actual_batch_size_mean") is None
            else None
        ),
        "actual_global_in_flight": (
            "requires Flock request start/completion timeline"
            if features.get("actual_global_in_flight_peak") is None
            else None
        ),
        "theoretical_global_in_flight_upper": (
            "requires terminal query traces with per-query request counts"
            if features.get("theoretical_global_in_flight_upper") is None
            else None
        ),
        "prefix_cache_hit_rate": (
            "requires vLLM prefix cache query/hit counters"
            if features.get("prefix_cache_hit_rate") is None
            else None
        ),
        "lmcache_cache_hit_rate": (
            "requires LMCache hit-token logs or vLLM external-prefix counters"
            if features.get("lmcache_cache_hit_rate") is None
            else None
        ),
        "tpot_distribution": (
            "requires request-level output token timing metrics"
            if features.get("tpot_p50_ms") is None
            else None
        ),
    }
    groups["missing_or_partial_features"] = {
        key: value for key, value in groups["missing_or_partial_features"].items() if value
    }
    return groups


def workload_features_for_model(features: dict[str, Any] | None) -> dict[str, Any]:
    """返回可直接喂给 BO/随机森林的扁平特征行。"""
    features = features or {}
    row: dict[str, Any] = {}
    for name in NUMERIC_WORKLOAD_FEATURES:
        # 模型输入使用 workload_ 前缀，避免和 vLLM knob 名称冲突。
        row[f"workload_{name}"] = _finite_float(features.get(name), -1.0)
    for name in CATEGORICAL_WORKLOAD_FEATURES:
        row[f"workload_{name}"] = str(features.get(name) or "__none__")
    return row
