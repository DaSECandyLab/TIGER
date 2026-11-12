"""Flock trial 运行的参数解析和配置决策。"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from ..config import resolve_model_path_for_quantization

REPOSITORY_ROOT = Path(__file__).resolve().parents[8]

DEFAULT_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct')
DEFAULT_AWQ_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct-AWQ')
DEFAULT_GPTQ_INT4_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct-GPTQ-Int4')
DEFAULT_VLLM_BIN = str(REPOSITORY_ROOT / 'vllm-0.20.0/.venv/bin/vllm')
DEFAULT_SEMBENCH_DIR = str(REPOSITORY_ROOT / 'SemBench')
DEFAULT_FLOCK_EXT = str(REPOSITORY_ROOT / 'flock/build/release/extension/flock/flock.duckdb_extension')
DEFAULT_RESULT_DIR = "tune_res/flock_trials"
DEFAULT_FLOCK_HTTP_MAX_CONNECTIONS = 2048
MAX_IN_FLIGHT_STRATEGIES = (
    "dynamic_active_terminal_partition",
    "global_slot_coordination",
    "fixed_equal_terminal_partition",
)
TUNING_REFERENCE_DEFAULT_TP = 2
TUNING_REFERENCE_DEFAULT_DATA_PARALLEL_SIZE = 1
TUNING_REFERENCE_DEFAULT_MAX_NUM_SEQS = 64
TUNING_REFERENCE_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
TUNING_REFERENCE_DEFAULT_BLOCK_SIZE = 16
TUNING_REFERENCE_DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD = 0
TUNING_REFERENCE_DEFAULT_MAX_MODEL_LEN = 2048
TUNING_REFERENCE_DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
TUNING_REFERENCE_DEFAULT_CPU_OFFLOAD_GB = 0.0
TUNING_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND = "native"
TUNING_REFERENCE_DEFAULT_KV_OFFLOADING_SIZE = 0.0
TUNING_REFERENCE_DEFAULT_DTYPE = "auto"
TUNING_REFERENCE_DEFAULT_KV_CACHE_DTYPE = "auto"
TUNING_REFERENCE_DEFAULT_ENABLE_CHUNKED_PREFILL = True
TUNING_REFERENCE_DEFAULT_ENABLE_PREFIX_CACHING = False


def parse_flock_max_in_flight_arg(value: str) -> int | str:
    """解析 Flock 请求上限参数。

    TPCC workload 和 legacy SemBench 都复用这个解析逻辑，
    其中 `0/inf/unlimited` 统一表示“不额外施加客户端上限”。
    """
    lowered = value.strip().lower()
    if lowered in {"", "0", "inf", "infinity", "unlimited"}:
        return "unlimited"
    if lowered in {"max_num_seqs", "match_max_num_seqs"}:
        return "match_max_num_seqs"
    try:
        parsed = int(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "flock max-in-flight must be a positive integer or one of: "
            "0, inf, infinity, unlimited, max_num_seqs"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("flock max-in-flight must be non-negative")
    return parsed


def resolve_flock_max_in_flight_arg(args: argparse.Namespace) -> int | str:
    if args.flock_max_in_flight != "match_max_num_seqs":
        return args.flock_max_in_flight
    if args.max_num_seqs is None or int(args.max_num_seqs) < 1:
        raise ValueError(
            "--flock-max-in-flight=max_num_seqs requires --max-num-seqs >= 1"
        )
    return int(args.max_num_seqs)


def normalize_query_id(query: str | int) -> str:
    raw = str(query).strip()
    if raw.lower().startswith("q"):
        raw = raw[1:]
    match = re.fullmatch(r"(\d+)([a-z]*)", raw)
    if match is None:
        raise ValueError(f"Invalid query ID: {query!r}")
    return f"{int(match[1])}{match[2]}"


def flatten_terminal_query_sequences(
    terminal_query_sequences: Iterable[Iterable[str]],
) -> list[str]:
    flattened: list[str] = []
    seen_queries: set[str] = set()
    for query_sequence in terminal_query_sequences:
        for query in query_sequence:
            normalized = normalize_query_id(query)
            if normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            flattened.append(normalized)
    return flattened


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析单轮 trial 参数；``argv`` 便于上层 Agent 直接复用。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--awq-model-path", default=DEFAULT_AWQ_MODEL_PATH)
    parser.add_argument("--gptq-model-path", default=DEFAULT_GPTQ_INT4_MODEL_PATH)
    parser.add_argument("--served-model-name", default="Qwen2-7B-Instruct")
    parser.add_argument(
        "--quantization",
        choices=["none", "awq", "gptq", "awq_marlin", "gptq_marlin"],
        default="none",
        help="Quantization method. Default keeps the unquantized base model.",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--kv-cache-dtype-skip-layers", nargs="*", default=[])
    parser.add_argument(
        "--kv-offloading-backend",
        choices=["native", "lmcache"],
        default=TUNING_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND,
    )
    parser.add_argument("--kv-offloading-size", type=float, default=None)
    parser.add_argument(
        "--cacheblend-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Toggle SQL-side cacheblend rewrite plus the default LMCache+CacheBlend server config.",
    )
    parser.add_argument("--cacheblend-allow-prefix-caching", action="store_true")
    parser.add_argument("--kv-transfer-config-json", default=None)
    parser.add_argument("--lmcache-chunk-size", type=int, default=16)
    parser.add_argument("--cacheblend-blend-special-str", default=" # # ")
    parser.add_argument("--cacheblend-blend-min-tokens", type=int, default=16)
    parser.add_argument(
        "--cacheblend-blend-check-layers", nargs="*", type=int, default=[1]
    )
    parser.add_argument(
        "--cacheblend-blend-recompute-ratios",
        nargs="*",
        type=float,
        default=[0.15],
    )
    parser.add_argument("--speculative-config-json", default=None)
    parser.add_argument("--speculative-method", default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument("--speculative-prompt-lookup-min", type=int, default=2)
    parser.add_argument("--speculative-prompt-lookup-max", type=int, default=5)
    parser.add_argument("--vllm-bin", default=DEFAULT_VLLM_BIN)
    parser.add_argument(
        "--vllm-api-server-count",
        type=int,
        default=None,
        help=(
            "可选的 vLLM HTTP 前端进程数；默认不传该参数，沿用 vLLM 按 "
            "data_parallel_size 自动选择的官方行为。显式设置时属于实验基础设施配置，"
            "不进入 BO。"
        ),
    )
    parser.add_argument("--sembench-dir", default=DEFAULT_SEMBENCH_DIR)
    parser.add_argument(
        "--sembench-files-override-root",
        default=None,
        help=(
            "Optional experiment-local SemBench files root. Query files found "
            "under <root>/<use-case>/query/flockmtl override the repository "
            "versions; missing query files fall back to SemBench/files."
        ),
    )
    parser.add_argument("--flock-ext", default=DEFAULT_FLOCK_EXT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--metrics-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scale-factor", type=int, default=2000)
    parser.add_argument(
        "--use-case",
        default="movie",
        help="SemBench use case/dataset name, for example: movie.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["1", "2", "3", "4", "5", "6", "7", "8"],
        help="SemBench query ids to run.",
    )
    parser.add_argument(
        "--flock-terminal-queries",
        action="append",
        nargs="+",
        metavar="QUERY",
        default=None,
        help=(
            "Per-terminal query sequence for TPCC workload mode. Repeat once per "
            "terminal, for example: --flock-terminal-queries 1 2 "
            "--flock-terminal-queries 3 4 8"
        ),
    )
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--trial-name", default=None)
    parser.add_argument(
        "--quality-thresholds-json",
        default=None,
        help=(
            "Per-query, per-metric quality gates passed to the TPCC evaluator. "
            "No aggregate quality score is computed."
        ),
    )
    parser.add_argument(
        "--baseline-config",
        default=None,
        help=(
            "Path to an audited baseline_config.json generated by "
            "the measured baseline. Its engine parameters are authoritative."
        ),
    )
    parser.add_argument(
        "--preserve-engine-defaults",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--no-start-vllm", action="store_true")
    parser.add_argument("--keep-vllm", action="store_true")

    # 这里的 vLLM 参数尽量复用 SCOOT 侧命名；未显式指定的值后续会用调优参考配置补齐，
    # 让独立运行和 BO 的稳定 baseline 尽量一致。
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--data-parallel-size", type=int, default=None)
    parser.add_argument("--model-impl", default="vllm")
    parser.add_argument("--decode-context-parallel-size", type=int, default=1)
    parser.add_argument("--cp-kv-cache-interleave-size", type=int, default=1)
    parser.add_argument("--data-parallel-backend", choices=["mp", "ray"], default="mp")
    parser.add_argument(
        "--disable-nccl-for-dp-synchronization",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pp", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--cpu-offload-gb", type=float, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-chunked-prefill",
        "--chunked-prefill",
        dest="enable_chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prefix-caching-hash-algo", default="sha256")
    parser.add_argument(
        "--scheduler-reserve-full-isl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--disable-cascade-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.set_defaults(scheduler_delay_factor=0.0, use_v2_block_manager=False)
    parser.add_argument(
        "--extra-vllm-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional raw arguments appended to `vllm serve`.",
    )

    # 这里是 Flock 负载整形参数。
    parser.add_argument("--flock-request-rate", default="inf")
    parser.add_argument("--flock-request-arrival", default="poisson")
    parser.add_argument(
        "--flock-max-in-flight",
        type=parse_flock_max_in_flight_arg,
        default="unlimited",
        help=(
            "Maximum simultaneous in-flight Flock requests. Use an integer, "
            "0/inf/unlimited for no extra client-side cap, or max_num_seqs "
            "to match the current --max-num-seqs."
        ),
    )
    parser.add_argument(
        "--flock-max-in-flight-strategy",
        choices=MAX_IN_FLIGHT_STRATEGIES,
        default="dynamic_active_terminal_partition",
        help=(
            "How a TPCC workload shares the global max-in-flight budget. "
            "Use global_slot_coordination for one cross-terminal slot pool."
        ),
    )
    parser.add_argument(
        "--flock-http-max-connections",
        type=int,
        default=DEFAULT_FLOCK_HTTP_MAX_CONNECTIONS,
        help=(
            "Flock 客户端同时建立的全局 HTTP 连接上限；它与逻辑请求上限 "
            "max-in-flight 分离，超出的请求仍保留并在 libcurl 内排队。"
            "默认 2048；设为 0 可关闭连接池上限并恢复旧行为。"
        ),
    )
    parser.add_argument("--flock-query-concurrency", type=int, default=1)
    parser.add_argument("--flock-query-start-stagger-ms", type=float, default=0.0)
    parser.add_argument(
        "--flock-workload-driver",
        choices=["sembench", "tpcc"],
        default="sembench",
        # `sembench` 保持原有按 query 执行方式；
        # `tpcc` 切到 terminal 风格 workload driver，由它负责 terminal 并发、
        # 包括 query 分类、请求数量控制和 trace 聚合。
        help="Use legacy SemBench query execution or the TPCC-style terminal workload driver.",
    )
    parser.add_argument("--flock-terminal-count", type=int, default=1)
    parser.add_argument(
        "--flock-request-control-mode",
        choices=["max_in_flight", "vllm_batch_size_div_terminals"],
        default="max_in_flight",
    )
    parser.add_argument("--flock-vllm-batch-size", type=int, default=None)
    parser.add_argument(
        "--flock-failure-policy",
        choices=["fail_fast", "continue"],
        default="fail_fast",
    )
    args = parser.parse_args(argv)
    if (
        args.vllm_api_server_count is not None
        and args.vllm_api_server_count < 1
    ):
        parser.error("--vllm-api-server-count must be at least 1")
    if args.flock_http_max_connections < 0:
        parser.error("--flock-http-max-connections must be non-negative")
    if (
        args.flock_http_max_connections > 0
        and args.flock_http_max_connections < args.flock_terminal_count
    ):
        parser.error(
            "--flock-http-max-connections must be 0 or at least "
            "--flock-terminal-count"
        )
    return args


def apply_generated_baseline_config(args: argparse.Namespace) -> None:
    """加载 Agent 为本轮初筛候选生成的 baseline 取值。"""

    raw_path = getattr(args, "baseline_config", None)
    if not raw_path:
        return
    path = Path(str(raw_path)).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"baseline config must be a JSON object: {path}")
    from .configuration import apply_baseline_config_to_namespace

    apply_baseline_config_to_namespace(args, payload)
    args.baseline_config = str(path)


def apply_tuning_reference_defaults(args: argparse.Namespace) -> None:
    if args.tp is None:
        args.tp = TUNING_REFERENCE_DEFAULT_TP
    if args.data_parallel_size is None:
        args.data_parallel_size = TUNING_REFERENCE_DEFAULT_DATA_PARALLEL_SIZE
    if args.max_num_seqs is None:
        args.max_num_seqs = TUNING_REFERENCE_DEFAULT_MAX_NUM_SEQS
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = TUNING_REFERENCE_DEFAULT_MAX_NUM_BATCHED_TOKENS
    if args.block_size is None:
        args.block_size = TUNING_REFERENCE_DEFAULT_BLOCK_SIZE
    if args.long_prefill_token_threshold is None:
        args.long_prefill_token_threshold = (
            TUNING_REFERENCE_DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD
        )
    if args.max_model_len is None:
        args.max_model_len = TUNING_REFERENCE_DEFAULT_MAX_MODEL_LEN
    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = TUNING_REFERENCE_DEFAULT_GPU_MEMORY_UTILIZATION
    if args.cpu_offload_gb is None:
        args.cpu_offload_gb = TUNING_REFERENCE_DEFAULT_CPU_OFFLOAD_GB
    if args.dtype is None:
        args.dtype = TUNING_REFERENCE_DEFAULT_DTYPE
    if args.kv_cache_dtype is None:
        args.kv_cache_dtype = TUNING_REFERENCE_DEFAULT_KV_CACHE_DTYPE
    if args.kv_offloading_backend is None:
        args.kv_offloading_backend = TUNING_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND
    if args.kv_offloading_size is None:
        args.kv_offloading_size = TUNING_REFERENCE_DEFAULT_KV_OFFLOADING_SIZE
    if args.enable_chunked_prefill is None:
        args.enable_chunked_prefill = TUNING_REFERENCE_DEFAULT_ENABLE_CHUNKED_PREFILL
    if args.enable_prefix_caching is None:
        args.enable_prefix_caching = TUNING_REFERENCE_DEFAULT_ENABLE_PREFIX_CACHING


def effective_query_ids(args: argparse.Namespace) -> list[str]:
    if (
        args.flock_workload_driver == "tpcc"
        and args.flock_terminal_queries is not None
    ):
        return flatten_terminal_query_sequences(args.flock_terminal_queries)
    return [normalize_query_id(query) for query in args.queries]


def validate_tpcc_terminal_query_args(args: argparse.Namespace) -> None:
    if args.flock_terminal_queries is None:
        return
    if args.flock_workload_driver != "tpcc":
        raise ValueError(
            "--flock-terminal-queries is only supported with "
            "--flock-workload-driver tpcc"
        )
    if len(args.flock_terminal_queries) != args.flock_terminal_count:
        raise ValueError(
            "flock terminal query sequence count must match flock terminal count "
            f"({len(args.flock_terminal_queries)} != {args.flock_terminal_count})"
        )
    for terminal_id, query_sequence in enumerate(args.flock_terminal_queries):
        if not query_sequence:
            raise ValueError(
                f"flock terminal query sequence for terminal {terminal_id} must not be empty"
            )


def parse_json_object_arg(raw_value: str | None, arg_name: str) -> dict[str, Any] | None:
    if raw_value is None:
        return None
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return parsed


def resolve_model_path(args: argparse.Namespace) -> str:
    model_path = resolve_model_path_for_quantization(
        model_path=args.model_path,
        awq_model_path=args.awq_model_path,
        gptq_model_path=args.gptq_model_path,
        quantization=args.quantization,
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Resolved model path does not exist for quantization={args.quantization}: {model_path}"
        )
    return model_path


def cacheblend_enabled(args: argparse.Namespace) -> bool:
    return args.cacheblend_mode == "on"


def lmcache_enabled(
    args: argparse.Namespace,
    kv_transfer_config: dict[str, Any] | None = None,
) -> bool:
    if kv_transfer_config is None:
        kv_transfer_config = resolve_kv_transfer_config(args)
    return (
        kv_transfer_config is not None
        and kv_transfer_config.get("kv_connector") == "LMCacheConnectorV1"
    )


def resolved_kv_offloading_backend(
    args: argparse.Namespace,
    kv_transfer_config: dict[str, Any] | None = None,
) -> str | None:
    backend = args.kv_offloading_backend
    if backend != "lmcache":
        return backend
    if lmcache_enabled(args, kv_transfer_config):
        return backend
    # 未启用 cacheblend 且没有显式 kv-transfer-config 时，不启动 LMCache。
    return None


def effective_prefix_caching_enabled(args: argparse.Namespace) -> bool:
    if cacheblend_enabled(args) and not args.cacheblend_allow_prefix_caching:
        return False
    return bool(args.enable_prefix_caching)


def resolve_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any] | None:
    explicit_config = parse_json_object_arg(
        args.kv_transfer_config_json, "--kv-transfer-config-json"
    )
    if explicit_config is not None:
        return explicit_config
    if not cacheblend_enabled(args):
        return None
    return {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "use_layerwise": True,
            "lmcache.use_layerwise": True,
            "lmcache.enable_blending": True,
            "lmcache.blend_special_str": args.cacheblend_blend_special_str,
            "lmcache.blend_min_tokens": args.cacheblend_blend_min_tokens,
            "lmcache.blend_check_layers": args.cacheblend_blend_check_layers,
            "lmcache.blend_recompute_ratios": args.cacheblend_blend_recompute_ratios,
            "lmcache.save_unfull_chunk": True,
        },
    }


def resolve_speculative_config(args: argparse.Namespace) -> dict[str, Any] | None:
    explicit_config = parse_json_object_arg(
        args.speculative_config_json, "--speculative-config-json"
    )
    if explicit_config is not None:
        return explicit_config
    if not args.speculative_method:
        return None
    if args.num_speculative_tokens is None or args.num_speculative_tokens < 1:
        raise ValueError(
            "--num-speculative-tokens must be >= 1 when --speculative-method is set"
        )
    config: dict[str, Any] = {
        "method": args.speculative_method,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    if args.speculative_method == "ngram":
        config["prompt_lookup_min"] = args.speculative_prompt_lookup_min
        config["prompt_lookup_max"] = args.speculative_prompt_lookup_max
    return config


def normalized_kv_offloading_size(args: argparse.Namespace) -> float:
    raw_value = args.kv_offloading_size
    if raw_value is None:
        return 0.0
    return max(0.0, float(raw_value))


def normalized_cpu_offload_gb(args: argparse.Namespace) -> float:
    raw_value = args.cpu_offload_gb
    if raw_value is None:
        return 0.0
    return max(0.0, float(raw_value))


def resolved_chunked_prefill_setting(
    args: argparse.Namespace,
) -> bool | None:
    # 保留 argparse 的三态值，以区分：
    # - None：交给 vLLM 使用模型相关默认值；
    # - True：强制开启 chunked prefill；
    # - False：强制关闭 chunked prefill。
    if args.enable_chunked_prefill is None:
        return None
    return bool(args.enable_chunked_prefill)


def resolved_prefix_caching_setting(
    args: argparse.Namespace,
) -> bool | None:
    if cacheblend_enabled(args) and not args.cacheblend_allow_prefix_caching:
        return False
    if args.enable_prefix_caching is not None:
        return bool(args.enable_prefix_caching)
    return None


def hma_disable_required(
    args: argparse.Namespace,
    kv_transfer_config: dict[str, Any] | None = None,
) -> bool:
    # 当 kv_offloading_size > 0 时，native 是有效默认后端。因此只要 offloading
    # size 为正，就按 HMA 不兼容处理，避免旧 BO 配置依赖隐式默认值绕过检查。
    if normalized_kv_offloading_size(args) > 0:
        return True
    return lmcache_enabled(args, kv_transfer_config)
