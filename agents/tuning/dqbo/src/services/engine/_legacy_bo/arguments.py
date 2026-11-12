"""Flock 贝叶斯优化的命令行参数。"""

from __future__ import annotations

import argparse
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[7]

DEFAULT_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct')
DEFAULT_AWQ_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct-AWQ')
DEFAULT_GPTQ_INT4_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct-GPTQ-Int4')
DEFAULT_MODEL_NAME = "Qwen2-7B-Instruct"
DEFAULT_VLLM_BIN = str(REPOSITORY_ROOT / 'vllm-0.20.0/.venv/bin/vllm')
DEFAULT_SEMBENCH_DIR = str(REPOSITORY_ROOT / 'SemBench')
DEFAULT_FLOCK_EXT = str(REPOSITORY_ROOT / 'flock/build/release/extension/flock/flock.duckdb_extension')
DEFAULT_FLOCK_HTTP_MAX_CONNECTIONS = 2048
MAX_IN_FLIGHT_STRATEGIES = (
    "dynamic_active_terminal_partition",
    "global_slot_coordination",
    "fixed_equal_terminal_partition",
)
DEFAULT_TUNED_PARAMS = frozenset({"tp", "data_parallel_size", "quantization", "max_num_seqs", "max_num_batched_tokens", "block_size", "enable_prefix_caching", "disable_custom_all_reduce"})
FOCUSED_30_TUNED_PARAMS = frozenset({"max_num_seqs", "max_num_batched_tokens", "block_size", "max_model_len", "gpu_memory_utilization", "enable_prefix_caching"})
TUNE_PARAM_CHOICES = (
    "tp", "data_parallel_size", "quantization", "dtype", "model_impl",
    "decode_context_parallel_size", "cp_kv_cache_interleave_size",
    "data_parallel_backend",
    "disable_nccl_for_dp_synchronization", "cpu_offload_gb", "kv_cache_dtype",
    "kv_offloading_backend", "kv_offloading_size", "cacheblend_mode",
    "lmcache_chunk_size", "cacheblend_blend_min_tokens",
    "cacheblend_blend_check_layers", "cacheblend_blend_recompute_ratios",
    "speculative_method", "num_speculative_tokens", "max_num_seqs",
    "max_num_batched_tokens", "block_size", "long_prefill_token_threshold",
    "max_model_len", "gpu_memory_utilization", "enforce_eager",
    "enable_prefix_caching", "prefix_caching_hash_algo",
    "scheduler_reserve_full_isl", "async_scheduling", "disable_cascade_attn",
    "enable_flashinfer_autotune", "disable_custom_all_reduce",
)
LEGACY_TUNE_FLAG_TO_PARAMS = {"tune_gpu_memory_utilization": {"gpu_memory_utilization"}, "tune_max_model_len": {"max_model_len"}, "tune_enforce_eager": {"enforce_eager"}, "tune_cacheblend_mode": {"cacheblend_mode"}, "tune_kv_cache_dtype": {"kv_cache_dtype"}, "tune_speculative_decoding": {"speculative_method", "num_speculative_tokens"}}
SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND = "native"
def parse_flock_max_in_flight_arg(value: str) -> int | str:
    """解析 BO 侧透传给 trial CLI 的 Flock 请求上限参数。"""
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


def parse_non_negative_int(value: str) -> int:
    """解析允许用 0 表示关闭的基础设施上限。"""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


# 解析并校验用户通过 --tune_params 指定的调参维度名称。
def parse_tune_param_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in TUNE_PARAM_CHOICES:
        supported = ", ".join(TUNE_PARAM_CHOICES)
        raise argparse.ArgumentTypeError(
            f"unsupported tune param '{value}'. Supported values: {supported}"
        )
    return normalized


# 注册 BO、vLLM、Flock workload、CacheBlend 等所有命令行参数。
def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """注册 SCOOT + Flock trial 所需参数。

    其中 TPCC workload 相关参数会原样透传给 trial CLI，用于切换
    workload driver、terminal 数量和请求数量控制方式。
    """
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--awq_model_path", default=DEFAULT_AWQ_MODEL_PATH)
    parser.add_argument("--gptq_model_path", default=DEFAULT_GPTQ_INT4_MODEL_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--quantization",
        choices=["none", "awq", "gptq", "awq_marlin", "gptq_marlin"],
        default="none",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--cpu_offload_gb", type=float, default=None)
    parser.add_argument("--kv_cache_dtype", default=None)
    parser.add_argument("--kv_cache_dtype_skip_layers", nargs="*", default=[])
    parser.add_argument(
        "--kv_offloading_backend",
        choices=["native", "lmcache"],
        default=SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND,
    )
    parser.add_argument("--kv_offloading_size", type=float, default=0.0)
    parser.add_argument(
        "--cacheblend_mode",
        choices=["auto", "on", "off"],
        default="auto",
    )
    parser.add_argument("--cacheblend_allow_prefix_caching", action="store_true")
    parser.add_argument("--kv_transfer_config_json", default=None)
    parser.add_argument("--lmcache_chunk_size", type=int, default=16)
    parser.add_argument("--cacheblend_blend_special_str", default=" # # ")
    parser.add_argument("--cacheblend_blend_min_tokens", type=int, default=16)
    parser.add_argument(
        "--cacheblend_blend_check_layers", nargs="*", type=int, default=[1]
    )
    parser.add_argument(
        "--cacheblend_blend_recompute_ratios",
        nargs="*",
        type=float,
        default=[0.15],
    )
    parser.add_argument("--speculative_config_json", default=None)
    parser.add_argument("--speculative_method", default=None)
    parser.add_argument("--num_speculative_tokens", type=int, default=None)
    parser.add_argument("--speculative_prompt_lookup_min", type=int, default=2)
    parser.add_argument("--speculative_prompt_lookup_max", type=int, default=5)
    parser.add_argument("--vllm_bin", default=DEFAULT_VLLM_BIN)
    parser.add_argument(
        "--vllm_api_server_count",
        type=int,
        default=None,
        help=(
            "Optional fixed vLLM HTTP frontend process count. By default the "
            "flag is omitted and vLLM derives it from data_parallel_size."
        ),
    )
    parser.add_argument("--sembench_dir", default=DEFAULT_SEMBENCH_DIR)
    parser.add_argument("--flock_ext", default=DEFAULT_FLOCK_EXT)
    parser.add_argument(
        "--run_name",
        default=None,
        help="Manual run directory name under scoot/tune_res/scoot_flock.",
    )
    parser.add_argument("--bo_loop", type=int, default=10)
    parser.add_argument("--bo_batch_size", type=int, default=1)
    parser.add_argument("--exp_num", type=int, default=1)
    parser.add_argument(
        "--num_obj",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help=(
            "Number of Pareto objectives in order: throughput, mean latency, "
            "p95 latency. Quality is enforced separately by per-query gates."
        ),
    )
    parser.add_argument("--gpu_num", type=int, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup_timeout", type=int, default=300)
    parser.add_argument("--scale_factor", type=int, default=2000)
    parser.add_argument("--use_case", default="movie")
    parser.add_argument("--queries", nargs="+", default=["1", "2", "3", "4", "8"])
    parser.add_argument(
        "--flock_terminal_queries",
        action="append",
        nargs="+",
        metavar="QUERY",
        default=None,
        help=(
            "Per-terminal query sequence for TPCC workload mode. Repeat once per "
            "terminal, for example: --flock_terminal_queries 1 2 "
            "--flock_terminal_queries 3 4 8"
        ),
    )
    parser.add_argument("--flock_request_rate", default="inf")
    parser.add_argument("--flock_request_arrival", default="poisson")
    parser.add_argument(
        "--flock_max_in_flight",
        type=parse_flock_max_in_flight_arg,
        default="unlimited",
        help=(
            "Maximum simultaneous in-flight Flock requests. Use an integer, "
            "0/inf/unlimited for no extra client-side cap, or max_num_seqs "
            "to follow each BO candidate's max_num_seqs."
        ),
    )
    parser.add_argument(
        "--flock_max_in_flight_strategy",
        choices=MAX_IN_FLIGHT_STRATEGIES,
        default="dynamic_active_terminal_partition",
        help=(
            "How TPCC terminals share max_in_flight. "
            "global_slot_coordination uses one cross-process slot pool."
        ),
    )
    parser.add_argument(
        "--flock_http_max_connections",
        type=parse_non_negative_int,
        default=DEFAULT_FLOCK_HTTP_MAX_CONNECTIONS,
        help=(
            "Fixed global Flock HTTP connection budget. Requests above this "
            "budget wait in libcurl; this is infrastructure, not a BO dimension. "
            "Use 0 only to restore the previous unlimited-connection behavior."
        ),
    )
    parser.add_argument("--flock_query_concurrency", type=int, default=1)
    parser.add_argument("--flock_query_start_stagger_ms", type=float, default=0.0)
    parser.add_argument(
        "--flock_workload_driver",
        choices=["sembench", "tpcc"],
        default="sembench",
    )
    parser.add_argument("--flock_terminal_count", type=int, default=1)
    parser.add_argument(
        "--flock_request_control_mode",
        choices=["max_in_flight", "vllm_batch_size_div_terminals"],
        default="max_in_flight",
    )
    parser.add_argument("--flock_vllm_batch_size", type=int, default=None)
    parser.add_argument(
        "--flock_failure_policy",
        choices=["fail_fast"],
        default="fail_fast",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument(
        "--screened-parameter-results",
        default=None,
        help=(
            "Parameter-level final-screening JSON. It selects parameter names only; "
            "candidate-value examples from this file never define the BO domain."
        ),
    )
    parser.add_argument(
        "--parameter-domain-spec",
        default=None,
        help=(
            "Audited bo_domain_spec.json produced by scoot.vllm_param_domain. "
            "Required together with --screened-parameter-results."
        ),
    )
    parser.add_argument(
        "--baseline-config",
        default=None,
        help=(
            "Audited baseline_config.json generated before measurement. BO runs "
            "this exact effective configuration once as the objective reference; "
            "the screened parameter file and domain spec still exclusively decide "
            "which knobs and ranges are searched."
        ),
    )
    parser.add_argument("--model_impl", default="vllm")
    parser.add_argument("--decode_context_parallel_size", type=int, default=1)
    parser.add_argument("--cp_kv_cache_interleave_size", type=int, default=1)
    parser.add_argument("--data_parallel_backend", choices=["mp", "ray"], default="mp")
    parser.add_argument(
        "--disable_nccl_for_dp_synchronization",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prefix_caching_hash_algo", default="sha256")
    parser.add_argument(
        "--scheduler_reserve_full_isl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--async_scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--disable_cascade_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable_flashinfer_autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--data_parallel_size_lb", type=int, default=1)
    parser.add_argument("--data_parallel_size_ub", type=int, default=None)
    parser.add_argument(
        "--trial_script",
        default=None,
        help=(
            "Optional Python trial script override. By default the package runs "
            "python -m scoot.flock.cli.trial."
        ),
    )
    parser.add_argument("--keep_vllm", action="store_true")
    parser.add_argument(
        "--tune-params",
        nargs="+",
        type=parse_tune_param_name,
        default=None,
        metavar="PARAM",
        help=(
            "Tune only the listed vLLM knobs and keep all others at their defaults. "
            "Example: --tune-params block_size max_num_seqs"
        ),
    )
    parser.add_argument(
        "--search_space_profile",
        choices=["none", "focused_30"],
        default="none",
        help=(
            "Optional preset that shrinks the BO search space for faster convergence. "
            "'focused_30' keeps only a small set of high-signal knobs and tightens "
            "bounds around the current stable baseline."
        ),
    )

    # 逐 query、逐代表指标门禁；JSON 结构与 trial CLI 保持一致。
    parser.add_argument("--quality_thresholds_json", default=None)

    # 默认值尽量贴近原 SCOOT 搜索空间，但保留 block_size >= 16；
    # 当前 vLLM/Flock 栈中 bs=8 不稳定。
    parser.add_argument("--max_num_seqs_lb", type=int, default=64)
    parser.add_argument("--max_num_seqs_ub", type=int, default=8192)
    parser.add_argument("--max_num_batched_tokens_lb", type=int, default=64)
    parser.add_argument("--max_num_batched_tokens_ub", type=int, default=None)
    parser.add_argument("--block_size_lb", type=int, default=16)
    parser.add_argument("--block_size_ub", type=int, default=64)
    parser.add_argument("--long_prefill_token_threshold", type=int, default=0)
    parser.add_argument("--long_prefill_token_threshold_lb", type=int, default=16)
    parser.add_argument("--long_prefill_token_threshold_ub", type=int, default=1024)
    parser.add_argument("--tune_gpu_memory_utilization", action="store_true")
    parser.add_argument("--gpu_memory_utilization_lb", type=int, default=75)
    parser.add_argument("--gpu_memory_utilization_ub", type=int, default=98)
    parser.add_argument("--gpu_memory_utilization_step", type=int, default=5)
    parser.add_argument(
        "--dtype_categories",
        nargs="+",
        default=["bfloat16", "half", "float"],
    )
    parser.add_argument("--cpu_offload_gb_lb", type=int, default=0)
    parser.add_argument("--cpu_offload_gb_ub", type=int, default=32)
    parser.add_argument("--cpu_offload_gb_step", type=int, default=2)
    parser.add_argument("--tune_max_model_len", action="store_true")
    parser.add_argument("--max_model_len_lb", type=int, default=0)
    parser.add_argument("--max_model_len_ub", type=int, default=32768)
    parser.add_argument("--tune_enforce_eager", action="store_true")
    parser.add_argument(
        "--kv_offloading_backend_categories",
        nargs="+",
        default=["native", "lmcache"],
    )
    parser.add_argument("--kv_offloading_size_lb", type=int, default=0)
    parser.add_argument("--kv_offloading_size_ub", type=int, default=16)
    parser.add_argument("--kv_offloading_size_step", type=int, default=2)
    parser.add_argument("--tune_cacheblend_mode", action="store_true")
    parser.add_argument(
        "--cacheblend_mode_categories",
        nargs="+",
        default=["auto", "on", "off"],
    )
    parser.add_argument("--lmcache_chunk_size_lb", type=int, default=8)
    parser.add_argument("--lmcache_chunk_size_ub", type=int, default=64)
    parser.add_argument("--cacheblend_blend_min_tokens_lb", type=int, default=8)
    parser.add_argument("--cacheblend_blend_min_tokens_ub", type=int, default=64)
    parser.add_argument("--cacheblend_blend_min_tokens_step", type=int, default=8)
    parser.add_argument(
        "--cacheblend_blend_check_layers_categories",
        nargs="+",
        default=["1", "1,2", "1,2,4"],
    )
    parser.add_argument(
        "--cacheblend_blend_recompute_ratios_categories",
        nargs="+",
        default=["0.15", "0.3", "0.5"],
    )
    parser.add_argument("--tune_kv_cache_dtype", action="store_true")
    parser.add_argument(
        "--kv_cache_dtype_categories",
        nargs="+",
        default=["bfloat16", "float16", "fp8_e4m3", "int8_per_token_head"],
    )
    parser.add_argument("--tune_speculative_decoding", action="store_true")
    parser.add_argument(
        "--speculative_method_categories",
        nargs="+",
        default=["none", "ngram"],
    )
    parser.add_argument("--num_speculative_tokens_lb", type=int, default=2)
    parser.add_argument("--num_speculative_tokens_ub", type=int, default=8)
    parser.add_argument("--extra_vllm_args", nargs=argparse.REMAINDER, default=[])
    return parser


# 根据搜索空间 preset 修改参数范围和固定项，例如 focused_30。
def apply_search_space_profile(args: argparse.Namespace) -> None:
    if args.search_space_profile == "none":
        return

    if args.tune_params is not None:
        raise ValueError(
            "--search_space_profile cannot be combined with explicit --tune-params"
        )

    legacy_flags = [
        flag_name
        for flag_name in LEGACY_TUNE_FLAG_TO_PARAMS
        if bool(getattr(args, flag_name))
    ]
    if legacy_flags:
        raise ValueError(
            "--search_space_profile cannot be combined with legacy --tune_* flags: "
            + ", ".join(f"--{flag_name}" for flag_name in legacy_flags)
        )

    if args.search_space_profile == "focused_30":
        # 中文说明：
        # 这个 preset 的目标不是覆盖所有可能路径，而是在 30~50 轮这种预算较紧的
        # 在 BO 里，优先围绕当前最稳定、最有希望超过参考配置的主导参数做局部搜索。
        #
        # 设计原则：
        # 1. 固定已经证明会明显拖慢或放大噪声的维度，如 quantization/offloading/
        #    cacheblend/speculative；
        # 2. 只保留少量对吞吐和延迟更敏感、且失败率相对可控的参数；
        # 3. 收紧上下界，让 BO 不必把大量轮次浪费在明显过激或无效的区域。
        args.tune_params = list(FOCUSED_30_TUNED_PARAMS)

        args.quantization = "none"
        args.dtype = "auto"
        args.cpu_offload_gb = 0.0
        args.kv_cache_dtype = "auto"
        args.kv_offloading_backend = "native"
        args.kv_offloading_size = 0.0
        args.cacheblend_mode = "off"
        args.speculative_method = None
        args.num_speculative_tokens = None

        args.max_num_seqs_lb = max(int(args.max_num_seqs_lb), 64)
        args.max_num_seqs_ub = min(int(args.max_num_seqs_ub), 256)

        args.max_num_batched_tokens_lb = max(int(args.max_num_batched_tokens_lb), 2048)
        if args.max_num_batched_tokens_ub is None:
            args.max_num_batched_tokens_ub = 8192
        else:
            args.max_num_batched_tokens_ub = min(
                int(args.max_num_batched_tokens_ub), 8192
            )

        args.block_size_lb = max(int(args.block_size_lb), 16)
        args.block_size_ub = min(int(args.block_size_ub), 64)

        args.max_model_len_lb = max(int(args.max_model_len_lb), 1024)
        args.max_model_len_ub = min(int(args.max_model_len_ub), 8192)

        args.gpu_memory_utilization_lb = max(
            int(args.gpu_memory_utilization_lb), 85
        )
        args.gpu_memory_utilization_ub = min(
            int(args.gpu_memory_utilization_ub), 92
        )
        args.gpu_memory_utilization_step = 5
        return

    raise ValueError(f"unsupported search space profile: {args.search_space_profile}")
