"""Movies workload input statistics from SQL/CSV/tokenizer, without run artifacts.

Only pure relational prompt producers are executed. No Flock extension, model
weights, HTTP endpoint, output labels, or baseline traces are needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def llm_calls(value):
    return [n for n in walk(value) if n.get('function_name', '').lower().startswith('llm_')]


def project_prompt_sql(con, sql):
    """Extract one supported scalar call's input; reject output-dependent SQL.

    This is deliberately a restricted compiler, not arbitrary SQL emulation.
    Top-level preceding pure CTEs are retained; downstream postprocessing is
    discarded. A WHERE consisting solely of CAST(llm_filter(...)) is supported.
    """
    tree = json.loads(con.execute('SELECT json_serialize_sql(?)', [sql]).fetchone()[0])
    if tree.get('error') or len(tree.get('statements', [])) != 1:
        raise ValueError('Expected one parseable SELECT')
    root = tree['statements'][0]['node']
    calls = llm_calls(root)
    if len(calls) != 1 or calls[0]['function_name'].lower() not in ('llm_filter', 'llm_complete'):
        raise ValueError('Only one scalar LLM call is supported; output-dependent/multiple calls rejected')
    call = calls[0]
    owner = None
    for node in walk(root):
        if node.get('type') == 'SELECT_NODE':
            local = {k: v for k, v in node.items() if k != 'cte_map'}
            if any(c is call for c in llm_calls(local)):
                owner = node  # deepest SELECT containing the call
    if owner is None or len(call.get('children', [])) != 2:
        raise ValueError('Unsupported LLM call structure')
    out = deepcopy(owner)
    if out.get('group_expressions') or out.get('having') or out.get('qualify') or out.get('modifiers'):
        raise ValueError('Grouping/limit/order at LLM call site requires explicit extraction support')
    where = owner.get('where_clause')
    if llm_calls(where):
        expr = where
        while expr.get('class') == 'CAST':
            expr = expr['child']
        if expr is not call or call['function_name'].lower() != 'llm_filter':
            raise ValueError('Compound semantic WHERE predicate is not supported')
        out['where_clause'] = None
    prompt = deepcopy(call['children'][1])
    prompt['alias'] = 'prompt_spec'
    out['select_list'] = [prompt]
    entries = []
    for entry in root.get('cte_map', {}).get('map', []):
        if any(n is owner for n in walk(entry)):
            break
        if llm_calls(entry):
            raise ValueError('Prompt depends on prior LLM output')
        entries.append(deepcopy(entry))
    if owner is not root and owner.get('cte_map', {}).get('map'):
        raise ValueError('Nested CTE scope is unsupported')
    out['cte_map'] = {'map': entries if owner is not root else out['cte_map']['map']}
    # No semantic function may remain anywhere in the executed projection.
    if llm_calls(out):
        raise ValueError('Semantic call survived input projection')
    # Limit sources to local tables; CSV ingestion happens separately.
    for n in walk(out):
        if n.get('type') == 'SUBQUERY' or (n.get('type') == 'TABLE_FUNCTION' and
                n.get('function', {}).get('function_name') != 'range'):
            raise ValueError('Unsupported table function/subquery in prompt producer')
    tree['statements'][0]['node'] = out
    projected = con.execute('SELECT json_deserialize_sql(?)', [json.dumps(tree)]).fetchone()[0]
    return projected


def render_segments(spec, delimiter):
    """Mirror Flock's split-before-placeholder-substitution for text inputs."""
    template = spec['prompt']
    parts = template.split(delimiter)
    for column in spec['context_columns']:
        if set(column) - {'name', 'data'}:
            raise ValueError('Only plain text context columns supported')
        value = column['data']
        text = '' if value is None else value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        key = '{{' + column['name'] + '}}'
        found = any(key in part for part in parts)
        parts = [part.replace(key, text) for part in parts]
        if not found:
            parts[-1] += '\n' + column['name'] + ':' + text + '; '
    if any(re.search(r'\{\{[^}]+\}\}', p) for p in parts):
        raise ValueError('Unresolved template placeholder')
    return parts


def split_tokens(tokens, separator):
    """LMCache 0.4.5 SegmentTokenDatabase sliding separator matching."""
    if not separator:
        raise ValueError('Empty LMCache separator')
    starts = [i for i in range(len(tokens) - len(separator) + 1)
              if tokens[i:i + len(separator)] == separator]
    start = 0
    for i in starts:
        yield tokens[start:i]
        start = i + len(separator)
    yield tokens[start:]


def distribution(counts):
    total = sum(counts.values())
    if not total:
        return {'count': 0}
    ordered = sorted(counts.items())
    result = {'count': total, 'mean': sum(k*v for k, v in ordered)/total,
              'min': ordered[0][0], 'max': ordered[-1][0],
              'histogram': {str(k): v for k, v in ordered}}
    for name, fraction in [('p50', .5), ('p95', .95), ('p99', .99)]:
        accumulated = 0
        for value, count in ordered:
            accumulated += count
            if accumulated >= total * fraction:
                result[name] = value
                break
    return result


def summarize(prompts, segments, segments_per_request, kv_bytes):
    requests = sum(prompts.values())
    total_input = sum(len(p)*n for p, n in prompts.items())
    segment_tokens = sum(len(p)*n for p, n in segments.items())
    unique_tokens = sum(map(len, segments))
    lengths = Counter()
    for p, n in prompts.items():
        lengths[len(p)] += n
    segment_lengths = Counter()
    for p, n in segments.items():
        segment_lengths[len(p)] += n
    # Unique token-trie edges = corpus prefix storage with perfect sharing.
    unique_prefix_tokens, previous = 0, ()
    for p in sorted(prompts):
        shared = 0
        for a, b in zip(previous, p):
            if a != b:
                break
            shared += 1
        unique_prefix_tokens += len(p) - shared
        previous = p
    return {'request_count': requests, 'input_tokens': distribution(lengths),
        'unique_prompt_count': len(prompts),
        'whole_prompt_repeat_fraction': (requests-len(prompts))/requests,
        'segments_per_request': distribution(segments_per_request),
        'segment_tokens': distribution(segment_lengths), 'unique_nonempty_segment_count': len(segments),
        'segment_occurrence_frequency_histogram': dict(sorted(Counter(segments.values()).items())),
        'segment_payload_tokens_total': segment_tokens,
        'unique_segment_payload_tokens': unique_tokens,
        'duplicate_segment_payload_tokens': segment_tokens-unique_tokens,
        'duplicate_segment_payload_fraction': (segment_tokens-unique_tokens)/segment_tokens if segment_tokens else 0,
        'duplicate_segment_tokens_over_input_tokens': (segment_tokens-unique_tokens)/total_input,
        'ideal_prefix_duplicate_tokens': total_input-unique_prefix_tokens,
        'ideal_prefix_duplicate_fraction': (total_input-unique_prefix_tokens)/total_input,
        'unique_segment_kv_payload_bytes': unique_tokens*kv_bytes}


def build_static(manifest_path, data_dir, model_dir, output):
    import duckdb
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT/'SemBench/src'))
    from flockmtl_cacheblend import apply_cacheblend_mode_to_sql
    raw = json.loads(manifest_path.read_text())
    # Do not import historical request counts, quality, results or baseline paths.
    workload = {k: raw[k] for k in ('query_override_root', 'terminal_queries')}
    multiplicities = Counter(q for terminal in workload['terminal_queries'] for q in terminal)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    cfg = json.loads((model_dir/'config.json').read_text())
    kv_bytes = 2 * cfg['num_hidden_layers'] * cfg['num_key_value_heads'] * (cfg['hidden_size']//cfg['num_attention_heads']) * 2
    delimiter = ' # # '
    separator = tuple(tokenizer.encode(delimiter, add_special_tokens=True))
    # Installed native LMCache 0.4.5 uses encode(delimiter)[1:], even for Qwen.
    cache_separator = separator[1:]
    @lru_cache(maxsize=100000)
    def encode(text):
        return tuple(tokenizer.encode(text, add_special_tokens=True))
    con = duckdb.connect(':memory:', config={'autoinstall_known_extensions': 'false', 'autoload_known_extensions': 'false'})
    source_paths = [manifest_path, model_dir/'config.json', Path(__file__),
                    ROOT/'SemBench/src/flockmtl_cacheblend.py',
                    ROOT/'flock/src/functions/scalar/scalar.cpp',
                    ROOT/'vllm-0.20.0/.venv/lib/python3.12/site-packages/lmcache/v1/token_database.py']
    source_paths += sorted(p for p in model_dir.iterdir() if p.is_file() and
                           (p.name.startswith('tokenizer') or p.name in ('vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json')))
    for table, filename in [('reviews_2000', 'Reviews.csv'), ('movies_2000', 'Movies.csv')]:
        path = data_dir/filename
        con.execute(f'CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?)', [str(path)])
        source_paths.append(path)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'static_features.json').exists():
        raise ValueError('Refusing to overwrite an existing static extraction')
    variants, coverage = {}, []
    for mode in ('off', 'on'):
        prompts, segments, segment_counts = Counter(), Counter(), Counter()
        for qid, weight in sorted(multiplicities.items()):
            path = Path(workload['query_override_root'])/f'movie/query/flockmtl/Q{qid}.sql'
            if mode == 'off':
                source_paths.append(path)
            sql = apply_cacheblend_mode_to_sql(path.read_text(), mode)
            projected = project_prompt_sql(con, sql)
            (output/f'{mode}_Q{qid}_inputs.sql').write_text(projected+'\n')
            rows = con.execute(projected).fetchall()
            coverage.append({'mode': mode, 'query': f'Q{qid}', 'instances': weight,
                             'input_rows_per_instance': len(rows), 'weighted_requests': len(rows)*weight})
            for (spec,) in rows:
                parts = render_segments(spec, delimiter)
                if mode == 'on':
                    ids = ()
                    for i, part in enumerate(parts):
                        if i:
                            ids += separator
                        ids += encode(part)
                    chunks = [p for p in split_tokens(ids, cache_separator) if p]
                else:
                    # This is user prompt content, not the chat/system wrapper.
                    ids = encode(delimiter.join(parts))
                    chunks = [ids] if ids else []
                prompts[ids] += weight
                segment_counts[len(chunks)] += weight
                for chunk in chunks:
                    segments[chunk] += weight
        variants[mode] = summarize(prompts, segments, segment_counts, kv_bytes)
    con.close()
    result = {'schema_version': 1, 'kind': 'offline_workload_static_features',
        'scope': 'whole workload; per-query counts are extraction coverage only',
        'baseline_artifacts_read': False, 'model_inference_calls': 0,
        'protocol': {'cacheblend_separator': delimiter, 'flock_separator_token_ids': separator,
                     'lmcache_separator_token_ids': cache_separator, 'remove_first_token': False,
                     'lmcache_segment_rule': '0.4.5 encode(separator)[1:] sliding matches; empty payloads excluded',
                     'kv_payload_bytes_per_token': kv_bytes, 'kv_dtype': 'bfloat16',
                     'tokenizer_class': type(tokenizer).__name__, 'duckdb_version': duckdb.__version__},
        'workload_query_instances': sum(multiplicities.values()),
        'variants': variants, 'coverage': coverage,
        'limitations': [
            'Only static input content is analyzed; no output lengths, actual hit rates, residency, timings or resource telemetry.',
            'Off input length covers rendered user text only, excluding provider chat/system wrappers. On covers Flock segment-assembled token IDs.',
            'Corpus duplication assumes reusable content and counts repeats beyond first occurrence; it is not cache-hit or avoided-prefill measurement.',
            'Ideal prefix sharing ignores block alignment, arrival order and cache availability; it is not the B0 prefix-cache hit rate.',
            'Do not subtract prefix and segment duplicate counts as CacheBlend incremental savings; overlap and computation semantics differ.',
            'KV payload excludes metadata, padding, workspace and replica overhead; no cache capacity simulation is performed.',
            'Source protocol is the installed LMCache 0.4.5/Flock text path; revalidate for different implementations.',
            'Multiple semantic calls and output-dependent request producers are rejected rather than guessed.'],
        'sources': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths]}
    (output/'static_features.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Movies 离线静态特征', '', '输入仅为 SQL、CSV、tokenizer、模型和静态源码；没有读取 baseline trace，也没有执行模型推理。', '',
             '| 特征 | CacheBlend off（用户文本） | CacheBlend on（拼接 token） |', '|---|---:|---:|']
    for key in ('request_count', 'unique_prompt_count', 'whole_prompt_repeat_fraction'):
        lines.append(f'| {key} | {variants["off"][key]} | {variants["on"][key]} |')
    lines += ['', '整个 workload 的分布与片段统计见 `static_features.json`；逐 Q 数据仅作输入覆盖审计，不建立逐 Q 成本模型。', '',
              '局限：off 长度不含聊天包装；on 按本机 Flock/LMCache 分隔规则计算。重复覆盖率不是实际命中率，也不是可避免的计算比例。',
              '旧 baseline 报告保持原始来源；本产物可以独立供新的成本预测器读取，不要求先生成 B0/B1。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'SemBench/files/movie/data/sf_2000')
    parser.add_argument('--model-dir', type=Path, default=ROOT/'models/Qwen2-7B-Instruct')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_static(args.manifest, args.data_dir, args.model_dir, args.output)
    print(json.dumps({'output': str(args.output/'static_features.json'),
                      'requests': {k: v['request_count'] for k, v in result['variants'].items()},
                      'inference_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
