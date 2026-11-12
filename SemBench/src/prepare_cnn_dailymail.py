"""Build a deterministic, length-stratified CNN/DailyMail 3.0.0 workload."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1] / "files" / "cnn_dailymail"
REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
SOURCE_SHA256 = "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e"
SOURCE_URL = f"https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/{REVISION}/3.0.0/test-00000-of-00001.parquet"
PROMPT = (
    "Summarize the main facts of the following news article in a concise summary. "
    "Preserve names, numbers and factual details. Use the article's wording where appropriate. "
    "Do not add facts or commentary. Return a JSON object with one string field summary.\n"
    "Article:\n{{article}}"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(article_ids, summary_ids, n):
    """Fraction of reference n-gram positions present in the article, not acceptance rate."""
    grams = {tuple(article_ids[i:i+n]) for i in range(len(article_ids)-n+1)}
    total = max(0, len(summary_ids)-n+1)
    return sum(tuple(summary_ids[i:i+n]) in grams for i in range(total)) / total if total else 0.0


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_sql(query_id, prompt=PROMPT, max_tokens=512):
    schema = {"type": "object", "properties": {"summary": {"type": "string"}},
              "required": ["summary"], "additionalProperties": False}
    parameters = {"max_tokens": max_tokens, "temperature": 0.0, "seed": 42,
                  "response_format": {"type": "json_schema", "strict": True,
                                      "json_schema": {"name": "news_summary", "schema": schema}}}
    return """SELECT row_id,
  json_extract_string(TRY_CAST(CAST(llm_complete(
    {'model_name': '<<model_name>>', 'model_parameters': '%s'},
    {'prompt': '%s',
     'context_columns': [{'data': article, 'name': 'article'}]}
  ) AS VARCHAR) AS JSON), '$.summary') AS summary
FROM cnn_articles
WHERE query_id = %d AND query_row <= __ROW_LIMIT__
ORDER BY row_id;
""" % (json.dumps(parameters).replace("'", "''"), prompt.replace("'", "''"), query_id)


def build(source, tokenizer_path, output, rows_per_query, seed):
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Source does not match the pinned CNN/DailyMail test parquet")
    if rows_per_query < 1:
        raise ValueError("rows_per_query must be positive")
    if (output / "manifest.json").exists() or (output / "data").exists():
        raise FileExistsError("Refusing to overwrite prepared data")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    source_rows = pq.read_table(source).to_pylist()
    if len({row['id'] for row in source_rows}) != len(source_rows):
        raise ValueError("Duplicate source IDs")
    source_rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    bounds = {1: (512, 1024), 2: (1024, 2048), 3: (2048, 4096)}
    selected = {q: [] for q in bounds}
    eligible = dict.fromkeys(bounds, 0)
    inputs, references, audit = [], [], []
    for row in source_rows:
        article_ids = tokenizer.encode(row["article"], add_special_tokens=False).ids
        query_id = next((q for q, (lo, hi) in bounds.items() if lo <= len(article_ids) < hi), None)
        if query_id is None:
            continue
        eligible[query_id] += 1
        if len(selected[query_id]) >= rows_per_query:
            continue
        selected[query_id].append(row['id'])
        query_row = len(selected[query_id])
        common = dict(row_id=row['id'], query_id=query_id, query_row=query_row)
        reference_ids = tokenizer.encode(row["highlights"], add_special_tokens=False).ids
        inputs.append(dict(common, article=row['article']))
        references.append(dict(common, summary=row['highlights']))
        audit.append(dict(common, article_tokens=len(article_ids), reference_tokens=len(reference_ids),
                          **{f"reference_{n}gram_overlap": overlap(article_ids, reference_ids, n)
                             for n in (2, 4, 8)}))
    if any(len(rows) != rows_per_query for rows in selected.values()):
        raise ValueError(f"Insufficient eligible rows: {eligible}")
    for part in ("data", "query/flockmtl"):
        (output / part).mkdir(parents=True, exist_ok=True)
    for name, rows in (("articles", inputs), ("references", references), ("selection", audit)):
        rows.sort(key=lambda row: (row['query_id'], row['query_row']))
        write_csv(output / "data" / f"{name}.csv", rows)
    for query_id in bounds:
        (output / "query/flockmtl" / f"Q{query_id}.sql").write_text(query_sql(query_id))
    stats = {}
    for q in bounds:
        group = [row for row in audit if row['query_id'] == q]
        stats[str(q)] = {"article_token_range_inclusive_exclusive": bounds[q],
                         "eligible_rows": eligible[q], "selected_rows": len(group),
                         **{key + "_mean": statistics.mean(row[key] for row in group)
                            for key in ("article_tokens", "reference_tokens", "reference_2gram_overlap",
                                        "reference_4gram_overlap", "reference_8gram_overlap")},
                         "reference_tokens_max": max(row['reference_tokens'] for row in group)}
    manifest = dict(dataset="abisee/cnn_dailymail", config="3.0.0", split="test",
                    revision=REVISION, source_url=SOURCE_URL, source_sha256=sha256(source),
                    source_row_count=len(source_rows), tokenizer_file=str(tokenizer_path.resolve()),
                    tokenizer_sha256=sha256(tokenizer_path), seed=seed, rows_per_query=rows_per_query,
                    total_rows=len(inputs), prompt=PROMPT, queries=stats,
                    selection="SHA256(seed:id) order, stratified only by article token length; no gold/output/overlap selection",
                    task="Original article-to-highlights summarization; unmodified articles and human references",
                    files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                           if p.is_file() and p.suffix in ('.csv', '.sql')})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "source/test.parquet")
    parser.add_argument("--tokenizer", type=Path, default=ROOT.parents[2] / "models/Qwen2-7B-Instruct/tokenizer.json")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--rows-per-query", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build(args.source, args.tokenizer, args.output, args.rows_per_query, args.seed)
    print(json.dumps({"total_rows": result['total_rows'], "queries": result['queries']}, indent=2))
