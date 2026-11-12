"""Build the experimental CNN/DailyMail replay: 12 SQLs, 60 requests per SQL."""
import csv
import json
from pathlib import Path
import random

from prepare_cnn_dailymail import ROOT, PROMPT, query_sql, sha256, write_csv

REPLAY_PROMPT = PROMPT.replace(
    "in a concise summary.", "in exactly 3 concise sentences, at most 100 words total."
)


def build(parent=ROOT, output=ROOT / "replay60_per_sql_v1", rows_per_query=60,
          starts=(1, 14, 27, 41)):
    if output.exists():
        raise FileExistsError(output)
    if len(starts) != 4 or any(start < 1 or start + rows_per_query - 1 > 100 for start in starts):
        raise ValueError("Four windows must fit within the 100 parent articles per length stratum")
    if set().union(*(set(range(start, start + rows_per_query)) for start in starts)) != set(range(1, 101)):
        raise ValueError("Windows must cover all 100 parent articles")
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    for relative, expected in parent_manifest["files"].items():
        if sha256(parent / relative) != expected:
            raise ValueError(f"Parent artifact changed: {relative}")
    tables = {}
    for name in ("articles", "references", "selection"):
        with (parent / "data" / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
            tables[name] = list(csv.DictReader(handle))

    # Four windows per length stratum cover all 100 parent articles.
    # Some articles recur across SQLs; every SQL contains distinct articles.
    mapping = {}
    for parent_query in range(1, 4):
        for batch, start in enumerate(starts):
            query_id = (parent_query - 1) * 4 + batch + 1
            for query_row, parent_row in enumerate(range(start, start + rows_per_query), 1):
                mapping.setdefault((parent_query, parent_row), []).append((query_id, query_row))

    (output / "data").mkdir(parents=True)
    (output / "query/flockmtl").mkdir(parents=True)
    for name, rows in tables.items():
        selected = []
        for row in rows:
            key = (int(row["query_id"]), int(row["query_row"]))
            for query_id, query_row in mapping.get(key, []):
                # A distinct request identity is needed even when the article recurs.
                selected.append(dict(row, source_row_id=row["row_id"],
                                     row_id=f"q{query_id:02d}_{query_row:02d}",
                                     query_id=query_id, query_row=query_row))
        selected.sort(key=lambda row: (int(row["query_id"]), int(row["query_row"])))
        total_rows = 12 * rows_per_query
        if len(selected) != total_rows:
            raise ValueError(f"Expected {total_rows} {name} rows, got {len(selected)}")
        write_csv(output / "data" / f"{name}.csv", selected)

    queries = {}
    for query_id in range(1, 13):
        (output / "query/flockmtl" / f"Q{query_id}.sql").write_text(
            query_sql(query_id, REPLAY_PROMPT, 1024)
        )
        parent_query = (query_id - 1) // 4 + 1
        queries[str(query_id)] = dict(
            parent_query=parent_query, batch=(query_id - 1) % 4 + 1,
            selected_rows=rows_per_query,
            article_token_range_inclusive_exclusive=parent_manifest["queries"][str(parent_query)][
                "article_token_range_inclusive_exclusive"
            ],
        )
    # Partition the 12 SQLs across three balanced terminals. Each terminal
    # gets four SQLs and four times rows_per_query requests, including all
    # three length strata.
    terminal_queries = ((1, 4, 5, 9), (2, 6, 7, 10), (3, 8, 11, 12))
    sequences = []
    for terminal, assigned in enumerate(terminal_queries):
        sequence = list(assigned)
        random.Random(20260918 + terminal).shuffle(sequence)
        sequences.append(sequence)
    manifest = dict(parent_manifest,
        rows_per_query=rows_per_query, total_rows=12 * rows_per_query, queries=queries,
        prompt=REPLAY_PROMPT, max_output_tokens=1024,
        parent_manifest_sha256=sha256(parent / "manifest.json"),
        selection=f"Four {rows_per_query}-article windows per length stratum, starts {'/'.join(map(str, starts))} in the parent's hash order; all 100 parent articles are covered, with partial overlap between SQLs",
        terminal_sequences=sequences, terminal_shuffle_seeds=[20260918 + i for i in range(3)],
        requests_per_workload=12 * rows_per_query, independent_articles=300,
        replay_policy=f"Twelve SQLs are partitioned across three terminals, four SQLs and {4 * rows_per_query} requests per terminal; each SQL has {rows_per_query} distinct articles, with partial article reuse across SQLs in a length stratum.",
        files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "replay60_per_sql_v1")
    args = parser.parse_args()
    print(json.dumps(build(parent=args.parent, output=args.output), indent=2))
