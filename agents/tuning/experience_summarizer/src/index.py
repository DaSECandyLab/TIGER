"""Summarize an existing BO run locally; this does not execute new trials."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True, help='Completed run bo_context.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--llm-mode', choices=['off', 'auto', 'required'], default='auto')
    args = parser.parse_args()
    from .services.cluster_experience import consolidate_run, summary_client
    result = consolidate_run(args.context, output_dir=args.output_dir,
                             client=summary_client(args.llm_mode))
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
