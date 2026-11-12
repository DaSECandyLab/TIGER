"""Prepared text-only MMQA sf1000 plus the frozen Ecomm four-query sample."""
import json
import os
from pathlib import Path


def files_root():
    # The trial runner's override is SQL-only; dataset/evaluator artifacts
    # remain in the prepared workload, independently of rewritten queries.
    parent = os.environ.get("MMQA_ECOMM_FILES_ROOT")
    return Path(parent) if parent else Path(__file__).resolve().parents[3] / "files" / "mmqa_ecomm"


class MMQAEcommScenario:
    def __init__(self, scale_factor=1000):
        if scale_factor != 1000:
            raise ValueError("This prepared mixed workload requires scale_factor=1000")
        self.scale_factor = scale_factor
        self.root = files_root()

    def get_data_dir(self):
        return str(self.root / "data" / "sf_1000")

    def setup_scenario(self, systems):
        if systems != ["flockmtl"]:
            raise ValueError("mmqa_ecomm currently supports only flockmtl")
        for relative in json.loads((self.root / "manifest.json").read_text())["required_files"]:
            if not (self.root / relative).is_file():
                raise FileNotFoundError(self.root / relative)

    def discover_available_queries(self, system_name=None):
        return list(range(1, 11))

    def get_query_text(self, query_id, system_name):
        if system_name != "flockmtl" or int(query_id) not in range(1, 11):
            raise ValueError((query_id, system_name))
        # Frozen SQL uses this token for portable references to Ecomm samples.
        override = os.environ.get("SEMBENCH_FILES_OVERRIDE_ROOT")
        query_root = Path(override) / "mmqa_ecomm" if override else self.root
        return (query_root / "query" / "flockmtl" / f"Q{query_id}.sql").read_text().replace(
            "__ECOMM_DATA__", str(self.root / "sources" / "ecomm" / "data").replace("'", "''")
        )
