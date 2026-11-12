"""Frozen CNN/DailyMail test articles, partitioned by input length."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "files" / "cnn_dailymail"


def files_root():
    override = os.environ.get("CNN_DAILYMAIL_FILES_ROOT")
    if override:
        return Path(override).resolve()
    replay = ROOT / "replay60_per_sql_v1"
    return replay if (replay / "manifest.json").exists() else ROOT


class CNNDailyMailScenario:
    def __init__(self, scale_factor=None):
        self.root = files_root()
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.scale_factor = manifest["rows_per_query"] if scale_factor is None else int(scale_factor)
        self.query_ids = sorted(map(int, manifest["queries"]))
        if not 1 <= self.scale_factor <= manifest["rows_per_query"]:
            raise ValueError(f"scale_factor must be 1..{manifest['rows_per_query']} (rows per query)")

    def get_data_dir(self):
        return str(self.root / "data")

    def setup_scenario(self, systems):
        if systems != ["flockmtl"]:
            raise ValueError("cnn_dailymail currently supports flockmtl only")
        for filename in ("articles.csv", "references.csv"):
            if not (self.root / "data" / filename).is_file():
                raise FileNotFoundError(self.root / "data" / filename)

    def discover_available_queries(self, system_name=None):
        return self.query_ids.copy()

    def get_query_text(self, query_id, system_name):
        query_id = int(str(query_id).removeprefix("Q").removeprefix("q"))
        if system_name != "flockmtl" or query_id not in self.query_ids:
            raise ValueError((query_id, system_name))
        override = os.environ.get("SEMBENCH_FILES_OVERRIDE_ROOT")
        root = Path(override) / "cnn_dailymail" if override else self.root
        return (root / "query" / "flockmtl" / f"Q{query_id}.sql").read_text().replace(
            "__ROW_LIMIT__", str(self.scale_factor)
        )
