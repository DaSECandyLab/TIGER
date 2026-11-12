"""HotpotQA queries and data for the CacheBlend runtime."""
import os
from pathlib import Path

from cacheblend_hotpotqa import dataset_dir, partition_query_sql


class CacheBlendRagScenario:
    def __init__(self, scale_factor: int = 1):
        self.scale_factor = scale_factor

    def setup_scenario(self, systems):
        if systems != ["flockmtl"]:
            raise ValueError("HotpotQA supports flockmtl only")
        from scenario.cacheblend_rag.setup.flockmtl import FlockMTLCacheBlendRagSetup
        setup = FlockMTLCacheBlendRagSetup()
        try:
            setup.setup_data()
        finally:
            setup.get_connection().close()

    @staticmethod
    def _query_dir(system_name="flockmtl"):
        if system_name != "flockmtl":
            raise ValueError(f"Unsupported HotpotQA system: {system_name}")
        override = os.environ.get("SEMBENCH_FILES_OVERRIDE_ROOT")
        root = Path(override) / "cacheblend_rag" if override else dataset_dir()
        return root / "query" / "flockmtl"

    def get_query_text(self, query_id, system_name):
        path = self._query_dir(system_name) / f"Q{query_id}.sql"
        return partition_query_sql(path.read_text(encoding="utf-8"))

    def discover_available_queries(self, system_name=None):
        return sorted(int(path.stem[1:]) for path in
                      self._query_dir(system_name or "flockmtl").glob("Q*.sql")
                      if path.stem[1:].isdigit())
