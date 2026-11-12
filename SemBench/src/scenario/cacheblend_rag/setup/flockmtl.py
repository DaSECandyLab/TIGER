import duckdb

from cacheblend_hotpotqa import dataset_dir, load_dataset
from flockmtl_config import (
    configure_flockmtl_openai_model,
    load_flockmtl_extension,
    resolve_flockmtl_db_path,
)


class FlockMTLCacheBlendRagSetup:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        db_path = resolve_flockmtl_db_path(
            str(dataset_dir() / "cacheblend_rag.duckdb")
        )
        self.flockmtl_conn = duckdb.connect(
            db_path,
            config={"allow_unsigned_extensions": "true"},
        )
        load_flockmtl_extension(self.flockmtl_conn)
        configure_flockmtl_openai_model(self.flockmtl_conn, model_name)

    def setup_data(self) -> None:
        load_dataset(self.flockmtl_conn, dataset_dir())

    def get_connection(self):
        return self.flockmtl_conn
