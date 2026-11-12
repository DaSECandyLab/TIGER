import duckdb

from flockmtl_config import (
    configure_flockmtl_openai_model, load_flockmtl_extension, resolve_flockmtl_db_path,
)
from runner.generic_flockmtl_runner.generic_flockmtl_runner import GenericFlockMTLRunner


class FlockMTLRunner(GenericFlockMTLRunner):
    def __init__(self, use_case, scale_factor, model_name="Qwen2-7B-Instruct",
                 concurrent_llm_worker=20, skip_setup=False):
        super().__init__(use_case, scale_factor, model_name, concurrent_llm_worker, skip_setup)
        from pathlib import Path
        self.data_path = Path(self.scenario_handler.get_data_dir())
        self.flockmtl_conn = duckdb.connect(
            resolve_flockmtl_db_path(str(self.data_path / "cnn_dailymail.duckdb")),
            config={"allow_unsigned_extensions": "true"},
        )
        load_flockmtl_extension(self.flockmtl_conn)
        configure_flockmtl_openai_model(self.flockmtl_conn, model_name)
        # Only inputs are loaded. References never enter the inference database.
        self.flockmtl_conn.execute(
            "CREATE OR REPLACE TABLE cnn_articles AS SELECT * FROM read_csv_auto(?)",
            [str(self.data_path / "articles.csv")],
        )
