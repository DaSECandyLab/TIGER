from pathlib import Path

from runner.generic_flockmtl_runner.generic_flockmtl_runner import GenericFlockMTLRunner
from scenario.mmqa_ecomm.setup.flockmtl import FlockMTLMMQAEcommSetup


class FlockMTLRunner(GenericFlockMTLRunner):
    def __init__(self, use_case, scale_factor, model_name="gpt-4o-mini",
                 concurrent_llm_worker=20, skip_setup=False):
        super().__init__(use_case, scale_factor, model_name, concurrent_llm_worker, skip_setup)
        self.data_path = Path(self.scenario_handler.get_data_dir())
        setup = FlockMTLMMQAEcommSetup(model_name=model_name, data_dir=self.data_path)
        try:
            setup.setup_data(str(self.data_path))
            self.flockmtl_conn = setup.get_connection()
            ecomm = self.scenario_handler.root / "sources" / "ecomm" / "data"
            self.flockmtl_conn.execute(
                "CREATE OR REPLACE TABLE styles_details AS SELECT * FROM read_parquet(?)",
                [str(ecomm / "styles_details.parquet")],
            )
            self.flockmtl_conn.execute("SET file_search_path = ?", [str(ecomm)])
        except Exception:
            setup.get_connection().close()
            raise
