import os

import duckdb

from flockmtl_config import (
    configure_flockmtl_openai_model,
    load_flockmtl_extension,
    resolve_flockmtl_db_path,
)

class FlockMTLMMQAEcommSetup:
    def __init__(self, model_name: str, data_dir):
        """
        Initializes the FlockMTL connection using environment variables.
        """
        db_path = resolve_flockmtl_db_path(
            os.path.join(data_dir, "mmqa.duckdb")
        )

        self.flockmtl_conn = duckdb.connect(
            db_path,
            config={"allow_unsigned_extensions": "true"},
        )

        load_flockmtl_extension(self.flockmtl_conn)
        configure_flockmtl_openai_model(self.flockmtl_conn, model_name)

    def _upload_file_to_db(self, csv_path: str, table_name: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}.")

        self.flockmtl_conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto(?)",
            [csv_path],
        )

    def setup_data(self, data_dir: str):
        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "ben_piazza.csv"),
            table_name="ben_piazza",
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "ben_piazza_text_data.csv"),
            table_name="ben_piazza_text_data",
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "lizzy_caplan_text_data.csv"),
            table_name="lizzy_caplan_text_data",
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "tampa_international_airport.csv"),
            table_name="tampa_international_airport",
        )

    def get_connection(self):
        """
        Returns the FlockMTL connection.
        """
        return self.flockmtl_conn
