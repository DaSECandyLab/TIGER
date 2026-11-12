from flockmtl_phase_timing import timed_phase


import os
from pathlib import Path

import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from flockmtl_config import (
    configure_flockmtl_openai_model,
    load_flockmtl_extension,
    resolve_flockmtl_db_path,
)

MOVIE_FILES_DIR = os.path.abspath(
    Path(__file__).resolve().parents[4] / "files" / "movie" / "data"
)

class FlockMTLMovieSetup:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the FlockMTL connection using environment variables.
        """
        db_path = resolve_flockmtl_db_path(
            os.path.join(MOVIE_FILES_DIR, "movie_database.duckdb")
        )
        with timed_phase("duckdb_connect"):
            self.flockmtl_conn = duckdb.connect(
                db_path,
                config={"allow_unsigned_extensions": "true"},
            )
        with timed_phase("load_extension"):
            load_flockmtl_extension(self.flockmtl_conn)
        with timed_phase("configure_model"):
            configure_flockmtl_openai_model(self.flockmtl_conn, model_name)


    def _upload_file_to_db(self, csv_path: str, table_name: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}. Please run the download script first.")

        with timed_phase("csv_import", table=table_name, path=csv_path):
            self.flockmtl_conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}');
                """)
        

    def _resolve_data_files(self, data_dir: str = None, scale_factor: int = 2000):
        search_roots = []
        if data_dir is not None:
            data_path = Path(data_dir)
            search_roots.extend(
                [
                    data_path,
                    data_path / "data",
                    data_path / "data" / f"sf_{scale_factor}",
                    data_path / f"sf_{scale_factor}",
                ]
            )

        movie_data_path = Path(MOVIE_FILES_DIR)
        search_roots.extend([movie_data_path / f"sf_{scale_factor}", movie_data_path])

        candidates = []
        for root in search_roots:
            candidates.extend(
                [
                    (root / "Movies.csv", root / "Reviews.csv"),
                    (
                        root / f"Movies_{scale_factor}.csv",
                        root / f"Reviews_{scale_factor}.csv",
                    ),
                ]
            )

        for movies_path, reviews_path in candidates:
            if movies_path.exists() and reviews_path.exists():
                return movies_path, reviews_path

        raise FileNotFoundError(
            "Could not find movie CSV files. Checked sf directory and legacy "
            f"Movies_{scale_factor}.csv/Reviews_{scale_factor}.csv layouts."
        )

    def setup_data(self, data_dir: str = None, scale_factor: int = 2000):
        movies_path, reviews_path = self._resolve_data_files(
            data_dir=data_dir,
            scale_factor=scale_factor,
        )

        # Query files currently reference these table names.
        self._upload_file_to_db(
            csv_path=str(movies_path),
            table_name="movies_2000"
        )

        self._upload_file_to_db(
            csv_path=str(reviews_path),
            table_name="reviews_2000"
        )


    def get_connection(self):
        """
        Returns the FlockMTL connection.
        """
        return self.flockmtl_conn
