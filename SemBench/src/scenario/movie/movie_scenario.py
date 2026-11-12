import os
from typing import List
import glob


MOVIE_FILES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "files", "movie")
)


def _resolve_movie_files_dir() -> str:
    override_root = os.environ.get("SEMBENCH_FILES_OVERRIDE_ROOT")
    if not override_root:
        return MOVIE_FILES_DIR
    return os.path.abspath(os.path.join(override_root, "movie"))


class MovieScenario:
    """
    Movie scenario handler.

    This class:
     * downloads and prepares data
     * retrieves queries
    """

    def __init__(self, scale_factor: int = 2000):
        self.data_dir = None
        self.scale_factor = scale_factor

    def setup_scenario(self, systems: List[str]) -> None:
        if systems != ["flockmtl"]:
            raise ValueError("Movies supports flockmtl only")
        # Download and prepare data if not already done
        from scenario.movie.preparation.generate_data import (
            download_from_google_drive,
            load_data,
            find_pattern_movie,
            get_negative_movie,
            get_top_movies_fast,
            sample_reviews,
            generate_movies_table,
        )
        from pathlib import Path

        # Check if data already exists
        data_folder = Path(MOVIE_FILES_DIR) / "data" / f"sf_{self.scale_factor}"
        movies_file = data_folder / "Movies.csv"
        reviews_file = data_folder / "Reviews.csv"

        if movies_file.exists() and reviews_file.exists():
            print(f"Data already exists at {data_folder}, skipping generation.")
            self.data_dir = str(data_folder)
        else:
            # Download source data
            data_path = download_from_google_drive()

            # Load data
            movies_df, reviews_df = load_data(data_path)

            # Find special movies
            pattern_movie, pattern_pattern = find_pattern_movie(reviews_df)
            negative_movie = get_negative_movie()

            # Get top movies
            top_movies = get_top_movies_fast(reviews_df)

            # Sample reviews
            selected_reviews = sample_reviews(
                reviews_df,
                pattern_movie,
                pattern_pattern,
                negative_movie,
                top_movies,
                self.scale_factor,
            )

            # Generate movies table
            selected_movies = generate_movies_table(movies_df, selected_reviews)

            # Save to data directory
            os.makedirs(data_folder, exist_ok=True)

            # Clean reviewText to prevent CSV formatting issues
            selected_reviews = selected_reviews.copy()
            selected_reviews["reviewText"] = (
                selected_reviews["reviewText"]
                .str.replace("\n", " ", regex=False)
                .str.replace("\r", " ", regex=False)
            )

            selected_movies.to_csv(movies_file, index=False)
            selected_reviews.to_csv(reviews_file, index=False)

            print(f"Data saved to {data_folder}")
            self.data_dir = str(data_folder)

        # Load data into the specified systems
        for system in systems:
            if system == "flockmtl":
                from scenario.movie.setup.flockmtl import FlockMTLMovieSetup

                setup = FlockMTLMovieSetup()
                setup.setup_data(
                    data_dir=Path(MOVIE_FILES_DIR) / "data",
                    scale_factor=self.scale_factor,
                )
            else:
                raise ValueError(f"Unsupported system: {system}")

    def get_query_text(self, query_id: int, system_name: str) -> str:
        """
        Get the SQL query text for a given query ID and system name.

        Args:
            query_id: ID of the query
            system_name: Name of the system
        Returns:
            SQL query text as a string
        """
        system_query_dirs = [
            os.path.abspath(
                os.path.join(_resolve_movie_files_dir(), "query", system_name)
            ),
            os.path.abspath(os.path.join(MOVIE_FILES_DIR, "query", system_name)),
        ]
        matching_files = []
        for system_query_dir in dict.fromkeys(system_query_dirs):
            matching_files = glob.glob(
                os.path.join(system_query_dir, f"Q{query_id}.*")
            )
            if matching_files:
                break

        if not matching_files:
            raise FileNotFoundError(
                f"No query implementation found for query {query_id} and system '{system_name}'"
            )
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple query files found for query {query_id} and system '{system_name}': {matching_files}"
            )

        with open(matching_files[0], "r") as f:
            return f.read()

    def get_data_dir(self) -> str:
        if self.data_dir:
            return self.data_dir
        else:
            return os.path.abspath(
                os.path.join(MOVIE_FILES_DIR, "data", f"sf_{self.scale_factor}")
            )

    def discover_available_queries(self, system_name: str = None) -> List[int]:
        """
        Discover available queries for the movie scenario.

        Returns:
            List of query IDs.
        """
        if system_name is None:
            # Discover the retained FlockMTL implementations.
            system_query_dirs = [
                os.path.abspath(
                    os.path.join(_resolve_movie_files_dir(), "query", "flockmtl")
                ),
                os.path.abspath(os.path.join(MOVIE_FILES_DIR, "query", "flockmtl")),
            ]
        else:
            system_query_dirs = [
                os.path.abspath(
                    os.path.join(_resolve_movie_files_dir(), "query", system_name)
                ),
                os.path.abspath(
                    os.path.join(MOVIE_FILES_DIR, "query", system_name)
                ),
            ]

        query_ids = []
        for system_query_dir in dict.fromkeys(system_query_dirs):
            if not os.path.exists(system_query_dir):
                continue

            query_files = glob.glob(os.path.join(system_query_dir, "Q*.*"))
            for f in query_files:
                try:
                    # Extract number from filename like Q1.py or Q1.sql
                    filename = os.path.basename(f)
                    query_id = int(filename[1:].split(".")[0])
                    query_ids.append(query_id)
                except ValueError:
                    continue
        return sorted(set(query_ids))
