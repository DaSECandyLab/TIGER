import json
import os
from pathlib import Path


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def resolve_flockmtl_db_path(default_path: str) -> str:
    override = os.environ.get("FLOCKMTL_DB_FILE")
    db_path = override or default_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return db_path


def load_flockmtl_extension(conn):
    extension_path = os.environ.get(
        "FLOCK_EXT",
        str(Path(__file__).resolve().parents[2] / "flock/build/release/extension/flock/flock.duckdb_extension"),
    )
    if os.path.exists(extension_path):
        conn.execute(f"LOAD '{_sql_quote(extension_path)}';")
    else:
        raise FileNotFoundError(
            f"Modified Flock extension not found: {extension_path}. "
            "From the Tiger root run: python3 -m scripts.build_flock; "
            "or set FLOCK_EXT to the compiled modified extension."
        )


def configure_flockmtl_openai_model(conn, model_name: str):
    api_key = (
        os.environ.get("FLOCK_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "dummy"
    )
    base_url = os.environ.get("FLOCK_OPENAI_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL"
    )
    temperature = float(os.environ.get("FLOCK_TEMPERATURE", "0.0"))
    model_parameters = {"temperature": temperature}
    extra_model_parameters_env = os.environ.get("FLOCK_MODEL_PARAMETERS_JSON")
    if extra_model_parameters_env:
        extra_model_parameters = json.loads(extra_model_parameters_env)
        if not isinstance(extra_model_parameters, dict):
            raise ValueError("FLOCK_MODEL_PARAMETERS_JSON must be a JSON object")
        model_parameters.update(extra_model_parameters)
    request_rate = os.environ.get("FLOCK_VLLM_REQUEST_RATE", "inf")
    request_arrival = os.environ.get("FLOCK_VLLM_REQUEST_ARRIVAL", "poisson")
    max_in_flight_env = os.environ.get("FLOCK_VLLM_MAX_IN_FLIGHT")
    max_in_flight = max_in_flight_env if max_in_flight_env is not None else "unlimited"

    base_url_clause = ""
    if base_url:
        base_url_clause = f",\n    BASE_URL '{_sql_quote(base_url)}'"

    conn.execute(
        f"""CREATE OR REPLACE SECRET (
    TYPE OPENAI{base_url_clause},
    API_KEY '{_sql_quote(api_key)}'
);"""
    )

    models_df = conn.execute("GET MODELS;").fetchdf()
    if "model_name" in models_df.columns:
        name_column = "model_name"
    elif "model" in models_df.columns:
        name_column = "model"
    else:
        name_column = None

    if name_column is None:
        named_models = models_df.iloc[0:0]
    else:
        named_models = models_df[models_df[name_column] == model_name]
    local_models = (
        named_models[named_models["scope"] == "local"]
        if "scope" in named_models.columns
        else named_models
    )

    if not named_models.empty and local_models.empty:
        return
    if not local_models.empty:
        conn.execute(f"DELETE MODEL '{_sql_quote(model_name)}';")

    print(
        "Configuring FlockMTL model with prompt batch_size=1 "
        "(vLLM dynamic batching happens server-side); "
        f"temperature={temperature}; "
        f"request_rate={request_rate}, arrival={request_arrival}, "
        f"max_in_flight={max_in_flight}"
    )

    conn.execute(
        f"""
        CREATE MODEL(
            '{_sql_quote(model_name)}',
            '{_sql_quote(model_name)}',
            'openai',
            {{
                "tuple_format": "json",
                "batch_size": 1,
                "model_parameters": {json.dumps(model_parameters)}
            }}
        );
        """
    )
