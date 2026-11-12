UV ?= uv
BOOTSTRAP_PYTHON ?= python3
PYTHON ?= .venv/bin/python
WORKLOAD_PYTHON ?= SemBench/.venvs/sembench/bin/python
JOBS ?= 4
MODEL ?= Qwen2-7B-Instruct
BASE_URL ?= http://127.0.0.1:8000/v1

.PHONY: install install-workload sources check-sources build-flock run-movies

install:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PYTHON) -r requirements.txt

install-workload:
	$(UV) venv --python 3.12 SemBench/.venvs/sembench
	$(UV) pip install --python $(WORKLOAD_PYTHON) -r requirements-workload.txt

sources:
	$(BOOTSTRAP_PYTHON) -m scripts.bootstrap_sources

check-sources:
	$(BOOTSTRAP_PYTHON) -m scripts.bootstrap_sources --check

build-flock:
	$(BOOTSTRAP_PYTHON) -m scripts.build_flock --jobs $(JOBS)

run-movies:
	FLOCK_OPENAI_BASE_URL="$(BASE_URL)" FLOCKMTL_DB_FILE="$(CURDIR)/outputs/movies.duckdb" $(WORKLOAD_PYTHON) SemBench/src/run.py --systems flockmtl --use-cases movie --scale-factor 40 --queries 1 9 10 --model "$(MODEL)" --cacheblend-mode off --skip-setup --no-isolation
