.PHONY: all data contexts benchmark collect uncertainty test lint

all: contexts benchmark collect uncertainty

data:         ; bash scripts/download_data.sh
contexts:     ; uv run python scripts/01_build_contexts.py
benchmark:    ; uv run python scripts/02_local_benchmark.py
collect:      ; uv run python scripts/03_collect_results.py
uncertainty:  ; uv run python scripts/04_uncertainty_analysis.py
test:         ; uv run pytest -q
lint:         ; uv run ruff check . && uv run ruff format --check .
