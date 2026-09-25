.PHONY: all data contexts coverage loso benchmark collect uncertainty submission test lint

all: contexts coverage loso benchmark collect uncertainty

data:         ; bash scripts/download_data.sh
contexts:     ; uv run python scripts/01_build_contexts.py
coverage:     ; uv run python scripts/07_coverage_audit.py
loso:         ; uv run python scripts/06_fit_loso.py
benchmark:    ; uv run python scripts/02_local_benchmark.py
collect:      ; uv run python scripts/03_collect_results.py
uncertainty:  ; uv run python scripts/04_uncertainty_analysis.py
submission:   ; uv run python scripts/05_make_submission.py --bundle data/raw/vcc --out results/submissions/submission.h5ad --model mean_transfer
test:         ; uv run pytest -q
lint:         ; uv run ruff check . && uv run ruff format --check .
