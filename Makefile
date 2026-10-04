.PHONY: all data contexts coverage loso benchmark collect uncertainty scale ceiling lbscale submission v5 test lint

all: contexts coverage loso benchmark collect uncertainty

data:         ; bash scripts/download_data.sh
contexts:     ; uv run python scripts/01_build_contexts.py
coverage:     ; uv run python scripts/07_coverage_audit.py
loso:         ; uv run python scripts/06_fit_loso.py
benchmark:    ; uv run python scripts/02_local_benchmark.py
collect:      ; uv run python scripts/03_collect_results.py
uncertainty:  ; uv run python scripts/04_uncertainty_analysis.py
scale:        ; uv run python scripts/08_response_scale.py
ceiling:      ; uv run python scripts/10_noise_ceiling.py
lbscale:      ; uv run python scripts/16_leaderboard_scale.py
submission:   ; uv run python scripts/05_make_submission.py --bundle data/raw/vcc --out results/submissions/submission.h5ad --model mean_transfer
# v5, the pre-registered final entry (results/prereg/final_test_protocol.md)
v5:           ; uv run python scripts/05_make_submission.py --bundle data/raw/vcc --out results/submissions/v5.h5ad --model norm_restored --extra-sources h1 cd4 --loso-table loso_grid_h1_cd4_challenge.csv
test:         ; uv run pytest -q
lint:         ; uv run ruff check . && uv run ruff format --check .
