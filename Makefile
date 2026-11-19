# Convenience targets. Paths can be overridden: make tables RUNS=runs OUT=tables
RUNS ?= runs
OUT  ?= tables
MANIFEST ?= data/manifest.csv
TRACES ?= cache/traces

.PHONY: install test dry-run tables stats efficiency all-results clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

dry-run:                     ## list every job of every suite without running it
	python scripts/run_suite.py all --manifest $(MANIFEST) --trace-root $(TRACES) --dry-run

tables:                      ## Tables 3-10 and the ablation table from finished runs
	python scripts/make_tables.py --runs $(RUNS) --out $(OUT)

stats:                       ## Table 12 and the forest plot (Fig. 7)
	python scripts/statistics.py --runs $(RUNS) --out $(OUT)

efficiency:                  ## Table 11
	python scripts/benchmark_efficiency.py --out $(OUT)/efficiency.json

all-results: tables stats efficiency

clean:
	rm -rf .pytest_cache build *.egg-info
	find . -name __pycache__ -prune -exec rm -rf {} +
