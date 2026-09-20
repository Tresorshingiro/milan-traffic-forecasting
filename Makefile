# Reproduce the whole study. Run `make all` on a clean clone after `make setup`.
PY := PYTHONPATH=src .venv/bin/python

# Phase-1 choices (see results/experiment_log.md, E2-E4). Override on the command line.
L ?= 144
LOSS ?= huber
LOG1P ?= --log1p
HUBER_DELTA ?= 0.3

.PHONY: all setup etl memory eda profile ladder grid final failure test test-fast clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

etl:                      ## 20.8 GB -> 357 MB memmap (~7 min)
	$(PY) scripts/build_matrix.py

memory:                   ## naive-vs-streaming memory evidence
	$(PY) scripts/measure_memory.py

eda:                      ## brief-mandated exploratory figures
	$(PY) scripts/run_eda.py

profile:                  ## forecastability profiles + pre-registered H1 prediction
	$(PY) scripts/profile_areas.py

ladder:                   ## Phase 1 is INTERACTIVE -- see README
	@echo "Run one stage at a time and record your reasoning between stages:"
	@echo "  $(PY) scripts/experiment_ladder.py E1"

grid:                     ## Phase 2 grid search, 60 runs (~3 h on 4 CPU threads); resumable
	$(PY) scripts/grid_search.py --L $(L) --loss $(LOSS) $(LOG1P) --huber-delta $(HUBER_DELTA)

final:                    ## final runs, 3 tables, 9 plots, hypothesis verdict
	$(PY) scripts/finalize.py

failure:                  ## failure diagnostics
	$(PY) scripts/failure_analysis.py

test:
	$(PY) -m pytest -v

test-fast:
	$(PY) -m pytest -v -m "not slow"

all: etl memory eda profile final failure

clean:                    ## delete regenerable artifacts; keeps raw data and ladder predictions
	rm -rf artifacts results/figures
