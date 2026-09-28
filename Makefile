# SPDX-License-Identifier: Apache-2.0
# CPU-only checks of the repository and the CPU-only reproduction of the paper's numbers.
#
#   make check        pytest on tests/core and tests/trainer (training environment) and tests/eval (evaluation
#                     environment); no GPU needed. Tests that need downloaded data or fetched graders skip.
#   make reproduce    python reproduce/reproduce.py --no-bootstrap (numpy only; prints ALL MATCH)
#
# The trainer and evaluation environments are separate (env/install.md), so the interpreters are parameters:
#   make check TRAIN_PYTHON=/path/to/train-env/bin/python EVAL_PYTHON=/path/to/eval-env/bin/python
# With neither set, every step uses `python` (the active environment).

PYTHON ?= python
TRAIN_PYTHON ?= $(PYTHON)
EVAL_PYTHON ?= $(PYTHON)
PYTEST_ARGS ?= -q

.PHONY: check check-core check-trainer check-eval reproduce

check: check-core check-trainer check-eval

check-core:
	$(TRAIN_PYTHON) -m pytest $(PYTEST_ARGS) tests/core

check-trainer:
	$(TRAIN_PYTHON) -m pytest $(PYTEST_ARGS) tests/trainer

check-eval:
	$(EVAL_PYTHON) -m pytest $(PYTEST_ARGS) tests/eval

reproduce:
	$(PYTHON) reproduce/reproduce.py --no-bootstrap
