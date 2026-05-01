# -*- mode: makefile -*-

# Short aliases
alias c := check
alias fc := fmt-check
alias l := lint
alias t := test
alias tq := test-quiet
alias f := fmt

# Shared globs for Python sources
PY_SRCS := "examples/*.py src/qemcmc/*.py tests/*.py"

# Show available recipes with their docs
default:
	@just --list

# Remove Python bytecode caches
pyclean:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +

# Format Python code with ruff (in-place)
fmt:
	ruff format {{PY_SRCS}}

# Check formatting without changing files
fmt-check:
	ruff format --check {{PY_SRCS}}

# Lint with ruff using project configuration
lint:
	ruff check {{PY_SRCS}}

# Apply safe autofixes from ruff
fix:
	ruff check --fix {{PY_SRCS}}

# Apply all autofixes, including unsafe ones
fix-unsafe:
	ruff check --fix --unsafe-fixes {{PY_SRCS}}

# Type-check with mypy (src, examples, tests)
type:
	mypy src/qemcmc examples tests

# Run unit tests (quiet)
test-quiet:
	pytest -q

# Run unit tests, not quiet
test:
	pytest

# Run tests with coverage report (terminal summary)
cov:
	pytest --cov=qemcmc --cov-report=term-missing

# Full local verification: format-check, lint, type, tests
check:
	just fmt-check
	just lint
	just type
	just test

# CI-friendly entry: clean caches, then run full check
ci:
	just pyclean
	just check

# Convert example script to notebook
nb:
	jupytext --to ipynb examples/workflows/mis_optimize.py
