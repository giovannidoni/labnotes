.PHONY: clean install dev test lint format format-check check fix run run-port unit-test test-file test-pattern endpoint-test endpoint-test-specific endpoint-test-auth help

## clean: remove Python caches, build artifacts, test reports, etc.
clean:
	@echo "Removing __pycache__, *.pyc, build/, dist/, .egg-info, pytest/mypy caches…"
	find . -type d -name "__pycache__"     -exec rm -rf {} +
	find . -type f -name "*.py[co]"        -delete
	find . -type f -name "*.log"           -delete
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .mypy_cache/

## install: install project dependencies
install:
	@echo "Installing dependencies with uv..."
	uv sync && uv pip install -e .

## lint: run linting checks
lint:
	@echo "Running linting checks..."
	uv run ruff check --fix labnotes/

## format: format code with ruff
format:
	@echo "Formatting code..."
	uv run ruff format labnotes/

## test: run tests
unit-test: install
	@echo "Running tests..."
	uv run pytest tests/ -v --disable-pytest-warnings

## test-pattern: run tests matching a pattern
test-pattern: install
	@echo "Running tests matching pattern $(PATTERN)..."
	uv run pytest tests/ -k $(PATTERN) -v

## test-coverage: run tests with coverage report
test-coverage: install
	@echo "Running tests with coverage..."
	uv run pytest tests/ --cov=backend --cov-report=html --cov-report=term-missing

## help: show this help message
help:
	@echo "Available commands:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/ /'