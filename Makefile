##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.DEFAULT_GOAL := check

PYTHON ?= python3
SOURCE_DIRS := src tests examples
REPORT_DIR ?= reports
JUNIT_SUITE_NAME ?= oracle-db-agent-sdk

.PHONY: check format format_check lint type_check test coverage compile_check security_check build install_dev clean help

check: format_check lint type_check coverage compile_check security_check ## Run all validation checks

format: ## Format source, tests, and examples with Ruff
	$(PYTHON) -m ruff format $(SOURCE_DIRS)

format_check: ## Check formatting without changing files
	$(PYTHON) -m ruff format --check $(SOURCE_DIRS)

lint: ## Check code with Ruff
	$(PYTHON) -m ruff check $(SOURCE_DIRS)

type_check: ## Run strict type checks on source
	$(PYTHON) -m mypy src

test: ## Run unit tests
	$(PYTHON) -m pytest -q

coverage: ## Run tests with JUnit XML, coverage XML, and browsable HTML coverage reports
	@mkdir -p "$(REPORT_DIR)"
	COVERAGE_FILE="$(REPORT_DIR)/.coverage" $(PYTHON) -m pytest -v \
		--junitxml="$(REPORT_DIR)/junit.xml" \
		-o junit_suite_name="$(JUNIT_SUITE_NAME)" \
		--cov=oracle_db_agent_sdk --cov-report=term-missing \
		--cov-report=xml:"$(REPORT_DIR)/coverage.xml" \
		--cov-report=html:"$(REPORT_DIR)/coverage-html"

compile_check: ## Compile Python files to check syntax
	$(PYTHON) -m compileall -q src examples

security_check: ## Scan source with Bandit
	$(PYTHON) -m bandit -q -r src

build: ## Build wheel and source distributions
	rm -rf build dist
	$(PYTHON) -m build

install_dev: ## Install this checkout and development dependencies using the selected Python
	$(PYTHON) -m pip install -e '.[dev]'

clean: ## Remove generated coverage reports and build artifacts
	rm -rf .coverage htmlcov build dist "$(REPORT_DIR)"

help: ## List available commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_][a-zA-Z_0-9]*:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo 'Use PYTHON=/path/to/venv/bin/python to select an environment.'
