.PHONY: all check test-unit test-integration test-e2e test-smoke test-all \
        lint-check lint-fix type-check link-check format-pr \
        sync lock build clean help

.DEFAULT_GOAL := check

# Default target runs fast local verification
all: check

check: test-unit lint-check type-check

# --- Testing ---
test-unit:
	tox -e test-unit

test-integration:
	tox -e test-integration

test-e2e:
	tox -e test-e2e

test-smoke:
	tox -e tests -- -m smoke

test-all:
	tox -e tests

# --- Quality & Formatting ---
lint-check:
	tox -e lint-check

lint-fix:
	tox -e lint-fix

type-check:
	tox -e type-check

link-check:
	tox -e link-check

format-pr:
	bash ./scripts/format_pr.sh

# --- Dependencies & Environment ---
sync:
	uv sync --all-groups

lock:
	tox -e lock

# --- Build & Cleanup ---
build:
	tox -e build

# Cross-platform cleanup using Python (works on Windows, macOS, and Linux)
clean:
	python -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in ('.tox', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.hypothesis', 'build', 'dist')]"
	python -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('*.egg-info')]"

help:
	@echo "GuideLLM Development Targets:"
	@echo "  make check            - Run unit tests, lint check, and type check"
	@echo "  make test-unit        - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-e2e         - Run end-to-end tests"
	@echo "  make test-smoke       - Run quick smoke tests"
	@echo "  make lint-check       - Check formatting and style"
	@echo "  make lint-fix         - Auto-fix style and markdown issues"
	@echo "  make type-check       - Run mypy type checking"
	@echo "  make link-check       - Verify documentation links"
	@echo "  make format-pr        - Prepare code and docs for PR submission"
	@echo "  make sync             - Sync local virtualenv via uv"
	@echo "  make lock             - Update uv dependency lockfile"
	@echo "  make build            - Build distribution package"
	@echo "  make clean            - Remove all build and test cache files"
