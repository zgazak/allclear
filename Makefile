# AllClear Makefile

PACKAGE   := allclear
DIST_NAME := allclear
VERSION   := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

.DEFAULT_GOAL := help

.PHONY: help install install-dev test coverage version build clean \
        check-clean-tree check-version-unpublished check-tag-free \
        tag publish-test publish

###################
# Setup           #
###################

install: ## Install package
	uv pip install -e .

install-dev: ## Install with dev dependencies
	uv pip install -e ".[dev]"

###################
# Testing         #
###################

test: ## Run the full test suite
	uv run pytest allclear/tests/ -v

coverage: ## Run tests with coverage + regenerate badges (tests.svg, coverage.svg)
	uv run pytest allclear/tests/ -v \
		--junitxml=reports/junit/junit.xml \
		--cov=$(PACKAGE) \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-report=html
	uv run genbadge tests -o tests.svg
	uv run genbadge coverage -i coverage.xml -o coverage.svg
	@echo "Coverage report: htmlcov/index.html"

###################
# Release         #
###################

version: ## Print the version from pyproject.toml
	@echo $(VERSION)

build: ## Build wheel and sdist (clean dist/ first)
	rm -rf dist/
	uv build

# Blocks on uncommitted *tracked* changes only. Untracked files are allowed
# (this repo keeps local-only dirs like .claude/ and benchmark data that are
# never part of the build — the sdist excludes them).
check-clean-tree:
	@git diff --quiet && git diff --cached --quiet || { echo "ERROR: uncommitted changes to tracked files; commit or stash first"; git status --short; exit 1; }

check-version-unpublished:
	@if curl -sf https://pypi.org/pypi/$(DIST_NAME)/$(VERSION)/json > /dev/null; then \
		echo "ERROR: $(DIST_NAME) $(VERSION) is already on PyPI; bump version in pyproject.toml first"; exit 1; \
	fi
	@echo "PyPI check OK: $(DIST_NAME) $(VERSION) not yet published"

check-tag-free:
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" > /dev/null; then \
		echo "ERROR: local tag v$(VERSION) already exists"; exit 1; \
	fi
	@if git ls-remote --exit-code --tags origin "v$(VERSION)" > /dev/null 2>&1; then \
		echo "ERROR: tag v$(VERSION) already exists on origin"; exit 1; \
	fi
	@echo "Tag check OK: v$(VERSION) is free"

tag: check-clean-tree check-tag-free ## git tag v<version> from pyproject.toml and push it
	git tag v$(VERSION)
	git push origin v$(VERSION)

publish-test: check-clean-tree test build ## Test + build + upload to TestPyPI (token from ~/.pypirc)
	uvx twine upload --repository testpypi dist/*

# Publish to PyPI. Guarded: refuses on a dirty tree or an already-published
# version, always rebuilds from scratch, and runs the test suite first.
# PyPI uploads are irreversible per version - bump pyproject.toml first.
publish: check-clean-tree check-version-unpublished test build ## Guarded upload to PyPI (token from ~/.pypirc)
	uvx twine upload dist/*

###################
# Cleanup         #
###################

clean: ## Remove build artifacts, caches, reports
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ htmlcov/ .coverage coverage.xml reports/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

###################
# Help            #
###################

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
