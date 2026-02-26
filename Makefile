# AllClear Makefile

.PHONY: help install install-dev test coverage badges build clean publish publish-test

PYTHON := uv run python

help:
	@echo "Available targets:"
	@echo "  install       - Install package"
	@echo "  install-dev   - Install with dev dependencies"
	@echo "  test          - Run tests"
	@echo "  coverage      - Run tests with coverage + generate badges"
	@echo "  build         - Build wheel and sdist"
	@echo "  clean         - Remove build artifacts"
	@echo "  publish-test  - Upload to TestPyPI"
	@echo "  publish       - Upload to PyPI"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

test:
	uv run pytest allclear/tests/ -v

coverage:
	uv run pytest allclear/tests/ -v \
		--junitxml=reports/junit/junit.xml \
		--cov=allclear \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-report=html
	uv run genbadge tests -o tests.svg
	uv run genbadge coverage -i coverage.xml -o coverage.svg
	@echo "Coverage report: htmlcov/index.html"

build: clean
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ htmlcov/ .coverage coverage.xml reports/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

publish-test: build
	uv run twine upload --repository testpypi dist/*

publish: build
	uv run twine upload dist/*
