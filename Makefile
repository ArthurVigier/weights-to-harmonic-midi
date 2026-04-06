.PHONY: test lint typecheck clean all

all: lint typecheck test

test:
	pytest tests/ --cov=src --cov-report=term-missing

lint:
	ruff check .

typecheck:
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -f .coverage
