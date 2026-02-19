.PHONY: install test test-coverage run-pipeline clean lint format setup-dirs clean-outputs

# Use uv run so that dependencies from uv's virtualenv are resolved automatically
UV_RUN = PYTHONPATH=. uv run

install:
	uv sync

test:
	$(UV_RUN) pytest tests/ -v --tb=short

test-coverage:
	$(UV_RUN) pytest tests/ --cov=src --cov-report=html

# Run the full pipeline via the clean entry point
run-pipeline:
	$(UV_RUN) python scripts/run_pipeline.py

# Validation smoke-test only
run-validation:
	$(UV_RUN) python -c "from src.validation.schemas import validate_seller_data; print('Validation OK')"

lint:
	$(UV_RUN) flake8 src/ tests/ scripts/
	$(UV_RUN) black --check src/ tests/ scripts/

format:
	$(UV_RUN) black src/ tests/ scripts/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/

# Clean generated outputs (keeps raw data and trained models)
clean-outputs:
	rm -f output/*.csv output/*.txt
	rm -f reports/*.md
	rm -f reports/figures/*.png

setup-dirs:
	mkdir -p data/raw data/processed output models reports/figures logs scripts/archive
