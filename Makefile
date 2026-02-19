.PHONY: install test test-coverage run-pipeline run-inference run-inference-custom clean lint format setup-dirs clean-outputs dashboard pre-commit-install pre-commit-run ci-check mlflow-ui \
        docker-build docker-pipeline docker-inference docker-mlflow docker-down docker-clean

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
	$(UV_RUN) isort --check-only src/ tests/ scripts/
	$(UV_RUN) bandit -c pyproject.toml -r src/

format:
	$(UV_RUN) black src/ tests/ scripts/
	$(UV_RUN) isort src/ tests/ scripts/

# Install the pre-commit hooks into .git/hooks/ (run once after cloning)
pre-commit-install:
	$(UV_RUN) pre-commit install
	@echo "✅  Pre-commit hooks installed. They will run automatically on git commit."

# Manually run all pre-commit hooks over the entire codebase
pre-commit-run:
	$(UV_RUN) pre-commit run --all-files

# Generate the stakeholder dashboard from pipeline outputs
dashboard:
	$(UV_RUN) python scripts/generate_dashboard.py
	@echo "Dashboard -> dashboard/index.html"

# Score new sellers using the trained models (no re-training)
run-inference:
	$(UV_RUN) python scripts/run_inference.py

# Score from a custom data directory; e.g.:
# DATA_PATH=/new/data OUTPUT=outputs/q2.csv make run-inference-custom
run-inference-custom:
	$(UV_RUN) python scripts/run_inference.py --data-path $(DATA_PATH) --output $(OUTPUT)

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/

# Clean generated outputs (keeps raw data and trained models)
clean-outputs:
	rm -f outputs/*.csv outputs/*.txt outputs/*.png outputs/figures/*.png
	rm -f dashboard/index.html

setup-dirs:
	mkdir -p data/raw data/processed outputs/figures models logs docs

# Simulate the full CI pipeline locally (lint → typecheck → tests)
# Run this before pushing to catch issues early.
ci-check: lint
	$(UV_RUN) mypy src/ --ignore-missing-imports --no-strict-optional
	$(UV_RUN) bandit -c pyproject.toml -r src/
	$(UV_RUN) pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=35
	@echo "✅  All CI checks passed locally."


# Open the MLflow tracking UI (http://localhost:5000)
mlflow-ui:
	$(UV_RUN) mlflow ui --backend-store-uri mlruns/ --host 0.0.0.0 --port 5000

# ── Docker ────────────────────────────────────────────────────────────────────
IMAGE_NAME ?= olist-churn
IMAGE_TAG  ?= latest

# Build the Docker image (uses multi-stage Dockerfile)
docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Run the full training pipeline inside Docker
docker-pipeline:
	docker compose --profile pipeline run --rm pipeline

# Run inference (scoring) inside Docker
docker-inference:
	docker compose --profile inference run --rm inference

# Start the MLflow UI container (http://localhost:5000)
docker-mlflow:
	docker compose --profile mlflow up mlflow

# Stop and remove all project containers
docker-down:
	docker compose down

# Remove all project containers AND named volumes (⚠ deletes outputs/models!)
docker-clean:
	docker compose down --volumes --remove-orphans
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	@echo "⚠️  All containers, volumes and the Docker image have been removed."
