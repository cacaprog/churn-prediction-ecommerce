# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder
#   Installs uv and resolves/installs all project dependencies into an
#   isolated virtual environment under /app/.venv.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Install system build dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager / lock-file resolver)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first – Docker layer cache stays valid
# as long as these files don't change.
COPY pyproject.toml uv.lock ./

# Sync production dependencies only (no dev/test extras) into .venv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv sync --frozen --no-dev --no-install-project

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime
#   Copies only the pre-built .venv and project source; no compiler toolchain.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the resolved virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Make the venv the active Python environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy project source code
COPY config/   ./config/
COPY src/      ./src/
COPY scripts/  ./scripts/

# Create runtime directories that the pipeline writes to
# (these are usually bind-mounted in docker-compose, created here as fallback)
RUN mkdir -p data/raw data/processed outputs/figures models logs mlruns \
    && chown -R appuser:appgroup /app

USER appuser

# Default command – run the full training pipeline.
# Override with `docker run … python scripts/run_inference.py` etc.
CMD ["python", "scripts/run_pipeline.py"]
