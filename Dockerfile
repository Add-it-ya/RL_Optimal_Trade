# CPU image for running the test suite, the experiment pipeline and the dashboard.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# Install dependencies first (better layer caching): copy only what the install needs.
COPY pyproject.toml README.md ./
COPY rl_execution ./rl_execution
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -e ".[rl,dashboard,dev]"

# Copy the remainder of the project (scripts, tests, config, dashboard).
COPY . .

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

CMD ["pytest", "-q"]
