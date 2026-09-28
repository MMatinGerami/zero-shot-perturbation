# Reproducible environment for this repository. Data and results live on the host:
#   docker build -t zsp .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" zsp make test
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN apt-get update && apt-get install -y --no-install-recommends make git curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/opt/venv/bin:$PATH"
CMD ["make", "test"]
