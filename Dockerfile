FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the rest of the project
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "hello_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
