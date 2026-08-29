FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy the whole project (uv needs src/hello_agent to build the package itself)
COPY . .

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "hello_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
