FROM python:3.12-slim

# Copy uv binary from official uv container into our image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /bin/uvx

# Set the working directory
# Set the working directory
WORKDIR /app

# Copy dependency files early for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies into .venv (default)
RUN uv sync --frozen --no-cache

# Copy application code after dependencies
COPY . .

# Optional: bytecode compilation for performance
ENV UV_COMPILE_BYTECODE=1

# Add .venv/bin to PATH so dependencies like uvicorn are found
ENV PATH="/app/.venv/bin:$PATH"

# Run FastAPI app via uv + uvicorn
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
