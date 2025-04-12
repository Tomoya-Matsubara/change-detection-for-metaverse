FROM ghcr.io/astral-sh/uv:bookworm-slim

RUN uv python install 3.12

# Install the application dependencies.
WORKDIR /app
COPY pyproject.toml* uv.lock* README.md ./
COPY src/mcd/__init__.py ./src/mcd/__init__.py
RUN if [[ -f pyproject.toml && -f uv.lock ]]; then uv sync --frozen --no-cache; fi

# Copy the application into the container.
COPY . /app
