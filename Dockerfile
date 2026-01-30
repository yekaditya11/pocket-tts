FROM python:3.12-slim

WORKDIR /app

# Intel CPU Optimizations (Critical for c7i/c8i)
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV KMP_BLOCKTIME=1


# Install uv
RUN pip install uv

# Copy project files
COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./README.md .
COPY ./.python-version .
COPY ./pocket_tts ./pocket_tts
COPY ./sample_audio ./sample_audio

# Create virtual environment and install dependencies
# We use --system since we are in a container, or we can rely on uv run
# The original dockerfile used `uv run`, we can stick to that.
# However, for caching, let's install dependencies first.
RUN uv sync --frozen --no-dev

# Run the help command to check and cache models/updates if needed
RUN uv run --no-dev pocket-tts serve --help

CMD ["uv", "run", "--no-dev", "pocket-tts", "serve", "--host", "0.0.0.0", "--port", "8000"]