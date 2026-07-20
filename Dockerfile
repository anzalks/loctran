FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

# Point to a running Ollama instance; defaults to localhost which is
# unreachable from inside the container. Override at runtime:
#   docker run -e OLLAMA_HOST=http://host.docker.internal:11434 ...
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000
CMD ["loctran", "serve", "--no-desktop", "--no-browser", "--host", "0.0.0.0"]
