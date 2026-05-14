FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["loctran", "serve", "--no-browser", "--host", "0.0.0.0"]
