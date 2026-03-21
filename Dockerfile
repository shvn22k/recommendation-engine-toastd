FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY src/ ./src/

# Set working directory to src
WORKDIR /app/src

# Expose port
EXPOSE 8080

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
