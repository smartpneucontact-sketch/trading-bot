FROM python:3.11-slim

WORKDIR /app

# Install system deps for lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY pipeline.py .
COPY dashboard.py .
COPY model/ model/

# Create persistent data directory (mount a Railway volume here)
RUN mkdir -p /app/data/state /app/data/logs
ENV DATA_DIR=/app/data

# Expose web port
EXPOSE 8080

# Run dashboard (includes scheduler + web UI)
CMD ["python", "dashboard.py"]
