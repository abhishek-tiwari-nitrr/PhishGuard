FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (before pip install)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY app.py .
COPY main.py .
COPY data_config/ ./data_config/
COPY .streamlit/ ./.streamlit/
COPY production_model/ ./production_model/
COPY pyproject.toml .

# Non-root user
RUN useradd -m -u 1000 phishguard && chown -R phishguard:phishguard /app
USER phishguard

EXPOSE 8000 8501

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]