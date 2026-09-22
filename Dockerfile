# ============================================================
# Dockerfile — Credit Default Classifier
# Base image: Python 3.11.5 slim
# Service: FastAPI inference API
# ============================================================

FROM python:3.11.5-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system build tools (required for LightGBM, XGBoost, SHAP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverage Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project source code into container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
