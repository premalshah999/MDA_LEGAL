# Use an official Python runtime as a parent image
FROM python:3.10-slim as base

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the backend code into the container
COPY backend /app/backend

# Create a working directory for data
RUN mkdir -p /app/data

# Expose port for FastAPI
EXPOSE 8000

# Set environment variables expected by the app (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1

# Start the FastAPI server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
