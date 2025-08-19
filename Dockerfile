# Base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

#rendered vs static comparision
RUN pip install playwright
RUN playwright install --with-deps chromium

# System deps (curl for healthcheck, build tools for lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \ 
    libxml2-dev libxslt1-dev \ 
    curl \ 
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \ 
    && pip install --no-cache-dir gunicorn

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Default envs
ENV TZ=Asia/Kolkata

# Start the app (use $PORT if provided by platform)
ENV PORT=8000
CMD exec gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 120
