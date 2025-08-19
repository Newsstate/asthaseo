# Chromium + system deps already baked in
FROM mcr.microsoft.com/playwright/python:v1.47.2-jammy

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Runtime env
ENV TZ=Asia/Kolkata
ENV ENABLE_RENDERED=1
# Optional tuning:
# ENV RENDER_TIMEOUT_MS=60000
# ENV RENDER_WAIT_UNTIL=networkidle

EXPOSE 8000
ENV PORT=8000

# Use 1 worker (SQLite + headless browser plays nicer)
CMD ["bash","-lc","gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 180"]
# Or uvicorn:
# CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
