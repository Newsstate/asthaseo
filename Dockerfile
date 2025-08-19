# ✅ Browsers + OS deps preinstalled here — no sudo needed
FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

WORKDIR /app

# Install Python deps (pin playwright to the same version as the image)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

ENV TZ=Asia/Kolkata
ENV ENABLE_RENDERED=1
EXPOSE 8000
ENV PORT=8000

# 1 worker plays nicer with SQLite + headless browser
CMD ["bash","-lc","gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 180"]
