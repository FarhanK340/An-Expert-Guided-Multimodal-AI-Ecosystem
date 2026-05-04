# =============================================================================
# Dockerfile — Hugging Face Spaces Edition (Final)
# =============================================================================

# ── Stage 1: Build Frontend ──────────────────────────────────────────────────
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
ENV VITE_API_URL=/api/v1
RUN npm run build

# ── Stage 2: Final Image ─────────────────────────────────────────────────────
FROM python:3.12-slim

# HF Spaces requirement: User 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev git wget nginx procps \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn whitenoise gdown

# Copy project files
COPY backend/ .
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY --from=frontend-builder /frontend/dist /app/frontend_dist

# ── Download Model Weights ──────────────────────────────────────────────────
# We do this as root during the build so the weights are baked into the image.
RUN mkdir -p /app/models/checkpoints/experts /app/models/report_generator/final \
    && chmod -R 777 /app/models \
    && bash /app/scripts/download_models.sh \
    && chown -R user:user /app/models

# Environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV DEBUG=False
ENV HF_SPACE=True
ENV DATABASE_URL=sqlite:///db.sqlite3

# Configure Nginx for HF (Port 7860)
RUN echo 'server { \
    listen 7860; \
    client_max_body_size 1G; \
    location / { \
        root /app/frontend_dist; \
        try_files $uri $uri/ /index.html; \
    } \
    location /api/ { \
        proxy_pass http://localhost:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
    } \
    location /admin/ { \
        proxy_pass http://localhost:8000; \
        proxy_set_header Host $host; \
    } \
    location /static/ { \
        alias /app/staticfiles/; \
    } \
    location /media/ { \
        alias /app/media/; \
    } \
}' > /etc/nginx/sites-available/default

# Fix permissions for User 1000
RUN mkdir -p /app/media /app/staticfiles /app/logs \
    && chown -R user:user /app /etc/nginx /var/log/nginx /var/lib/nginx /run

USER user

# Entrypoint script
RUN echo '#!/bin/bash\n\
python manage.py migrate\n\
python manage.py collectstatic --noinput\n\
# Start Gunicorn\n\
gunicorn medical_ai_backend.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 600 &\n\
# Start Nginx\n\
/usr/sbin/nginx -g "daemon off;"' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

EXPOSE 7860

CMD ["/app/entrypoint.sh"]
