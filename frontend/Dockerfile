# =============================================================================
# Frontend Dockerfile — Multi-Stage Production Build
#
# Stage 1 (builder): Node 18 — installs deps, builds Vite bundle → dist/
# Stage 2 (production): Nginx Alpine — serves the static dist/ files
#
# VITE_API_URL is baked in at build time via docker-compose build arg.
# On the production server this should be your actual domain, e.g.:
#   https://medai-fyp.duckdns.org/api/v1
# =============================================================================

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# Cache npm install layer separately so it only re-runs when package files change
COPY package*.json ./
RUN npm ci --legacy-peer-deps

# Copy all source files
COPY . .

# Inject backend API URL at build time (Vite replaces import.meta.env.VITE_*)
ARG VITE_API_URL=http://localhost/api/v1
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ── Stage 2: Production Serve ────────────────────────────────────────────────
FROM nginx:alpine AS production

# Copy the compiled bundle from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy SPA routing config (try_files → index.html for React Router)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
