# Frontend Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy project files
COPY . .

# Build the app (for production)
# RUN npm run build

EXPOSE 5173

# Development server (for production, use nginx to serve build)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
