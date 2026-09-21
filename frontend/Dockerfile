# Frontend: React + Vite dev server (proxies /predict etc. to the backend)
FROM node:20-slim

WORKDIR /app

# Install deps first for better layer caching.
COPY package.json package-lock.json ./
RUN npm install

# App source.
COPY . .

EXPOSE 5173

# Bind to all interfaces so the port is reachable from the host.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
