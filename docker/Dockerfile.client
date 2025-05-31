# Dockerfile.client

# 1. Build della web app
FROM node:18-alpine AS build
WORKDIR /app

# Copia solo i file necessari per install
COPY client/package*.json ./
RUN npm install

# Copia tutto il resto e compila
COPY client/ .
ENV NODE_ENV=production
RUN npm run build

# 2. Serve con Nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
