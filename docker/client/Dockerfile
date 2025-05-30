# Fase 1: Build della web app
FROM node:18-alpine AS build
WORKDIR /app
COPY client/package*.json ./
RUN npm install
COPY client/ .
ENV NODE_ENV=production
RUN npm run build

# Fase 2: Serve con Nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
