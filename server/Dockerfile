FROM node:18-slim

RUN apt-get update && apt-get install -y \
      wget gnupg ca-certificates fonts-liberation \
      libatk1.0-0 libatk-bridge2.0-0 libcairo2 libpango-1.0-0 \
      libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
      libasound2 libnss3 libxss1 xvfb \
      --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package*.json ./
RUN npm install --omit=dev --verbose
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
