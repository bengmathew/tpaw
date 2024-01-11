# syntax=docker/dockerfile:1.2

# src: https://dev.to/cloudx/how-to-use-puppeteer-inside-a-docker-container-568c

FROM node:20-slim
USER root
WORKDIR /root


# We don't need the standalone Chromium
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD true

# Install Google Chrome Stable and fonts
# Note: this installs the necessary libs to make the browser work with Puppeteer.
RUN apt-get update && apt-get install gnupg wget -y && \
    wget --quiet --output-document=- https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/google-archive.gpg && \
    sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' && \
    apt-get update && \
    apt-get install google-chrome-stable -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*


COPY . .
RUN --mount=type=secret,id=TPAW_NPM_AUTH_TOKEN,required \
    TPAW_NPM_AUTH_TOKEN=$(cat /run/secrets/TPAW_NPM_AUTH_TOKEN) \
    npm ci
RUN npm run build

ENV PORT 8080
EXPOSE 8080
ENTRYPOINT node . serve
