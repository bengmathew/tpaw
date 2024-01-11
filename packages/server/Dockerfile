# syntax=docker/dockerfile:1.2

# Primsa needs some libs not there in -slim images.
FROM node:20
USER root
WORKDIR /root

COPY . .
RUN --mount=type=secret,id=TPAW_NPM_AUTH_TOKEN,required \
    TPAW_NPM_AUTH_TOKEN=$(cat /run/secrets/TPAW_NPM_AUTH_TOKEN) \
    npm ci
RUN npm run build

ENV PORT 8080
EXPOSE 8080
ENTRYPOINT node . serve
