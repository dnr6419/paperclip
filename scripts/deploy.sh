#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "==> Local changes detected, stashing..."
    git stash save "pre-deploy-$(date +%Y%m%d-%H%M%S)"
fi

echo "==> Pulling latest from origin/main..."
git pull origin main

echo "==> Building dashboard image..."
docker-compose build dashboard

echo "==> Restarting dashboard container..."
docker-compose up -d dashboard

echo "==> Done. Dashboard is running at http://localhost:8000"
docker-compose ps dashboard
