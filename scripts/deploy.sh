#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Pulling latest from origin/main..."
git pull origin main

echo "==> Building dashboard image..."
docker compose build dashboard

echo "==> Restarting dashboard container..."
docker compose up -d dashboard

echo "==> Done. Dashboard is running at http://localhost:8000"
docker compose ps dashboard
