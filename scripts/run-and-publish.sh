#!/usr/bin/env bash
# Run daily backtest, commit results, push to GitHub.
# Usage: ./scripts/run-and-publish.sh
# The Synology side pulls via ./scripts/deploy.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> [1/4] Running backtest..."
python3 -m backtesting.run_backtest

echo "==> [2/4] Checking for changes..."
if git diff --quiet backtesting/results.json 2>/dev/null; then
    echo "    No changes in results.json — skipping commit."
    exit 0
fi

DATE=$(date +%Y-%m-%d)

echo "==> [3/4] Committing results..."
git add backtesting/results.json
git commit -m "data: daily backtest results ${DATE}

Co-Authored-By: Paperclip <noreply@paperclip.ing>"

echo "==> [4/4] Pushing to origin/main..."
git push origin main

echo "==> Done. Run ./scripts/deploy.sh on Synology to update the dashboard."
