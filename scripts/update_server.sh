#!/usr/bin/env bash
# Production update entrypoint. Run on the server from the cloned repository.
# The server keeps .env and Docker volumes locally; git only updates code.
set -euo pipefail

cd "$(dirname "$0")/.."

branch="${DEPLOY_BRANCH:-master}"
previous_commit="$(git rev-parse HEAD 2>/dev/null || true)"

rollback() {
  if [ -z "$previous_commit" ]; then
    echo "Rollback skipped: previous commit is unknown." >&2
    return 1
  fi
  echo "Rolling back to ${previous_commit}..." >&2
  git checkout -B "$branch" "$previous_commit"
  git reset --hard "$previous_commit"
  docker compose up -d --build
}

if [ "$(id -u)" -eq 0 ]; then
  echo "WARNING: deploy is running as root; use a dedicated deploy user." >&2
fi

if [ -f .env ]; then
  mode="$(stat -c '%a' .env 2>/dev/null || stat -f '%Lp' .env 2>/dev/null || echo '')"
  if [ -n "$mode" ]; then
    last_two="${mode: -2}"
    if [ "$last_two" != "00" ]; then
      echo "WARNING: .env is readable by group/others; run: chmod 600 .env" >&2
    fi
  fi
fi

echo "Fetching origin/${branch}..."
git fetch origin "$branch"
target_commit="$(git rev-parse "origin/${branch}")"

echo "Switching working tree to ${target_commit}..."
git checkout -B "$branch" "$target_commit"
git reset --hard "$target_commit"

echo "Rebuilding and restarting Docker service..."
if ! docker compose up -d --build; then
  echo "Deploy build/start failed." >&2
  rollback || true
  exit 1
fi

echo "Running container smoke check..."
sleep 5
smoke_ok=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  if docker compose exec -T rqm-nav python -m app.smoke; then
    smoke_ok=1
    break
  fi
  sleep 3
done

if [ "$smoke_ok" != "1" ]; then
  echo "Smoke check failed after retries." >&2
  if [ "$previous_commit" != "$target_commit" ]; then
    rollback || true
  fi
  exit 1
fi

echo "Cleaning unused Docker images..."
docker image prune -f >/dev/null 2>&1 || true

echo "Current containers:"
docker compose ps
