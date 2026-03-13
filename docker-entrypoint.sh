#!/bin/sh
set -eu

mkdir -p /app/data /app/data/jobs /app/data/job-index /app/data/u2net
chown -R appuser:appgroup /app/data

if [ "${1:-}" = "uvicorn" ]; then
  if [ -n "${ENGINE_UVICORN_WORKERS:-}" ]; then
    set -- "$@" --workers "${ENGINE_UVICORN_WORKERS}"
  fi
  if [ -n "${ENGINE_UVICORN_LIMIT_CONCURRENCY:-}" ]; then
    set -- "$@" --limit-concurrency "${ENGINE_UVICORN_LIMIT_CONCURRENCY}"
  fi
fi

exec "$@"
