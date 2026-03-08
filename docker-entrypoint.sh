#!/bin/sh
set -eu

mkdir -p /app/data /app/data/jobs /app/data/job-index /app/data/u2net
chown -R appuser:appgroup /app/data

exec "$@"
