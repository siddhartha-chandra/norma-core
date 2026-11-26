#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p ./station_data

echo "Starting NormaCore Station..."
echo "  NormFS TCP : 0.0.0.0:8888"
echo "  Web Viewer : http://localhost:8889"
exec station \
  --config ./station.yaml \
  --normfs-base-folder ./station_data \
  --tcp 0.0.0.0:8888 \
  --web 0.0.0.0:8889
