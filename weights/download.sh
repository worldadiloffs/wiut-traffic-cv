#!/usr/bin/env bash
# Re-download the open weights (only needed if weights/ was stripped). Run once, with internet.
set -euo pipefail
cd "$(dirname "$0")"
base=https://github.com/ultralytics/assets/releases/download/v8.3.0
for m in yolo11s.pt yolo11n.pt; do
  [ -s "$m" ] || curl -L --fail -o "$m" "$base/$m"
done
sha256sum yolo11s.pt yolo11n.pt
