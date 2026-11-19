#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

case "$MODE" in
    download)
        python /app/download_synloc.py
        ;;
    convert)
        python /app/convert_coco_to_yolo.py
        ;;
    train)
        python /app/train_yolo26.py
        ;;
    all)
        python /app/download_synloc.py
        python /app/convert_coco_to_yolo.py
        python /app/train_yolo26.py
        ;;
    *)
        echo "Usage: $0 {download|convert|train|all}" >&2
        exit 2
        ;;
esac
