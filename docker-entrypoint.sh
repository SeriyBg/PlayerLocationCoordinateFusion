#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-idle}"

case "$COMMAND" in
    idle)
        echo "SynLoc RunPod container is ready."
        echo "Run one of:"
        echo "  /app/run_pipeline.sh download"
        echo "  /app/run_pipeline.sh convert"
        echo "  /app/run_pipeline.sh train"
        echo "  /app/run_pipeline.sh all"
        exec sleep infinity
        ;;
    pipeline|all)
        exec /app/run_pipeline.sh all
        ;;
    download|convert|train)
        exec /app/run_pipeline.sh "$COMMAND"
        ;;
    *)
        exec "$@"
        ;;
esac
