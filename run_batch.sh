#!/bin/bash
set -e

AMAT_DIR="videos/training"
RUNNER="scripts/runner.py"

echo "=== Batch processing amateur videos from: $AMAT_DIR ==="

shopt -s nullglob
for vid in "$AMAT_DIR"/*.mp4 "$AMAT_DIR"/*.mov; do
    echo "=== Running runner.py on $vid ==="
    python3 "$RUNNER" --amateur_video "$vid"
done
