#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")/docker/sandbox"
IMAGE_NAME="${SANDBOX_IMAGE:-kronos-sandbox:latest}"

echo "Building sandbox Docker image..."
docker build -t "$IMAGE_NAME" "$DOCKER_DIR"
echo "Done. Image: $IMAGE_NAME"
