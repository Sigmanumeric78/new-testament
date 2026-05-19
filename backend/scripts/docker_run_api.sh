#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$BACKEND_DIR/.." && pwd)"
if [[ ! -d "$REPO_ROOT/backend" ]]; then
  REPO_ROOT="$BACKEND_DIR"
fi

IMAGE_NAME="alcohol-intelligence-api:local"
CONTAINER_NAME="alcohol-intelligence-api-local"
ENV_FILE="$REPO_ROOT/.env"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Image $IMAGE_NAME not found. Building first..."
  bash "$REPO_ROOT/backend/scripts/docker_build.sh"
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

RUN_ARGS=(
  -d
  --name "$CONTAINER_NAME"
  -e "NEO4J_URI=${NEO4J_URI:-bolt://host.docker.internal:7687}"
  -e "WEAVIATE_URL=${WEAVIATE_URL:-http://host.docker.internal:8080}"
  -e "WEAVIATE_GRPC_HOST=${WEAVIATE_GRPC_HOST:-host.docker.internal}"
  -e "WEAVIATE_GRPC_PORT=${WEAVIATE_GRPC_PORT:-50051}"
  -e "OLLAMA_HOST=${OLLAMA_HOST:-http://host.docker.internal:11434}"
  -e "PROJECT_ROOT=/app"
  -e "PYTHONPATH=/app/backend"
  --add-host=host.docker.internal:host-gateway
  -p 8000:8000
)

if [[ -f "$ENV_FILE" ]]; then
  RUN_ARGS+=(--env-file "$ENV_FILE")
fi

cd "$REPO_ROOT"
docker run "${RUN_ARGS[@]}" "$IMAGE_NAME"

echo "Container started: $CONTAINER_NAME"
