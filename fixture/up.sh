#!/usr/bin/env bash
# Local Qdrant 1.19.1 under podman, seeded to match schema_contract.md.
#
# Plain `podman run` rather than podman-compose: it is one container with two
# ports and a volume, and this way there is no second tool to install.
#
#   bash fixture/up.sh            start + seed if empty
#   bash fixture/up.sh --recreate drop the collection and reseed
#   bash fixture/up.sh --down     stop and remove the container (keeps data)
#   bash fixture/up.sh --destroy  remove the container AND the data volume
set -euo pipefail

NAME=workshop-qdrant
IMAGE=docker.io/qdrant/qdrant:v1.19.1   # matches the cluster's server version
VOLUME=workshop-qdrant-storage
URL=http://localhost:6333
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  --down)    podman rm -f "$NAME" >/dev/null 2>&1 || true; echo "stopped $NAME"; exit 0 ;;
  --destroy) podman rm -f "$NAME" >/dev/null 2>&1 || true
             # `podman rm -f` can return before the container is really gone, and
             # then the volume is still "in use". Wait it out rather than
             # swallowing the error and leaving stale data behind.
             for _ in $(seq 1 30); do
               podman container exists "$NAME" || break
               sleep 1
             done
             if podman volume rm "$VOLUME" >/dev/null 2>&1; then
               echo "removed $NAME and volume $VOLUME"
             elif podman volume exists "$VOLUME" 2>/dev/null; then
               echo "removed $NAME, but volume $VOLUME could not be removed:" >&2
               podman volume rm "$VOLUME" 2>&1 | sed 's/^/  /' >&2
               exit 1
             else
               echo "removed $NAME and volume $VOLUME"
             fi
             exit 0 ;;
esac

command -v podman >/dev/null || {
  echo "podman not found. Install it: https://podman.io/docs/installation" >&2; exit 1; }

# On macOS and Windows podman runs containers inside a VM that has to be up
# first, and it does not always survive a laptop sleep.
if podman machine list --format '{{.Name}}' 2>/dev/null | grep -q .; then
  podman machine start 2>/dev/null || true
fi

if ! podman info >/dev/null 2>&1; then
  echo "podman is installed but not reachable." >&2
  echo "  try: podman machine stop && podman machine start" >&2
  exit 1
fi

if podman container exists "$NAME"; then
  podman start "$NAME" >/dev/null
else
  # --restart=always so it comes back by itself after the VM restarts.
  podman run -d --name "$NAME" --restart=always \
    -p 6333:6333 -p 6334:6334 -v "$VOLUME":/qdrant/storage "$IMAGE" >/dev/null
fi

printf "waiting for qdrant"
for _ in $(seq 1 60); do
  if curl -sf "$URL/readyz" >/dev/null 2>&1 || curl -sf "$URL/" >/dev/null 2>&1; then
    echo " -- up at $URL"; break
  fi
  printf "."; sleep 1
done
curl -sf "$URL/" >/dev/null || { echo; echo "qdrant did not come up; podman logs $NAME" >&2; exit 1; }

PY=python3
[ -x "$HERE/../.venv/bin/python" ] && PY="$HERE/../.venv/bin/python"
exec "$PY" "$HERE/seed_fixture.py" --url "$URL" ${1:+"$1"}
