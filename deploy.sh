#!/usr/bin/env bash
#
# deploy.sh - Build, push, and deploy the Edinburgh worker to Colossus
# Usage: ./deploy.sh
#
# Same shape as Dublin's deploy: build here, push to Docker Hub, then over SSH
# pull and restart the container. Two things the worker needs on top:
# host networking, because PAIR on Colossus takes requests only from its own
# loopback, and a data folder on the host for the worker's state and logs.
set -euo pipefail

IMAGE="thebigduck/edinburgh"
COLOSSUS="heinrich@192.168.0.98"
CONTAINER="edinburgh-worker"
# Both live in heinrich's home so deploys never need sudo on Colossus.
# Provision/update the env with: scp .env heinrich@192.168.0.98:edinburgh/.env
# (set OLLAMA_BASE_URL in it to PAIR's proxy on Colossus, http://127.0.0.1:11434)
ENV_FILE="/home/heinrich/edinburgh/.env"
DATA_DIR="/home/heinrich/edinburgh/data"
GIT_SHA=$(git rev-parse --short HEAD)

# Refuse to deploy uncommitted work - the image tag must match GitHub
if [[ -n $(git status --porcelain) ]]; then
    echo "ERROR: Uncommitted changes. Commit to GitHub first."
    exit 1
fi

echo "==> Building $IMAGE:$GIT_SHA"
docker build -t "$IMAGE:$GIT_SHA" -t "$IMAGE:latest" .

echo "==> Pushing to Docker Hub"
docker push "$IMAGE:$GIT_SHA"
docker push "$IMAGE:latest"

echo "==> Deploying to Colossus"
ssh "$COLOSSUS" bash -s <<EOF
set -euo pipefail
# Fail before touching the running container: a bad pull or missing env
# file must not take down the current deployment.
if [[ ! -f $ENV_FILE ]]; then
    echo "ERROR: $ENV_FILE not found on Colossus."
    echo "Provision it with: scp .env $COLOSSUS:edinburgh/.env"
    exit 1
fi
mkdir -p $DATA_DIR
docker pull $IMAGE:latest
docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true
# --network host: PAIR answers plaintext only on Colossus's own loopback.
# The data folder keeps worker state, the daily logs and the hedge record
# across deploys; the image runs as uid 1000, the same as heinrich.
docker run -d --name $CONTAINER --restart unless-stopped \\
    --network host \\
    --log-opt max-size=10m --log-opt max-file=5 \\
    --env-file $ENV_FILE \\
    -v $DATA_DIR:/home/appuser/.tradingagents \\
    $IMAGE:latest worker
echo "--- Last 20 log lines ---"
sleep 3
docker logs $CONTAINER --tail 20
EOF

echo "==> Deployed $GIT_SHA successfully"
