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

IMAGE="mobyius/edinburgh-kilty-machaggis"
COLOSSUS="heinrich@192.168.0.98"
CONTAINER="edinburgh-worker"
WEB_CONTAINER="edinburgh-web"
WEB_PORT=8010               # Dublin's logbook has 8000
# Both live in heinrich's home so deploys never need sudo on Colossus.
# Provision/update the env with: scp .env heinrich@192.168.0.98:edinburgh/.env
# (set OLLAMA_BASE_URL in it to PAIR's proxy on Colossus, http://127.0.0.1:11434)
ENV_FILE="/home/heinrich/edinburgh/.env"
DATA_DIR="/home/heinrich/edinburgh/data"
# PAIR's config folder on Colossus: its workload history names the node that
# served each call, which the positions page shows. Mounted read-only into
# the web container when it exists.
PAIR_DIR="/home/heinrich/.config/Nvidia Corporation/Personal AI Router"
KEEP_IMAGES=3               # builds of $IMAGE kept on each side; older ones go
GIT_SHA=$(git rev-parse --short HEAD)
# Build version, Dublin's scheme: major . year . month . day . hour-of-build,
# major 1 = Edinburgh (Dublin is 4); see tradingagents/build.py. One build an
# hour is distinguishable, which is plenty for a hand-run deploy. Baked into
# the image AND used as its tag, so `docker ps` on Colossus names the build it
# is running instead of the uninformative "latest".
BUILD_VERSION="1.$(date +%Y.%m.%d.%H)"

# Keep the newest $KEEP_IMAGES tags of $IMAGE (":latest" rides on the newest
# sha, so it is not counted) and drop the rest with their dangling layers, so
# a deploy a day does not fill the drive. Runs here and on Colossus.
PRUNE_IMAGES="docker images $IMAGE --format '{{.Repository}}:{{.Tag}}' | grep -v ':latest\$' | tail -n +$((KEEP_IMAGES + 1)) | xargs -r docker rmi >/dev/null 2>&1 || true; docker image prune -f >/dev/null 2>&1 || true; echo \"images kept: \$(docker images $IMAGE --format '{{.Tag}}' | grep -v '^latest\$' | tr '\\n' ' ')\""

# Refuse to deploy uncommitted work - the image tag must match GitHub
if [[ -n $(git status --porcelain) ]]; then
    echo "ERROR: Uncommitted changes. Commit to GitHub first."
    exit 1
fi

echo "==> Building $IMAGE:$BUILD_VERSION ($GIT_SHA)"
docker build \
    --build-arg BUILD_VERSION="$BUILD_VERSION" \
    --build-arg GIT_SHA="$GIT_SHA" \
    -t "$IMAGE:$BUILD_VERSION" -t "$IMAGE:latest" .

echo "==> Pushing to Docker Hub"
docker push "$IMAGE:$BUILD_VERSION"
docker push "$IMAGE:latest"
echo "==> Pruning local images"
eval "$PRUNE_IMAGES"

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
# Pull the build and point :latest at it, so Colossus keeps the last few
# builds too (a pull of :latest alone leaves the old image untagged) and a
# rollback is a rerun of these lines with an earlier build's tag.
docker pull $IMAGE:$BUILD_VERSION
docker tag $IMAGE:$BUILD_VERSION $IMAGE:latest
docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true
# --network host: PAIR answers plaintext only on Colossus's own loopback.
# The data folder keeps worker state, the daily logs and the hedge record
# across deploys; the image runs as uid 1000, the same as heinrich. The :z
# labels it for containers: Colossus is Fedora, and SELinux refuses a
# container's access to a bind mount without it. Lower-case z is the shared
# label; :Z is private, and the second container to mount the folder would
# relabel it and lock the first out (which happened on the first deploy).
# PAIR's history, read-only, for both containers: the worker's client looks up
# the node that served each call and checks its endpoint against it; the web
# page shows the shares. The folder's name has spaces: an array keeps it one
# argument on Colossus.
PAIR_ARGS=()
if [[ -f "$PAIR_DIR/workloads-history.json" ]]; then
    PAIR_ARGS=(-v "$PAIR_DIR:/home/appuser/pair:ro,z" -e TRADINGAGENTS_PAIR_DIR=/home/appuser/pair)
    echo "PAIR history mounted from $PAIR_DIR"
fi
docker run -d --name $CONTAINER --restart unless-stopped \\
    --network host \\
    --log-opt max-size=10m --log-opt max-file=5 \\
    --env-file $ENV_FILE \\
    -v $DATA_DIR:/home/appuser/.tradingagents:z \\
    "\${PAIR_ARGS[@]}" \\
    $IMAGE:$BUILD_VERSION worker
# The positions page: the same image, read-only on the same data folder.
docker stop $WEB_CONTAINER 2>/dev/null || true
docker rm $WEB_CONTAINER 2>/dev/null || true
docker run -d --name $WEB_CONTAINER --restart unless-stopped \\
    --network host \\
    --log-opt max-size=10m --log-opt max-file=5 \\
    --env-file $ENV_FILE \\
    -v $DATA_DIR:/home/appuser/.tradingagents:ro,z \\
    "\${PAIR_ARGS[@]}" \\
    $IMAGE:$BUILD_VERSION web --port $WEB_PORT
echo "--- Last 20 log lines ---"
sleep 3
docker logs $CONTAINER --tail 20
docker logs $WEB_CONTAINER --tail 5
echo "--- Pruning images on Colossus ---"
$PRUNE_IMAGES
echo "Positions page: http://192.168.0.98:$WEB_PORT/"
EOF

echo "==> Deployed build $BUILD_VERSION ($GIT_SHA) successfully"
