#!/usr/bin/env bash
# Publishes a directory of logs to the branch ${PUBLISH_PREFIX:-ci-logs-cts-}<name> (forced, one branch per job: the job logs of
# GitHub are not readable from the dev sandbox). Usage: publish-logs.sh <name> <dir>
set -uo pipefail
NAME="$1"; DIR="$2"
BRANCH="${PUBLISH_PREFIX:-ci-logs-cts-}${NAME}"
OUT=$(mktemp -d)
cp -r "$DIR"/. "$OUT"/ 2>/dev/null || true
cd "$OUT"
git init -q && git checkout -q -b ci-logs
git -c user.name=Claude -c user.email=noreply@anthropic.com add -A
git -c user.name=Claude -c user.email=noreply@anthropic.com commit -qm "logs ${GITHUB_SHA::7} ${NAME}" --allow-empty
for a in 1 2 3; do
  git push -qf "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "ci-logs:${BRANCH}" && exit 0
  sleep $((a * 4))
done
echo "could not publish the logs" >&2
exit 0
