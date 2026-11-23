#!/usr/bin/env bash
# Flood /predict with Locust against 1, 2 and 4 inference containers.
#
# Local (Docker on this machine):
#   bash scripts/run_load_tests.sh
# Remote (the stack runs on AWS, Locust runs on THIS machine, containers are scaled over SSH):
#   SSH_TARGET=ubuntu@<EC2-IP> SSH_KEY=~/key.pem HOST=http://<EC2-IP>:8000 OUT_DIR=results/aws \
#   REMOTE_DIR='~/summative-pneumonia-mlops' bash scripts/run_load_tests.sh
#
# Other knobs: USERS=50 SPAWN=10 DURATION=60s COUNTS="1 2 4"
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-http://localhost:8000}"
USERS="${USERS:-50}"
SPAWN="${SPAWN:-10}"
DURATION="${DURATION:-60s}"
COUNTS="${COUNTS:-1 2 4}"
OUT_DIR="${OUT_DIR:-results}"
SSH_TARGET="${SSH_TARGET:-}"
SSH_KEY="${SSH_KEY:-}"
REMOTE_DIR="${REMOTE_DIR:-~/summative-pneumonia-mlops}"
mkdir -p "$OUT_DIR"

scale() {   # scale the inference service to $1 containers (locally or over SSH)
  if [ -n "$SSH_TARGET" ]; then
    ssh ${SSH_KEY:+-i "$SSH_KEY"} -o StrictHostKeyChecking=accept-new "$SSH_TARGET" \
      "cd $REMOTE_DIR && docker compose up -d --scale inference=$1 --wait"
  else
    docker compose up -d --scale inference="$1" --wait
  fi
}

for N in $COUNTS; do
  echo
  echo "=================  $N inference container(s)  ================="
  scale "$N"
  sleep 12      # let Nginx refresh its DNS view of the replicas and the models warm up
  SERVED_OUT="$OUT_DIR/served_n${N}.json" locust -f locust/locustfile.py --headless \
      -u "$USERS" -r "$SPAWN" -t "$DURATION" --host "$HOST" \
      --csv "$OUT_DIR/n${N}" --only-summary
done

echo
echo "Done. Now run:  python scripts/summarize_load_tests.py $OUT_DIR"