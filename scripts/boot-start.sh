#!/usr/bin/env bash
# Boot-start fast path for the DeepSeek-V4.1 service. Run once at boot by scripts/systemd/dsv41-boot-start.service on
# the head (a user unit with linger on); the watchdog stays the fallback. Decides from the state the previous boot left:
#   - BOOT_LEASE (optional file) younger than BOOT_LEASE_MIN minutes -> do nothing (you are working on the pair)
#   - rank 0 already running                                           -> do nothing
#   - rank 0 container present (exited: power loss / crash)            -> start
#   - rank 0 absent and serve.sh's stop marker is from an earlier boot -> start
#   - rank 0 absent and no such marker (removed by hand, or stopped during this boot) -> do nothing
# Before the start it waits (bounded, BOOT_WAIT s) for docker, the worker over ssh with docker up, and the RoCE ports
# ACTIVE on both nodes; then `scripts/serve.sh restart` (drops caches, memory gate, slot check, canary).
# Log: $STATE_DIR/boot-start.log. BOOT_DRY_RUN=1 prints the decision and waits, starts nothing.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
CONFIG=${CONFIG:-config/prod.env}
STATE_DIR="${STATE_DIR:-${XDG_STATE_HOME:-${HOME:-/tmp}/.local/state}/dsv41-tf}"
BOOT_WAIT=${BOOT_WAIT:-300}
BOOT_LEASE=${BOOT_LEASE:-}
BOOT_LEASE_MIN=${BOOT_LEASE_MIN:-20}
cfg() { awk -F= -v k="$1" '$1==k {sub(/[ \t]+#.*/, "", $2); print $2}' "$CONFIG" | tail -1; }
NAME=$(cfg NAME); NAME=${NAME:-dsv41-tf}
W=$(cfg WORKER_SSH); W=${W//\"/}
[[ -n "$W" && "$W" != *"<"* ]] || { echo "boot-start: set WORKER_SSH in $CONFIG" >&2; exit 2; }
HCAS=$(cfg NCCL_IB_HCA); ROCE_DEVS=${ROCE_DEVS:-${HCAS//,/ }}
PORT=$(cfg PORT); PORT=${PORT:-8000}
mkdir -p "$STATE_DIR"
[[ "${BOOT_STDOUT:-0}" == 1 ]] || exec >> "$STATE_DIR/boot-start.log" 2>&1
up() { awk '{printf "%.0f", $1}' /proc/uptime; }
log() { echo "[$(date '+%F %T')] dsv41 boot-start (+$(up)s): $*"; }
boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)

if [[ -n "$BOOT_LEASE" && -f "$BOOT_LEASE" ]] && (( $(date +%s) - $(stat -c %Y "$BOOT_LEASE") < BOOT_LEASE_MIN * 60 )); then
    log "$BOOT_LEASE is fresh: not starting (the watchdog takes over when it ages)"; exit 0
fi
end=$(( $(date +%s) + BOOT_WAIT ))
until docker info >/dev/null 2>&1; do (( $(date +%s) < end )) || { log "docker not up after ${BOOT_WAIT}s"; exit 1; }; sleep 2; done
state=$(docker inspect -f '{{.State.Status}}' "$NAME-r0" 2>/dev/null || echo absent)
case "$state" in
    running) log "rank 0 already running: nothing to do"; exit 0 ;;
    absent)
        read -r mb _ < "$STATE_DIR/stopped" 2>/dev/null || mb=""
        if [[ -z "$mb" || "$mb" == "$boot_id" ]]; then log "rank 0 absent and not stopped before this boot: nothing to do"; exit 0; fi
        log "rank 0 absent, stopped before the reboot (marker $STATE_DIR/stopped): starting" ;;
    *) log "rank 0 container $state (the previous boot was serving): starting" ;;
esac
ssh_ok() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$W" "docker info >/dev/null 2>&1 && for d in $ROCE_DEVS; do grep -q ACTIVE /sys/class/infiniband/\$d/ports/1/state || exit 1; done" 2>/dev/null; }
roce_ok() { local d; for d in $ROCE_DEVS; do grep -q ACTIVE "/sys/class/infiniband/$d/ports/1/state" 2>/dev/null || return 1; done; }
until ssh_ok && roce_ok; do
    (( $(date +%s) < end )) || { log "worker ($W) / RoCE not ready after ${BOOT_WAIT}s: leaving it to the watchdog"; exit 1; }
    sleep 3
done
log "worker $W answers, docker up, RoCE ports ACTIVE on both nodes ($ROCE_DEVS)"
[[ "${BOOT_DRY_RUN:-0}" == 1 ]] && { log "BOOT_DRY_RUN=1: not starting"; exit 0; }
# restart, not start: it removes both rank containers first (after a head-only reboot the worker's rank 1 still runs)
log "serve.sh restart ($CONFIG)"
CONFIG="$CONFIG" scripts/serve.sh restart; rc=$?
log "serve.sh restart rc=$rc; serving: $(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | cut -c1-80)"
exit $rc
