#!/usr/bin/env bash
# Launch a long job so that it SURVIVES LOGOUT on this machine.
# Why: logind has KillUserProcesses=yes -> at logout every process in the login session is killed, including
# nohup/setsid ones (Phase-1 E03 died twice this way). With `loginctl enable-linger` (done, Linger=yes) the systemd
# *user manager* persists, so jobs started with `systemd-run --user` live outside the session and survive.
# A machine reboot still kills them -> all long sweeps checkpoint and resume.
# Usage: scripts/launch.sh <name> <command...>      log: .tmp/<name>.log   status: systemctl --user status ns-<name>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$1"; shift
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
systemd-run --user --collect --unit="ns-$NAME" --working-directory="$ROOT" \
  --setenv=PYTHONPATH="$ROOT/src" --setenv=HOME="$HOME" --setenv=PATH="$PATH" \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /bin/bash -c "nohup $* > '$ROOT/.tmp/$NAME.log' 2>&1 < /dev/null"
echo "launched ns-$NAME -> .tmp/$NAME.log"
