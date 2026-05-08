#!/bin/bash
# deploy.sh — Deploy Kronos Agent OS to a remote host or local runner host
#
# Usage: deploy.sh [--first-run]
#
# Safe deploy: syncs code via rsync, preserves all config and state.
# NEVER use `git reset --hard` on the remote host. This script is the only
# deployment path because it preserves local config and runtime state.

set -euo pipefail

# Load env vars from .env if present (for KAOS_REMOTE etc.)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value || [ -n "$key" ]; do
        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"
        case "$key" in
          ''|\#*) continue ;;
        esac
        if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
fi

DEPLOY_MODE="${KAOS_DEPLOY_MODE:-remote}"
REMOTE_DIR="${KAOS_REMOTE_DIR:-/opt/kaos}"
AGENTS="${KAOS_AGENTS:-kaos}"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$DEPLOY_MODE" != "local" ] && [ "$DEPLOY_MODE" != "remote" ]; then
  echo "FATAL: KAOS_DEPLOY_MODE must be 'local' or 'remote'."
  exit 1
fi

if [ "$DEPLOY_MODE" = "remote" ]; then
  REMOTE="${KAOS_REMOTE:?Set KAOS_REMOTE=user@host in .env or environment}"
else
  REMOTE=""
fi

echo "=== Deploying Kronos Agent OS ==="
echo "Mode: $DEPLOY_MODE"
echo "Target dir: $REMOTE_DIR"
echo "Agents: $AGENTS"

sync_files() {
  local target

  if [ "$DEPLOY_MODE" = "local" ]; then
    sudo mkdir -p "$REMOTE_DIR/app"
    sudo chown -R "$(id -un):$(id -gn)" "$REMOTE_DIR"
    target="$REMOTE_DIR/app/"
  else
    target="$REMOTE:$REMOTE_DIR/app/"
  fi

  # Sync code — explicitly exclude everything that must survive deploy.
  echo "Syncing files..."
  rsync -avz --delete \
    --exclude='.DS_Store' \
    --exclude='.git/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='__pycache__/' \
    --exclude='*.egg-info/' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='mcp-server.log' \
    --exclude='node_modules/' \
    --exclude='data/' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='*.session' \
    --exclude='*.session-*' \
    --exclude='.venv/' \
    --exclude='workspaces/' \
    "$SOURCE_DIR/" "$target"
}

target_bash() {
  if [ "$DEPLOY_MODE" = "local" ]; then
    KAOS_REMOTE_DIR="$REMOTE_DIR" \
    KAOS_AGENTS="$AGENTS" \
    KAOS_HEALTH_URL="${KAOS_HEALTH_URL:-}" \
    KAOS_HEALTH_REQUIRED="${KAOS_HEALTH_REQUIRED:-true}" \
    bash -s
  else
    ssh "$REMOTE" "KAOS_REMOTE_DIR='$REMOTE_DIR' KAOS_AGENTS='$AGENTS' KAOS_HEALTH_URL='${KAOS_HEALTH_URL:-}' KAOS_HEALTH_REQUIRED='${KAOS_HEALTH_REQUIRED:-true}' bash -s"
  fi
}

sync_files

if [ "${1:-}" = "--first-run" ]; then
  echo "First run setup..."
  target_bash <<'TARGET_SCRIPT'
    set -euo pipefail
    cd "$KAOS_REMOTE_DIR"

    if ! python3 - <<'PY' >/dev/null 2>&1
import ensurepip
PY
    then
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
      else
        echo "FATAL: python3 venv/ensurepip is unavailable and apt-get was not found."
        exit 1
      fi
    fi

    if [ -d app/.venv ] && [ ! -x app/.venv/bin/pip ]; then
      echo "Removing incomplete virtualenv from previous failed setup."
      rm -rf app/.venv
    fi

    # Create venv
    python3 -m venv app/.venv
    app/.venv/bin/pip install -e "app/.[dev]"
    app/.venv/bin/pip install edge-tts

    # Install systemd units (replace default User=kronos with actual remote user)
    REMOTE_USER=$(whoami)
    for f in app/systemd/*.service app/systemd/*.timer; do
      sudo sed "s/User=kronos/User=$REMOTE_USER/" "$f" | sudo tee "/etc/systemd/system/$(basename "$f")" >/dev/null
    done
    sudo systemctl daemon-reload

    echo "First run setup complete."
    echo "Next steps:"
    echo "  1. Create .env files from .env.example"
    echo "  2. Run auth-userbot.py for each agent"
    echo "  3. sudo systemctl enable --now kaos"
TARGET_SCRIPT
else
  echo "Deploying to target host..."
  target_bash <<'TARGET_SCRIPT'
    set -euo pipefail
    cd "$KAOS_REMOTE_DIR"

    # === Safety checks ===

    # Verify .env exists
    if [ ! -f app/.env ]; then
      echo "FATAL: app/.env not found! Aborting."
      exit 1
    fi

    # Verify session files exist for all agents
    MISSING_SESSIONS=""
    for agent in $KAOS_AGENTS; do
      if [ ! -f "app/${agent}.session" ]; then
        MISSING_SESSIONS="$MISSING_SESSIONS $agent"
      fi
    done
    if [ -n "$MISSING_SESSIONS" ]; then
      echo "WARNING: Missing session files:$MISSING_SESSIONS"
      echo "Run: AGENT_NAME=<name> .venv/bin/python scripts/auth-userbot.py"
    fi

    # Verify TG_BOT_TOKEN is NOT in agent-specific .env files
    for agent in $KAOS_AGENTS; do
      f="app/.env.$agent"
      if [ -f "$f" ] && grep -qP '^TG_BOT_TOKEN=.+' "$f" 2>/dev/null; then
        echo "WARNING: $f contains TG_BOT_TOKEN — agents should use userbot, not bot!"
      fi
    done

    # Update systemd units if changed (replace default User=kronos with actual remote user)
    REMOTE_USER=$(whoami)
    for f in app/systemd/*.service app/systemd/*.timer; do
      [ -f "$f" ] && sudo sed "s/User=kronos/User=$REMOTE_USER/" "$f" | sudo tee "/etc/systemd/system/$(basename "$f")" >/dev/null
    done
    sudo systemctl daemon-reload

    # Reinstall package (in case deps changed)
    app/.venv/bin/python -m pip install -e "app/." --quiet 2>/dev/null || true

    # Restart all agents
    echo "Restarting all agents..."
    sudo systemctl restart $KAOS_AGENTS

    sleep 3

    # Verify all agents are running
    echo ""
    echo "Agent status:"
    for svc in $KAOS_AGENTS; do
      if ! STATUS=$(systemctl is-active "$svc"); then
        echo "  $svc: $STATUS"
        echo ""
        echo "Last logs for $svc:"
        journalctl -u "$svc" -n 80 --no-pager || true
        exit 1
      fi
      echo "  $svc: $STATUS"
    done

    if [ -n "${KAOS_HEALTH_URL:-}" ]; then
      echo ""
      echo "Health check: $KAOS_HEALTH_URL"
      HEALTH_OK=false
      for attempt in {1..12}; do
        if curl -fsS --max-time 10 "$KAOS_HEALTH_URL"; then
          HEALTH_OK=true
          break
        fi
        if [ "$attempt" -lt 12 ]; then
          sleep 3
        fi
      done
      if [ "$HEALTH_OK" != "true" ]; then
        echo ""
        echo "Health check failed: $KAOS_HEALTH_URL"
        if [ "${KAOS_HEALTH_REQUIRED:-true}" = "true" ]; then
          exit 1
        fi
      fi
      echo ""
    fi

    echo ""
    echo "Deploy complete."
TARGET_SCRIPT
fi

echo "=== Done ==="
