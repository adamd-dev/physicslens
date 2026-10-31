#!/usr/bin/env bash
# Put the PhysicsLens app on the internet from this Mac: the full app in public mode behind a Cloudflare quick
# tunnel (no account, no card). Online only while this script runs; Ctrl+C stops the app and the tunnel.
#
#   deploy/tunnel.sh             start, print the public link
#   deploy/tunnel.sh --publish   also point the landing page's /app link at it (commits web/vercel.json, pushes main)
#
# Needs cloudflared on PATH (brew install cloudflared), or CLOUDFLARED=/path/to/cloudflared.
# The Meta key and budget come from .env as for local use: everyone who opens the link spends from that budget.
set -euo pipefail
cd "$(dirname "$0")/.."
CLOUDFLARED="${CLOUDFLARED:-cloudflared}"
[ -d "$HOME/.local/node20/bin" ] && PATH="$HOME/.local/node20/bin:$PATH"
command -v "$CLOUDFLARED" >/dev/null || { echo "cloudflared not found: brew install cloudflared" >&2; exit 1; }
mkdir -p cache

echo "building the web app..."
(cd web && npm run build >/dev/null)

export PHYSICSLENS_PUBLIC=1 PHYSICSLENS_MAX_UPLOAD_MB=100 PHYSICSLENS_MAX_UPLOADS=40 PHYSICSLENS_PORT=8000
.venv/bin/physicslens-ui >cache/server.log 2>&1 &
SERVER=$!
# The Mac is the server: no display, idle or system sleep while it runs (caffeinate ends with the server).
command -v caffeinate >/dev/null && caffeinate -dims -w "$SERVER" &
TUNNEL=""
trap 'kill $SERVER ${TUNNEL:+$TUNNEL} 2>/dev/null' EXIT INT TERM
for _ in $(seq 30); do curl -sf http://127.0.0.1:8000/api/config >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:8000/api/config >/dev/null || { echo "the app did not start; see cache/server.log" >&2; exit 1; }

"$CLOUDFLARED" tunnel --no-autoupdate --url http://127.0.0.1:8000 >cache/tunnel.log 2>&1 &
TUNNEL=$!
URL=""
for _ in $(seq 60); do
  URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' cache/tunnel.log | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { echo "no tunnel link; see cache/tunnel.log" >&2; exit 1; }
echo
echo "PhysicsLens is online at $URL"
echo "(public: every upload and run is visible to anyone with the link)"

if [ "${1:-}" = "--publish" ]; then
  if [ "$(git branch --show-current)" != "main" ]; then
    echo "not on main: the landing link was not changed" >&2
  else
    .venv/bin/python - "$URL" <<'EOF'
import json, sys
from pathlib import Path
path = Path("web/vercel.json")
config = json.loads(path.read_text())
config["redirects"] = [{"source": "/app", "destination": sys.argv[1] + "/", "permanent": False}]
path.write_text(json.dumps(config, indent=2) + "\n")
EOF
    git add web/vercel.json
    git commit -q -m "deploy: the landing's /app link points to $URL"
    git push -q origin main
    echo "landing /app now redirects there (Vercel redeploys in about a minute)"
  fi
fi

echo "Ctrl+C to stop."
wait "$SERVER"
