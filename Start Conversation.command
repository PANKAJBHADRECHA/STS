#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
STS_LIVE_HOME="${STS_LIVE_HOME:-$HOME/Library/Application Support/STS}"
export STS_LIVE_HOME
export PYTHONPYCACHEPREFIX="$STS_LIVE_HOME/pycache"
if [[ ! -x "$STS_LIVE_HOME/venv-live/bin/python" || ! -f "$STS_LIVE_HOME/app/scripts/live_server.py" ]]; then
  print 'Run Setup Conversation.command once before starting.'
  read -k1
  exit 1
fi
exec "$STS_LIVE_HOME/venv-live/bin/python" "$STS_LIVE_HOME/app/scripts/live_server.py" --port 8765 --open
