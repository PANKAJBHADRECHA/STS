#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
STS_LIVE_HOME="${STS_LIVE_HOME:-$HOME/Library/Application Support/STS}"
export STS_LIVE_HOME
python3.12 -m venv "$STS_LIVE_HOME/venv-live"
"$STS_LIVE_HOME/venv-live/bin/python" -m pip install -r requirements-live.txt
if [[ ! -e .venv-live ]]; then ln -s "$STS_LIVE_HOME/venv-live" .venv-live; fi
if [[ ! -f /opt/homebrew/lib/libspeexdsp.dylib && ! -f /usr/local/lib/libspeexdsp.dylib ]]; then
  HOMEBREW_NO_AUTO_UPDATE=1 brew install speexdsp
fi
"$STS_LIVE_HOME/venv-live/bin/python" scripts/setup_conversation.py
