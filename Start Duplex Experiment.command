#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
export VECLIB_MAXIMUM_THREADS=1
exec ./.venv-sts/bin/python scripts/duplex_server.py
