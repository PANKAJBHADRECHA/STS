#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
exec ./.venv-sts/bin/python -m sts_lab serve --port 8769
