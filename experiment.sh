#!/usr/bin/env bash
# Run an experiment on the current HEAD commit.
# Pushes first so Modal can clone the exact commit.
# Usage: ./experiment.sh [modal_app.py args...]
set -e
git push
exec .venv/bin/modal run modal_app.py "$@"
