#!/usr/bin/env bash
# MLX Master Trainer — start the local backend (pure-local, 127.0.0.1:8808).
# Open http://127.0.0.1:8808 in a browser, or use the menu-bar app (desktop/src-tauri).
set -e
cd "$(dirname "$0")"
exec .venv/bin/python backend/server.py
