#!/usr/bin/env bash
set -euo pipefail

# Builds nicegal-server for release. Windows uses build-server.cmd instead (see there for why).
# macOS and Linux have no accelerated execution provider wired up yet, so this builds CPU-only
# until one is added here.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

exec ./dev.sh build --release --locked -p nicegal-server
