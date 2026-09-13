#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
    printf 'Usage: %s <cargo arguments...>\n' "${0##*/}" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$repo_root/dev.cmd" "$@"
