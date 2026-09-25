#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
    printf 'Usage: %s <cargo arguments...>\n' "${0##*/}" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CARGO_BUILD_WARNINGS="${CARGO_BUILD_WARNINGS:-deny}"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) exec "$repo_root/dev.cmd" "$@" ;;
    Linux|Darwin)
        case "$1" in
            build|check|clippy|run|test|bench) native_build=true ;;
            *) native_build=false ;;
        esac
        if [[ "$native_build" == true ]]; then
            "$repo_root/scripts/prepare-ort-posix.sh"
            has_features=false
            for argument in "$@"; do
                case "$argument" in
                    --features|--features=*|--no-default-features|--all-features|-F|-F*) has_features=true ;;
                esac
            done
            if [[ "$has_features" == false ]]; then
                if [[ "$(uname -s)" == Linux ]]; then
                    features=nicegal-server/regex
                    for provider in ${NICEGAL_LINUX_PROVIDERS:-webgpu}; do
                        features+=",nicegal-server/ort-$provider"
                    done
                else
                    features=nicegal-server/regex,nicegal-server/ort-coreml
                fi
                set -- "$@" --no-default-features --features "$features"
            fi
        fi
        if [[ -n "${NICEGAL_FFMPEG_DIR:-}" ]]; then
            export FFMPEG_DIR="$NICEGAL_FFMPEG_DIR"
        elif [[ -z "${FFMPEG_DIR:-}" && "$native_build" == true ]]; then
            "$repo_root/scripts/prepare-ffmpeg.sh"
            export FFMPEG_DIR="$repo_root/.deps/ffmpeg"
        fi
        exec cargo "$@"
        ;;
    *) exec cargo "$@" ;;
esac
