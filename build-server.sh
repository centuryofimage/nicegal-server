#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

if [[ "$(uname -s)" == Linux ]]; then
    features=regex
    for provider in ${NICEGAL_LINUX_PROVIDERS:-webgpu}; do
        case "$provider" in
            openvino|webgpu) features+=",ort-$provider" ;;
            *) echo "Unknown Linux runtime: $provider" >&2; exit 1 ;;
        esac
    done
    exec ./dev.sh build --release --locked -p nicegal-server --no-default-features --features "$features"
fi

if [[ "$(uname -s)" == Darwin ]]; then
    exec ./dev.sh build --release --locked -p nicegal-server --no-default-features --features regex,ort-coreml
fi

exec ./dev.sh build --release --locked -p nicegal-server
