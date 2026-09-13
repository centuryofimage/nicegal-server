#!/usr/bin/env bash
set -euo pipefail

# Builds nicegal-server for release. Windows uses build-server.cmd instead (see there for why).
# Linux bundles OpenVINO, which also includes the CPU execution provider.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

if [[ "$(uname -s)" == Linux ]]; then
    if [[ ! -x .venv-openvino/bin/python ]]; then
        uv venv --python 3.13 .venv-openvino
    fi
    uv pip install --python .venv-openvino/bin/python -r requirements-openvino.txt
    runtime_directory=$(.venv-openvino/bin/python -c 'import pathlib, onnxruntime; print(pathlib.Path(onnxruntime.__file__).parent / "capi")')
    # Keep both profiles usable, including cargo test executables under deps/.
    for profile in debug release; do
        for directory in "${CARGO_TARGET_DIR:-target}/$profile" "${CARGO_TARGET_DIR:-target}/$profile/deps"; do
            mkdir -p "$directory/onnxruntime/openvino"
            cp -L "$runtime_directory"/lib*.so* "$directory/onnxruntime/openvino/"
            cp -L "$runtime_directory"/libonnxruntime.so.* "$directory/onnxruntime/openvino/libonnxruntime.so"
        done
    done
    exec ./dev.sh build --release --locked -p nicegal-server --no-default-features --features regex,ort-openvino
fi

exec ./dev.sh build --release --locked -p nicegal-server
