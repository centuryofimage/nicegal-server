#!/usr/bin/env bash
set -euo pipefail

# Builds nicegal-server for release. Windows uses build-server.cmd instead (see there for why).
# Each Linux wheel supplies one ONNX Runtime distribution with CPU and its own accelerator.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

if [[ "$(uname -s)" == Linux ]]; then
    providers=${NICEGAL_LINUX_PROVIDERS:-webgpu}
    features=regex
    for provider in $providers; do
        case "$provider" in
            openvino|webgpu) ;;
            *) echo "Unknown Linux runtime: $provider" >&2; exit 1 ;;
        esac
        features+=",ort-$provider"
        venv=".venv-$provider"
        if [[ -d "$venv/Scripts" ]]; then
            echo "A Windows virtual environment exists at $venv; build from a separate Linux checkout" >&2
            exit 1
        fi
        if [[ ! -x "$venv/bin/python" ]]; then
            uv venv --python 3.13 "$venv"
        fi
        uv pip install --python "$venv/bin/python" -r "requirements-$provider.txt"
        # Resolve the package without importing it: GPU wheels may need host drivers to import.
        runtime_directory=$("$venv/bin/python" -c 'import pathlib, site; print(next(pathlib.Path(p).joinpath("onnxruntime/capi") for p in site.getsitepackages() if pathlib.Path(p).joinpath("onnxruntime/capi").is_dir()))')
        runtime_libraries=("$runtime_directory"/libonnxruntime.so.*)
        [[ -f "${runtime_libraries[0]}" ]] || {
            echo "Missing ONNX Runtime shared library in $runtime_directory" >&2
            exit 1
        }
        # Keep both profiles usable, including cargo test executables under deps/.
        for profile in debug release; do
            for directory in "${CARGO_TARGET_DIR:-target}/$profile" "${CARGO_TARGET_DIR:-target}/$profile/deps"; do
                destination="$directory/onnxruntime/$provider"
                mkdir -p "$destination"
                cp -L "$runtime_directory"/lib*.so* "$destination/"
                ln -sfn "$(basename "${runtime_libraries[0]}")" "$destination/libonnxruntime.so"
                if [[ "$provider" == webgpu ]]; then
                    plugin_library=$("$venv/bin/python" -c 'import onnxruntime_ep_webgpu; print(onnxruntime_ep_webgpu.get_library_path())')
                    cp -L "$plugin_library" "$destination/"
                fi
            done
        done
    done
    exec ./dev.sh build --release --locked -p nicegal-server --no-default-features --features "$features"
fi

if [[ "$(uname -s)" == Darwin ]]; then
    case "$(uname -m)" in
        x86_64) requirements=requirements-coreml-intel.txt ;;
        arm64) requirements=requirements-coreml.txt ;;
        *) echo "Unsupported macOS architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    venv=".venv-coreml"
    if [[ ! -x "$venv/bin/python" ]]; then
        uv venv --python 3.13 "$venv"
    fi
    uv pip install --python "$venv/bin/python" -r "$requirements"
    # Resolve the package without importing it: a release build may run on a Mac without the
    # hardware an execution provider needs to initialize.
    runtime_directory=$("$venv/bin/python" -c 'import pathlib, site; print(next(pathlib.Path(p).joinpath("onnxruntime/capi") for p in site.getsitepackages() if pathlib.Path(p).joinpath("onnxruntime/capi").is_dir()))')
    runtime_libraries=("$runtime_directory"/libonnxruntime*.dylib)
    [[ -f "${runtime_libraries[0]}" ]] || {
        echo "Missing ONNX Runtime shared library in $runtime_directory" >&2
        exit 1
    }
    # Keep release executables and test executables self-contained. In this macOS wheel CoreML is
    # compiled into libonnxruntime itself, which is the only runtime dylib it ships.
    for profile in debug release; do
        for directory in "${CARGO_TARGET_DIR:-target}/$profile" "${CARGO_TARGET_DIR:-target}/$profile/deps"; do
            destination="$directory/onnxruntime/coreml"
            mkdir -p "$destination"
            cp -L "$runtime_directory"/*.dylib "$destination/"
            ln -sfn "$(basename "${runtime_libraries[0]}")" "$destination/libonnxruntime.dylib"
        done
    done
    exec ./dev.sh build --release --locked -p nicegal-server --no-default-features --features regex,ort-coreml
fi

exec ./dev.sh build --release --locked -p nicegal-server
