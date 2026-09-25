#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
target_root=${CARGO_TARGET_DIR:-$repo_root/target}
if [[ "$target_root" != /* ]]; then
    target_root="$PWD/$target_root"
fi
cd "$repo_root"

if [[ "$(uname -s)" == Linux ]]; then
    providers=${NICEGAL_LINUX_PROVIDERS:-webgpu}
    for provider in $providers; do
        case "$provider" in
            openvino|webgpu) ;;
            *) echo "Unknown Linux runtime: $provider" >&2; exit 1 ;;
        esac
        venv=".venv-$provider"
        if [[ -d "$venv/Scripts" ]]; then
            echo "A Windows virtual environment exists at $venv; build from a separate Linux checkout" >&2
            exit 1
        fi
        if [[ ! -x "$venv/bin/python" ]]; then
            uv venv --python 3.13 "$venv"
        fi
        uv pip install --python "$venv/bin/python" -r "requirements-$provider.txt"
        # Resolve the wheel without importing ONNX Runtime, which may need host GPU drivers.
        runtime_directory=$("$venv/bin/python" -c 'import pathlib, site; print(next(pathlib.Path(p).joinpath("onnxruntime/capi") for p in site.getsitepackages() if pathlib.Path(p).joinpath("onnxruntime/capi").is_dir()))')
        runtime_libraries=("$runtime_directory"/libonnxruntime.so.*)
        [[ -f "${runtime_libraries[0]}" ]] || {
            echo "Missing ONNX Runtime shared library in $runtime_directory" >&2
            exit 1
        }
        for profile in debug release; do
            for directory in "$target_root/$profile" "$target_root/$profile/deps"; do
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
elif [[ "$(uname -s)" == Darwin ]]; then
    case "$(uname -m)" in
        x86_64) requirements=requirements-coreml-intel.txt ;;
        arm64) requirements=requirements-coreml.txt ;;
        *) echo "Unsupported macOS architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    venv=.venv-coreml
    if [[ ! -x "$venv/bin/python" ]]; then
        uv venv --python 3.13 "$venv"
    fi
    uv pip install --python "$venv/bin/python" -r "$requirements"
    runtime_directory=$("$venv/bin/python" -c 'import pathlib, site; print(next(pathlib.Path(p).joinpath("onnxruntime/capi") for p in site.getsitepackages() if pathlib.Path(p).joinpath("onnxruntime/capi").is_dir()))')
    runtime_libraries=("$runtime_directory"/libonnxruntime*.dylib)
    [[ -f "${runtime_libraries[0]}" ]] || {
        echo "Missing ONNX Runtime shared library in $runtime_directory" >&2
        exit 1
    }
    for profile in debug release; do
        for directory in "$target_root/$profile" "$target_root/$profile/deps"; do
            destination="$directory/onnxruntime/coreml"
            mkdir -p "$destination"
            cp -L "$runtime_directory"/*.dylib "$destination/"
            ln -sfn "$(basename "${runtime_libraries[0]}")" "$destination/libonnxruntime.dylib"
        done
    done
fi
