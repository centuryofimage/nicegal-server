#!/usr/bin/env bash
set -euo pipefail

# Match the pinned POSIX recipe used by the parent repository's GitHub Actions.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
parent_root=$(cd -- "$repo_root/.." && pwd)
args_file="$parent_root/.github/ffmpeg/posix.args"
if [[ ! -f "$args_file" ]]; then
    args_file="$repo_root/scripts/ffmpeg-posix.args"
fi
source_root="$repo_root/.deps/ffmpeg-src"
build_root="$repo_root/.deps/ffmpeg-build"
install_root="$repo_root/.deps/ffmpeg"
stamp="$install_root/.nicegal-build"
version=n9.0.1

options=()
while IFS= read -r option || [[ -n "$option" ]]; do
    [[ -z "${option//[[:space:]]/}" || "$option" == \#* ]] && continue
    options+=("$option")
done < "$args_file"
if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
    options+=(--enable-neon)
fi

recipe=$(printf '%s\n' "$version" "$(uname -s)" "$(uname -m)" "${options[@]}" | cksum)
if [[ -f "$stamp" && -f "$install_root/lib/libavcodec.a" && -f "$install_root/include/libavcodec/avcodec.h" && "$(<"$stamp")" == "$recipe" ]]; then
    exit 0
fi

for tool in git make nasm cc; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "FFmpeg requires $tool; install it and retry" >&2
        exit 1
    fi
done

mkdir -p "$repo_root/.deps"
if [[ ! -d "$source_root/.git" ]]; then
    git clone --depth 1 --branch "$version" https://github.com/FFmpeg/FFmpeg.git "$source_root"
fi
if [[ "$(git -C "$source_root" describe --tags --exact-match 2>/dev/null)" != "$version" ]]; then
    echo "Expected FFmpeg $version in $source_root; remove that source directory and retry" >&2
    exit 1
fi

mkdir -p "$build_root" "$install_root"
cd "$build_root"
"$source_root/configure" --prefix="$install_root" "${options[@]}"
if [[ "$(uname -s)" == Darwin ]]; then
    jobs=$(sysctl -n hw.ncpu)
else
    jobs=$(nproc)
fi
make -j "$jobs"
make install
mkdir -p "$install_root/share/licenses/ffmpeg"
cp "$source_root/COPYING.LGPLv2.1" "$install_root/share/licenses/ffmpeg/"
printf '%s\n' "$recipe" > "$stamp"
