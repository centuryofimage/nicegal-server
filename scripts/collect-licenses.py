#!/usr/bin/env python3
"""Refresh the checked-in Cargo dependency metadata inventory (Python 3 stdlib only)."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent.parent
SCOPE = (
    "Cargo workspace dependencies, including transitive, build, development, and "
    "target-specific crates resolved by cargo metadata --locked. License expressions "
    "are package-author metadata, not a legal compatibility assessment or complete "
    "attribution bundle. Available root license/notice files and declared license "
    "files are retained as text. Native runtimes are listed separately; downloaded "
    "model weights are not included."
)


def collect_notices(directory: Path, declared_file: str | None = None) -> list[dict[str, str]]:
    candidates = {
        path for path in directory.iterdir()
        if path.is_file() and path.name.upper().startswith(
            ("LICENSE", "LICENCE", "COPYING", "NOTICE", "THIRDPARTYNOTICES")
        )
    }
    if declared_file:
        candidates.add(directory / declared_file.replace("\\", "/"))
    notices = []
    for path in sorted(candidates, key=lambda path: path.as_posix()):
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            print(f"Skipping non-UTF-8 notice: {path}")
            continue
        if "\0" in content:
            print(f"Skipping binary notice: {path}")
            continue
        try:
            filename = path.relative_to(directory).as_posix()
        except ValueError:
            filename = path.name
        notices.append({"file": filename, "text": content})
    return notices


def package_url(package: dict) -> str:
    """Return browser-openable HTTPS links, retaining repository preference."""
    for candidate in (package.get("repository"), package.get("homepage")):
        if not candidate:
            continue
        if candidate.startswith("git+"):
            candidate = candidate[4:]
        parsed = urlsplit(candidate)
        if parsed.scheme in ("http", "https", "git") and parsed.hostname and not parsed.username:
            return urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return f"https://crates.io/crates/{package['name']}/{package['version']}"


def package_license(package: dict) -> str:
    if package.get("license"):
        return package["license"]
    license_file = package.get("license_file")
    if not license_file:
        return "Unknown (not declared in Cargo metadata)"
    # nom-exif declares only license-file; this exact published file was reviewed.
    if (package["name"], package["version"]) == ("nom-exif", "3.7.0"):
        # Cargo on Windows may be invoked from MSYS Python, whose Path is POSIX.
        path = Path(package["manifest_path"].replace("\\", "/")).parent / license_file.replace("\\", "/")
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != "9cfbbedc535b1a4b73b2f7f51fd683ac6cae914f6d4bc7f271b9b230dd4e2808":
            raise ValueError("nom-exif license file changed; review before refreshing inventory")
        return "MIT"
    return "See package license file (no SPDX expression declared)"


def runtime_packages() -> list[dict]:
    """Pinned runtime metadata; DirectML's version is not pinned independently."""
    requirements = "\n".join(
        (ROOT / filename).read_text(encoding="utf-8")
        for filename in (
            "requirements-directml.txt", "requirements-openvino.txt",
            "requirements-webgpu.txt",
        )
    )
    packages = []
    for name, license_name, url in (
        ("onnxruntime-directml", "MIT", "https://github.com/microsoft/onnxruntime"),
        ("onnxruntime-openvino", "MIT", "https://github.com/microsoft/onnxruntime"),
        ("onnxruntime", "MIT", "https://github.com/microsoft/onnxruntime"),
        ("onnxruntime-ep-webgpu", "MIT", "https://github.com/microsoft/onnxruntime"),
        ("openvino", "Apache-2.0", "https://github.com/openvinotoolkit/openvino"),
    ):
        match = re.search(rf"^{re.escape(name)}==([^;\s]+)", requirements, re.MULTILINE)
        if match is None:
            raise ValueError(f"Expected a pinned runtime requirement for {name}")
        entry = {"name": name, "version": match[1], "license": license_name, "url": url}
        snapshot = ROOT / "third-party-notices" / f"{name}-{match[1]}"
        if not snapshot.is_dir():
            raise ValueError(f"Missing pinned runtime notice snapshot: {snapshot.name}")
        entry["notices"] = collect_notices(snapshot)
        packages.append(entry)
    packages.extend([{
        "name": "Microsoft DirectML",
        "version": "Bundled with onnxruntime-directml (Windows)",
        "license": "Microsoft Software License Terms (DirectML)",
        "url": "https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4/License",
    }, {
        "name": "FFmpeg",
        "version": "9.0.1",
        "license": " LGPL-2.1-or-later",
        "url": "https://ffmpeg.org/",
        "notices": collect_notices(ROOT / "third-party-notices" / "ffmpeg-9.0.1"),
    }])
    return packages


def main() -> None:
    wrapper = str(ROOT / ("dev.cmd" if os.name == "nt" else "dev.sh"))
    command = [wrapper, "metadata", "--locked", "--format-version", "1"]
    if os.name == "nt":
        command = ["cmd.exe", "/d", "/c", *command]
    result = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    # The Windows build wrapper prints the Visual Studio environment banner before
    # forwarding Cargo's JSON output.
    metadata = json.loads(result.stdout[result.stdout.index("{"):])
    own_packages = {"nicegal-core", "nicegal-cli", "nicegal-server"}
    packages = []
    for package in metadata["packages"]:
        if package["name"] in own_packages and package["source"] is None:
            continue
        entry = {
            "name": package["name"],
            "version": package["version"],
            "license": package_license(package),
            "url": package_url(package),
            "notices": collect_notices(
                Path(package["manifest_path"].replace("\\", "/")).parent,
                package.get("license_file"),
            ),
        }
        packages.append(entry)
    packages.sort(key=lambda package: (package["name"].lower(), package["version"], package["url"]))
    lockfile = (ROOT / "Cargo.lock").read_bytes().replace(b"\r\n", b"\n")
    inventory = {
        "schemaVersion": 1,
        "scope": SCOPE,
        "lockfileSha256": hashlib.sha256(lockfile).hexdigest(),
        "packages": packages,
        "runtimeScope": (
            "Separately provisioned runtime components. ONNX Runtime and OpenVINO "
            "versions come from requirements files; the separately installed OpenVINO "
            "pin applies to Windows. Linux uses libraries bundled in its ONNX Runtime "
            "wheel. Linux defaults to the native WebGPU runtime and plugin. DirectML 1.15.4 license "
            "terms were verified against the Windows "
            "runtime DLL; its version is controlled by the ONNX Runtime wheel. "
            "These entries do not enumerate every component bundled in native runtimes."
        ),
        "runtimePackages": runtime_packages(),
    }
    (ROOT / "third-party-licenses.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(packages)} dependencies to third-party-licenses.json")


if __name__ == "__main__":
    main()
