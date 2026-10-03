"""Regression tests for build-time license generation; no Cargo compilation required."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "collect_licenses", Path(__file__).with_name("collect-licenses.py")
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class LicenseGenerationTests(unittest.TestCase):
    def test_stale_build_output_is_replaced_without_changing_the_lockfile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lockfile = root / "Cargo.lock"
            lockfile.write_bytes(b"# Fixture lockfile\r\nversion = 4\r\n")
            original_lock = lockfile.read_bytes()
            output = root / "target" / "third-party-licenses.json"
            output.parent.mkdir()
            output.write_text('{"lockfileSha256": "stale"}', encoding="utf-8")
            result = subprocess.CompletedProcess([], 0, stdout='Banner\n{"packages": []}')
            with (
                patch.object(collector, "ROOT", root),
                patch.object(collector, "runtime_packages", return_value=[]),
                patch.object(collector.subprocess, "run", return_value=result) as run,
                redirect_stdout(io.StringIO()),
            ):
                collector.main()
            command = run.call_args.args[0]
            self.assertIn("--locked", command)
            self.assertIn("metadata", command)
            self.assertTrue(any(Path(arg).name in {"dev.cmd", "dev.sh"} for arg in command))
            self.assertEqual(lockfile.read_bytes(), original_lock)
            self.assertNotEqual(json.loads(output.read_text())["lockfileSha256"], "stale")
            self.assertNotIn(b"\r\n", output.read_bytes())
            self.assertFalse((root / "third-party-licenses.json").exists())

    def test_cargo_failure_keeps_its_diagnostic_and_stops_generation(self):
        diagnostic = "error: the lock file needs to be updated but --locked was passed\n"
        failure = subprocess.CalledProcessError(101, ["cargo"], stderr=diagnostic)
        stderr = io.StringIO()
        with (
            patch.object(collector.subprocess, "run", side_effect=failure),
            redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as raised:
                collector.main()
        self.assertEqual(raised.exception.code, 101)
        self.assertEqual(stderr.getvalue(), diagnostic)

    def test_changed_license_text_uses_the_declared_file_without_blocking_the_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "LICENSE").write_text("Changed upstream license text", encoding="utf-8")
            package = {
                "name": "nom-exif",
                "version": "3.7.0",
                "license_file": "LICENSE",
                "manifest_path": str(root / "Cargo.toml"),
            }
            self.assertEqual(
                collector.package_license(package),
                "See package license file (no SPDX expression declared)",
            )
            self.assertEqual(
                collector.collect_notices(root)[0]["text"], "Changed upstream license text"
            )

    def test_runtime_pin_changes_do_not_require_an_inventory_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "requirements-directml.txt", "requirements-openvino.txt", "requirements-webgpu.txt"
            ):
                (root / name).write_text("onnxruntime==99.0.0\n", encoding="utf-8")
            ffmpeg = root / "third-party-notices" / "ffmpeg-9.0.1"
            ffmpeg.mkdir(parents=True)
            (ffmpeg / "LICENSE").write_text("FFmpeg notice", encoding="utf-8")
            with patch.object(collector, "ROOT", root):
                packages = collector.runtime_packages()
            runtime = next(package for package in packages if package["name"] == "onnxruntime")
            self.assertEqual(runtime["version"], "99.0.0")
            self.assertEqual(runtime["notices"], [])
            self.assertTrue(runtime["url"].startswith("https://"))
            ffmpeg_entry = next(package for package in packages if package["name"] == "FFmpeg")
            self.assertEqual(ffmpeg_entry["notices"][0]["text"], "FFmpeg notice")


if __name__ == "__main__":
    unittest.main()
