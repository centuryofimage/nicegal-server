# Native runtime notice snapshots

These unmodified upstream files were copied from the installed Windows runtime
packages on September 13, 2026, matching the pinned requirements:

- `onnxruntime-directml-1.24.4`: `.venv-directml/Lib/site-packages/onnxruntime/LICENSE`
  and `ThirdPartyNotices.txt` from the `onnxruntime-directml==1.24.4` wheel.
- `onnxruntime-openvino-1.24.1`: `.venv-openvino/Lib/site-packages/onnxruntime/LICENSE`
  and `ThirdPartyNotices.txt` from the `onnxruntime-openvino==1.24.1` wheel.
- `openvino-2025.4.1`: `.venv-openvino/Lib/site-packages/openvino-2025.4.1.dist-info/licenses/LICENSE`
  from the `openvino==2025.4.1` wheel.
- `onnxruntime-1.30.0`: Linux wheel's `onnxruntime/LICENSE` and
  `ThirdPartyNotices.txt` from the native WebGPU test environment.
- `onnxruntime-ep-webgpu-0.3.0`: `LICENSE` and `ThirdPartyNotices.txt` from
  the Linux plugin wheel's `.dist-info/licenses` directory.

The collector reads these snapshots so building the notices document does not need
runtime installations. On runtime upgrades, copy the corresponding upstream text
files into a directory matching the new package name and version, and refresh the
inventory. The collector fails if a pinned package has no snapshot directory.

These are the available notices from these installed wheels, not an exhaustive
native dependency audit. Linux wheel contents may differ. The DirectML DLL has
separate [Microsoft terms](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4/License);
the ONNX Runtime MIT license does not replace them. Model weights are outside this
directory's scope.
