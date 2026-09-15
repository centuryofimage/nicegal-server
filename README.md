# nicegal-server

The Rust library, local HTTP server, and search CLI for [Nicegal](https://github.com/centuryofimage/nicegal), a searchable desktop image gallery. The server provides OCR, text and image search, and thumbnails.

Download the desktop app from [Nicegal releases](https://github.com/centuryofimage/nicegal/releases), or the standalone server from [backend releases](https://github.com/centuryofimage/nicegal-server/releases).

## Build from source

On Windows x64, install Rust 1.98.0 (MSVC), Visual Studio C++ build tools, and [uv](https://docs.astral.sh/uv/). On Linux x64, install Rust 1.98.0, uv, `build-essential`, `pkg-config`, `libssl-dev`, `libclang-dev`, `cmake`, and `nasm`. Then clone this repository:

```text
git clone https://github.com/centuryofimage/nicegal-server.git
cd nicegal-server
```

Build a deployable server with `build-server.cmd` on Windows or `./build-server.sh` on Linux. Run `nicegal-server --help` or `nicegal-cli --help` for standalone usage. The desktop app configures and starts its bundled server automatically. The [HTTP API reference](INTERNAL_API.md) covers integration.

## Image search models

MetaCLIP2 B/32 is the default. The model selector offers these five choices:

| Model | Input | Dimensions | Model license |
| --- | --- | --- | --- |
| [MetaCLIP2 B/32](https://huggingface.co/bep256/metaclip-2-worldwide-b32-ONNX) | 224 | 512 | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| [MetaCLIP2 B/16](https://huggingface.co/bep256/metaclip-2-worldwide-b16-ONNX) | 224 | 512 | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| [SigLIP2 Base B/16](https://huggingface.co/bep256/siglip2-base-patch16-256-ONNX) | 256 | 768 | Apache-2.0 |
| [SigLIP beta SwinV2 Base (experimental)](https://huggingface.co/deepghs/siglip_beta/tree/main/smilingwolf/siglip_swinv2_base_2025_02_22_18h56m54s) | 448 | 1024 | Apache-2.0 |
| [DINOv3 B/16 (experimental)](https://huggingface.co/bep256/dinov3-vitb16-pretrain-lvd1689m-ONNX) | 224 | 768 | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license) |

DINOv3 accepts image examples only; it does not support text queries. Model weights retain their publishers' licenses, including MetaCLIP2's noncommercial restriction.

## License

Original application code is [AGPL-3.0-only with a DirectML linking exception](LICENSE), Copyright 2026 bep. Modified third-party code in `vendor/` remains Apache-2.0 under its retained license files and notices. The [dependency license inventory](third-party-licenses.json) and [native runtime notices](third-party-notices/README.md) identify other separately licensed components.

To verify a downloaded server ZIP's GitHub build provenance, run:

```text
gh attestation verify PATH_TO_DOWNLOADED_ZIP --repo centuryofimage/nicegal-server
```
