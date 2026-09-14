# FastEmbed for Nicegal

Modified for Nicegal; changes remain licensed under Apache-2.0.

This directory vendors the FastEmbed 6.0.0 Rust crate from
[fastembed-rs](https://github.com/Anush008/fastembed-rs). Upstream authors are
listed in `Cargo.toml`. The package retains its [Apache-2.0 license](LICENSE).

Nicegal uses FastEmbed for BGE text embeddings and CLIP image/text embeddings
through ONNX Runtime. The application initializes the runtime and supplies
execution-provider settings before constructing embedding sessions.

The optional `webgpu` feature recognizes WebGPU dispatches and registers the
native plugin's discovered device. The optional `ort-profiling` feature enables
bounded image-inference diagnostics through `NICEGAL_ORT_PROFILE_DIR` and requires
ONNX Runtime API 25. Both are disabled unless selected by the parent build.

The local integration exposes a cloneable `ImagePreprocessor` through
`ImageEmbedding::preprocessor()`. Decode workers preprocess images into owned
`ndarray::Array3<f32>` tensors; `ImageEmbedding::embed_preprocessed` runs batches
of those tensors through the model. Cached model-loading paths allow query
encoders to reload without a network request.

`ImagePreprocessor::with_resize` lets the host supply its shared SIMD resizer.
Nicegal uses float intermediates to limit changes to model inputs. The fork still
owns each model's dimensions, crop offsets, Pillow pass order, and normalization;
the callback is shared without locking the parallel preprocessing workers.

Build this package through the parent Nicegal workspace. See the backend
[README](../../README.md) and [CLIP benchmark](../../benches/CLIP_INDEX.md).

Local paired encoder loading uses `try_new_from_path` to resolve external ONNX
tensors from disk without duplicating the entire checkpoint in Rust buffers.
The paired text loader pads to its fixed context length and selects `text_embeds`.
Local CLIP preprocessors can opt into aspect-preserving shortest-edge resizing;
legacy configs retain their previous behavior. SigLIP image processor configs
are also accepted for exact-square resizing and model-specific normalization.

Local preprocessors honor the declared bilinear/bicubic interpolation and use
horizontal then vertical RGB8 passes to match Pillow's clipping and rounding.
A per-model center-crop option implements torchvision's ties-to-even offsets.
Text inference only sends attention masks when the ONNX graph declares that input.
