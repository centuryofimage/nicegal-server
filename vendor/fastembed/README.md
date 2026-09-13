# FastEmbed for Nicegal

Modified for Nicegal; changes remain licensed under Apache-2.0.

This directory vendors the FastEmbed 6.0.0 Rust crate from
[fastembed-rs](https://github.com/Anush008/fastembed-rs). Upstream authors are
listed in `Cargo.toml`. The package retains its [Apache-2.0 license](LICENSE).

Nicegal uses FastEmbed for BGE text embeddings and CLIP image/text embeddings
through ONNX Runtime. The application initializes the runtime and supplies
execution-provider settings before constructing embedding sessions.

The local integration exposes a cloneable `ImagePreprocessor` through
`ImageEmbedding::preprocessor()`. Decode workers preprocess images into owned
`ndarray::Array3<f32>` tensors; `ImageEmbedding::embed_preprocessed` runs batches
of those tensors through the model. Cached model-loading paths allow query
encoders to reload without a network request.

Build this package through the parent Nicegal workspace. See the backend
[README](../../README.md) and [CLIP benchmark](../../benches/CLIP_INDEX.md).
