//! Windows PowerShell is a second HTTP transport; the Hub cache format stays unchanged.
use super::{DownloadCancelled, ModelSource};
use anyhow::{Context, Result, bail};
use hf_hub::Cache;
use std::{
    io::{BufRead, BufReader, Read},
    os::windows::process::CommandExt,
    path::PathBuf,
    process::{Command, Stdio},
};

pub(super) fn is_connection_error(error: &anyhow::Error) -> bool {
    for cause in error.chain() {
        if let Some(io) = cause.downcast_ref::<std::io::Error>()
            && matches!(
                io.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::UnexpectedEof
                    | std::io::ErrorKind::BrokenPipe
            )
        {
            return true;
        }
        if let Some(hf_hub::api::tokio::ApiError::RequestError(request)) =
            cause.downcast_ref::<hf_hub::api::tokio::ApiError>()
        {
            return request.status().is_none() && !request.is_builder();
        }
        if let Some(hf_hub::api::sync::ApiError::RequestError(request)) =
            cause.downcast_ref::<hf_hub::api::sync::ApiError>()
        {
            // ureq's HTTP status errors must not trigger a second download (401/404 etc.).
            return std::error::Error::source(request.as_ref()).is_some();
        }
    }
    false
}

pub(super) fn fallback(
    source: &ModelSource,
    cache: &Cache,
    original: anyhow::Error,
    progress: impl FnMut(usize, usize) -> bool,
) -> Result<PathBuf> {
    tracing::warn!(model_id = %source.model_id, filename = %source.filename,
        error_chain = %format!("{original:#}"), "Retrying Hugging Face download with powershell.exe");
    download_with_progress(source, cache, progress).with_context(|| {
        format!("Windows PowerShell fallback failed after hf-hub failed: {original:#}")
    })
}

fn safe_relative(value: &str) -> bool {
    !value.is_empty()
        && value.split('/').all(|part| {
            !part.is_empty() && part != "." && part != ".." && !part.contains(['\\', ':', '\0'])
        })
}

#[cfg(test)]
pub(super) fn download(source: &ModelSource, cache: &Cache) -> Result<PathBuf> {
    download_with_progress(source, cache, |_, _| true)
}

fn download_with_progress(
    source: &ModelSource,
    cache: &Cache,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<PathBuf> {
    let revision = source.revision.as_deref().unwrap_or("main");
    if !safe_relative(&source.model_id)
        || !safe_relative(&source.filename)
        || !safe_relative(revision)
    {
        bail!("invalid Hugging Face repository path");
    }
    // All variable inputs are environment values, never interpolated into PowerShell code.
    let mut child = Command::new("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-OutputFormat",
            "Text",
            "-Command",
            include_str!("download-windows.ps1"),
        ])
        .env("NICEGAL_HF_REPO", &source.model_id)
        .env("NICEGAL_HF_REVISION", revision)
        .env("NICEGAL_HF_FILE", &source.filename)
        .env("NICEGAL_HF_CACHE", cache.path())
        .env("NICEGAL_HF_TOKEN", cache.token().unwrap_or_default())
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting powershell.exe for Hugging Face download")?;
    let mut stderr = child.stderr.take().context("missing download stderr")?;
    // Drain both pipes while running, so errors cannot block a large download.
    let errors = std::thread::spawn(move || {
        let mut message = String::new();
        stderr.read_to_string(&mut message).map(|_| message)
    });
    let stdout = child.stdout.take().context("missing download stdout")?;
    let reports = (|| -> Result<()> {
        for line in BufReader::new(stdout).lines() {
            let line = line?;
            let mut fields = line.split_whitespace();
            if fields.next() == Some("PROGRESS") {
                let downloaded: usize =
                    fields.next().context("missing downloaded bytes")?.parse()?;
                let total: usize = fields.next().context("missing total bytes")?.parse()?;
                if !progress(downloaded, total) {
                    return Err(DownloadCancelled.into());
                }
            }
        }
        Ok(())
    })();
    if reports.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let stderr = errors
        .join()
        .map_err(|_| anyhow::anyhow!("download stderr reader panicked"))??;
    reports?;
    if !status.success() {
        bail!("powershell.exe download failed: {}", stderr.trim());
    }
    cache
        .repo(source.repo())
        .get(&source.filename)
        .context("PowerShell download did not publish the requested cache file")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cache_path_traversal() {
        for path in ["../model", "/model", "a/../b", "C:/model", "a\\b", "a//b"] {
            assert!(!safe_relative(path), "{path}");
        }
        assert!(safe_relative("onnx/model.onnx"));
    }

    #[test]
    fn retries_connection_reset_but_not_disk_errors() {
        assert!(is_connection_error(
            &std::io::Error::from(std::io::ErrorKind::ConnectionReset).into()
        ));
        assert!(!is_connection_error(
            &std::io::Error::from(std::io::ErrorKind::PermissionDenied).into()
        ));
    }

    #[test]
    fn rejects_corrupt_download_without_publishing_cache() -> Result<()> {
        use std::io::{BufRead, Write};
        let server = std::net::TcpListener::bind("127.0.0.1:0")?;
        let endpoint = format!("http://{}", server.local_addr()?);
        let worker = std::thread::spawn(move || {
            let metadata = serde_json::json!({
                "sha": "0123456789012345678901234567890123456789",
                "siblings": [{"rfilename": "model.onnx", "size": 3,
                    "lfs": {"sha256": "0".repeat(64)}}]
            })
            .to_string();
            for body in [metadata.as_str(), "bad"] {
                let (mut socket, _) = server.accept().unwrap();
                socket
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut reader = std::io::BufReader::new(&socket);
                loop {
                    let mut line = String::new();
                    assert!(reader.read_line(&mut line).unwrap() > 0);
                    if line == "\r\n" {
                        break;
                    }
                }
                write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
            }
        });
        let temp = tempfile::tempdir()?;
        let output = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                include_str!("download-windows.ps1"),
            ])
            .env("HF_ENDPOINT", endpoint)
            .env("NICEGAL_HF_REPO", "test/model")
            .env("NICEGAL_HF_REVISION", "main")
            .env("NICEGAL_HF_FILE", "model.onnx")
            .env("NICEGAL_HF_CACHE", temp.path())
            .env("NICEGAL_HF_TOKEN", "")
            .creation_flags(0x08000000)
            .output()?;
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("SHA-256 mismatch"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        worker.join().unwrap();
        assert!(
            Cache::new(temp.path().into())
                .model("test/model".into())
                .get("model.onnx")
                .is_none()
        );
        assert!(
            walkdir::WalkDir::new(temp.path())
                .into_iter()
                .filter_map(std::result::Result::ok)
                .all(|entry| !entry.file_type().is_file())
        );
        Ok(())
    }

    #[test]
    #[ignore = "downloads models from Hugging Face and loads ONNX Runtime"]
    fn powershell_downloads_and_loads_models() -> Result<()> {
        use crate::runtime::{self, RuntimeOptions};
        let _ = tracing_subscriber::fmt()
            .with_env_filter("warn,nicegal_core=info")
            .try_init();
        let temp = tempfile::tempdir()?;
        let cache = Cache::new(temp.path().join("cache with spaces"));
        let source = |model: &str, filename: &str| ModelSource {
            model_id: model.into(),
            filename: filename.into(),
            revision: None,
        };
        let det = source("PaddlePaddle/PP-OCRv6_small_det_onnx", "inference.onnx");
        let rec = source("PaddlePaddle/PP-OCRv6_small_rec_onnx", "inference.onnx");
        let det_path = download(&det, &cache)?;
        let rec_path = download(&rec, &cache)?;
        let det_config = source(&det.model_id, "inference.yml");
        // Force the normal async client to fail, exercising the actual fallback branch.
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_endpoint("http://127.0.0.1:1".into())
            .with_progress(false)
            .build()?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let det_config_path = rt.block_on(det_config.get_with_progress(&api, &cache, ()))?;
        let rec_config_path = download(&source(&rec.model_id, "inference.yml"), &cache)?;
        assert_eq!(
            cache.repo(det.repo()).get("inference.onnx"),
            Some(det_path.clone())
        );
        runtime::initialize_bundled_runtime(runtime::ExecutionProvider::Cpu)?;
        let _ocr = crate::ocr::PaddleOcrModels::load_files(
            crate::ocr::OcrModelFiles {
                source: &det,
                model_path: &det_path,
                config_path: &det_config_path,
            },
            crate::ocr::OcrModelFiles {
                source: &rec,
                model_path: &rec_path,
                config_path: &rec_config_path,
            },
            RuntimeOptions::default(),
        )?;
        for model in [fastembed::EmbeddingModel::BGESmallENV15] {
            let info = fastembed::TextEmbedding::get_model_info(&model)?;
            for file in [
                info.model_file.as_str(),
                "tokenizer.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ] {
                download(&source(&info.model_code, file), &cache)?;
            }
            let mut loaded = fastembed::TextEmbedding::try_new_cached(
                fastembed::TextInitOptions::new(model).with_cache_dir(cache.path().clone()),
            )?
            .expect("fallback files must be discoverable offline");
            assert_eq!(loaded.embed(vec!["a photo of a cat"], None)?.len(), 1);
        }
        Ok(())
    }
}
