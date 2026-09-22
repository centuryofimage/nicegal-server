use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, de::DeserializeOwned};

#[derive(Deserialize)]
pub struct TextCase {
    pub text: String,
    pub ids: Vec<u32>,
    pub embedding: Vec<f32>,
}

#[derive(Deserialize)]
pub struct ImageCase {
    pub path: std::path::PathBuf,
    pub embedding: Vec<f32>,
}

pub fn read_cases<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("reading fixtures: {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing fixtures: {}", path.display()))
}

pub fn compare(actual: &[f32], expected: &[f32]) -> Result<(f32, f32)> {
    ensure!(actual.len() == expected.len(), "vector dimensions differ");
    ensure!(
        actual.iter().all(|value| value.is_finite()),
        "nonfinite native vector"
    );
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let dot = actual.iter().zip(expected).map(|(a, b)| a * b).sum();
    Ok((max_error, dot))
}
