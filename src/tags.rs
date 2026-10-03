//! Zero-shot image tags from the `bep256/clip-tags` vocabulary.
//!
//! The repository holds one shared extensionless `vocabulary` and, for each supported image model, a
//! safetensors file with a float16 text embedding per tag (`embeddings`) and the mean image vector
//! of a public reference set (`reference_mean`). A tag's score for an image is its cosine
//! similarity to the image minus its similarity to that mean, so tags that match nearly every
//! image do not crowd the list.
use anyhow::{Context, Result, bail, ensure};
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use std::path::Path;

use crate::embedding::ImageEmbeddingModel;
use crate::hub::ModelSource;

const REPOSITORY: &str = "bep256/clip-tags";
const REVISION: &str = "1973bb7963b91efeee3d0afb139beb2db74a2492";
const VOCABULARY: &str = "vocabulary";
const HEADER: &str = "term\tkind\tsource\tsensitivity";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    /// A common everyday word, like car or happy.
    Simple,
    Subject,
    Vibe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TagSource {
    Metaclip,
    Wordnet,
}

/// `Blocked` covers slurs and crude or sexualizing labels; `Mature` is descriptive adult terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Sensitivity {
    Ok,
    Mature,
    Blocked,
}

#[derive(Debug, Clone)]
pub struct Tag {
    pub term: String,
    pub kind: TagKind,
    pub source: TagSource,
    pub sensitivity: Sensitivity,
}

#[derive(Debug, Clone)]
pub struct ScoredTag<'a> {
    pub tag: &'a Tag,
    /// Cosine similarity between the tag's text vector and the image.
    pub similarity: f32,
    /// `similarity` minus the tag's mean similarity over the reference images.
    pub score: f32,
}

/// Best-scoring tags of each kind for one image.
#[derive(Debug, Clone, Default)]
pub struct TagRanking<'a> {
    pub simple: Vec<ScoredTag<'a>>,
    pub subjects: Vec<ScoredTag<'a>>,
    pub vibes: Vec<ScoredTag<'a>>,
}

/// One model's tags, with its embeddings mapped from the Hugging Face cache.
pub struct TagSet {
    model: ImageEmbeddingModel,
    tags: Vec<Tag>,
    embeddings: Mmap,
    /// Each tag's mean similarity over the reference images, in vocabulary order.
    baselines: Vec<f32>,
}

impl TagSet {
    /// Whether published or explicitly staged local tags exist for `model`.
    pub fn supports(model: ImageEmbeddingModel) -> bool {
        if let Some(directory) = local_directory() {
            return directory.join(VOCABULARY).is_file()
                && directory.join(embedding_filename(model)).is_file();
        }
        model.published_tags()
    }

    /// Load `model`'s tags, downloading the vocabulary and the model's embeddings on a cache miss.
    pub fn load(model: ImageEmbeddingModel) -> Result<Self> {
        ensure!(Self::supports(model), "no tags are published for {model}");
        let (vocabulary, embeddings) = if let Some(directory) = local_directory() {
            (
                directory.join(VOCABULARY),
                directory.join(embedding_filename(model)),
            )
        } else {
            (
                source(VOCABULARY).get_sync()?,
                source(&embedding_filename(model)).get_sync()?,
            )
        };
        let tags = parse_vocabulary(
            &std::fs::read_to_string(&vocabulary)
                .with_context(|| format!("reading {}", vocabulary.display()))?,
        )?;
        let embeddings = map(&embeddings)?;
        let baselines = baselines(&embeddings, tags.len(), model.dimensions())?;
        Ok(Self {
            model,
            tags,
            embeddings,
            baselines,
        })
    }

    pub fn model(&self) -> ImageEmbeddingModel {
        self.model
    }

    /// The `per_kind` best tags of each kind for a unit image vector from this set's model.
    pub fn rank(
        &self,
        image: &[f32],
        per_kind: usize,
        hide_blocked: bool,
    ) -> Result<TagRanking<'_>> {
        let dimensions = self.model.dimensions();
        ensure!(
            image.len() == dimensions,
            "image vector has {} dimensions, tags expect {dimensions}",
            image.len()
        );
        let file = SafeTensors::deserialize(&self.embeddings).context("reading tag embeddings")?;
        let rows = file.tensor("embeddings").context("tag embeddings")?;
        let similarities: Vec<f32> = rows
            .data()
            .par_chunks_exact(dimensions * 2)
            .map(|row| dot_f16(row, image))
            .collect();
        let mut ranking = TagRanking::default();
        for (index, tag) in self.tags.iter().enumerate() {
            if hide_blocked && tag.sensitivity == Sensitivity::Blocked {
                continue;
            }
            let similarity = similarities[index];
            let scored = ScoredTag {
                tag,
                similarity,
                score: similarity - self.baselines[index],
            };
            match tag.kind {
                TagKind::Simple => ranking.simple.push(scored),
                TagKind::Subject => ranking.subjects.push(scored),
                TagKind::Vibe => ranking.vibes.push(scored),
            }
        }
        for list in [
            &mut ranking.simple,
            &mut ranking.subjects,
            &mut ranking.vibes,
        ] {
            let count = per_kind.min(list.len());
            if count < list.len() {
                list.select_nth_unstable_by(count, |a, b| b.score.total_cmp(&a.score));
                list.truncate(count);
            }
            list.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
        Ok(ranking)
    }
}

fn local_directory() -> Option<std::path::PathBuf> {
    std::env::var_os("NICEGAL_TAGS_DIR").map(std::path::PathBuf::from)
}

fn embedding_filename(model: ImageEmbeddingModel) -> String {
    format!("{}.safetensors", model.id().replace('/', "--"))
}

fn source(filename: &str) -> ModelSource {
    ModelSource::pinned(REPOSITORY, REVISION, filename)
}

fn map(path: &Path) -> Result<Mmap> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: Hugging Face cache files are content-addressed. Explicit local tag exports must
    // likewise remain unchanged while loaded; the map is only read.
    unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))
}

fn parse_vocabulary(text: &str) -> Result<Vec<Tag>> {
    let mut lines = text.lines();
    ensure!(
        lines.next() == Some(HEADER),
        "unexpected tag vocabulary header"
    );
    lines
        .enumerate()
        .map(|(index, line)| {
            let mut fields = line.split('\t');
            let (Some(term), Some(kind), Some(source), Some(sensitivity), None) = (
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
            ) else {
                bail!(
                    "tag vocabulary line {} does not have four columns",
                    index + 2
                );
            };
            Ok(Tag {
                term: term.to_owned(),
                kind: match kind {
                    "simple" => TagKind::Simple,
                    "subject" => TagKind::Subject,
                    "vibe" => TagKind::Vibe,
                    other => bail!("unknown tag kind {other:?}"),
                },
                source: match source {
                    "metaclip" => TagSource::Metaclip,
                    "wordnet" => TagSource::Wordnet,
                    other => bail!("unknown tag source {other:?}"),
                },
                sensitivity: match sensitivity {
                    "ok" => Sensitivity::Ok,
                    "mature" => Sensitivity::Mature,
                    "blocked" => Sensitivity::Blocked,
                    other => bail!("unknown tag sensitivity {other:?}"),
                },
            })
        })
        .collect()
}

/// Check the embedding file's shapes and compute each tag's reference similarity.
fn baselines(embeddings: &[u8], tags: usize, dimensions: usize) -> Result<Vec<f32>> {
    let file = SafeTensors::deserialize(embeddings).context("reading tag embeddings")?;
    let rows = file.tensor("embeddings").context("tag embeddings")?;
    ensure!(
        rows.dtype() == Dtype::F16 && rows.shape() == [tags, dimensions],
        "tag embeddings are {:?} {:?}, expected F16 [{tags}, {dimensions}]",
        rows.dtype(),
        rows.shape()
    );
    let mean = file.tensor("reference_mean").context("reference mean")?;
    ensure!(
        mean.dtype() == Dtype::F32 && mean.shape() == [dimensions],
        "reference mean is {:?} {:?}, expected F32 [{dimensions}]",
        mean.dtype(),
        mean.shape()
    );
    let mean: Vec<f32> = mean
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    Ok(rows
        .data()
        .par_chunks_exact(dimensions * 2)
        .map(|row| dot_f16(row, &mean))
        .collect())
}

/// Dot product of a little-endian float16 row with `vector`.
fn dot_f16(row: &[u8], vector: &[f32]) -> f32 {
    row.as_chunks::<2>()
        .0
        .iter()
        .zip(vector)
        .map(|(bytes, value)| f16::from_le_bytes(*bytes).to_f32() * value)
        .sum()
}

impl std::fmt::Debug for TagSet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TagSet")
            .field("model", &self.model)
            .field("tags", &self.tags.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_requires_the_header_and_four_known_columns() {
        let tags = parse_vocabulary(
            "term\tkind\tsource\tsensitivity\ncozy\tvibe\tmetaclip\tok\nokapi\tsubject\twordnet\tok\n",
        )
        .unwrap();
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[1].kind, TagKind::Subject);
        assert!(parse_vocabulary("cozy\tvibe\tmetaclip\tok\n").is_err());
        assert!(parse_vocabulary(&format!("{HEADER}\ncozy\tvibe\tmetaclip\n")).is_err());
        assert!(parse_vocabulary(&format!("{HEADER}\ncozy\tvibe\tmetaclip\tok\textra\n")).is_err());
        assert!(parse_vocabulary(&format!("{HEADER}\ncozy\tmood\tmetaclip\tok\n")).is_err());
    }

    #[test]
    #[ignore = "downloads the pinned published vocabulary and every supported model's tags"]
    fn published_tags_load_for_every_supported_model() {
        assert!(local_directory().is_none());
        let models: Vec<_> = ImageEmbeddingModel::ALL
            .into_iter()
            .filter(|model| TagSet::supports(*model))
            .collect();
        assert_eq!(models.len(), 6);
        for model in models {
            let tags = TagSet::load(model).unwrap();
            assert!(!tags.tags.is_empty());
            assert_eq!(tags.baselines.len(), tags.tags.len());
        }
    }
}
