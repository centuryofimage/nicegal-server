use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use image::imageops::{FilterType, resize, rotate90};
use image::{Rgb, RgbImage};
use ort::session::Session;
use ort::value::TensorRef;
use serde::Deserialize;
use tracing::{Span, debug, field, instrument};

const DETECTION_TARGET_SIDE: u32 = 736;
/// Default longest detector input side. Lower values reduce work but may miss small text.
const DETECTION_MAX_SIDE: u32 = 960;
const DETECTION_MIN_MAX_SIDE: u32 = 320;
const DETECTION_LIMIT_MAX_SIDE: u32 = 8_192;
const RECOGNITION_MAX_WIDTH: usize = 3_200;
const MIN_COMPONENT_PIXELS: usize = 3;
/// Recognition maps 0..=255 onto -1..=1, so `v / 127.5 - 1.0` folds to one multiply-add.
const RECOGNITION_SCALE: f32 = 1.0 / 127.5;

#[derive(Debug, Clone, Copy)]
pub struct PaddleOcrOptions {
    pub recognition_batch_size: usize,
    pub minimum_recognition_score: f32,
    /// Longest side the detector input is scaled to fit. Lower trades recall on small text for
    /// close to quadratic savings in detection time; see [`DETECTION_MAX_SIDE`].
    pub detection_max_side: u32,
}

impl Default for PaddleOcrOptions {
    fn default() -> Self {
        Self {
            recognition_batch_size: 8,
            minimum_recognition_score: 0.5,
            detection_max_side: DETECTION_MAX_SIDE,
        }
    }
}

impl PaddleOcrOptions {
    pub fn validate(self) -> Result<Self> {
        if self.recognition_batch_size == 0 {
            bail!("PaddleOCR recognition batch size must be greater than zero");
        }
        if !self.minimum_recognition_score.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_recognition_score)
        {
            bail!("PaddleOCR minimum recognition score must be between zero and one");
        }
        if !(DETECTION_MIN_MAX_SIDE..=DETECTION_LIMIT_MAX_SIDE).contains(&self.detection_max_side) {
            bail!(
                "PaddleOCR detection max side must be between {DETECTION_MIN_MAX_SIDE} and {DETECTION_LIMIT_MAX_SIDE}"
            );
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PaddleOcrOutput {
    pub width: u32,
    pub height: u32,
    pub contents: String,
    pub lines: usize,
}

#[derive(Debug)]
pub(super) struct PaddleOcrConfig {
    detector: DetectorConfig,
    recognizer: RecognizerConfig,
}

#[derive(Debug)]
struct DetectorConfig {
    threshold: f32,
    box_threshold: f32,
    max_candidates: usize,
    unclip_ratio: f32,
    mean: [f32; 3],
    std: [f32; 3],
}

#[derive(Debug)]
struct RecognizerConfig {
    channels: usize,
    height: usize,
    base_width: usize,
    characters: Vec<String>,
}

#[derive(Deserialize)]
struct ModelConfig<P> {
    #[serde(rename = "PreProcess")]
    preprocess: PreprocessConfig,
    #[serde(rename = "PostProcess")]
    postprocess: P,
}

#[derive(Deserialize)]
struct PreprocessConfig {
    transform_ops: Vec<TransformConfig>,
}

#[derive(Deserialize)]
struct TransformConfig {
    #[serde(rename = "NormalizeImage")]
    normalize: Option<NormalizeConfig>,
    #[serde(rename = "RecResizeImg")]
    resize: Option<RecognizerResize>,
}

#[derive(Deserialize)]
struct NormalizeConfig {
    mean: [f32; 3],
    std: [f32; 3],
}

#[derive(Deserialize)]
struct DetectorPostprocess {
    thresh: f32,
    box_thresh: f32,
    max_candidates: usize,
    unclip_ratio: f32,
}

#[derive(Deserialize)]
struct RecognizerResize {
    image_shape: [usize; 3],
}

#[derive(Deserialize)]
struct RecognizerPostprocess {
    character_dict: Vec<String>,
}

impl PaddleOcrConfig {
    pub(super) fn load(detection_path: &Path, recognition_path: &Path) -> Result<Self> {
        let detection = fs::read_to_string(detection_path).with_context(|| {
            format!(
                "reading PaddleOCR detection configuration: {}",
                detection_path.display()
            )
        })?;
        let recognition = fs::read_to_string(recognition_path).with_context(|| {
            format!(
                "reading PaddleOCR recognition configuration: {}",
                recognition_path.display()
            )
        })?;

        Ok(Self {
            detector: parse_detector_config(&detection)?,
            recognizer: parse_recognizer_config(&recognition)?,
        })
    }
}

fn parse_detector_config(document: &str) -> Result<DetectorConfig> {
    let config: ModelConfig<DetectorPostprocess> =
        yaml_serde::from_str(document).context("invalid PaddleOCR detection configuration")?;
    let normalize = config
        .preprocess
        .transform_ops
        .into_iter()
        .find_map(|op| op.normalize)
        .context("PaddleOCR detection configuration is missing NormalizeImage")?;
    Ok(DetectorConfig {
        threshold: config.postprocess.thresh,
        box_threshold: config.postprocess.box_thresh,
        max_candidates: config.postprocess.max_candidates,
        unclip_ratio: config.postprocess.unclip_ratio,
        mean: normalize.mean,
        std: normalize.std,
    })
}

fn parse_recognizer_config(document: &str) -> Result<RecognizerConfig> {
    let config: ModelConfig<RecognizerPostprocess> =
        yaml_serde::from_str(document).context("invalid PaddleOCR recognition configuration")?;
    let image_shape = config
        .preprocess
        .transform_ops
        .into_iter()
        .find_map(|op| op.resize)
        .context("PaddleOCR recognition configuration is missing RecResizeImg")?
        .image_shape;
    if image_shape[0] != 3 || image_shape[1] == 0 || image_shape[2] == 0 {
        bail!("PaddleOCR recognition image_shape must be [3, height, width]");
    }

    let characters = config.postprocess.character_dict;
    if characters.is_empty() {
        bail!("PaddleOCR recognition configuration has no character_dict entries");
    }

    Ok(RecognizerConfig {
        channels: image_shape[0],
        height: image_shape[1],
        base_width: image_shape[2],
        characters,
    })
}

pub(super) struct PaddleOcrEngine<'a> {
    detection: &'a mut Session,
    recognition: &'a mut Session,
    config: &'a PaddleOcrConfig,
    scratch: &'a mut PaddleOcrScratch,
}

#[derive(Default)]
pub(super) struct PaddleOcrScratch {
    detector_input: Vec<f32>,
    recognizer_input: Vec<f32>,
}

impl<'a> PaddleOcrEngine<'a> {
    pub(super) fn new(
        detection: &'a mut Session,
        recognition: &'a mut Session,
        config: &'a PaddleOcrConfig,
        scratch: &'a mut PaddleOcrScratch,
    ) -> Self {
        Self {
            detection,
            recognition,
            config,
            scratch,
        }
    }

    #[instrument(
        name = "ocr_image",
        skip_all,
        fields(width = image.width(), height = image.height(), boxes = field::Empty, lines = field::Empty)
    )]
    pub(super) fn scan(
        &mut self,
        image: &RgbImage,
        options: PaddleOcrOptions,
    ) -> Result<PaddleOcrOutput> {
        let options = options.validate()?;
        let boxes = self.detect(image, options)?;
        Span::current().record("boxes", boxes.len());
        if boxes.is_empty() {
            Span::current().record("lines", 0);
            return Ok(PaddleOcrOutput {
                width: image.width(),
                height: image.height(),
                contents: String::new(),
                lines: 0,
            });
        }

        let crop_started = Instant::now();
        let crops = boxes
            .iter()
            .filter_map(|text_box| crop_text_line(image, *text_box))
            .collect::<Vec<_>>();
        debug!(
            stage = "crop_text_lines",
            elapsed_ms = crop_started.elapsed().as_secs_f64() * 1_000.0,
            boxes = boxes.len(),
            crops = crops.len(),
            "PaddleOCR stage timing"
        );
        let recognized = self.recognize(&crops, options.recognition_batch_size)?;
        let lines = recognized
            .into_iter()
            .filter(|line| {
                !line.text.trim().is_empty() && line.score >= options.minimum_recognition_score
            })
            .map(|line| line.text)
            .collect::<Vec<_>>();
        Span::current().record("lines", lines.len());
        debug!("PaddleOCR image inference complete");
        Ok(PaddleOcrOutput {
            width: image.width(),
            height: image.height(),
            contents: lines.join("\n"),
            lines: lines.len(),
        })
    }

    fn detect(&mut self, image: &RgbImage, options: PaddleOcrOptions) -> Result<Vec<TextBox>> {
        let preprocess_started = Instant::now();
        let previous_capacity = self.scratch.detector_input.capacity();
        let resized = detection_input(
            image,
            &self.config.detector,
            options.detection_max_side,
            &mut self.scratch.detector_input,
        )?;
        let resized_width = resized.width();
        let resized_height = resized.height();
        let input_bytes = self.scratch.detector_input.len() * size_of::<f32>();
        let input = TensorRef::from_array_view((
            vec![
                1_i64,
                3,
                i64::from(resized_height),
                i64::from(resized_width),
            ],
            self.scratch.detector_input.as_slice(),
        ))?;
        debug!(
            stage = "detection_preprocess",
            elapsed_ms = preprocess_started.elapsed().as_secs_f64() * 1_000.0,
            input_width = resized_width,
            input_height = resized_height,
            input_bytes,
            input_capacity_bytes = self.scratch.detector_input.capacity() * size_of::<f32>(),
            input_reallocated = self.scratch.detector_input.capacity() != previous_capacity,
            "PaddleOCR stage timing"
        );
        let inference_started = Instant::now();
        let outputs = self
            .detection
            .run(ort::inputs![input])
            .context("running PaddleOCR detection model")?;
        debug!(
            stage = "detection_inference",
            elapsed_ms = inference_started.elapsed().as_secs_f64() * 1_000.0,
            input_width = resized_width,
            input_height = resized_height,
            "PaddleOCR stage timing"
        );
        let postprocess_started = Instant::now();
        let output = outputs
            .values()
            .next()
            .ok_or_else(|| anyhow!("PaddleOCR detection model returned no output"))?;
        let (shape, probabilities) = output
            .try_extract_tensor::<f32>()
            .context("reading PaddleOCR detection output")?;
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
            bail!(
                "PaddleOCR detection output must have shape [1, 1, height, width], got {shape:?}"
            );
        }
        let map_height = usize::try_from(shape[2]).context("invalid detection output height")?;
        let map_width = usize::try_from(shape[3]).context("invalid detection output width")?;
        if probabilities.len() != map_height.saturating_mul(map_width) {
            bail!("PaddleOCR detection output size does not match its shape");
        }
        let boxes = boxes_from_probability_map(
            probabilities,
            map_width,
            map_height,
            image.width(),
            image.height(),
            &self.config.detector,
        );
        debug!(
            stage = "detection_postprocess",
            elapsed_ms = postprocess_started.elapsed().as_secs_f64() * 1_000.0,
            map_width,
            map_height,
            boxes = boxes.len(),
            "PaddleOCR stage timing"
        );
        Ok(boxes)
    }

    fn recognize(&mut self, crops: &[RgbImage], batch_size: usize) -> Result<Vec<RecognizedLine>> {
        let ordering_started = Instant::now();
        let mut order = (0..crops.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            aspect_ratio(&crops[*left]).total_cmp(&aspect_ratio(&crops[*right]))
        });
        debug!(
            stage = "recognition_ordering",
            elapsed_ms = ordering_started.elapsed().as_secs_f64() * 1_000.0,
            crops = crops.len(),
            configured_batch_size = batch_size,
            batches = crops.len().div_ceil(batch_size),
            "PaddleOCR stage timing"
        );
        let mut recognized = vec![RecognizedLine::default(); crops.len()];

        for batch in order.chunks(batch_size) {
            let max_ratio = batch.iter().map(|index| aspect_ratio(&crops[*index])).fold(
                self.config.recognizer.base_width as f32 / self.config.recognizer.height as f32,
                f32::max,
            );
            let width = ((self.config.recognizer.height as f32 * max_ratio).ceil() as usize)
                .clamp(self.config.recognizer.base_width, RECOGNITION_MAX_WIDTH);
            let preprocess_started = Instant::now();
            let previous_capacity = self.scratch.recognizer_input.capacity();
            recognition_input(
                crops,
                batch,
                width,
                &self.config.recognizer,
                &mut self.scratch.recognizer_input,
            );
            let input_bytes = self.scratch.recognizer_input.len() * size_of::<f32>();
            let tensor = TensorRef::from_array_view((
                vec![
                    i64::try_from(batch.len())?,
                    i64::try_from(self.config.recognizer.channels)?,
                    i64::try_from(self.config.recognizer.height)?,
                    i64::try_from(width)?,
                ],
                self.scratch.recognizer_input.as_slice(),
            ))?;
            debug!(
                stage = "recognition_preprocess",
                elapsed_ms = preprocess_started.elapsed().as_secs_f64() * 1_000.0,
                batch_size = batch.len(),
                configured_batch_size = batch_size,
                input_width = width,
                input_height = self.config.recognizer.height,
                input_bytes,
                input_capacity_bytes = self.scratch.recognizer_input.capacity() * size_of::<f32>(),
                input_reallocated = self.scratch.recognizer_input.capacity() != previous_capacity,
                "PaddleOCR stage timing"
            );
            let inference_started = Instant::now();
            let outputs = self
                .recognition
                .run(ort::inputs![tensor])
                .context("running PaddleOCR recognition model")?;
            debug!(
                stage = "recognition_inference",
                elapsed_ms = inference_started.elapsed().as_secs_f64() * 1_000.0,
                batch_size = batch.len(),
                configured_batch_size = batch_size,
                input_width = width,
                input_height = self.config.recognizer.height,
                "PaddleOCR stage timing"
            );
            let decode_started = Instant::now();
            let output = outputs
                .values()
                .next()
                .ok_or_else(|| anyhow!("PaddleOCR recognition model returned no output"))?;
            let (shape, logits) = output
                .try_extract_tensor::<f32>()
                .context("reading PaddleOCR recognition output")?;
            if shape.len() != 3 || usize::try_from(shape[0]).ok() != Some(batch.len()) {
                bail!(
                    "PaddleOCR recognition output must have shape [batch, steps, classes], got {shape:?}"
                );
            }
            let steps = usize::try_from(shape[1]).context("invalid recognition step count")?;
            let classes = usize::try_from(shape[2]).context("invalid recognition class count")?;
            let space_index = match classes.checked_sub(self.config.recognizer.characters.len()) {
                Some(1) => None,
                Some(2) => Some(classes - 1),
                _ => {
                    bail!(
                        "PaddleOCR recognition output has {classes} classes but its character_dict has {} entries",
                        self.config.recognizer.characters.len()
                    );
                }
            };
            for (batch_index, crop_index) in batch.iter().copied().enumerate() {
                recognized[crop_index] = decode_ctc(
                    &logits[batch_index * steps * classes..(batch_index + 1) * steps * classes],
                    steps,
                    classes,
                    &self.config.recognizer.characters,
                    space_index,
                );
            }
            debug!(
                stage = "recognition_decode",
                elapsed_ms = decode_started.elapsed().as_secs_f64() * 1_000.0,
                batch_size = batch.len(),
                steps,
                classes,
                "PaddleOCR stage timing"
            );
        }
        Ok(recognized)
    }
}

fn detection_input(
    image: &RgbImage,
    config: &DetectorConfig,
    max_side_limit: u32,
    input: &mut Vec<f32>,
) -> Result<RgbImage> {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        bail!("cannot OCR an empty image");
    }
    let min_side = width.min(height) as f32;
    let max_side = width.max(height) as f32;
    let mut ratio = if min_side < DETECTION_TARGET_SIDE as f32 {
        DETECTION_TARGET_SIDE as f32 / min_side
    } else {
        1.0
    };
    if max_side * ratio > max_side_limit as f32 {
        ratio = max_side_limit as f32 / max_side;
    }
    let resize_width = aligned_dimension(width, ratio);
    let resize_height = aligned_dimension(height, ratio);
    let resized = resize(image, resize_width, resize_height, FilterType::Triangle);
    let plane = usize::try_from(resize_width)? * usize::try_from(resize_height)?;
    input.resize(plane * 3, 0.0);
    // `(v / 255 - mean) / std` is two divisions per channel per pixel — over five million of them
    // on a 1.7 MP input. The same line as one multiply-add folds both constants in beforehand.
    let scale = [
        1.0 / (255.0 * config.std[0]),
        1.0 / (255.0 * config.std[1]),
        1.0 / (255.0 * config.std[2]),
    ];
    let bias = [
        -config.mean[0] / config.std[0],
        -config.mean[1] / config.std[1],
        -config.mean[2] / config.std[2],
    ];
    let (blue, rest) = input.split_at_mut(plane);
    let (green, red) = rest.split_at_mut(plane);
    for (index, pixel) in resized.pixels().enumerate() {
        // PaddleOCR's DecodeImage uses BGR, so the Image crate's RGB pixels are reversed here.
        blue[index] = f32::from(pixel[2]).mul_add(scale[0], bias[0]);
        green[index] = f32::from(pixel[1]).mul_add(scale[1], bias[1]);
        red[index] = f32::from(pixel[0]).mul_add(scale[2], bias[2]);
    }
    Ok(resized)
}

fn aligned_dimension(value: u32, ratio: f32) -> u32 {
    let scaled = value as f32 * ratio;
    ((scaled / 32.0).round().max(1.0) as u32) * 32
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy)]
struct TextBox {
    /// Corners in principal-axis order: min/min, max/min, max/max, min/max.
    points: [Point; 4],
}

fn boxes_from_probability_map(
    probabilities: &[f32],
    width: usize,
    height: usize,
    destination_width: u32,
    destination_height: u32,
    config: &DetectorConfig,
) -> Vec<TextBox> {
    let mut mask = probabilities
        .iter()
        .map(|probability| u8::from(*probability > config.threshold))
        .collect::<Vec<_>>();
    let mut boxes = Vec::new();
    let mut queue = VecDeque::new();
    let mut component = Vec::new();

    for start in 0..mask.len() {
        if mask[start] == 0 || boxes.len() >= config.max_candidates {
            continue;
        }
        mask[start] = 0;
        queue.push_back(start);
        component.clear();
        while let Some(index) = queue.pop_front() {
            component.push(index);
            let x = index % width;
            let y = index / width;
            for offset_y in -1_i32..=1 {
                for offset_x in -1_i32..=1 {
                    if offset_x == 0 && offset_y == 0 {
                        continue;
                    }
                    let neighbor_x = x as i32 + offset_x;
                    let neighbor_y = y as i32 + offset_y;
                    if neighbor_x < 0
                        || neighbor_y < 0
                        || neighbor_x >= width as i32
                        || neighbor_y >= height as i32
                    {
                        continue;
                    }
                    let neighbor = neighbor_y as usize * width + neighbor_x as usize;
                    if mask[neighbor] != 0 {
                        mask[neighbor] = 0;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        if component.len() < MIN_COMPONENT_PIXELS {
            continue;
        }
        let score = component
            .iter()
            .map(|index| probabilities[*index])
            .sum::<f32>()
            / component.len() as f32;
        if score < config.box_threshold {
            continue;
        }
        let Some(mut text_box) = oriented_component_box(&component, width, config.unclip_ratio)
        else {
            continue;
        };
        let scale_x = destination_width as f32 / width as f32;
        let scale_y = destination_height as f32 / height as f32;
        for point in &mut text_box.points {
            point.x = (point.x * scale_x).clamp(0.0, destination_width.saturating_sub(1) as f32);
            point.y = (point.y * scale_y).clamp(0.0, destination_height.saturating_sub(1) as f32);
        }
        if box_width(text_box) > 3.0 && box_height(text_box) > 3.0 {
            boxes.push(text_box);
        }
    }

    sort_into_reading_order(&mut boxes);
    boxes
}

/// Order boxes top-to-bottom, then left-to-right within a row.
///
/// Row membership is assigned before horizontal sorting because a tolerance-based comparator is
/// not transitive. `total_cmp` also gives malformed coordinates a deterministic order.
fn sort_into_reading_order(boxes: &mut [TextBox]) {
    const ROW_TOLERANCE: f32 = 10.0;

    boxes.sort_by(|left, right| {
        left.points[0]
            .y
            .total_cmp(&right.points[0].y)
            .then_with(|| left.points[0].x.total_cmp(&right.points[0].x))
    });
    let mut start = 0;
    while start < boxes.len() {
        let row_top = boxes[start].points[0].y;
        let mut end = start + 1;
        while end < boxes.len() && (boxes[end].points[0].y - row_top).abs() < ROW_TOLERANCE {
            end += 1;
        }
        boxes[start..end].sort_by(|left, right| left.points[0].x.total_cmp(&right.points[0].x));
        start = end;
    }
}

fn oriented_component_box(component: &[usize], width: usize, unclip_ratio: f32) -> Option<TextBox> {
    let count = component.len() as f32;
    let (sum_x, sum_y) = component.iter().fold((0.0_f32, 0.0_f32), |sum, index| {
        (
            sum.0 + (index % width) as f32,
            sum.1 + (index / width) as f32,
        )
    });
    let center = Point {
        x: sum_x / count,
        y: sum_y / count,
    };
    let (cov_xx, cov_xy, cov_yy) =
        component
            .iter()
            .fold((0.0_f32, 0.0_f32, 0.0_f32), |covariance, index| {
                let x = (index % width) as f32 - center.x;
                let y = (index / width) as f32 - center.y;
                (
                    covariance.0 + x * x,
                    covariance.1 + x * y,
                    covariance.2 + y * y,
                )
            });
    let angle = 0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy);
    // The dominant covariance eigenvector gives the component's principal text axis.
    let axis_x = Point {
        x: angle.cos(),
        y: angle.sin(),
    };
    let axis_y = Point {
        x: -axis_x.y,
        y: axis_x.x,
    };
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for index in component {
        let delta_x = (*index % width) as f32 - center.x;
        let delta_y = (*index / width) as f32 - center.y;
        // Project into the principal-axis basis to measure an oriented bounding rectangle.
        let projected_x = delta_x * axis_x.x + delta_y * axis_x.y;
        let projected_y = delta_x * axis_y.x + delta_y * axis_y.y;
        min_x = min_x.min(projected_x);
        max_x = max_x.max(projected_x);
        min_y = min_y.min(projected_y);
        max_y = max_y.max(projected_y);
    }
    let box_width = max_x - min_x + 1.0;
    let box_height = max_y - min_y + 1.0;
    if box_width.min(box_height) < MIN_COMPONENT_PIXELS as f32 {
        return None;
    }
    // DB-style "unclip": expand by area × ratio / perimeter before mapping back to image space.
    let expansion = box_width * box_height * unclip_ratio / (2.0 * (box_width + box_height));
    min_x -= expansion;
    max_x += expansion;
    min_y -= expansion;
    max_y += expansion;

    let point = |projected_x: f32, projected_y: f32| Point {
        x: center.x + projected_x * axis_x.x + projected_y * axis_y.x,
        y: center.y + projected_x * axis_x.y + projected_y * axis_y.y,
    };
    Some(TextBox {
        points: [
            point(min_x, min_y),
            point(max_x, min_y),
            point(max_x, max_y),
            point(min_x, max_y),
        ],
    })
}

fn box_width(text_box: TextBox) -> f32 {
    distance(text_box.points[0], text_box.points[1])
        .max(distance(text_box.points[2], text_box.points[3]))
}

fn box_height(text_box: TextBox) -> f32 {
    distance(text_box.points[0], text_box.points[3])
        .max(distance(text_box.points[1], text_box.points[2]))
}

fn distance(left: Point, right: Point) -> f32 {
    (left.x - right.x).hypot(left.y - right.y)
}

fn crop_text_line(image: &RgbImage, text_box: TextBox) -> Option<RgbImage> {
    let width = box_width(text_box).round().max(1.0) as u32;
    let height = box_height(text_box).round().max(1.0) as u32;
    if width <= 1 || height <= 1 {
        return None;
    }
    let mut crop = RgbImage::new(width, height);
    for destination_y in 0..height {
        let v = if height == 1 {
            0.0
        } else {
            destination_y as f32 / (height - 1) as f32
        };
        for destination_x in 0..width {
            let u = if width == 1 {
                0.0
            } else {
                destination_x as f32 / (width - 1) as f32
            };
            let top = interpolate(text_box.points[0], text_box.points[1], u);
            let bottom = interpolate(text_box.points[3], text_box.points[2], u);
            let source = interpolate(top, bottom, v);
            crop.put_pixel(destination_x, destination_y, sample_bilinear(image, source));
        }
    }
    if crop.height() as f32 / crop.width() as f32 >= 1.5 {
        Some(rotate90(&crop))
    } else {
        Some(crop)
    }
}

fn interpolate(start: Point, end: Point, amount: f32) -> Point {
    Point {
        x: start.x + (end.x - start.x) * amount,
        y: start.y + (end.y - start.y) * amount,
    }
}

fn sample_bilinear(image: &RgbImage, point: Point) -> Rgb<u8> {
    let x = point.x.clamp(0.0, image.width().saturating_sub(1) as f32);
    let y = point.y.clamp(0.0, image.height().saturating_sub(1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let x_weight = x - x0 as f32;
    let y_weight = y - y0 as f32;
    let mut channels = [0_u8; 3];
    for (channel, value) in channels.iter_mut().enumerate() {
        let top = f32::from(image.get_pixel(x0, y0)[channel]) * (1.0 - x_weight)
            + f32::from(image.get_pixel(x1, y0)[channel]) * x_weight;
        let bottom = f32::from(image.get_pixel(x0, y1)[channel]) * (1.0 - x_weight)
            + f32::from(image.get_pixel(x1, y1)[channel]) * x_weight;
        *value = (top * (1.0 - y_weight) + bottom * y_weight).round() as u8;
    }
    Rgb(channels)
}

fn aspect_ratio(image: &RgbImage) -> f32 {
    image.width() as f32 / image.height().max(1) as f32
}

fn recognition_input(
    crops: &[RgbImage],
    batch: &[usize],
    width: usize,
    config: &RecognizerConfig,
    input: &mut Vec<f32>,
) {
    let plane = config.height * width;
    let item_size = config.channels * plane;
    input.resize(batch.len() * item_size, 0.0);
    input.fill(0.0);
    for (batch_index, crop_index) in batch.iter().copied().enumerate() {
        let crop = &crops[crop_index];
        let resized_width =
            ((config.height as f32 * aspect_ratio(crop)).ceil() as usize).clamp(1, width);
        let resized = resize(
            crop,
            resized_width as u32,
            config.height as u32,
            FilterType::Triangle,
        );
        let item_offset = batch_index * item_size;
        for (pixel_index, pixel) in resized.pixels().enumerate() {
            let row = pixel_index / resized_width;
            let column = pixel_index % resized_width;
            let destination = row * width + column;
            input[item_offset + destination] = f32::from(pixel[2]).mul_add(RECOGNITION_SCALE, -1.0);
            input[item_offset + plane + destination] =
                f32::from(pixel[1]).mul_add(RECOGNITION_SCALE, -1.0);
            input[item_offset + plane * 2 + destination] =
                f32::from(pixel[0]).mul_add(RECOGNITION_SCALE, -1.0);
        }
    }
}

#[derive(Debug, Clone, Default)]
struct RecognizedLine {
    text: String,
    score: f32,
}

fn decode_ctc(
    logits: &[f32],
    steps: usize,
    classes: usize,
    characters: &[String],
    space_index: Option<usize>,
) -> RecognizedLine {
    let mut text = String::new();
    let mut score = 0.0_f32;
    let mut selected = 0_usize;
    let mut previous = usize::MAX;
    for step in 0..steps {
        let values = &logits[step * classes..(step + 1) * classes];
        let (index, confidence) = values
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Less))
            .unwrap_or((0, 0.0));
        if index != 0 && index != previous {
            if Some(index) == space_index {
                text.push(' ');
            } else {
                text.push_str(&characters[index - 1]);
            }
            score += confidence;
            selected += 1;
        }
        previous = index;
    }
    RecognizedLine {
        text,
        score: if selected == 0 {
            0.0
        } else {
            score / selected as f32
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_box(x: f32, y: f32) -> TextBox {
        TextBox {
            points: [
                Point { x, y },
                Point { x: x + 8.0, y },
                Point {
                    x: x + 8.0,
                    y: y + 8.0,
                },
                Point { x, y: y + 8.0 },
            ],
        }
    }

    /// Exercises row chains that make pairwise tolerance comparisons intransitive.
    #[test]
    fn reading_order_survives_scattered_boxes() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2_000 {
            let count = 20 + (next() % 60) as usize;
            let mut boxes = (0..count)
                .map(|_| text_box((next() % 1000) as f32 / 10.0, (next() % 1000) as f32 / 10.0))
                .collect::<Vec<_>>();
            sort_into_reading_order(&mut boxes);
            assert_eq!(boxes.len(), count);
            assert!(
                boxes.windows(2).all(|pair| {
                    let (above, below) = (pair[0].points[0].y, pair[1].points[0].y);
                    above <= below || (below - above).abs() < 10.0
                }),
                "y may only go backwards inside one row"
            );
        }
    }

    #[test]
    fn reading_order_sorts_a_row_left_to_right() {
        let mut boxes = vec![
            text_box(50.0, 4.0),
            text_box(10.0, 0.0),
            text_box(30.0, 8.0),
            text_box(20.0, 40.0),
        ];
        sort_into_reading_order(&mut boxes);
        let xs = boxes
            .iter()
            .map(|text_box| text_box.points[0].x)
            .collect::<Vec<_>>();
        assert_eq!(xs, [10.0, 30.0, 50.0, 20.0]);
    }

    #[test]
    fn reading_order_tolerates_non_finite_coordinates() {
        let mut boxes = vec![
            text_box(10.0, f32::NAN),
            text_box(20.0, 0.0),
            text_box(30.0, f32::INFINITY),
        ];
        sort_into_reading_order(&mut boxes);
        assert_eq!(boxes.len(), 3);
    }

    #[test]
    fn recognition_config_preserves_quoted_characters() -> Result<()> {
        let config = r#"
PreProcess:
  transform_ops:
  - RecResizeImg:
      image_shape: [3, 48, 320]
PostProcess:
  character_dict:
  - '!'
  - ''''
  - \
  - ' '
  - "\u4E2D"
"#;
        let parsed = parse_recognizer_config(config)?;
        assert_eq!(parsed.characters, ["!", "'", "\\", " ", "中"]);
        assert_eq!(
            (parsed.channels, parsed.height, parsed.base_width),
            (3, 48, 320)
        );
        Ok(())
    }

    #[test]
    fn recognition_config_rejects_invalid_shapes_and_empty_dictionary() {
        for shape in [
            "[]",
            "[3]",
            "[3, 48]",
            "[3, 48, 320, 1]",
            "[1, 48, 320]",
            "[3, 0, 320]",
        ] {
            let config = format!(
                "PreProcess:\n  transform_ops:\n  - RecResizeImg:\n      image_shape: {shape}\nPostProcess:\n  character_dict: ['a']"
            );
            assert!(
                parse_recognizer_config(&config).is_err(),
                "accepted {shape}"
            );
        }
        assert!(parse_recognizer_config(
            "PreProcess:\n  transform_ops:\n  - RecResizeImg:\n      image_shape: [3, 48, 320]\nPostProcess:\n  character_dict: []"
        ).is_err());
    }

    #[test]
    fn detection_config_reads_nested_yaml_and_aliases() -> Result<()> {
        let config = r#"
unrelated: {thresh: 99}
PreProcess:
  transform_ops:
  - DecodeImage: {img_mode: BGR}
  - NormalizeImage:
      mean: &values [0.1, 0.2, 0.3]
      std: *values
PostProcess:
  thresh: 0.2
  box_thresh: 0.45
  max_candidates: 3000
  unclip_ratio: 1.4
"#;
        let parsed = parse_detector_config(config)?;
        assert_eq!(parsed.threshold, 0.2);
        assert_eq!(parsed.mean, [0.1, 0.2, 0.3]);
        assert_eq!(parsed.std, parsed.mean);
        assert!(parse_detector_config(&config.replace("[0.1, 0.2, 0.3]", "[0.1]")).is_err());
        Ok(())
    }

    #[test]
    fn ctc_decoder_removes_blanks_and_repeated_classes() {
        let logits = [
            0.1, 0.8, 0.1, 0.0, // A
            0.1, 0.7, 0.2, 0.0, // repeated A
            0.9, 0.05, 0.05, 0.0, // blank
            0.1, 0.2, 0.7, 0.0, // B
        ];
        let result = decode_ctc(&logits, 4, 4, &["A".to_owned(), "B".to_owned()], Some(3));
        assert_eq!(result.text, "AB");
        assert!((result.score - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn probability_components_become_reading_order_boxes() {
        let mut map = vec![0.0_f32; 20 * 12];
        for y in 2..5 {
            for x in 10..18 {
                map[y * 20 + x] = 0.9;
            }
        }
        for y in 7..10 {
            for x in 1..9 {
                map[y * 20 + x] = 0.9;
            }
        }
        let boxes = boxes_from_probability_map(
            &map,
            20,
            12,
            200,
            120,
            &DetectorConfig {
                threshold: 0.2,
                box_threshold: 0.45,
                max_candidates: 3000,
                unclip_ratio: 1.4,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
        );
        assert_eq!(boxes.len(), 2);
        assert!(boxes[0].points[0].y < boxes[1].points[0].y);
    }
}
