#![forbid(unsafe_code)]
//! FFmpeg is confined to this synchronous reader on the caller's thread.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use camino::{Utf8Path, Utf8PathBuf};
use ffmpeg_the_third::{self as ffmpeg, codec, ffi, format, frame, media, software};

use crate::imaging::{ExifOrientation, Raster, fit_within};
use crate::video::{VideoMetadata, VideoSample, VideoSamplingOptions};

const MAX_PACKETS_PER_SAMPLE: usize = 20_000;
const MAX_SOURCE_PIXELS: u64 = 64 * 1024 * 1024;

pub(crate) enum ProbeBudget {
    /// Catalog metadata is best effort; MPEG program streams otherwise scan up to 5 MiB.
    Catalog,
    /// Decoding and detailed inspection need reliable stream metadata for seeking.
    Full,
}

impl ProbeBudget {
    fn size_bytes(&self) -> &'static str {
        match self {
            Self::Catalog => "65536",
            Self::Full => "5242880",
        }
    }
}

struct Interrupt {
    cancelled: Arc<AtomicBool>,
    aborted: Arc<AtomicBool>,
    deadline: Instant,
}

impl Interrupt {
    fn stopped(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
            || self.aborted.load(Ordering::Relaxed)
            || Instant::now() >= self.deadline
    }
    fn check(&self) -> Result<()> {
        ensure!(!self.stopped(), "video decoding cancelled or timed out");
        Ok(())
    }
}

pub(crate) struct VideoReader {
    path: Utf8PathBuf,
    input: format::context::Input,
    control: Arc<Interrupt>,
    decoder: ffmpeg::decoder::Video,
    metadata: VideoMetadata,
    stream: usize,
    time_base: ffmpeg::Rational,
    start: i64,
    orientation: ExifOrientation,
    aspect: ffmpeg::Rational,
    transfer: ffmpeg::color::TransferCharacteristic,
}

impl VideoReader {
    #[tracing::instrument(level = "trace", skip_all, fields(path = %path))]
    pub(crate) fn open(
        path: &Utf8Path,
        cancelled: Arc<AtomicBool>,
        aborted: Arc<AtomicBool>,
        probe_budget: ProbeBudget,
    ) -> Result<Self> {
        // Verify this is a local file before passing a string to FFmpeg's URL-oriented API.
        ensure!(
            std::fs::metadata(path)?.is_file(),
            "video source is not a file"
        );
        let control = Arc::new(Interrupt {
            cancelled,
            aborted,
            deadline: Instant::now() + Duration::from_secs(30),
        });
        control.check()?;
        let input = open_input(path, Arc::clone(&control), probe_budget)?;
        let stream = input
            .streams()
            .filter(|stream| {
                stream.parameters().medium() == media::Type::Video
                    && !stream
                        .disposition()
                        .contains(format::stream::Disposition::ATTACHED_PIC)
            })
            .min_by_key(|stream| {
                !stream
                    .disposition()
                    .contains(format::stream::Disposition::DEFAULT)
            })
            .context("no video stream")?;
        let parameters = stream.parameters();
        ensure!(
            codec::decoder::find(parameters.id()).is_some(),
            "no decoder in the selected FFmpeg build for {:?}",
            parameters.id()
        );
        validate_dimensions(parameters.width(), parameters.height())?;
        let orientation = display_orientation(&parameters)?;
        let mut context = codec::Context::from_parameters(stream.parameters())?;
        context.set_threading(codec::threading::Config::count(1));
        let decoder = context.decoder().video()?;
        let time_base = stream.time_base();
        ensure!(
            time_base.numerator() > 0 && time_base.denominator() > 0,
            "invalid video time base"
        );
        let start = if stream.start_time() == ffi::AV_NOPTS_VALUE {
            0
        } else {
            stream.start_time()
        };
        let duration_ms = if stream.duration() > 0 {
            u64::try_from(to_millis(stream.duration(), time_base)).ok()
        } else if input.duration() > 0 {
            u64::try_from(input.duration() / 1000).ok()
        } else {
            None
        }
        .filter(|value| *value > 0);
        let aspect = parameters.sample_aspect_ratio();
        let transfer = parameters.color_transfer_characteristic();
        let (width, height) =
            display_dimensions(parameters.width(), parameters.height(), aspect, orientation);
        let tags = input.metadata();
        let stream_tags = stream.metadata();
        let tag = |name: &str| {
            stream_tags
                .get(name)
                .or_else(|| tags.get(name))
                .map(|value| value.chars().take(4096).collect::<String>())
        };
        let rate = stream.avg_frame_rate();
        let metadata = VideoMetadata {
            width,
            height,
            duration_ms,
            frame_count: u32::try_from(stream.frames())
                .ok()
                .filter(|value| *value > 0),
            codec: decoder
                .codec()
                .map(|codec| codec.name().to_owned())
                .unwrap_or_default(),
            frame_rate: (rate.numerator() > 0 && rate.denominator() > 0)
                .then(|| f64::from(rate.numerator()) / f64::from(rate.denominator())),
            bit_rate: u64::try_from(input.bit_rate())
                .ok()
                .filter(|value| *value > 0),
            creation_time: tag("creation_time"),
            title: tag("title"),
            is_hdr: hdr_transfer(transfer),
        };
        let stream = stream.index();
        Ok(Self {
            path: path.to_owned(),
            input,
            control,
            decoder,
            metadata,
            stream,
            time_base,
            start,
            orientation,
            aspect,
            transfer,
        })
    }

    pub(crate) fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    #[tracing::instrument(level = "trace", skip_all, fields(path = %self.path, max_edge = options.max_edge, targets = options.seek_percentages.len()))]
    pub(crate) fn samples(&mut self, options: &VideoSamplingOptions) -> Result<Vec<VideoSample>> {
        let targets = sample_targets(self.metadata.duration_ms, &options.seek_percentages);
        // Keep a usable fallback before seeking: a failed seek can leave a demuxer
        // at an unspecified position. Unknown-duration files need only this sample.
        let first = self.read_sample(options.max_edge, true)?;
        if targets.is_empty() {
            return Ok(vec![first]);
        }
        let mut samples: Vec<VideoSample> = Vec::with_capacity(targets.len());
        if options.first_frame_poster {
            samples.push(first.clone());
        }
        for target in targets {
            self.control.check()?;
            let start_us = to_micros(self.start, self.time_base);
            let seek_us = start_us.saturating_add(target.saturating_mul(1000));
            if self.input.seek(seek_us, ..=seek_us).is_err() {
                if samples.is_empty() {
                    samples.push(first);
                }
                break;
            }
            self.decoder.flush();
            self.decoder.skip_frame(ffmpeg::Discard::NonKey);
            let keyframe = self.read_sample(options.max_edge, false);
            let sample = match keyframe.or_else(|key_error| {
                self.control.check()?;
                tracing::debug!(%key_error, target_ms = target, "retrying video sample without keyframe restriction");
                self.decoder.skip_frame(ffmpeg::Discard::None);
                self.input.seek(seek_us, ..=seek_us)?;
                self.decoder.flush();
                self.read_sample(options.max_edge, false)
            }) {
                Ok(sample) => sample,
                Err(error) => {
                    // A target can land beyond the video track while the container still
                    // advertises a longer duration. Keep frames decoded before that target.
                    // Cancellation and deadlines must still abort the request.
                    self.control.check()?;
                    tracing::debug!(%error, target_ms = target, "stopping video samples after a failed target");
                    if samples.is_empty() {
                        samples.push(first);
                    }
                    break;
                }
            };
            if !samples
                .iter()
                .any(|old| old.timestamp_ms == sample.timestamp_ms)
            {
                samples.push(sample);
            }
        }
        ensure!(!samples.is_empty(), "no usable video samples");
        Ok(samples)
    }

    #[tracing::instrument(level = "trace", skip_all, fields(path = %self.path, max_edge, allow_missing_timestamp))]
    fn read_sample(&mut self, max_edge: u32, allow_missing_timestamp: bool) -> Result<VideoSample> {
        let mut decoded = frame::Video::empty();
        let mut draining = false;
        for _ in 0..MAX_PACKETS_PER_SAMPLE {
            self.control.check()?;
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => return self.convert(&decoded, max_edge, allow_missing_timestamp),
                Err(ffmpeg::Error::Eof) => bail!("no usable frame before end of video"),
                Err(error) if again(error) => {}
                Err(error) => return Err(error).context("receiving video frame"),
            }
            ensure!(!draining, "video decoder requested input after EOF");
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut self.input) {
                Ok(()) if packet.stream() == self.stream => {
                    self.decoder
                        .send_packet(&packet)
                        .context("sending video packet")?;
                }
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) => {
                    self.decoder.send_eof()?;
                    draining = true;
                }
                Err(error) => return Err(error).context("reading video packet"),
            }
        }
        bail!("video frame scan exceeded packet budget")
    }

    #[tracing::instrument(level = "trace", skip_all, fields(path = %self.path, width = decoded.width(), height = decoded.height(), max_edge))]
    fn convert(
        &self,
        decoded: &frame::Video,
        max_edge: u32,
        allow_missing_timestamp: bool,
    ) -> Result<VideoSample> {
        validate_dimensions(decoded.width(), decoded.height())?;
        ensure!(
            decoded
                .side_data(frame::side_data::Type::DOVI_RPU_BUFFER)
                .is_none()
                && decoded
                    .side_data(frame::side_data::Type::DOVI_METADATA)
                    .is_none(),
            "Dolby Vision is not supported"
        );
        let timestamp_ms = match decoded.timestamp().or_else(|| decoded.pts()) {
            Some(timestamp) => {
                to_millis(timestamp.saturating_sub(self.start), self.time_base).max(0)
            }
            None if allow_missing_timestamp => 0,
            None => bail!("video frame has no timestamp"),
        };
        let (display_w, display_h) = display_dimensions(
            decoded.width(),
            decoded.height(),
            self.aspect,
            ExifOrientation::Identity,
        );
        let (width, height) = fit_within(display_w, display_h, max_edge, max_edge);
        let frame_transfer = decoded.color_transfer_characteristic();
        let transfer = if hdr_transfer(frame_transfer) {
            frame_transfer
        } else {
            self.transfer
        };
        let is_hdr = hdr_transfer(transfer);
        let mut scaler = software::scaling::Context::get(
            decoded.format(),
            decoded.width(),
            decoded.height(),
            if is_hdr {
                ffmpeg::format::Pixel::RGB48LE
            } else {
                ffmpeg::format::Pixel::RGB24
            },
            width,
            height,
            software::scaling::Flags::BILINEAR,
        )?;
        let mut rgb = frame::Video::empty();
        scaler.run(decoded, &mut rgb)?;
        let row_bytes = width as usize * if is_hdr { 6 } else { 3 };
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for row in rgb.data(0).chunks(rgb.stride(0)).take(height as usize) {
            ensure!(row.len() >= row_bytes, "short video RGB row");
            if is_hdr {
                for pixel in row[..row_bytes].as_chunks::<6>().0 {
                    let channels = [
                        u16::from_le_bytes([pixel[0], pixel[1]]),
                        u16::from_le_bytes([pixel[2], pixel[3]]),
                        u16::from_le_bytes([pixel[4], pixel[5]]),
                    ];
                    pixels.extend(channels.map(|channel| tone_map_channel(channel, transfer)));
                }
            } else {
                pixels.extend_from_slice(&row[..row_bytes]);
            }
        }
        ensure!(
            pixels.len() == width as usize * height as usize * 3,
            "short video RGB frame"
        );
        let raster = Raster::Rgb {
            width,
            height,
            pixels,
        }
        .orient(self.orientation);
        Ok(VideoSample {
            timestamp_ms,
            raster,
        })
    }
}

fn hdr_transfer(transfer: ffmpeg::color::TransferCharacteristic) -> bool {
    matches!(
        transfer,
        ffmpeg::color::TransferCharacteristic::SMPTE2084
            | ffmpeg::color::TransferCharacteristic::ARIB_STD_B67
    )
}

/// Map HDR light to an SDR channel without throwing away values above SDR white.
/// This intentionally uses a simple Reinhard curve; thumbnail color precision is
/// less important than retaining bright detail.
fn tone_map_channel(value: u16, transfer: ffmpeg::color::TransferCharacteristic) -> u8 {
    let encoded = f64::from(value) / 65535.0;
    let linear = match transfer {
        ffmpeg::color::TransferCharacteristic::SMPTE2084 => {
            // SMPTE ST 2084 inverse EOTF, in units of 100 cd/m² SDR white.
            let m1 = 2610.0 / 16384.0;
            let m2 = 2523.0 / 32.0;
            let c1 = 3424.0 / 4096.0;
            let c2 = 2413.0 / 128.0;
            let c3 = 2392.0 / 128.0;
            let power = encoded.powf(1.0 / m2);
            let ratio = (power - c1).max(0.0) / (c2 - c3 * power).max(f64::EPSILON);
            ratio.powf(1.0 / m1) * 100.0
        }
        ffmpeg::color::TransferCharacteristic::ARIB_STD_B67 => {
            // HLG inverse OETF, with a nominal 1000 cd/m² display peak.
            let a: f64 = 0.178_832_77;
            let b = 1.0 - 4.0 * a;
            let c = 0.5 - a * (4.0 * a).ln();
            let relative = if encoded <= 0.5 {
                encoded * encoded / 3.0
            } else {
                (((encoded - c) / a).exp() + b) / 12.0
            };
            relative * 10.0
        }
        _ => encoded,
    };
    let mapped = linear / (1.0 + linear);
    let sdr = if mapped < 0.018 {
        4.5 * mapped
    } else {
        1.099 * mapped.powf(0.45) - 0.099
    };
    (sdr.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn again(error: ffmpeg::Error) -> bool {
    matches!(error, ffmpeg::Error::Other { errno } if errno == libc::EAGAIN)
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    ensure!(
        width > 0
            && height > 0
            && width <= 32768
            && height <= 32768
            && u64::from(width) * u64::from(height) <= MAX_SOURCE_PIXELS,
        "video dimensions exceed decode limits"
    );
    Ok(())
}

fn to_micros(value: i64, base: ffmpeg::Rational) -> i64 {
    let scaled = i128::from(value) * i128::from(base.numerator()) * 1_000_000
        / i128::from(base.denominator());
    scaled.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}
fn to_millis(value: i64, base: ffmpeg::Rational) -> i64 {
    to_micros(value, base) / 1000
}

fn sample_targets(duration: Option<u64>, percentages: &[u8]) -> Vec<i64> {
    match duration.filter(|value| *value > 0) {
        Some(duration) => percentages
            .iter()
            .map(|percent| {
                (u128::from(duration) * u128::from(*percent) / 100).min(i64::MAX as u128 / 1000)
                    as i64
            })
            .collect(),
        None => Vec::new(),
    }
}

fn display_dimensions(
    width: u32,
    height: u32,
    aspect: ffmpeg::Rational,
    orientation: ExifOrientation,
) -> (u32, u32) {
    let width = if aspect.numerator() > 0 && aspect.denominator() > 0 {
        (u64::from(width) * aspect.numerator() as u64 / aspect.denominator() as u64).clamp(1, 32768)
            as u32
    } else {
        width
    };
    if orientation.swaps_dimensions() {
        (height, width)
    } else {
        (width, height)
    }
}

fn display_orientation(parameters: &codec::ParametersRef<'_>) -> Result<ExifOrientation> {
    use codec::packet::side_data::Type;
    ensure!(
        parameters.side_data(Type::DOVI_CONF).is_none(),
        "Dolby Vision is not supported"
    );
    let Some(data) = parameters.side_data(Type::DisplayMatrix) else {
        return Ok(ExifOrientation::Identity);
    };
    ensure!(data.len() >= 36, "invalid video display matrix");
    let mut matrix = [0_i32; 9];
    for (value, bytes) in matrix.iter_mut().zip(data[..36].as_chunks::<4>().0) {
        *value = i32::from_ne_bytes(*bytes);
    }
    orientation_from_matrix(matrix)
}

fn orientation_from_matrix(matrix: [i32; 9]) -> Result<ExifOrientation> {
    let sign = |value: i32| {
        if value.abs_diff(0) < 64 {
            0
        } else {
            value.signum()
        }
    };
    Ok(
        match (
            sign(matrix[0]),
            sign(matrix[1]),
            sign(matrix[3]),
            sign(matrix[4]),
        ) {
            (1, 0, 0, 1) => ExifOrientation::Identity,
            (-1, 0, 0, 1) => ExifOrientation::FlipHorizontal,
            (1, 0, 0, -1) => ExifOrientation::FlipVertical,
            (-1, 0, 0, -1) => ExifOrientation::Rotate180,
            (0, 1, -1, 0) => ExifOrientation::Rotate90Clockwise,
            (0, -1, 1, 0) => ExifOrientation::Rotate270Clockwise,
            (0, 1, 1, 0) => ExifOrientation::Transpose,
            (0, -1, -1, 0) => ExifOrientation::Transverse,
            _ => bail!("unsupported non-orthogonal video rotation"),
        },
    )
}

fn open_input(
    path: &Utf8Path,
    control: Arc<Interrupt>,
    probe_budget: ProbeBudget,
) -> Result<format::context::Input> {
    let mut options = ffmpeg::dict! {
        "protocol_whitelist" => "file",
        "analyzeduration" => "5000000",
    };
    options.set("probesize", probe_budget.size_bytes());
    let probe_options = ffmpeg::dict! { "threads" => "1" };
    format::input_with_dictionary_and_interrupt(path.as_str(), options, &probe_options, move || {
        control.stopped()
    })
    .context("opening and probing video input")
}

#[cfg(test)]
mod tests {
    use super::*;

    use ffmpeg::filter;

    #[test]
    fn linked_transpose_filter_rotates_a_synthetic_frame() -> Result<()> {
        ffmpeg::init()?;
        let mut graph = filter::Graph::new();
        let source = filter::find("buffer").context("buffer filter is unavailable")?;
        filter::find("transpose").context("transpose filter is unavailable")?;
        let sink = filter::find("buffersink").context("buffersink filter is unavailable")?;
        graph.add(
            &source,
            "in",
            "video_size=2x3:pix_fmt=rgb24:time_base=1/1000:pixel_aspect=1/1",
        )?;
        graph.add(&sink, "out", "")?;
        graph
            .output("in", 0)?
            .input("out", 0)?
            .parse("transpose=clock")?;
        graph.validate()?;

        let mut input = frame::Video::new(ffmpeg::format::Pixel::RGB24, 2, 3);
        for (y, row) in [[1_u8, 2], [3, 4], [5, 6]].into_iter().enumerate() {
            let stride = input.stride(0);
            for (x, value) in row.into_iter().enumerate() {
                input.data_mut(0)[y * stride + x * 3..y * stride + x * 3 + 3].fill(value);
            }
        }
        input.set_pts(Some(0));
        graph
            .get("in")
            .context("missing buffer source")?
            .source()
            .add(&input)?;
        let mut output = frame::Video::empty();
        graph
            .get("out")
            .context("missing buffer sink")?
            .sink()
            .frame(&mut output)?;

        assert_eq!((output.width(), output.height()), (3, 2));
        for (y, expected) in [[5_u8, 3, 1], [6, 4, 2]].into_iter().enumerate() {
            for (x, value) in expected.into_iter().enumerate() {
                assert_eq!(
                    &output.data(0)[y * output.stride(0) + x * 3..][..3],
                    &[value; 3]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn targets_and_display_geometry_are_bounded() {
        assert_eq!(
            sample_targets(Some(10000), &[10, 50, 90]),
            vec![1000, 5000, 9000]
        );
        assert_eq!(sample_targets(None, &[10, 50, 90]), Vec::<i64>::new());
        assert_eq!(
            display_dimensions(
                720,
                480,
                ffmpeg::Rational(4, 3),
                ExifOrientation::Rotate90Clockwise
            ),
            (480, 960)
        );
    }

    #[test]
    fn display_matrices_cover_rotation_and_mirroring() -> Result<()> {
        let quarter_turn = [0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30];
        assert_eq!(
            orientation_from_matrix(quarter_turn)?,
            ExifOrientation::Rotate90Clockwise
        );
        let mirrored = [0, 65536, 0, 65536, 0, 0, 0, 0, 1 << 30];
        assert_eq!(
            orientation_from_matrix(mirrored)?,
            ExifOrientation::Transpose
        );
        let diagonal = [46341, 46341, 0, -46341, 46341, 0, 0, 0, 1 << 30];
        assert!(orientation_from_matrix(diagonal).is_err());
        Ok(())
    }
}
