#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Raster {
    Rgb {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
    Rgba {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
}

/// The EXIF orientation values defined by the TIFF specification.
///
/// Invalid and missing EXIF orientation values intentionally map to [`Self::Identity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExifOrientation {
    Identity,
    FlipHorizontal,
    Rotate180,
    FlipVertical,
    Transpose,
    Rotate90Clockwise,
    Transverse,
    Rotate270Clockwise,
}

impl ExifOrientation {
    pub(crate) fn from_value(value: u16) -> Self {
        match value {
            2 => Self::FlipHorizontal,
            3 => Self::Rotate180,
            4 => Self::FlipVertical,
            5 => Self::Transpose,
            6 => Self::Rotate90Clockwise,
            7 => Self::Transverse,
            8 => Self::Rotate270Clockwise,
            _ => Self::Identity,
        }
    }

    pub(crate) const fn swaps_dimensions(self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90Clockwise | Self::Transverse | Self::Rotate270Clockwise
        )
    }
}

impl Raster {
    pub fn width(&self) -> u32 {
        match self {
            Self::Rgb { width, .. } | Self::Rgba { width, .. } => *width,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            Self::Rgb { height, .. } | Self::Rgba { height, .. } => *height,
        }
    }

    pub fn pixels(&self) -> &[u8] {
        match self {
            Self::Rgb { pixels, .. } | Self::Rgba { pixels, .. } => pixels,
        }
    }

    /// Whether any pixel actually uses transparency. An RGBA source with a fully opaque alpha
    /// channel reports `false`, so callers can still pick the smaller opaque encoding for it.
    pub fn has_alpha(&self) -> bool {
        match self {
            Self::Rgb { .. } => false,
            Self::Rgba { pixels, .. } => pixels.chunks_exact(4).any(|pixel| pixel[3] != 255),
        }
    }

    /// Add an opaque alpha channel, producing flat RGBA8 bytes.
    pub fn into_rgba_bytes(self) -> Vec<u8> {
        match self {
            Self::Rgba { pixels, .. } => pixels,
            Self::Rgb { pixels, .. } => {
                let mut rgba = Vec::with_capacity(pixels.len() / 3 * 4);
                for pixel in pixels.chunks_exact(3) {
                    rgba.extend_from_slice(pixel);
                    rgba.push(255);
                }
                rgba
            }
        }
    }

    /// Drop the alpha channel, producing flat RGB8 bytes.
    pub fn into_rgb_bytes(self) -> Vec<u8> {
        match self {
            Self::Rgb { pixels, .. } => pixels,
            Self::Rgba { pixels, .. } => {
                let mut rgb = Vec::with_capacity(pixels.len() / 4 * 3);
                for pixel in pixels.chunks_exact(4) {
                    rgb.extend_from_slice(&pixel[..3]);
                }
                rgb
            }
        }
    }

    /// Flatten onto an opaque background, producing flat RGB8 bytes. JPEG has no alpha channel,
    /// so a transparent source needs real compositing here rather than a naive channel drop —
    /// otherwise "don't care" RGB left behind fully transparent pixels (common in exported PNGs)
    /// would show through as garbage color.
    pub fn flatten_rgb(self, background: [u8; 3]) -> Vec<u8> {
        match self {
            Self::Rgb { pixels, .. } => pixels,
            Self::Rgba { pixels, .. } => {
                let mut rgb = Vec::with_capacity(pixels.len() / 4 * 3);
                for pixel in pixels.chunks_exact(4) {
                    let alpha = u16::from(pixel[3]);
                    for channel in 0..3 {
                        let foreground = u16::from(pixel[channel]);
                        let background = u16::from(background[channel]);
                        let blended = (foreground * alpha + background * (255 - alpha)) / 255;
                        rgb.push(blended as u8);
                    }
                }
                rgb
            }
        }
    }

    /// Apply a TIFF/EXIF orientation without intermediate full-resolution rasters.
    pub(crate) fn orient(self, orientation: ExifOrientation) -> Self {
        if orientation == ExifOrientation::Identity {
            return self;
        }

        match self {
            Self::Rgb {
                width,
                height,
                pixels,
            } => {
                let (width, height, pixels) = orient_pixels(width, height, pixels, 3, orientation);
                Self::Rgb {
                    width,
                    height,
                    pixels,
                }
            }
            Self::Rgba {
                width,
                height,
                pixels,
            } => {
                let (width, height, pixels) = orient_pixels(width, height, pixels, 4, orientation);
                Self::Rgba {
                    width,
                    height,
                    pixels,
                }
            }
        }
    }
}

fn orient_pixels(
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    channels: usize,
    orientation: ExifOrientation,
) -> (u32, u32, Vec<u8>) {
    let (destination_width, destination_height) = if orientation.swaps_dimensions() {
        (height, width)
    } else {
        (width, height)
    };
    let mut destination = vec![0; pixels.len()];
    let width = width as usize;
    let height = height as usize;
    let destination_width = destination_width as usize;

    for source_y in 0..height {
        for source_x in 0..width {
            let (destination_x, destination_y) = match orientation {
                ExifOrientation::Identity => (source_x, source_y),
                ExifOrientation::FlipHorizontal => (width - 1 - source_x, source_y),
                ExifOrientation::Rotate180 => (width - 1 - source_x, height - 1 - source_y),
                ExifOrientation::FlipVertical => (source_x, height - 1 - source_y),
                ExifOrientation::Transpose => (source_y, source_x),
                ExifOrientation::Rotate90Clockwise => (height - 1 - source_y, source_x),
                ExifOrientation::Transverse => (height - 1 - source_y, width - 1 - source_x),
                ExifOrientation::Rotate270Clockwise => (source_y, width - 1 - source_x),
            };
            let source_offset = (source_y * width + source_x) * channels;
            let destination_offset = (destination_y * destination_width + destination_x) * channels;
            destination[destination_offset..destination_offset + channels]
                .copy_from_slice(&pixels[source_offset..source_offset + channels]);
        }
    }

    (
        destination_width as u32,
        destination_height as u32,
        destination,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster() -> Raster {
        Raster::Rgb {
            width: 3,
            height: 2,
            pixels: (0..6).flat_map(|value| [value, 0, 0]).collect(),
        }
    }

    fn values(raster: Raster) -> Vec<u8> {
        raster
            .pixels()
            .chunks_exact(3)
            .map(|pixel| pixel[0])
            .collect()
    }

    #[test]
    fn orientations_rearrange_rgb_pixels_without_changing_channels() {
        let cases = [
            (
                ExifOrientation::FlipHorizontal,
                (3, 2),
                vec![2, 1, 0, 5, 4, 3],
            ),
            (ExifOrientation::Rotate180, (3, 2), vec![5, 4, 3, 2, 1, 0]),
            (
                ExifOrientation::FlipVertical,
                (3, 2),
                vec![3, 4, 5, 0, 1, 2],
            ),
            (ExifOrientation::Transpose, (2, 3), vec![0, 3, 1, 4, 2, 5]),
            (
                ExifOrientation::Rotate90Clockwise,
                (2, 3),
                vec![3, 0, 4, 1, 5, 2],
            ),
            (ExifOrientation::Transverse, (2, 3), vec![5, 2, 4, 1, 3, 0]),
            (
                ExifOrientation::Rotate270Clockwise,
                (2, 3),
                vec![2, 5, 1, 4, 0, 3],
            ),
        ];

        for (orientation, dimensions, expected) in cases {
            let oriented = raster().orient(orientation);
            assert_eq!((oriented.width(), oriented.height()), dimensions);
            assert_eq!(values(oriented), expected, "{orientation:?}");
        }
    }

    #[test]
    fn orientation_preserves_rgba_pixels() {
        let raster = Raster::Rgba {
            width: 2,
            height: 1,
            pixels: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };

        assert_eq!(
            raster.orient(ExifOrientation::FlipHorizontal),
            Raster::Rgba {
                width: 2,
                height: 1,
                pixels: vec![5, 6, 7, 8, 1, 2, 3, 4],
            }
        );
    }
}
