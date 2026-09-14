// Modified for Nicegal; changes remain licensed under Apache-2.0.
use crate::common::{Error, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array, Array3};
use std::ops::{Div, Sub};
#[cfg(feature = "hf-hub")]
use std::{fs::read_to_string, path::Path};

pub enum TransformData {
    Image(DynamicImage),
    NdArray(Array3<f32>),
}

impl TransformData {
    pub fn image(self) -> Result<DynamicImage> {
        match self {
            TransformData::Image(img) => Ok(img),
            _ => Err(Error::ImageTransform("TransformData convert error".into())),
        }
    }

    pub fn array(self) -> Result<Array3<f32>> {
        match self {
            TransformData::NdArray(array) => Ok(array),
            _ => Err(Error::ImageTransform("TransformData convert error".into())),
        }
    }
}

pub(crate) type ResizeFn =
    dyn Fn(DynamicImage, u32, u32, FilterType) -> Result<DynamicImage> + Send + Sync;

fn default_resize(
    image: DynamicImage,
    width: u32,
    height: u32,
    filter: FilterType,
) -> Result<DynamicImage> {
    Ok(image.resize_exact(width, height, filter))
}

pub trait Transform: Send + Sync {
    fn transform(&self, images: TransformData) -> Result<TransformData> {
        self.transform_with_resize(images, &default_resize)
    }
    fn transform_with_resize(
        &self,
        images: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData>;
}

struct ConvertToRGB;

impl Transform for ConvertToRGB {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        let image = data.image()?;
        let image = image.into_rgb8().into();
        Ok(TransformData::Image(image))
    }
}

pub struct Resize {
    pub size: (u32, u32),
    pub resample: FilterType,
}

impl Transform for Resize {
    fn transform_with_resize(
        &self,
        data: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData> {
        let image = data.image()?;
        let image = resize(image, self.size.1, self.size.0, self.resample)?;
        Ok(TransformData::Image(image))
    }
}

// Pillow performs horizontal filtering into RGB8 before its vertical pass. Separate
// single-axis passes preserve that intermediate rounding and clipping.
fn pillow_resize(
    image: DynamicImage,
    width: u32,
    height: u32,
    filter: FilterType,
    resize: &ResizeFn,
) -> Result<DynamicImage> {
    let source_height = image.height();
    let horizontal = resize(image, width, source_height, filter)?;
    resize(horizontal, width, height, filter)
}
struct ResizePillow {
    size: (u32, u32),
    filter: FilterType,
}
impl Transform for ResizePillow {
    fn transform_with_resize(
        &self,
        data: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData> {
        Ok(TransformData::Image(pillow_resize(
            data.image()?,
            self.size.1,
            self.size.0,
            self.filter,
            resize,
        )?))
    }
}

/// Resize the shorter side without distorting the image before a center crop.
struct ResizeShortestEdge {
    size: u32,
}

/// The exported DeepGHS pipeline resizes the short side to `size`, caps the long side at
/// `max_size`, then center-crops (padding the shorter side black). Both steps truncate to int.
struct ResizeDeepGhs {
    size: u32,
    max_size: u32,
}
impl Transform for ResizeDeepGhs {
    fn transform_with_resize(
        &self,
        data: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData> {
        let image = data.image()?;
        let (width, height) = image.dimensions();
        let (mut output_width, mut output_height) = if width < height {
            (
                self.size,
                (u64::from(self.size) * u64::from(height) / u64::from(width)) as u32,
            )
        } else {
            (
                (u64::from(self.size) * u64::from(width) / u64::from(height)) as u32,
                self.size,
            )
        };
        if output_width.max(output_height) > self.max_size {
            if output_height > output_width {
                output_width = (u64::from(self.max_size) * u64::from(output_width)
                    / u64::from(output_height)) as u32;
                output_height = self.max_size;
            } else {
                output_height = (u64::from(self.max_size) * u64::from(output_height)
                    / u64::from(output_width)) as u32;
                output_width = self.max_size;
            }
        }
        if (width, height) == (output_width, output_height) {
            Ok(TransformData::Image(image))
        } else {
            Ok(TransformData::Image(pillow_resize(
                image,
                output_width.max(1),
                output_height.max(1),
                FilterType::CatmullRom,
                resize,
            )?))
        }
    }
}
impl Transform for ResizeShortestEdge {
    fn transform_with_resize(
        &self,
        data: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData> {
        let image = data.image()?;
        let (width, height) = image.dimensions();
        let shorter = width.min(height) as u64;
        let size = self.size as u64;
        let (width, height) = (
            (width as u64 * size / shorter) as u32,
            (height as u64 * size / shorter) as u32,
        );
        Ok(TransformData::Image(pillow_resize(
            image,
            width,
            height,
            FilterType::CatmullRom,
            resize,
        )?))
    }
}

pub struct CenterCrop {
    pub size: (u32, u32),
    pub round_even: bool,
}

impl Transform for CenterCrop {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        let mut image = data.image()?;
        let (mut origin_width, mut origin_height) = image.dimensions();
        let (crop_width, crop_height) = self.size;
        if origin_width >= crop_width && origin_height >= crop_height {
            // cropped area is within image boundaries
            let offset = |difference: u32| {
                let floor = difference / 2;
                if self.round_even && difference % 2 == 1 && floor % 2 == 1 {
                    floor + 1
                } else {
                    floor
                }
            };
            let x = offset(origin_width - crop_width);
            let y = offset(origin_height - crop_height);
            let image = image.crop_imm(x, y, crop_width, crop_height);
            Ok(TransformData::Image(image))
        } else {
            if origin_width > crop_width || origin_height > crop_height {
                let (new_width, new_height) =
                    (origin_width.min(crop_width), origin_height.min(crop_height));
                let (x, y) = if origin_width > crop_width {
                    ((origin_width - crop_width) / 2, 0)
                } else {
                    (0, (origin_height - crop_height) / 2)
                };
                image = image.crop_imm(x, y, new_width, new_height);
                (origin_width, origin_height) = image.dimensions();
            }
            let mut pixels_array =
                Array3::zeros((3usize, crop_height as usize, crop_width as usize));
            let offset_x = (crop_width - origin_width) / 2;
            let offset_y = (crop_height - origin_height) / 2;
            // whc -> chw
            for (x, y, pixel) in image.to_rgb8().enumerate_pixels() {
                pixels_array[[0, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[0] as f32;
                pixels_array[[1, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[1] as f32;
                pixels_array[[2, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[2] as f32;
            }
            Ok(TransformData::NdArray(pixels_array))
        }
    }
}

struct PILToNDarray;

impl Transform for PILToNDarray {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        match data {
            TransformData::Image(image) => {
                let image = image.to_rgb8();
                let (width, height) = image.dimensions();
                // whc -> chw
                let mut pixels_array = Array3::zeros((3usize, height as usize, width as usize));
                for (x, y, pixel) in image.enumerate_pixels() {
                    pixels_array[[0, y as usize, x as usize]] = pixel[0] as f32;
                    pixels_array[[1, y as usize, x as usize]] = pixel[1] as f32;
                    pixels_array[[2, y as usize, x as usize]] = pixel[2] as f32;
                }
                Ok(TransformData::NdArray(pixels_array))
            }
            ndarray => Ok(ndarray),
        }
    }
}

pub struct Rescale {
    pub scale: f32,
}

impl Transform for Rescale {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        let array = data.array()?;
        let array = array * self.scale;
        Ok(TransformData::NdArray(array))
    }
}

pub struct Normalize {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Transform for Normalize {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        let array = data.array()?;
        let mean = Array::from_vec(self.mean.clone())
            .into_shape_with_order((3, 1, 1))
            .map_err(|e| Error::InvalidShape(format!("Failed to reshape mean array: {e}")))?;
        let std = Array::from_vec(self.std.clone())
            .into_shape_with_order((3, 1, 1))
            .map_err(|e| Error::InvalidShape(format!("Failed to reshape std array: {e}")))?;

        let shape = array.shape().to_vec();
        match shape.as_slice() {
            [c, h, w] => {
                let mean_broadcast = mean.broadcast((*c, *h, *w)).ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "Failed to broadcast mean array to shape {:?}",
                        (*c, *h, *w)
                    ))
                })?;
                let std_broadcast = std.broadcast((*c, *h, *w)).ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "Failed to broadcast std array to shape {:?}",
                        (*c, *h, *w)
                    ))
                })?;
                let array_normalized = array.sub(mean_broadcast).div(std_broadcast);
                Ok(TransformData::NdArray(array_normalized))
            }
            _ => Err(Error::ImageTransform(
                "Transformer convert error. Normalize operator got error shape.".into(),
            )),
        }
    }
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    #[cfg(feature = "hf-hub")]
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        let config = serde_json::from_str(&content)
            .map_err(|e| Error::PreprocessorConfig(format!("Invalid preprocessor JSON: {e}")))?;
        load_preprocessor(config)
    }

    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Compose> {
        let config = serde_json::from_slice(bytes.as_ref())
            .map_err(|e| Error::PreprocessorConfig(format!("Invalid preprocessor JSON: {e}")))?;
        load_preprocessor(config)
    }

    /// Read the five-stage pipeline bundled with DeepGHS SigLIP checkpoints. Validate its
    /// parameters rather than silently changing preprocessing if upstream revises the file.
    pub fn from_deepghs_bytes(bytes: &[u8]) -> Result<Compose> {
        let config: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| Error::PreprocessorConfig(e.to_string()))?;
        let stages = config["stages"]
            .as_array()
            .ok_or_else(|| Error::PreprocessorConfig("DeepGHS stages are missing".into()))?;
        if stages.len() != 5
            || stages[0]["type"] != "convert_rgb"
            || stages[0]["force_background"] != "white"
            || stages[1]["type"] != "resize"
            || stages[1]["interpolation"] != "bicubic"
            || stages[1]["antialias"] != true
            || stages[2]["type"] != "center_crop"
            || stages[3]["type"] != "maybe_to_tensor"
            || stages[4]["type"] != "normalize"
        {
            return Err(Error::PreprocessorConfig(
                "unsupported DeepGHS preprocessing stages".into(),
            ));
        }
        let size = stages[1]["size"]
            .as_u64()
            .ok_or_else(|| Error::PreprocessorConfig("DeepGHS resize size is missing".into()))?
            as u32;
        let max_size = stages[1]["max_size"]
            .as_u64()
            .ok_or_else(|| Error::PreprocessorConfig("DeepGHS maximum size is missing".into()))?
            as u32;
        let crop = stages[2]["size"]
            .as_u64()
            .ok_or_else(|| Error::PreprocessorConfig("DeepGHS crop size is missing".into()))?
            as u32;
        if size == 0 || max_size == 0 || crop == 0 {
            return Err(Error::PreprocessorConfig(
                "DeepGHS image size must be positive".into(),
            ));
        }
        let channels = |key: &str| -> Result<Vec<f32>> {
            let values = stages[4][key]
                .as_array()
                .ok_or_else(|| Error::PreprocessorConfig(format!("DeepGHS {key} is missing")))?;
            if values.len() != 3 {
                return Err(Error::PreprocessorConfig(format!(
                    "DeepGHS {key} must have three channels"
                )));
            }
            values
                .iter()
                .map(|value| {
                    value.as_f64().map(|v| v as f32).ok_or_else(|| {
                        Error::PreprocessorConfig(format!("DeepGHS {key} must be numeric"))
                    })
                })
                .collect()
        };
        let mean = channels("mean")?;
        let std = channels("std")?;
        if std.contains(&0.0) {
            return Err(Error::PreprocessorConfig(
                "DeepGHS standard deviation is zero".into(),
            ));
        }
        Ok(Self::new(vec![
            Box::new(ConvertToRGB),
            Box::new(ResizeDeepGhs { size, max_size }),
            Box::new(CenterCrop {
                size: (crop, crop),
                round_even: false,
            }),
            Box::new(PILToNDarray),
            Box::new(Rescale { scale: 1.0 / 255.0 }),
            Box::new(Normalize { mean, std }),
        ]))
    }

    pub fn preprocess_image(&self, image: DynamicImage) -> Result<Array3<f32>> {
        Self::pixels(self.transform(TransformData::Image(image))?)
    }

    pub(crate) fn preprocess_image_with_resize(
        &self,
        image: DynamicImage,
        resize: &ResizeFn,
    ) -> Result<Array3<f32>> {
        Self::pixels(self.transform_with_resize(TransformData::Image(image), resize)?)
    }

    fn pixels(data: TransformData) -> Result<Array3<f32>> {
        match data {
            TransformData::NdArray(array) => Ok(array),
            _ => Err(Error::PreprocessorConfig(
                "Preprocessor configuration did not produce image pixels".into(),
            )),
        }
    }
}

impl Transform for Compose {
    fn transform_with_resize(
        &self,
        mut image: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData> {
        for transform in &self.transforms {
            image = transform.transform_with_resize(image, resize)?;
        }
        Ok(image)
    }
}

fn load_preprocessor(config: serde_json::Value) -> Result<Compose> {
    let mut transformers: Vec<Box<dyn Transform>> = vec![];
    transformers.push(Box::new(ConvertToRGB));

    let mode = config["image_processor_type"]
        .as_str()
        .unwrap_or("CLIPImageProcessor");
    match mode {
        "CLIPImageProcessor"
        | "SiglipImageProcessor"
        | "Siglip2ImageProcessor"
        | "DINOv3ViTImageProcessorFast" => {
            if config["do_resize"].as_bool().unwrap_or(false) {
                let size = config["size"].clone();
                let shortest_edge = size["shortest_edge"].as_u64();
                let (height, width) = (size["height"].as_u64(), size["width"].as_u64());

                if let Some(shortest_edge) = shortest_edge {
                    if config["nicegal_preserve_aspect_ratio"]
                        .as_bool()
                        .unwrap_or(false)
                    {
                        transformers.push(Box::new(ResizeShortestEdge {
                            size: shortest_edge as u32,
                        }));
                    } else {
                        let size = (shortest_edge as u32, shortest_edge as u32);
                        transformers.push(Box::new(Resize {
                            size,
                            resample: FilterType::CatmullRom,
                        }));
                    }
                } else if let (Some(height), Some(width)) = (height, width) {
                    let size = (height as u32, width as u32);
                    if config["nicegal_pillow_resize"].as_bool().unwrap_or(false) {
                        transformers.push(Box::new(ResizePillow {
                            size,
                            filter: if config["resample"].as_u64() == Some(2) {
                                FilterType::Triangle
                            } else {
                                FilterType::CatmullRom
                            },
                        }));
                    } else {
                        transformers.push(Box::new(Resize {
                            size,
                            resample: FilterType::CatmullRom,
                        }));
                    }
                } else {
                    return Err(Error::PreprocessorConfig(
                        "Size must contain either 'shortest_edge' or 'height' and 'width'.".into(),
                    ));
                }
            }

            if config["do_center_crop"].as_bool().unwrap_or(false) {
                let crop_size = config["crop_size"].clone();
                let (height, width) = if crop_size.is_u64() {
                    let size = crop_size.as_u64().ok_or_else(|| {
                        Error::PreprocessorConfig("crop_size must be a valid u64".into())
                    })? as u32;
                    (size, size)
                } else if crop_size.is_object() {
                    (
                        crop_size["height"]
                            .as_u64()
                            .map(|height| height as u32)
                            .ok_or_else(|| {
                                Error::PreprocessorConfig(
                                    "crop_size height must be contained".into(),
                                )
                            })?,
                        crop_size["width"]
                            .as_u64()
                            .map(|width| width as u32)
                            .ok_or_else(|| {
                                Error::PreprocessorConfig(
                                    "crop_size width must be contained".into(),
                                )
                            })?,
                    )
                } else {
                    return Err(Error::PreprocessorConfig(format!(
                        "Invalid crop size: {crop_size:?}"
                    )));
                };
                transformers.push(Box::new(CenterCrop {
                    size: (width, height),
                    round_even: config["nicegal_center_crop_round"]
                        .as_bool()
                        .unwrap_or(false),
                }));
            }
        }
        "ConvNextFeatureExtractor" => {
            let shortest_edge = config["size"]["shortest_edge"].as_u64();
            if shortest_edge.is_none() {
                return Err(Error::PreprocessorConfig(
                    "Size dictionary must contain 'shortest_edge' key.".into(),
                ));
            }
            let shortest_edge = shortest_edge.unwrap() as u32;
            let crop_pct = config["crop_pct"].as_f64().unwrap_or(0.875);
            if shortest_edge < 384 {
                let resize_shortet_edge = shortest_edge as f64 / crop_pct;
                transformers.push(Box::new(Resize {
                    size: (resize_shortet_edge as u32, resize_shortet_edge as u32),
                    resample: FilterType::CatmullRom,
                }));
                transformers.push(Box::new(CenterCrop {
                    size: (shortest_edge, shortest_edge),
                    round_even: false,
                }))
            } else {
                transformers.push(Box::new(Resize {
                    size: (shortest_edge, shortest_edge),
                    resample: FilterType::CatmullRom,
                }));
            }
        }
        "BitImageProcessor" => {
            if config["do_convert_rgb"].as_bool().unwrap_or(false) {
                transformers.push(Box::new(ConvertToRGB));
            }
            if config["do_resize"].as_bool().unwrap_or(false) {
                let size = config["size"].clone();
                let shortest_edge = size["shortest_edge"].as_u64();
                let (height, width) = (size["height"].as_u64(), size["width"].as_u64());

                if let Some(shortest_edge) = shortest_edge {
                    let size = (shortest_edge as u32, shortest_edge as u32);
                    transformers.push(Box::new(Resize {
                        size,
                        resample: FilterType::CatmullRom,
                    }));
                } else if let (Some(height), Some(width)) = (height, width) {
                    let size = (height as u32, width as u32);
                    transformers.push(Box::new(Resize {
                        size,
                        resample: FilterType::CatmullRom,
                    }));
                } else {
                    return Err(Error::PreprocessorConfig(
                        "Size must contain either 'shortest_edge' or 'height' and 'width'.".into(),
                    ));
                }
            }

            if config["do_center_crop"].as_bool().unwrap_or(false) {
                let crop_size = config["crop_size"].clone();
                let (height, width) = if crop_size.is_u64() {
                    let size = crop_size.as_u64().ok_or_else(|| {
                        Error::PreprocessorConfig("crop_size must be a valid u64".into())
                    })? as u32;
                    (size, size)
                } else if crop_size.is_object() {
                    (
                        crop_size["height"]
                            .as_u64()
                            .map(|height| height as u32)
                            .ok_or_else(|| {
                                Error::PreprocessorConfig(
                                    "crop_size height must be contained".into(),
                                )
                            })?,
                        crop_size["width"]
                            .as_u64()
                            .map(|width| width as u32)
                            .ok_or_else(|| {
                                Error::PreprocessorConfig(
                                    "crop_size width must be contained".into(),
                                )
                            })?,
                    )
                } else {
                    return Err(Error::PreprocessorConfig(format!(
                        "Invalid crop size: {crop_size:?}"
                    )));
                };
                transformers.push(Box::new(CenterCrop {
                    size: (width, height),
                    round_even: config["nicegal_center_crop_round"]
                        .as_bool()
                        .unwrap_or(false),
                }));
            }
        }
        mode => {
            return Err(Error::PreprocessorConfig(format!(
                "Preprocessor {mode} is not supported"
            )));
        }
    }

    transformers.push(Box::new(PILToNDarray));

    if config["do_rescale"].as_bool().unwrap_or(true) {
        let rescale_factor = config["rescale_factor"].as_f64().unwrap_or(1.0f64 / 255.0);
        transformers.push(Box::new(Rescale {
            scale: rescale_factor as f32,
        }));
    }

    if config["do_normalize"].as_bool().unwrap_or(false) {
        let mean = config["image_mean"]
            .as_array()
            .ok_or_else(|| Error::PreprocessorConfig("image_mean must be contained".into()))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|num| num as f32)
                    .ok_or_else(|| Error::PreprocessorConfig("image_mean must be float".into()))
            })
            .collect::<Result<Vec<f32>>>()?;
        let std = config["image_std"]
            .as_array()
            .ok_or_else(|| Error::PreprocessorConfig("image_std must be contained".into()))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|num| num as f32)
                    .ok_or_else(|| Error::PreprocessorConfig("image_std must be float".into()))
            })
            .collect::<Result<Vec<f32>>>()?;
        transformers.push(Box::new(Normalize { mean, std }));
    }

    Ok(Compose::new(transformers))
}

#[cfg(test)]
mod nicegal_tests {
    use super::*;
    #[test]
    fn deepghs_fit_long_side_pads_the_short_side_black() {
        let config = serde_json::json!({"stages": [
            {"type": "convert_rgb", "force_background": "white"},
            {"type": "resize", "size": 4, "max_size": 4,
                "interpolation": "bicubic", "antialias": true},
            {"type": "center_crop", "size": 4},
            {"type": "maybe_to_tensor"},
            {"type": "normalize", "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5]},
        ]});
        let preprocessor =
            Compose::from_deepghs_bytes(&serde_json::to_vec(&config).unwrap()).unwrap();
        let image = image::RgbImage::from_pixel(8, 4, image::Rgb([255, 0, 0]));
        let pixels = preprocessor.preprocess_image(image.into()).unwrap();
        assert_eq!(pixels.shape(), &[3, 4, 4]);
        assert_eq!(pixels[[0, 0, 0]], -1.0); // black top padding
        assert_eq!(pixels[[0, 1, 0]], 1.0); // red image
        assert_eq!(pixels[[1, 1, 0]], -1.0);
        assert_eq!(pixels[[0, 3, 0]], -1.0); // black bottom padding
    }
    #[test]
    fn local_clip_preserves_aspect_while_legacy_config_stays_unchanged() {
        // Wide three-color image: a true center crop should exclude the outer columns.
        let mut image = image::RgbImage::new(12, 4);
        for (x, _, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb(if x < 4 {
                [255, 0, 0]
            } else if x < 8 {
                [0, 255, 0]
            } else {
                [0, 0, 255]
            });
        }
        let base = serde_json::json!({"do_resize": true, "size": {"shortest_edge": 4},
            "do_center_crop": true, "crop_size": 4, "do_rescale": false, "do_normalize": false,
            "nicegal_preserve_aspect_ratio": true});
        let tensor = load_preprocessor(base.clone())
            .unwrap()
            .preprocess_image(DynamicImage::ImageRgb8(image.clone()))
            .unwrap();
        assert_eq!(tensor.shape(), &[3, 4, 4]);
        assert!(tensor
            .index_axis(ndarray::Axis(0), 1)
            .iter()
            .all(|value| *value == 255.0));
        let mut legacy = base;
        legacy["nicegal_preserve_aspect_ratio"] = false.into();
        let tensor = load_preprocessor(legacy)
            .unwrap()
            .preprocess_image(DynamicImage::ImageRgb8(image))
            .unwrap();
        assert!(tensor[[0, 0, 0]] > 100.0);
    }
    #[test]
    fn torchvision_center_crop_rounds_half_offsets_to_even() {
        let mut pixels = image::RgbImage::new(7, 4);
        for (x, _, pixel) in pixels.enumerate_pixels_mut() {
            *pixel = image::Rgb([x as u8, 0, 0]);
        }
        let image = DynamicImage::ImageRgb8(pixels);
        let rounded = CenterCrop {
            size: (4, 4),
            round_even: true,
        }
        .transform(TransformData::Image(image.clone()))
        .unwrap()
        .image()
        .unwrap();
        let floored = CenterCrop {
            size: (4, 4),
            round_even: false,
        }
        .transform(TransformData::Image(image))
        .unwrap()
        .image()
        .unwrap();
        assert_eq!(rounded.to_rgb8().get_pixel(0, 0)[0], 2);
        assert_eq!(floored.to_rgb8().get_pixel(0, 0)[0], 1);
    }

    #[test]
    fn siglip_resizes_to_exact_square_and_normalizes() {
        let config = serde_json::json!({"image_processor_type": "SiglipImageProcessor",
            "do_resize": true, "size": {"height": 256, "width": 256},
            "do_rescale": true, "rescale_factor": 0.00392156862745098,
            "do_normalize": true, "image_mean": [0.5,0.5,0.5], "image_std": [0.5,0.5,0.5]});
        let tensor = load_preprocessor(config)
            .unwrap()
            .preprocess_image(DynamicImage::ImageRgb8(image::RgbImage::new(12, 4)))
            .unwrap();
        assert_eq!(tensor.shape(), &[3, 256, 256]);
        assert!(tensor.iter().all(|value| *value == -1.0));
    }
}
