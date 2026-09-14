// Modified for Nicegal; changes remain licensed under Apache-2.0.
use super::utils::Compose;
use crate::{init::InitOptions, ImageEmbeddingModel};
use image::DynamicImage;
use ndarray::Array3;
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};
use std::sync::Arc;

/// Options for initializing the ImageEmbedding model
pub type ImageInitOptions = InitOptions<ImageEmbeddingModel>;

/// Options for initializing UserDefinedImageEmbeddingModel
///
/// Model files are held by the UserDefinedImageEmbeddingModel struct
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ImageInitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    /// Number of intra-op threads for ONNX Runtime. `None` (the default) uses
    /// every available CPU core via `std::thread::available_parallelism`.
    /// Set this to cap CPU usage (e.g. on laptops) at the cost of throughput.
    pub intra_threads: Option<usize>,
}

impl ImageInitOptionsUserDefined {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    /// Set the number of intra-op threads ONNX Runtime uses. By default
    /// (`None`) all available CPU cores are used; capping this limits CPU
    /// usage at the cost of per-inference throughput.
    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = Some(intra_threads);
        self
    }
}

/// Convert ImageInitOptions to ImageInitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<ImageInitOptions> for ImageInitOptionsUserDefined {
    fn from(options: ImageInitOptions) -> Self {
        ImageInitOptionsUserDefined {
            execution_providers: options.execution_providers,
            intra_threads: options.intra_threads,
        }
    }
}

/// Struct for "bring your own" embedding models
///
/// The onnx_file and preprocessor_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedImageEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub preprocessor_file: Vec<u8>,
}

impl UserDefinedImageEmbeddingModel {
    pub fn new(onnx_file: Vec<u8>, preprocessor_file: Vec<u8>) -> Self {
        Self {
            onnx_file,
            preprocessor_file,
        }
    }
}

/// A model-specific image preprocessor that can be shared with CPU worker threads.
///
/// The returned arrays are ready to pass to [`ImageEmbedding::embed_preprocessed`]. Cloning this
/// type is cheap and does not clone image data or ONNX Runtime state.
#[derive(Clone)]
pub struct ImagePreprocessor {
    inner: Arc<Compose>,
    resize: Option<Arc<super::utils::ResizeFn>>,
}

impl ImagePreprocessor {
    pub(crate) fn new(inner: Compose) -> Self {
        Self {
            inner: Arc::new(inner),
            resize: None,
        }
    }

    /// Resize, crop, rescale, and normalize an image according to this model's configuration.
    pub fn preprocess(&self, image: DynamicImage) -> crate::Result<Array3<f32>> {
        match &self.resize {
            Some(resize) => self
                .inner
                .preprocess_image_with_resize(image, resize.as_ref()),
            None => self.inner.preprocess_image(image),
        }
    }

    /// Use the host's resizer while retaining model-specific geometry and normalization.
    pub fn with_resize<F>(mut self, resize: F) -> Self
    where
        F: Fn(DynamicImage, u32, u32, image::imageops::FilterType) -> crate::Result<DynamicImage>
            + Send
            + Sync
            + 'static,
    {
        self.resize = Some(Arc::new(resize));
        self
    }
}

/// Rust representation of the ImageEmbedding model.
pub struct ImageEmbedding {
    pub(crate) preprocessor: ImagePreprocessor,
    pub(crate) session: Session,
    pub(crate) output_key: Option<&'static str>,
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, RgbImage};

    use super::*;

    #[test]
    fn host_resize_keeps_crop_and_normalization() {
        let config = br#"{"do_resize":true,"size":{"height":4,"width":4},
            "do_center_crop":true,"crop_size":{"height":2,"width":2},
            "do_rescale":true,"rescale_factor":0.5,
            "do_normalize":true,"image_mean":[1,1,1],"image_std":[2,2,2]}"#;
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let preprocessor = ImagePreprocessor::new(Compose::from_bytes(config).unwrap())
            .with_resize(move |_, width, height, filter| {
                assert_eq!((width, height), (4, 4));
                assert_eq!(filter, image::imageops::FilterType::CatmullRom);
                observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(RgbImage::from_pixel(width, height, image::Rgb([5, 7, 9])).into())
            });
        let pixels = preprocessor
            .clone()
            .preprocess(RgbImage::new(10, 20).into())
            .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(pixels.shape(), &[3, 2, 2]);
        assert_eq!(pixels[[0, 0, 0]], 0.75);
        assert_eq!(pixels[[1, 1, 1]], 1.25);
        assert_eq!(pixels[[2, 0, 1]], 1.75);
    }

    #[test]
    fn preprocessor_is_cloneable_and_normalizes_images() {
        let config = br#"{
            "image_processor_type": "CLIPImageProcessor",
            "do_resize": true,
            "size": { "height": 2, "width": 2 },
            "do_center_crop": true,
            "crop_size": { "height": 2, "width": 2 },
            "do_rescale": true,
            "rescale_factor": 0.5,
            "do_normalize": true,
            "image_mean": [1.0, 1.0, 1.0],
            "image_std": [2.0, 2.0, 2.0]
        }"#;
        let preprocessor =
            ImagePreprocessor::new(Compose::from_bytes(config).expect("valid test config"));
        let image = RgbImage::from_pixel(1, 1, image::Rgb([5, 7, 9]));

        let pixels = preprocessor
            .clone()
            .preprocess(DynamicImage::ImageRgb8(image))
            .expect("test image preprocesses");

        assert_eq!(pixels.shape(), &[3, 2, 2]);
        assert!((pixels[[0, 0, 0]] - 0.75).abs() < f32::EPSILON);
        assert!((pixels[[1, 0, 0]] - 1.25).abs() < f32::EPSILON);
        assert!((pixels[[2, 0, 0]] - 1.75).abs() < f32::EPSILON);
    }

    #[test]
    fn preprocessor_respects_non_square_resize_dimensions() {
        let config = br#"{
            "image_processor_type": "CLIPImageProcessor",
            "do_resize": true,
            "size": { "height": 3, "width": 5 },
            "do_center_crop": false,
            "do_rescale": false,
            "do_normalize": false
        }"#;
        let preprocessor =
            ImagePreprocessor::new(Compose::from_bytes(config).expect("valid test config"));
        let image = RgbImage::from_pixel(1, 1, image::Rgb([9, 7, 5]));

        let pixels = preprocessor
            .preprocess(DynamicImage::ImageRgb8(image))
            .expect("test image preprocesses");

        assert_eq!(pixels.shape(), &[3, 3, 5]);
    }

    #[test]
    fn preprocessor_respects_non_square_crop_dimensions() {
        let config = br#"{
            "image_processor_type": "CLIPImageProcessor",
            "do_resize": false,
            "do_center_crop": true,
            "crop_size": { "height": 3, "width": 5 },
            "do_rescale": false,
            "do_normalize": false
        }"#;
        let preprocessor =
            ImagePreprocessor::new(Compose::from_bytes(config).expect("valid test config"));
        let image = RgbImage::from_pixel(1, 1, image::Rgb([9, 7, 5]));

        let pixels = preprocessor
            .preprocess(DynamicImage::ImageRgb8(image))
            .expect("test image preprocesses");

        assert_eq!(pixels.shape(), &[3, 3, 5]);
        assert_eq!(pixels[[0, 1, 2]], 9.0);
        assert_eq!(pixels[[1, 1, 2]], 7.0);
        assert_eq!(pixels[[2, 1, 2]], 5.0);
    }
}
