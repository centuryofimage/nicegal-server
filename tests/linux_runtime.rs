#![cfg(all(target_os = "linux", feature = "ort-openvino"))]

use nicegal_core::runtime::{
    ExecutionProvider, RuntimeOptions, SessionSpec, initialize_bundled_runtime, load_sessions,
};
use ort::value::Tensor;

/// Run after build-server.sh has staged the native runtime next to the test executables.
#[test]
#[ignore = "requires the bundled Linux OpenVINO runtime; run build-server.sh first"]
fn bundled_openvino_and_cpu_execute_inference() -> anyhow::Result<()> {
    // ONNX Relu graph: float32[3] -> float32[3], opset 13, IR 8. No downloaded model needed.
    let model: &[u8] = &[
        8, 8, 58, 69, 10, 12, 10, 1, 120, 18, 1, 121, 34, 4, 82, 101, 108, 117, 18, 19, 108, 105,
        110, 117, 120, 45, 114, 117, 110, 116, 105, 109, 101, 45, 115, 109, 111, 107, 101, 90, 15,
        10, 1, 120, 18, 10, 10, 8, 8, 1, 18, 4, 10, 2, 8, 3, 98, 15, 10, 1, 121, 18, 10, 10, 8, 8,
        1, 18, 4, 10, 2, 8, 3, 66, 2, 16, 13,
    ];
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("relu.onnx");
    std::fs::write(&path, model)?;
    initialize_bundled_runtime(ExecutionProvider::OpenVino)?;
    for provider in [ExecutionProvider::OpenVino, ExecutionProvider::Cpu] {
        let mut loaded = load_sessions(
            [SessionSpec::new("smoke", &path)],
            RuntimeOptions {
                execution_provider: provider,
                allow_cpu_fallback: false,
                ..RuntimeOptions::default()
            },
        )?;
        assert_eq!(loaded.execution_provider, provider);
        let input = Tensor::from_array(([3usize], vec![-1.0f32, 0.0, 2.0]))?;
        let outputs = loaded.sessions[0].run(ort::inputs![input])?;
        assert_eq!(outputs[0].try_extract_tensor::<f32>()?.1, &[0.0, 0.0, 2.0]);
    }
    Ok(())
}
