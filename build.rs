#[cfg(windows)]
use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
};

/// The Rust-side provider registrations that mean the server needs both dynamically loaded
/// ONNX Runtime distributions bundled beside it.
#[cfg(windows)]
const ACCELERATED_ORT_FEATURES: &[&str] = &[
    "CARGO_FEATURE_ORT_OPENVINO",
    "CARGO_FEATURE_ORT_DIRECTML",
    "CARGO_FEATURE_ORT_CUDA",
];

#[cfg(windows)]
fn main() {
    // FFMPEG_DIR bypasses ffmpeg-sys's vcpkg system-library discovery. These
    // dependencies are listed by the custom static build's pkg-config files.
    for library in ["ole32", "user32", "bcrypt"] {
        println!("cargo:rustc-link-lib={library}");
    }
    for feature in ACCELERATED_ORT_FEATURES {
        println!("cargo:rerun-if-env-changed={feature}");
    }
    if !ACCELERATED_ORT_FEATURES
        .iter()
        .any(|feature| env::var_os(feature).is_some())
    {
        return;
    }

    for env_var in [
        "NICEGAL_DIRECTML_ORT_LIB_PATH",
        "NICEGAL_OPENVINO_ORT_LIB_PATH",
        "NICEGAL_OPENVINO_LIB_PATH",
        "NICEGAL_CUDA_ORT_LIB_PATH",
        "NICEGAL_CUDA_LIB_PATH",
        "NICEGAL_CRT_LIB_PATH",
        "VCToolsRedistDir",
        "ProgramFiles(x86)",
    ] {
        println!("cargo:rerun-if-env-changed={env_var}");
    }

    if let Err(error) = copy_runtime_distributions() {
        panic!("failed to copy Windows runtime distributions: {error}");
    }
}

#[cfg(not(windows))]
fn main() {}

/// Copy provider-specific ONNX Runtime distributions into namespaced directories. `ort`
/// dynamically loads precisely one at startup, so the DLLs must never overwrite each other.
#[cfg(windows)]
fn copy_runtime_distributions() -> io::Result<()> {
    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "OUT_DIR is not set by Cargo")
        })?);
    let profile_dir = profile_dir(&out_dir)?;
    copy_crt(profile_dir)?;
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "CARGO_MANIFEST_DIR is not set by Cargo",
        )
    })?);

    if env::var_os("CARGO_FEATURE_ORT_DIRECTML").is_some() {
        copy_distribution(
            "directml",
            "NICEGAL_DIRECTML_ORT_LIB_PATH",
            manifest_dir.join(".venv-directml/Lib/site-packages/onnxruntime/capi"),
            profile_dir,
            &[],
        )?;
    }
    if env::var_os("CARGO_FEATURE_ORT_OPENVINO").is_some() {
        copy_distribution(
            "openvino",
            "NICEGAL_OPENVINO_ORT_LIB_PATH",
            manifest_dir.join(".venv-openvino/Lib/site-packages/onnxruntime/capi"),
            profile_dir,
            &[(
                "NICEGAL_OPENVINO_LIB_PATH",
                manifest_dir.join(".venv-openvino/Lib/site-packages/openvino/libs"),
            )],
        )?;
    }
    if env::var_os("CARGO_FEATURE_ORT_CUDA").is_some() {
        copy_distribution(
            "cuda",
            "NICEGAL_CUDA_ORT_LIB_PATH",
            manifest_dir.join(".venv-cuda/Lib/site-packages/onnxruntime/capi"),
            profile_dir,
            &[],
        )?;
        let cuda_destinations = [
            profile_dir.join("onnxruntime/cuda"),
            profile_dir.join("deps/onnxruntime/cuda"),
        ];
        let cuda_libraries = env::var_os("NICEGAL_CUDA_LIB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest_dir.join(".venv-cuda/Lib/site-packages/nvidia"));
        copy_dll_tree("NICEGAL_CUDA_LIB_PATH", &cuda_libraries, &cuda_destinations)?;
    }
    Ok(())
}

/// Use Visual Studio's redistributable files, never DLLs from System32.
#[cfg(windows)]
fn copy_crt(profile_dir: &Path) -> io::Result<()> {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let platform = match arch.as_str() {
        "x86_64" => "x64",
        "x86" => "x86",
        "aarch64" => "arm64",
        _ => {
            return Err(io::Error::other(format!(
                "unsupported CRT target architecture: {arch}"
            )));
        }
    };
    let library_dir = if let Some(path) = env::var_os("NICEGAL_CRT_LIB_PATH") {
        PathBuf::from(path)
    } else {
        let redist = if let Some(path) = env::var_os("VCToolsRedistDir") {
            PathBuf::from(path)
        } else {
            let program_files =
                env::var_os("ProgramFiles(x86)").unwrap_or_else(|| "C:/Program Files (x86)".into());
            let vswhere =
                PathBuf::from(program_files).join("Microsoft Visual Studio/Installer/vswhere.exe");
            let output = std::process::Command::new(vswhere)
                .args([
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                    "-utf8",
                ])
                .output()?;
            let installation = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !output.status.success() || installation.is_empty() {
                return Err(io::Error::other(
                    "cannot locate Visual Studio CRT; set NICEGAL_CRT_LIB_PATH to its target-architecture Microsoft.VC145.CRT directory",
                ));
            }
            let vc = PathBuf::from(installation).join("VC");
            let version = fs::read_to_string(
                vc.join("Auxiliary/Build/Microsoft.VCRedistVersion.default.txt"),
            )?;
            vc.join("Redist/MSVC").join(version.trim())
        };
        redist.join(platform).join("Microsoft.VC145.CRT")
    };
    for required in [
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
    ] {
        if required == "vcruntime140_1.dll" && platform == "x86" {
            continue;
        }
        if !library_dir.join(required).is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "CRT directory {} is missing {required}; set NICEGAL_CRT_LIB_PATH to a complete redistributable directory",
                    library_dir.display()
                ),
            ));
        }
    }
    copy_dll_directory(
        "NICEGAL_CRT_LIB_PATH",
        library_dir,
        &[profile_dir.to_owned(), profile_dir.join("deps")],
    )
}

#[cfg(windows)]
fn copy_distribution(
    name: &str,
    ort_env_var: &str,
    default_ort_dir: PathBuf,
    profile_dir: &Path,
    extra_dll_directories: &[(&str, PathBuf)],
) -> io::Result<()> {
    let destinations = [
        profile_dir.join("onnxruntime").join(name),
        profile_dir.join("deps").join("onnxruntime").join(name),
    ];
    copy_dll_directory(ort_env_var, default_ort_dir, &destinations)?;
    for (env_var, default_dir) in extra_dll_directories {
        copy_dll_directory(env_var, default_dir.clone(), &destinations)?;
    }
    Ok(())
}

#[cfg(windows)]
fn copy_dll_directory(env_var: &str, default: PathBuf, destinations: &[PathBuf]) -> io::Result<()> {
    let library_dir = env::var_os(env_var).map(PathBuf::from).unwrap_or(default);
    copy_dlls_from(env_var, library_dir, destinations)
}

#[cfg(windows)]
fn copy_dlls_from(
    source_name: &str,
    library_dir: PathBuf,
    destinations: &[PathBuf],
) -> io::Result<()> {
    if !library_dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{source_name} does not name a directory: {}",
                library_dir.display()
            ),
        ));
    }

    println!("cargo:rerun-if-changed={}", library_dir.display());
    let mut found_dll = false;
    for entry in fs::read_dir(&library_dir)? {
        let source = entry?.path();
        if !source.is_file() || !is_dll(&source) {
            continue;
        }
        copy_dll(source_name, &source, destinations)?;
        found_dll = true;
    }

    if !found_dll {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{source_name} contains no DLL files: {}",
                library_dir.display()
            ),
        ));
    }
    Ok(())
}

/// CUDA extras install DLLs in separate `nvidia/<package>/bin` directories.
#[cfg(windows)]
fn copy_dll_tree(source_name: &str, root: &Path, destinations: &[PathBuf]) -> io::Result<()> {
    if !root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{source_name} does not name a directory: {}",
                root.display()
            ),
        ));
    }
    let mut directories = vec![root.to_owned()];
    let mut found_dll = false;
    while let Some(directory) = directories.pop() {
        println!("cargo:rerun-if-changed={}", directory.display());
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.is_dir() {
                directories.push(path);
            } else if path.is_file() && is_dll(&path) {
                copy_dll(source_name, &path, destinations)?;
                found_dll = true;
            }
        }
    }
    if !found_dll {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{source_name} contains no DLL files: {}", root.display()),
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn copy_dll(source_name: &str, source: &Path, destinations: &[PathBuf]) -> io::Result<()> {
    println!("cargo:rerun-if-changed={}", source.display());
    let file_name = source.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DLL path has no file name: {}", source.display()),
        )
    })?;
    // The GPU wheel also ships a TensorRT provider, and NVIDIA's extra wheels contain
    // standalone wrappers and an alternate NVRTC build. None are reached by the CUDA
    // provider's imports or its dependency DLLs' runtime-loaded references.
    let excluded = match source_name {
        "NICEGAL_CUDA_ORT_LIB_PATH" => {
            ["onnxruntime_providers_tensorrt.dll"].contains(&file_name.to_string_lossy().as_ref())
        }
        "NICEGAL_CUDA_LIB_PATH" => [
            "cufftw64_11.dll",
            "curand64_10.dll",
            "nvblas64_12.dll",
            "nvrtc64_120_0.alt.dll",
        ]
        .contains(&file_name.to_string_lossy().as_ref()),
        _ => false,
    };
    if excluded {
        for destination_dir in destinations {
            let destination = destination_dir.join(file_name);
            if destination.exists() {
                fs::remove_file(destination)?;
            }
        }
        return Ok(());
    }
    for destination_dir in destinations {
        fs::create_dir_all(destination_dir)?;
        let destination = destination_dir.join(file_name);
        if should_copy(source, &destination)? {
            fs::copy(source, destination)?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn is_dll(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case(OsStr::new("dll")))
}

#[cfg(windows)]
fn profile_dir(out_dir: &Path) -> io::Result<&Path> {
    let package_dir = out_dir
        .parent()
        .filter(|_| out_dir.file_name() == Some(OsStr::new("out")));
    let build_dir = package_dir
        .and_then(Path::parent)
        .filter(|build_dir| build_dir.file_name() == Some(OsStr::new("build")));
    let profile_dir = build_dir.and_then(Path::parent);

    profile_dir.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "OUT_DIR does not match Cargo's target/<profile>/build/<package-hash>/out layout: {}",
                out_dir.display()
            ),
        )
    })
}

#[cfg(windows)]
fn should_copy(source: &Path, destination: &Path) -> io::Result<bool> {
    let source_metadata = fs::metadata(source)?;
    let Ok(destination_metadata) = fs::metadata(destination) else {
        return Ok(true);
    };

    if source_metadata.len() != destination_metadata.len() {
        return Ok(true);
    }

    let (Ok(source_modified), Ok(destination_modified)) =
        (source_metadata.modified(), destination_metadata.modified())
    else {
        return Ok(true);
    };

    Ok(destination_modified < source_modified)
}
