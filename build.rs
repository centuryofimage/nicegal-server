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
const ACCELERATED_ORT_FEATURES: &[&str] =
    &["CARGO_FEATURE_ORT_OPENVINO", "CARGO_FEATURE_ORT_DIRECTML"];

#[cfg(windows)]
fn main() {
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

/// Copy both provider-specific ONNX Runtime distributions into namespaced directories. `ort`
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

    copy_distribution(
        "directml",
        "NICEGAL_DIRECTML_ORT_LIB_PATH",
        manifest_dir.join(".venv-directml/Lib/site-packages/onnxruntime/capi"),
        profile_dir,
        &[],
    )?;
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
                    "cannot locate Visual Studio CRT; set NICEGAL_CRT_LIB_PATH to its target-architecture Microsoft.VC143.CRT directory",
                ));
            }
            let vc = PathBuf::from(installation).join("VC");
            let version = fs::read_to_string(
                vc.join("Auxiliary/Build/Microsoft.VCRedistVersion.default.txt"),
            )?;
            vc.join("Redist/MSVC").join(version.trim())
        };
        redist.join(platform).join("Microsoft.VC143.CRT")
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
    if !library_dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{env_var} does not name a directory: {} (create the provider venv with build-server.cmd or set {env_var} explicitly)",
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

        if !found_dll {
            for destination_dir in destinations {
                fs::create_dir_all(destination_dir)?;
            }
            found_dll = true;
        }

        println!("cargo:rerun-if-changed={}", source.display());
        let file_name = source.file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("DLL path has no file name: {}", source.display()),
            )
        })?;

        for destination_dir in destinations {
            let destination = destination_dir.join(file_name);
            if should_copy(&source, &destination)? {
                fs::copy(&source, destination)?;
            }
        }
    }

    if !found_dll {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{env_var} contains no DLL files: {}", library_dir.display()),
        ));
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
