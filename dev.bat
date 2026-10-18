@echo off
if not defined VCPKG_ROOT set "VCPKG_ROOT=%USERPROFILE%\vcpkg"
set "VCPKGRS_TRIPLET=x64-win-llvm-static-md-release"
if not defined LIBCLANG_PATH set "LIBCLANG_PATH=%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin"
REM Do not accidentally link Scoop's shared import libraries via a global FFMPEG_DIR.
REM Use the app-specific override when building with another static FFmpeg package.
if defined NICEGAL_FFMPEG_DIR (
    set "FFMPEG_DIR=%NICEGAL_FFMPEG_DIR%"
) else (
    set "FFMPEG_DIR=%USERPROFILE%\vcpkg\packages\ffmpeg_%VCPKGRS_TRIPLET%"
)
REM if not defined RUSTFLAGS set "RUSTFLAGS=-C target-feature=+crt-static"
REM build.rs reads the provider venvs by default (see build-server.cmd). Set
REM NICEGAL_DIRECTML_ORT_LIB_PATH, NICEGAL_OPENVINO_ORT_LIB_PATH,
REM NICEGAL_OPENVINO_LIB_PATH, NICEGAL_CUDA_ORT_LIB_PATH, or
REM NICEGAL_CUDA_LIB_PATH here only to use a different DLL source directory.
