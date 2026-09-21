@echo off
if not defined VCPKG_ROOT set "VCPKG_ROOT=%USERPROFILE%\vcpkg"
set "VCPKGRS_TRIPLET=x64-win-llvm-lto-static-md-rel"
if not defined LIBCLANG_PATH set "LIBCLANG_PATH=%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin"
REM if not defined RUSTFLAGS set "RUSTFLAGS=-C target-feature=+crt-static"
REM build.rs reads the two provider venvs by default (see build-server.cmd). Set
REM NICEGAL_DIRECTML_ORT_LIB_PATH, NICEGAL_OPENVINO_ORT_LIB_PATH, or
REM NICEGAL_OPENVINO_LIB_PATH here only to use a different DLL source directory.
