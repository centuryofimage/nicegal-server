@echo off
if not defined VCToolsInstallDir (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    if errorlevel 1 exit /b 1
)
call "%~dp0dev.bat"
if errorlevel 1 exit /b %errorlevel%
cargo %*
