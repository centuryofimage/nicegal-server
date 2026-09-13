@echo off
setlocal
cd /d "%~dp0"

if not exist .venv-directml\.deps-installed (
    uv venv --python 3.13 .venv-directml
    if errorlevel 1 exit /b %errorlevel%
    uv pip install --python .venv-directml -r requirements-directml.txt
    if errorlevel 1 exit /b %errorlevel%
    type nul > .venv-directml\.deps-installed
)

if not exist .venv-openvino\.deps-installed (
    uv venv --python 3.13 .venv-openvino
    if errorlevel 1 exit /b %errorlevel%
    uv pip install --python .venv-openvino -r requirements-openvino.txt
    if errorlevel 1 exit /b %errorlevel%
    type nul > .venv-openvino\.deps-installed
)

call "%~dp0dev.cmd" build --release --locked -p nicegal-server
if errorlevel 1 exit /b %errorlevel%
