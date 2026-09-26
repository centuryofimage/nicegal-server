@echo off
setlocal
cd /d "%~dp0"

if not exist .venv-directml\pyvenv.cfg uv venv --python 3.13 .venv-directml
if errorlevel 1 exit /b 1
uv pip install --python .venv-directml -r requirements-directml.txt
if errorlevel 1 exit /b 1

if not exist .venv-openvino\pyvenv.cfg uv venv --python 3.13 .venv-openvino
if errorlevel 1 exit /b 1
uv pip install --python .venv-openvino -r requirements-openvino.txt
if errorlevel 1 exit /b 1

set "NICEGAL_CUDA_FEATURE="
if /I "%NICEGAL_ENABLE_CUDA%"=="1" (
    if not exist .venv-cuda\pyvenv.cfg uv venv --python 3.13 .venv-cuda
    if errorlevel 1 exit /b 1
    uv pip install --python .venv-cuda -r requirements-cuda.txt
    if errorlevel 1 exit /b 1
    set "NICEGAL_CUDA_FEATURE=--features ort-cuda"
)

call "%~dp0dev.cmd" build --release --locked -p nicegal-server %NICEGAL_CUDA_FEATURE%
if errorlevel 1 exit /b 1
