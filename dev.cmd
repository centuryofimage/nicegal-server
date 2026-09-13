@echo off
call "%~dp0dev.bat"
if errorlevel 1 exit /b %errorlevel%
cargo %*
