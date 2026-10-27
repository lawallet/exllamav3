@echo off
rem Include the Windows toolchain for upstream tests that JIT-build tiny probes.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=8.9;12.0
set DISTUTILS_USE_SDK=1
if not defined MAX_JOBS set MAX_JOBS=8
cd /d "%~dp0.."
"D:\tabbyAPI\venv\Scripts\python.exe" util\validate_v165_integration.py %*
exit /b %errorlevel%
