@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set MAX_JOBS=8
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=8.9;12.0
set DISTUTILS_USE_SDK=1
set EXLLAMA_EXT_COMPRESS=require
set EXLLAMA_NOCOMPILE=
set PYTHONPATH=
cd /d "%~dp0.."
if not exist native-results\deepseek-v41\package mkdir native-results\deepseek-v41\package
"D:\tabbyAPI\venv\Scripts\python.exe" setup.py build --build-base native-results\deepseek-v41\package\build > native-results\deepseek-v41\package\build.log 2>&1
exit /b %errorlevel%
