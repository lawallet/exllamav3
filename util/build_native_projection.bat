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
set DIST_EXTRA_CONFIG=
cd /d "%~dp0.."
rem Reuse only compilation intermediates; the installed and old release binaries
rem are never a build destination. The controller preserves the old cache first.
"D:\tabbyAPI\venv\Scripts\python.exe" setup.py build_ext --build-temp native-results\deepseek-v41\package\build\temp.win-amd64-cpython-313 --build-lib native-results\deepseek-v41\attention-optimization\native-build\stage build_py --build-lib native-results\deepseek-v41\attention-optimization\native-build\stage > native-results\deepseek-v41\attention-optimization\native-build\build.log 2>&1
exit /b %errorlevel%
