@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set MAX_JOBS=4
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=8.9;12.0
set DISTUTILS_USE_SDK=1
cd /d D:\exllamav3
"D:\tabbyAPI\venv\Scripts\python.exe" -m pip wheel -v --no-deps --no-build-isolation --wheel-dir native-results\glm53-concurrency\wheels . > native-results\glm53-concurrency\build.log 2>&1
exit /b %errorlevel%
