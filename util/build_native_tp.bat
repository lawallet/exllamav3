@echo off
rem Build an isolated wheel; do not install into the live backend.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
if not defined MAX_JOBS set MAX_JOBS=32
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=8.9;12.0
set DISTUTILS_USE_SDK=1
set EXLLAMA_EXT_COMPRESS=require
set EXLLAMA_NOCOMPILE=
set PYTHONPATH=
cd /d "%~dp0.."
if not exist native-results\tp\package mkdir native-results\tp\package
rem Separate invocations avoid BuildExtension mutating flags and rebuilding twice.
"D:\tabbyAPI\venv\Scripts\python.exe" setup.py build --build-base native-results\tp\package\build > native-results\tp\package\build.log 2>&1
if errorlevel 1 exit /b 1
set DIST_EXTRA_CONFIG=%CD%\native-results\tp\package\build.ini
> "%DIST_EXTRA_CONFIG%" echo [build]
>> "%DIST_EXTRA_CONFIG%" echo build_base = native-results/tp/package/build
"D:\tabbyAPI\venv\Scripts\python.exe" setup.py bdist_wheel --skip-build --bdist-dir native-results\tp\package\bdist --dist-dir native-results\tp\package\wheels > native-results\tp\package\wheel.log 2>&1
exit /b %errorlevel%
