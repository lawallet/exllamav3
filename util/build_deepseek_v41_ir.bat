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
if /I "%~1"=="--python-only" goto refresh_python
"D:\tabbyAPI\venv\Scripts\python.exe" setup.py build --build-base native-results\deepseek-v41\package\build > native-results\deepseek-v41\package\build.log 2>&1
if errorlevel 1 exit /b 1
goto package
:refresh_python
rem Keep the exact tested native binary when only Python/release metadata changed.
if not exist native-results\deepseek-v41\package\build\lib.win-amd64-cpython-313\exllamav3_ext.cp313-win_amd64.pyd exit /b 1
"D:\tabbyAPI\venv\Scripts\python.exe" setup.py build_py --build-lib native-results\deepseek-v41\package\build\lib.win-amd64-cpython-313 > native-results\deepseek-v41\package\python-refresh.log 2>&1
if errorlevel 1 exit /b 1
:package
set DIST_EXTRA_CONFIG=%CD%\native-results\deepseek-v41\package\build.ini
> "%DIST_EXTRA_CONFIG%" echo [build]
>> "%DIST_EXTRA_CONFIG%" echo build_base = native-results/deepseek-v41/package/build
"D:\tabbyAPI\venv\Scripts\python.exe" setup.py bdist_wheel --skip-build --bdist-dir native-results\deepseek-v41\package\bdist --dist-dir native-results\deepseek-v41\package\wheels > native-results\deepseek-v41\package\wheel.log 2>&1
exit /b %errorlevel%
