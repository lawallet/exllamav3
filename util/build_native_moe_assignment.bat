@echo off
rem Isolated build: does not replace or install the live inference extension.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
if not defined MAX_JOBS set MAX_JOBS=8
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=8.9;12.0
set DISTUTILS_USE_SDK=1
set EXLLAMA_EXT_COMPRESS=require
set EXLLAMA_NOCOMPILE=
set PYTHONPATH=
cd /d "%~dp0.."
if not exist native-results\native-ir-moe-assignment mkdir native-results\native-ir-moe-assignment
"D:\text-generation-webui\installer_files\env\python.exe" setup.py build_ext --build-temp native-results\native-ir-moe-assignment\build --build-lib native-results\native-ir-moe-assignment\extension > native-results\native-ir-moe-assignment\build.log 2>&1
exit /b %errorlevel%
