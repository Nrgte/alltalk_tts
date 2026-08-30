@echo off
cd /D "%~dp0"
set CONDA_ROOT_PREFIX=%~dp0alltalk_environment\conda
set INSTALL_ENV_DIR=%~dp0alltalk_environment\env
set DS_BUILD_AIO=0
set DS_BUILD_GDS=0
if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
call python script.py
pause