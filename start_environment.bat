@echo off
cd /D "%~dp0"
set CONDA_ROOT_PREFIX=%~dp0alltalk_environment\conda
set INSTALL_ENV_DIR=%~dp0alltalk_environment\env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
cmd /k