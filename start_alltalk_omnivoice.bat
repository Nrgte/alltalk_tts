@echo off
cd /D "%~dp0"
set CONDA_ROOT_PREFIX=%~dp0alltalk_environment\conda
set INSTALL_ENV_DIR=%~dp0alltalk_environment\env
set DS_BUILD_AIO=0
set DS_BUILD_GDS=0
if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"

echo Setting active TTS engine to omnivoice (k2-fsa/OmniVoice) ...
python -c "import json, pathlib; p = pathlib.Path(r'system/tts_engines/tts_engines.json'); d = json.loads(p.read_text()); d['engine_loaded'] = 'omnivoice'; d['selected_model'] = 'omnivoice - k2-fsa/OmniVoice'; [e.update(selected_model='omnivoice - k2-fsa/OmniVoice') for e in d['engines_available'] if e['name'] == 'omnivoice']; p.write_text(json.dumps(d, indent=4)); print('Active engine: ' + d['engine_loaded'] + ' (' + d['selected_model'] + ')')"
if errorlevel 1 (
    echo Failed to set the omnivoice engine in tts_engines.json.
    pause
    exit /b 1
)

call python script.py
pause
