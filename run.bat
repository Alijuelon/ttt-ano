@echo off
REM Launcher Image-to-Voices TTS menggunakan virtual environment Python
set VENV_PY=%~dp0.venv\Scripts\python.exe
if exist "%VENV_PY%" (
    "%VENV_PY%" "%~dp0main.py" %*
) else (
    python "%~dp0main.py" %*
)
