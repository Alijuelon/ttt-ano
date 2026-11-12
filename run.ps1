# PowerShell Launcher Image-to-Voices TTS
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    & $venvPy (Join-Path $PSScriptRoot "main.py") @args
} else {
    python (Join-Path $PSScriptRoot "main.py") @args
}
