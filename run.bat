@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

REM =================================================
REM 1. Check Python
REM =================================================
echo [STATUS] Checking Python...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your PATH.
    pause
    exit /b 1
)

REM =================================================
REM 2. Virtual Env Setup
REM =================================================
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

call "venv\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM =================================================
REM 3. Install Requirements
REM =================================================
echo.
echo [INFO] Checking dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PIP Install failed.
    pause
    exit /b 1
)

REM =================================================
REM 3.5. Check Ollama
REM =================================================
echo.
echo [INFO] Checking Ollama status...

REM Check if Ollama is reachable
ollama list >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ollama is not running!
    echo [FIX] Please launch the "Ollama" app from your Start Menu.
    pause
    exit /b 1
)

echo [INFO] Ollama is running. Verifying model...
echo.
echo --- YOUR OLLAMA MODELS ---
ollama list
echo --------------------------
echo.

REM Precise check for qwen3:8b
ollama list | findstr /C:"qwen3:8b" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Script could not detect 'qwen3:8b' automatically.
    echo [INFO] Since you confirmed it exists, we will proceed anyway.
) else (
    echo [SUCCESS] Model 'qwen3:8b' detected!
)

REM =================================================
REM 4. Run App
REM =================================================
echo.
echo [SUCCESS] Launching Transcriber...
python main.py
pause