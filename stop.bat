@echo off
setlocal

echo [STOP] Killing Transcription App...
REM 1. Kill the Python script and the CMD window running it
taskkill /F /IM python.exe /T >nul 2>&1

echo [STOP] Unloading AI Models...
REM 2. Send signal to Ollama to unload qwen3
curl -X POST http://localhost:11434/api/generate -d "{\"model\": \"qwen3:8b\", \"keep_alive\": 0}" >nul 2>&1

REM Optional: Kill Ollama entirely if you want to save maximum RAM
REM taskkill /F /IM ollama_app.exe /T >nul 2>&1

echo.
echo [DONE] Python stopped. GPU memory freed.
timeout /t 2