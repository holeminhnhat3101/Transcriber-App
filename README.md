``` Brief description
# Audio Transcriber

Lightweight local transcription web app using Faster Whisper for ASR and a local Ollama LLM for text refinement. The app can optionally save transcripts to Google Docs if `credentials.json` is present and Google OAuth is configured.

## Quick summary
- Frontend: `static/index.html` + `static/CSS-styles/app.js`
- Backend: `main.py` (FastAPI)
- ASR: `faster-whisper`
- LLM: Local Ollama instance (configured via `OLLAMA_BASE_URL`, default `http://localhost:11434`)
- Google Docs (optional): requires `credentials.json` and enabling the Google Docs API

## Prerequisites
- Python 3.10+ (create a venv recommended)
- GPU with CUDA (recommended for Whisper). Minimum 8 GB VRAM recommended — larger Whisper/LLM models may require more VRAM. CPU-only will work but be substantially slower.

## Run (local)
Recommended (Windows): use the bundled `run.bat` which automates venv creation, dependency install, Ollama checks, and app launch.

To start the app (double-click or run in PowerShell/CMD):

```powershell
.\run.bat

```

**What `run.bat` does:**

* Creates/uses a `venv` virtual environment in the project folder if missing.
* Activates the venv and installs `requirements.txt`.
* Checks that the Ollama service is reachable and tries to detect the `qwen3:8b` model.
* Finally launches the backend with `python main.py`.

**Manual alternative (if you prefer to control each step):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
python main.py

```

### Stopping the app:

* To stop the transcription process entirely, run the included `stop.bat` (double-click or run `.\stop.bat`).
* `stop.bat` performs a forceful termination of Python processes (`taskkill /F /IM python.exe`) and then attempts to unload the Ollama model by calling the Ollama API.

> **Warning:** `stop.bat` forcefully kills all `python.exe` processes on the machine. If you have other Python apps running, close them first or stop the specific process manually instead of running `stop.bat`.

If you launched the app manually with `python main.py`, you can usually stop it with Ctrl+C in the terminal.

The web UI will be available at `http://127.0.0.1:8001/` by default.

## What if `credentials.json` is missing?

* The app will still transcribe audio locally (Whisper) and refine with the local LLM (Ollama).
* Google Docs upload/creation will be disabled if `credentials.json` is not present.
* To enable Google Docs integration:
1. Enable the Google Docs API in the Google Cloud Console.
2. Create OAuth 2.0 credentials (Desktop / Installed application) and download the `credentials.json` file into the project root.
3. You would need to add your email or the email that you want this app to write on in "test users" section.
4. The app will create `token.json` automatically after the first successful OAuth flow.



> **Tip:** For a step-by-step visual walkthrough on creating the `credentials.json` (Google OAuth 2.0 client), search YouTube for terms like "create credentials.json Google OAuth desktop" or "Google OAuth 2.0 credentials tutorial" and watch a recent guide — it shows the Console navigation and how to download the file.

**Notes on `credentials.json`:**

* **Do NOT leak this file.** It contains sensitive client information.
* The typical flow: when you first run a Google-scoped action, a browser window will open to authorize the app and create `token.json`.

## Ollama (local LLM) — quick guidance

This project expects a local Ollama-compatible service reachable at `OLLAMA_BASE_URL` (default `http://localhost:11434`). The backend posts to `/api/chat` and `/api/generate`.

1. Install Ollama following the official instructions: [https://ollama.com](https://ollama.com)
2. Pull or install the LLM you want to use (the repository uses `qwen3:8b` by default). Example (after installing ollama):

```powershell
# Pull the model (if available)
ollama pull qwen3:8b

```

3. Start the Ollama service / daemon per the official docs and ensure it is reachable at the URL in `.env` or `OLLAMA_BASE_URL`.
4. If you run Ollama on a different host/port, update `.env` or set the environment variable `OLLAMA_BASE_URL` before starting the Python app.

> **Important:** if Ollama is not running or the model is unavailable, the app will log errors when trying to call the LLM; transcription (Whisper) still works.

## Configuration

* `.env` — created automatically if missing. Defaults to:

```ini
OLLAMA_BASE_URL=http://localhost:11434

```

* `LLM_MODEL` is set in `main.py` as `qwen3:8b` (change there if you prefer a different model name).

## Common troubleshooting

* If you get CUDA OOM errors: the app aggressively clears GPU memory between steps, but you may need a larger GPU or lower model sizes.
* If Google OAuth flow fails: confirm `credentials.json` is a valid OAuth client and the Docs API is enabled.
* If Ollama calls return non-200 or fail: ensure Ollama daemon is running and the model is pulled/loaded.

## Files to look at

* `main.py` — backend server and processing pipeline
* `static/index.html` — web UI
* `static/CSS-styles/app.js` — frontend request flow