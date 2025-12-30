import os
import time
import asyncio
import re
import logging
import gc
import sys
import json
from typing import List
from functools import partial

import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# AI / ML Imports
from faster_whisper import WhisperModel
import static_ffmpeg

# Google Imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# --------------------------------------------------
# LOGGING & CONFIGURATION
# --------------------------------------------------

# Suppress verbose logs from third-party libraries to keep the console output clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("transcriber")

# Initialize local environment config if missing
if not os.path.exists(".env"):
    with open(".env", "w") as f:
        f.write("# Local LLM Configuration\nOLLAMA_BASE_URL=http://localhost:11434\n")
load_dotenv()

# Attempt to load static ffmpeg binaries for system independence
try:
    static_ffmpeg.add_paths()
except Exception as e:
    logger.warning(f"Could not set static ffmpeg paths: {e}. Relying on system PATH.")

SCOPES = ["https://www.googleapis.com/auth/documents"]
MODEL_SIZE = "large-v3"
LLM_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

app = FastAPI()
os.makedirs("static/CSS-styles", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------
# SYSTEM HELPERS
# --------------------------------------------------

def force_cuda_cleanup():
    """
    Aggressively clears GPU cache. 
    Crucial when switching between Whisper (VRAM heavy) and Ollama (VRAM heavy)
    to prevent Out-Of-Memory errors on consumer cards.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("🧹 GPU Memory Flushed")

def smart_chunk_text(text, max_chars=1500):
    """
    Splits text into chunks that fit within the LLM's context window.
    Uses regex to split strictly at sentence boundaries (. ? !) to avoid 
    cutting context in the middle of a thought.
    """
    # Look behind for punctuation followed by whitespace
    sentences = re.split(r'(?<=[.?!])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds the limit, push the current chunk
        if current_length + sentence_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# --------------------------------------------------
# GOOGLE INTEGRATION
# --------------------------------------------------

def get_google_service():
    """Authenticates with Google and returns the Docs service object."""
    creds = None
    token_path = "token.json"
    
    # Load existing token
    if os.path.exists(token_path):
        try: creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception: creds = None
            
    # Refresh or Create new token via OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try: creds.refresh(Request())
            except Exception: creds = None
        else:
            if not os.path.exists("credentials.json"): return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception: return None
        if creds:
            with open(token_path, "w") as f: f.write(creds.to_json())
            
    return build("docs", "v1", credentials=creds) if creds else None

def create_doc(service, title):
    if not service: return None
    try: return service.documents().create(body={"title": title}).execute()["documentId"]
    except Exception as e:
        logger.error(f"Failed to create Google Doc: {e}")
        return None

def append_to_doc(service, doc_id, title, text, is_first):
    """Appends text to a Google Doc with simple formatting (Headers vs Body)."""
    if not service or not doc_id: return
    try:
        doc = service.documents().get(documentId=doc_id).execute()
        # Determine where to append (end of document)
        end_index = 1
        try:
            if "body" in doc and "content" in doc["body"]:
                end_index = int(doc["body"]["content"][-1].get("endIndex", 1))
        except: pass

        insert_index = max(1, end_index - 1)
        requests_body = []

        # Insert page break for subsequent files
        if not is_first:
            requests_body.append({"insertPageBreak": {"location": {"index": insert_index}}})
            insert_index += 1

        # Format Title as Heading 1
        requests_body.append({"insertText": {"location": {"index": insert_index}, "text": f"{title}\n"}})
        title_end = insert_index + len(title) + 1
        requests_body.append({
            "updateParagraphStyle": {
                "range": {"startIndex": insert_index, "endIndex": title_end},
                "paragraphStyle": {"namedStyleType": "HEADING_1"},
                "fields": "namedStyleType"
            }
        })

        # Insert main content
        requests_body.append({"insertText": {"location": {"index": title_end}, "text": f"\n{text}\n"}})
        service.documents().batchUpdate(documentId=doc_id, body={"requests": requests_body}).execute()
    except Exception as e:
        logger.error(f"Error appending to doc: {e}")

# --------------------------------------------------
# CORE LOGIC: WHISPER & LLM
# --------------------------------------------------

def run_whisper_job(file_path, prompt):
    """
    Runs local transcription.
    Note: 'initial_prompt' is used to bias Whisper towards specific vocabulary or style.
    """
    logger.info(f"🎧 Loading Whisper ({MODEL_SIZE})...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        logger.info("🎙️ Transcribing...")
        segments, _ = model.transcribe(
            file_path, beam_size=5, language="vi",
            initial_prompt=prompt, condition_on_previous_text=False, temperature=0.0
        )
        text = " ".join(s.text for s in segments)
        del model
    except Exception as e:
        logger.error(f"Whisper Error: {e}")
        text = ""
    force_cuda_cleanup()
    return text

def clean_with_qwen(full_text):
    if not full_text: return ""
    
    text_chunks = smart_chunk_text(full_text, max_chars=2000)
    cleaned_parts = []
    logger.info(f"🧠 {LLM_MODEL} processing {len(text_chunks)} chunks...")

    # System Prompt: Enforces strict text processing (grammar/structure) 
    # and explicitly forbids conversational responses (hallucination prevention).
    system_instruction = """
ROLE: You are a TEXT PROCESSING SOFTWARE, NOT A CHATBOT. TASK: Receive Input (Raw Transcript) -> Fix Grammar/Structure -> Return Output (Clean Text).
ABSOLUTE SILENCE PROTOCOL (CRITICAL - DO NOT VIOLATE):
ONLY RETURN THE PROCESSED TEXT. Do not add any conversational filler, intro, or outro.
FORBIDDEN PHRASES: Do not use phrases like "Here is the corrected text", "Note:", "Output:", or "Sure".
NO EXPLANATIONS: Do not explain why changes were made. Do not reply with "Understood" or "Ok".
NO MARKDOWN FORMATTING: Do not use bolding, headers, or blockquotes in the final output. Return standard, plain text only.
FORMATTING: Apply proper punctuation, capitalization, line breaks, and paragraphing based on the context flow.
CONTENT PROCESSING RULES:
SENTENCE REPAIR (Fix Segmentation):
Connect fragmented sentences caused by speech-to-text errors.
Input: "This project. Very potential. Good development."
Output: "This project is very potential and has good development."
LOOP KILLING (De-duplication):
Remove filler words (uh, um, ah, like) if they clutter the meaning.
If a phrase is repeated continuously (stuttering or glitch), DELETE REPETITIONS, KEEP ONLY 1 INSTANCE.
ENTITY AND TERMINOLOGY HANDLING:
Proper Names/Places: Capitalize correctly (e.g., Vietnam, Hanoi, Samsung, Apple, Bitcoin).
Technical Terms/Slang/Foreign Words: KEEP ORIGINAL (Copy-Paste).
SAFETY FIRST: If you encounter a technical term, slang, or a foreign word you are unsure about, LEAVE IT ALONE. Do not attempt to translate or "fix" it into a similar-sounding word.
EXAMPLE (INPUT TO OUTPUT):
Input: "now let's talk about AI artificial intelligence it is growing very fast very fast chatgpt or midjourney it helps a lot helps a lot for work however however there are risks risks about data privacy engineers need to pay close attention to this"
Output: "Now let's talk about AI (Artificial Intelligence); it is growing very fast. ChatGPT or Midjourney helps a lot for work. However, there are risks regarding data privacy. Engineers need to pay close attention to this."
BEGIN PROCESSING INPUT:
"""
    url = f"{OLLAMA_BASE_URL}/api/chat"

    for i, chunk in enumerate(text_chunks):
        original_len = len(chunk)
        max_retries = 3
        attempt = 0
        final_chunk_result = chunk # Default to original if LLM fails
        
        while attempt < max_retries:
            try:
                payload = {
                    "model": LLM_MODEL,
                    "stream": False,
                    "messages": [
                        { "role": "system", "content": system_instruction },
                        { "role": "user", "content": f"Fix the following text:\n\n{chunk}" }
                    ],
                    "options": { "num_predict": -1, "temperature": 0.2, "num_ctx": 8192 }
                }

                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    result = response.json().get("message", {}).get("content", "").strip()
                    result_len = len(result)
                    
                    # Sanity Check: Validation Logic
                    # If the output length is suspiciously different from input, the LLM likely 
                    # hallucinated or stripped too much content.
                    
                    # Case 1: Too short (< 80%). Likely accidental deletion.
                    if result_len < original_len * 0.80:
                        logger.warning(f"⚠️ REJECT Chunk {i+1}: Output too short. Retry {attempt+1}...")
                        attempt += 1
                        continue
                    
                    # Case 2: Too long (> 110%). Likely added conversational filler.
                    elif result_len > original_len * 1.10:
                        logger.warning(f"⚠️ REJECT Chunk {i+1}: Output too long. Retry {attempt+1}...")
                        attempt += 1
                        continue
                    
                    # Case 3: Acceptable range.
                    else:
                        final_chunk_result = result
                        break 
                else:
                    logger.error(f"Ollama Error: {response.status_code}")
                    attempt += 1
                    
            except Exception as e:
                logger.error(f"LLM Connection Error: {e}")
                attempt += 1

        cleaned_parts.append(final_chunk_result)

    # Ensure model is unloaded to free VRAM for the next request/Whisper
    try:
        requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": LLM_MODEL, "keep_alive": 0})
    except: pass

    return "\n\n".join(cleaned_parts)

# --------------------------------------------------
# API ROUTES
# --------------------------------------------------

@app.get("/")
def ui():
    index_path = "static/index.html"
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Error: static/index.html not found</h1>", status_code=404)

@app.post("/transcribe")
async def transcribe(
    prompt: str = Form(""), 
    single_doc: bool = Form(True), # <--- NEW: Logic toggle
    files: List[UploadFile] = File(...)
):
    loop = asyncio.get_running_loop()
    results = []
    operation_log = []
    service = None
    
    # 0. Setup Identifiers & Paths
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    master_doc_id = None
    
    # Local Directory Setup
    local_dir = "transcriptions"
    os.makedirs(local_dir, exist_ok=True)
    local_batch_path = os.path.join(local_dir, f"Batch_Transcripts_{timestamp}.txt")

    # 1. Attempt Google Login
    if os.path.exists("credentials.json"):
        try:
            service = get_google_service()
            # [Merge Logic] If merging, create the Master Google Doc ONCE now
            if service and single_doc:
                doc_title = f"Transcript Batch {timestamp}"
                master_doc_id = create_doc(service, doc_title)
        except Exception as e:
            operation_log.append(f"Google Doc Error: {e}")
            logger.warning("Google Login Failed. Will fall back to local .txt files.")

    # Inject domain-specific vocabulary into Whisper prompt
    base_prompt = "Begin transcription."
    final_prompt = f"{prompt}. {base_prompt}"

    for idx, f in enumerate(files):
        file_logs = [f"Processing: {f.filename}"]
        if os.path.splitext(f.filename)[1].lower() not in ALLOWED_EXTENSIONS: continue
        
        temp_path = f"__tmp_{time.time_ns()}_{f.filename}"
        
        try:
            file_bytes = await f.read()
            with open(temp_path, "wb") as out: out.write(file_bytes)

            file_logs.append(f"Step 1: Whisper Transcription ({MODEL_SIZE})...")
            raw_text = await loop.run_in_executor(None, run_whisper_job, temp_path, final_prompt)
            
            file_logs.append(f"Step 2: LLM Refinement ({LLM_MODEL})...")
            refined_text = await loop.run_in_executor(None, clean_with_qwen, raw_text)

            # --- BRANCHING LOGIC: SAVE RESULTS ---
            individual_doc_id = None # Store ID if creating separate docs

            # OPTION A: Google Docs (Primary)
            if service:
                if single_doc and master_doc_id:
                    # [Merge] Append to Master Doc
                    file_logs.append("Step 3: Appending to Master Google Doc...")
                    await loop.run_in_executor(None, append_to_doc, service, master_doc_id, f.filename, refined_text, idx == 0)
                else:
                    # [Separate] Create NEW Doc for this file
                    file_logs.append("Step 3: Creating Separate Google Doc...")
                    indiv_title = f"{f.filename} - {time.strftime('%H:%M')}"
                    individual_doc_id = create_doc(service, indiv_title)
                    if individual_doc_id:
                        await loop.run_in_executor(None, append_to_doc, service, individual_doc_id, f.filename, refined_text, True)

            # OPTION B: Local File (Fallback)
            else:
                if single_doc:
                    # [Merge] Append to Batch .txt
                    file_logs.append(f"Step 3: Appending to local batch file...")
                    with open(local_batch_path, "a", encoding="utf-8") as text_file:
                        # Add a separator so you know where one file ends and next begins
                        text_file.write(f"\n{'='*50}\n")
                        text_file.write(f"FILE: {f.filename}\n")
                        text_file.write(f"DATE: {time.strftime('%Y-%m-%d %H:%M')}\n")
                        text_file.write(f"{'='*50}\n\n")
                        text_file.write(refined_text + "\n\n")
                    logger.info(f"💾 Appended to: {local_batch_path}")
                else:
                    # [Separate] Create Individual .txt
                    file_logs.append(f"Step 3: Saving individual local file...")
                    safe_filename = os.path.splitext(f.filename)[0] + "_cleaned.txt"
                    local_path = os.path.join(local_dir, safe_filename)
                    
                    with open(local_path, "w", encoding="utf-8") as text_file:
                        text_file.write(f"TITLE: {f.filename}\n")
                        text_file.write("-" * 40 + "\n\n")
                        text_file.write(refined_text)
                    logger.info(f"💾 Saved: {local_path}")

            results.append({
                "filename": f.filename, 
                "text": refined_text, 
                "logs": file_logs,
                "individual_doc_id": individual_doc_id
            })
            
        except Exception as e:
            logger.exception("Error during processing")
            results.append({"filename": f.filename, "text": f"Error: {e}", "logs": file_logs})
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)
            force_cuda_cleanup()

    return { "results": results, "google_doc_id": master_doc_id, "operation_log": operation_log }

if __name__ == "__main__":
    import uvicorn
    print("\n--- APP STARTED ---")
    uvicorn.run(app, host="127.0.0.1", port=8001)