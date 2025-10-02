# Meeting Assistant (Streamlit)

## Overview
Meeting Assistant is a lightweight, local-first web app that turns your meeting audio into structured insights in minutes. Upload an audio file (mp3/wav/m4a/mov), the app transcribes it using OpenAI Whisper-1, then asks an OpenAI Chat model (gpt-4o-mini by default) to extract a concise summary, a list of decisions, and actionable tasks with owners and due dates. A built-in Q&A section lets you ask questions strictly grounded in the transcript. The app is designed to be simple, fast, and production-lean: single-page Streamlit UI, minimal dependencies, environment-based configuration, robust JSON parsing with safe fallbacks, and temporary-file management for large audio.

## Flow
- Audio → Transcription (OpenAI Whisper-1)
- Transcript → Summary / Decisions / Actions (OpenAI Chat, gpt-4o-mini)
- Optional: Q&A grounded in the transcript

## Setup
1) Create and activate a Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Install ffmpeg (required by pydub)
- macOS: `brew install ffmpeg`
- Linux (Debian/Ubuntu): `sudo apt-get update && sudo apt-get install -y ffmpeg`

4) Create a .env file
Copy `.env.example` and fill in your key:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

5) (Optional) Increase upload limit
Set a higher upload limit in `.streamlit/config.toml` (example below sets 1GB):
```toml
[server]
maxUploadSize = 1024
```

## Run
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

## Notes
- Model names are managed via `.env`. `OPENAI_MODEL_CHAT` defaults to `gpt-4o-mini`.
- Whisper-1 requires internet access.
- JSON robustness: If the model returns extra text around JSON, the app trims from the first `{` to the last `}` and tries `json.loads`. If parsing still fails, it falls back to an empty decisions/actions with a summary made from the first 500 chars of the transcript.
- API keys are never logged; keys are read only from environment.
- Temporary files are properly cleaned up after processing.
- Supported uploads: `mp3`, `wav`, `m4a`, `mov`.

## Q&A (Transcript-grounded)
- Ask questions about the content of your transcript. The app retrieves the most relevant transcript chunks (via OpenAI embeddings with a robust fallback) and answers strictly from that context.
- If the information is not present in the transcript, the answer will be: “Transkriptte bilgi bulunamadı.”
- The temperature is set low (0.0–0.2) to minimize speculation.

## Troubleshooting
- ffmpeg/ffprobe not found: install ffmpeg as above. On macOS with Homebrew: `brew install ffmpeg`. You can also export specific paths via `FFMPEG_BINARY` and `FFPROBE_BINARY` env vars if needed.
- Upload limit still 200MB: ensure `.streamlit/config.toml` exists with `[server] maxUploadSize = 1024` and restart Streamlit. A fresh browser reload may be required.
- Port already in use: stop existing processes on `8501` (e.g., `lsof -ti tcp:8501 | xargs kill -9`) or run on another port with `--server.port`.
