import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import re
import unicodedata

import streamlit as st
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import which
from openai import OpenAI
import numpy as np


# -----------------------------
# Helpers for text normalization
# -----------------------------

def _normalize_text(value: str) -> str:
    """Lowercase, strip accents/punctuation for simple matching (Turkish-friendly)."""
    if not value:
        return ""
    # Normalize accents
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    # Lowercase
    value = value.lower()
    # Remove punctuation (keep spaces and alphanumerics)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    # Collapse spaces
    value = re.sub(r"\s+", " ", value).strip()
    return value

# -----------------------------
# Config & Environment
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")
OPENAI_MODEL_WHISPER = "whisper-1"


@dataclass
class ActionItem:
    owner: str
    task: str
    due: str


def save_uploaded_audio(uploaded_file: Any) -> str:
    """Save the uploaded audio file to a temporary location and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def _configure_pydub_binaries() -> None:
    """Configure pydub to use system ffmpeg/ffprobe if available, else raise."""
    ffmpeg_path = os.getenv("FFMPEG_BINARY") or which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    ffprobe_path = os.getenv("FFPROBE_BINARY") or which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
    if ffprobe_path:
        AudioSegment.ffprobe = ffprobe_path
    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError(
            "ffmpeg/ffprobe bulunamadı. macOS: 'brew install ffmpeg', Linux: 'sudo apt-get install -y ffmpeg'"
        )


def ensure_wav_16k(input_path: str) -> str:
    """Convert the audio to 16kHz mono WAV if needed, returning a new temp path."""
    _configure_pydub_binaries()
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    audio.export(wav_path, format="wav")
    return wav_path


def transcribe_openai_whisper(file_path: str) -> str:
    """Transcribe audio using OpenAI Whisper-1 and return the text."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY eksik. Lütfen .env dosyanızı kontrol edin.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=OPENAI_MODEL_WHISPER,
            file=f,
        )
    # SDK returns an object with text attribute
    return getattr(resp, "text", "").strip()


def run_openai_chat(transcript_text: str) -> str:
    """Call OpenAI Chat to produce structured JSON from transcript."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY eksik. Lütfen .env dosyanızı kontrol edin.")

    system_prompt = (
        "Sen deneyimli bir toplantı asistanısın. Görevlerin:\n"
        " (1) Kısa ama kapsayıcı özet,\n"
        " (2) 'Kararlar' listesi (maddeler),\n"
        " (3) 'Aksiyonlar' listesi (owner, task, due).\n"
        " Sadece geçerli JSON döndür:\n"
        " {\n"
        "   \"summary\": \"...\",\n"
        "   \"decisions\": [\"...\"],\n"
        "   \"actions\": [{\"owner\":\"...\", \"task\":\"...\", \"due\":\"YYYY-MM-DD veya boş\"}]\n"
        " }"
    )

    user_msg = f"Toplantı transkripti: {transcript_text}"

    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=OPENAI_MODEL_CHAT,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    content = completion.choices[0].message.content if completion.choices else ""
    return content or ""


def _safe_json_loads(data: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(data)
    except Exception:
        return None


def extract_structured(raw_or_transcript: str) -> Dict[str, Any]:
    """Extract JSON structure; fallback to safe defaults if parsing fails."""
    text = raw_or_transcript.strip()
    first = text.find("{")
    last = text.rfind("}")
    candidate = text[first:last + 1] if first != -1 and last != -1 and last > first else text

    parsed = _safe_json_loads(candidate)

    if not isinstance(parsed, dict):
        # Fallback: produce minimal structure from the original content
        excerpt = raw_or_transcript[:500]
        return {
            "summary": excerpt,
            "decisions": [],
            "actions": [],
        }

    summary = parsed.get("summary") if isinstance(parsed.get("summary"), str) else ""
    decisions = parsed.get("decisions") if isinstance(parsed.get("decisions"), list) else []

    actions_raw = parsed.get("actions")
    actions: List[ActionItem] = []
    if isinstance(actions_raw, list):
        for item in actions_raw:
            if isinstance(item, dict):
                owner = str(item.get("owner", ""))
                task = str(item.get("task", ""))
                due = str(item.get("due", ""))
                actions.append(ActionItem(owner=owner, task=task, due=due))

    # Convert back to dicts for rendering ease
    return {
        "summary": summary,
        "decisions": [str(x) for x in decisions],
        "actions": [
            {"owner": a.owner, "task": a.task, "due": a.due} for a in actions
        ],
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _split_text(text: str, chunk_size: int = 2000, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks by characters."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def retrieve_relevant_context(transcript_text: str, question: str, client: OpenAI, top_k: int = 3) -> Dict[str, Any]:
    """Return most relevant transcript chunks to the question using embeddings; robust fallback if unavailable."""
    transcript_text = transcript_text.strip()
    question = question.strip()
    if not transcript_text or not question:
        return {"context": "", "matches": []}

    # Prepare chunks
    chunks = _split_text(transcript_text, chunk_size=2000, overlap=150)
    if not chunks:
        return {"context": "", "matches": []}

    try:
        q_emb_resp = client.embeddings.create(model="text-embedding-3-small", input=question)
        q_vec = np.array(q_emb_resp.data[0].embedding, dtype=np.float32)

        c_emb_resp = client.embeddings.create(model="text-embedding-3-small", input=chunks)
        c_vecs = [np.array(item.embedding, dtype=np.float32) for item in c_emb_resp.data]

        scored = []
        for chunk, vec in zip(chunks, c_vecs):
            score = _cosine_similarity(q_vec, vec)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k] if scored else []
        # Lower threshold and always include at least top-1
        threshold = 0.08
        filtered = [c for s, c in top if s >= threshold]
        if not filtered and top:
            filtered = [top[0][1]]
        context = "\n\n".join(filtered)
        return {"context": context, "matches": top}
    except Exception:
        # Fallback: simple keyword filter with normalization
        q_norm = _normalize_text(question)
        q_terms = [t for t in q_norm.split(" ") if len(t) > 2]
        scored = []
        for chunk in chunks:
            c_norm = _normalize_text(chunk)
            score = sum(c_norm.count(t) for t in q_terms)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Always include at least one chunk
        top = [c for s, c in scored[:top_k] if s > 0]
        if not top and scored:
            top = [scored[0][1]]
        context = "\n\n".join(top)
        return {"context": context, "matches": scored[:top_k]}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Toplantı Asistanı", page_icon="📝", layout="wide")

st.title("📝 Toplantı Asistanı")
st.write(
    "Ses dosyanızı (mp3/wav/m4a/mov) yükleyin. Uygulama Whisper-1 ile transkribe eder,"
    " ardından GPT ile özet/kararlar/aksiyonlar üretir."
)

# Session state for transcript
if "transcript_text" not in st.session_state:
    st.session_state["transcript_text"] = ""
if "structured_data" not in st.session_state:
    st.session_state["structured_data"] = None

with st.sidebar:
    st.subheader("Sağlayıcı")
    st.text("OpenAI (sabit)")
    st.caption(f"Chat modeli: {OPENAI_MODEL_CHAT}")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY eksik. Lütfen .env dosyanızı ayarlayın ve sayfayı yenileyin.")

uploaded = st.file_uploader("Ses dosyası yükleyin", type=["mp3", "wav", "m4a", "mov"])
run = st.button("Çalıştır", type="primary")

if run:
    if not uploaded:
        st.warning("Lütfen bir dosya yükleyin.")
    elif not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY eksik. Lütfen .env dosyasını kontrol edin.")
    else:
        transcript_text = ""
        raw_model_output = ""
        saved_path: Optional[str] = None
        wav_path: Optional[str] = None
        try:
            with st.spinner("Dosya kaydediliyor..."):
                saved_path = save_uploaded_audio(uploaded)

            with st.spinner("16k mono WAV'a dönüştürülüyor..."):
                wav_path = ensure_wav_16k(saved_path)

            with st.spinner("Transkripsiyon yapılıyor (Whisper-1)..."):
                transcript_text = transcribe_openai_whisper(wav_path)
                st.session_state["transcript_text"] = transcript_text

            with st.spinner("Özet/karar/aksiyon üretiliyor..."):
                raw_model_output = run_openai_chat(transcript_text)

            data = extract_structured(raw_model_output)
            st.session_state["structured_data"] = data

        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
        finally:
            # Cleanup temp files
            for p in [wav_path, saved_path]:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

# Always render current transcript & structured sections if available
current_transcript = st.session_state.get("transcript_text", "")
if current_transcript:
    st.subheader("Transkript")
    st.text_area("Transkript", value=current_transcript, height=200)

current_data = st.session_state.get("structured_data")
if isinstance(current_data, dict):
    st.subheader("Özet")
    st.write(current_data.get("summary", ""))

    st.subheader("Kararlar")
    decisions: List[str] = current_data.get("decisions", [])
    if decisions:
        for d in decisions:
            st.markdown(f"- {d}")
    else:
        st.caption("Karar bulunamadı.")

    st.subheader("Aksiyonlar")
    actions: List[Dict[str, str]] = current_data.get("actions", [])
    if actions:
        for a in actions:
            owner = a.get("owner", "")
            task = a.get("task", "")
            due = a.get("due", "")
            mid = " • "
            st.markdown(f"- **{owner}** {mid} {task} {mid} {due}")
    else:
        st.caption("Aksiyon bulunamadı.")


# Q&A Section based on transcript
st.markdown("---")
st.subheader("Soru-Cevap")
qa_col1, qa_col2 = st.columns([4, 1])
with qa_col1:
    user_question = st.text_input("Sorunuzu yazın", value="", placeholder="Transkripte göre bir soru sorun…")
with qa_col2:
    ask = st.button("Sor", type="secondary")

if ask:
    transcript_ctx = st.session_state.get("transcript_text", "").strip()
    if not transcript_ctx:
        st.warning("Önce bir ses dosyası işleyip transkript oluşturun.")
    elif not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY eksik. Lütfen .env dosyasını kontrol edin.")
    elif not user_question.strip():
        st.warning("Lütfen bir soru yazın.")
    else:
        try:
            with st.spinner("Yanıt üretiliyor..."):
                client = OpenAI(api_key=OPENAI_API_KEY)
                retrieval = retrieve_relevant_context(transcript_ctx, user_question, client)
                context = retrieval.get("context", "").strip()

                system_qna = (
                    "Yalnızca verilen BAĞLAM'a dayanarak kısa ve doğrudan yanıt ver. "
                    "Varsayım yapma. Eğer yanıt BAĞLAM'da yoksa sadece şunu yaz: 'Transkriptte bilgi bulunamadı.'"
                )
                messages = [
                    {"role": "system", "content": system_qna},
                    {"role": "user", "content": f"BAĞLAM:\n{context if context else '(boş)'}"},
                    {"role": "user", "content": f"Soru: {user_question}"},
                ]
                client = OpenAI(api_key=OPENAI_API_KEY)
                completion = client.chat.completions.create(
                    model=OPENAI_MODEL_CHAT,
                    temperature=0.0,
                    messages=messages,
                )
                answer = completion.choices[0].message.content if completion.choices else ""
            st.success("Yanıt")
            st.write(answer)
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
