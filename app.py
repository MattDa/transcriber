import os
import tempfile
from pathlib import Path
import logging
import subprocess

import streamlit as st
from openai import OpenAI
import whisper
import torch
import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


centgpt_url = os.environ.get("CENTGPT_URL", "http://localhost:8000/v1")

SUMMARY_PROMPT = "<SUMMARY_PROMPT_PLACEHOLDER>"
GEM_PROMPT = "<GEM_PROMPT_PLACEHOLDER>"

@st.cache_resource
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Whisper model on %s", device)
        try:
            model = whisper.load_model("large-v3", device=device)
        except TypeError:
            model = whisper.load_model("large-v3")
        if device == "cuda":
            model = torch.compile(model, mode="reduce-overhead")
        return model
    except Exception:
        logger.exception("Failed to load Whisper model")
        raise


model = load_model()

st.title("Transcriber")

if "transcripts" not in st.session_state:
    st.session_state["transcripts"] = {}

uploaded = st.file_uploader(
    "Upload audio/video/text",
    type=["mp4", "mp3", "wav", "m4a", "ogg", "flac", "txt"],
    accept_multiple_files=False,
    key="uploader",
)



def validate_media_file(path: str) -> bool:
    """Run ffprobe to verify the file is a readable media container."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=format_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("ffprobe failed for %s: %s", path, result.stderr.strip())
            return False
    except Exception:
        logger.exception("ffprobe invocation failed for %s", path)
        return False
    return True


def process_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if not validate_media_file(tmp_path):
            st.error("Invalid or corrupted media file")
            return None
        progress_bar = st.progress(0)
        original_tqdm = tqdm.tqdm

        class StreamlitTqdm(original_tqdm):
            def update(self, n=1):
                super().update(n)
                if self.total:
                    progress = int(self.n / self.total * 100)
                    progress_bar.progress(min(progress, 100))

            def close(self):
                progress_bar.progress(100)
                super().close()

        tqdm.tqdm = StreamlitTqdm
        try:
            try:
                result = model.transcribe(tmp_path, verbose=None)
            except TypeError:
                result = model.transcribe(tmp_path)
            return result["text"]
        finally:
            tqdm.tqdm = original_tqdm
            progress_bar.empty()
    finally:
        os.unlink(tmp_path)


def handle_new_upload(uploaded_file):
    if uploaded_file is None or uploaded_file.name in st.session_state.transcripts:
        return
    try:
        with st.spinner("Processing file..."):
            text = process_uploaded_file(uploaded_file)
            if text is not None:
                st.session_state.transcripts[uploaded_file.name] = text

def validate_media_file(path: str) -> bool:
    """Run ffprobe to verify the file is a readable media container."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=format_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("ffprobe failed for %s: %s", path, result.stderr.strip())
            return False
    except Exception:
        logger.exception("ffprobe invocation failed for %s", path)
        return False
    return True


def process_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if not validate_media_file(tmp_path):
            st.error("Invalid or corrupted media file")
            return None
        result = model.transcribe(tmp_path)
        return result["text"]
    finally:
        os.unlink(tmp_path)


if uploaded is not None:
    try:
        with st.spinner("Processing file..."):
            text = process_uploaded_file(uploaded)
            if text is not None:
                st.session_state.transcripts[uploaded.name] = text

                st.success("File processed")
    except Exception:
        logger.exception("Failed to process uploaded file")
        st.error("Error processing file")
    finally:
        st.session_state.uploader = None

handle_new_upload(uploaded)



st.sidebar.header("Uploaded files")
all_files = list(st.session_state.transcripts.keys())
selected = st.sidebar.multiselect(
    "Select transcripts",
    options=all_files,
    format_func=lambda x: x if len(x) <= 20 else x[:17] + "...",
)

prompt_choice = st.sidebar.radio("Prompt", ["Summarize", "GEM"])

if st.sidebar.button("Execute") and selected:
    full_text = "\n\n".join(st.session_state.transcripts[name] for name in selected)
    prompt = SUMMARY_PROMPT if prompt_choice == "Summarize" else GEM_PROMPT
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": full_text},
    ]

    try:
        client = OpenAI(base_url=centgpt_url)
        response = client.chat.completions.create(
            model="llama3-70b-instruct",
            messages=messages,
            stream=True,
        )
        placeholder = st.empty()
        collected = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                collected += delta
                placeholder.markdown(collected)
    except Exception:
        logger.exception("LLM request failed")
        st.error("Error contacting language model")


