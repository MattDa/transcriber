import os
import tempfile
from pathlib import Path
import logging

import streamlit as st
import openai
import whisper
import torch

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
        model = whisper.load_model("large-v3")
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        logger.exception("Failed to load Whisper model")
        raise

    return whisper.load_model("large-v3")


model = load_model()

st.title("Transcriber")

if "transcripts" not in st.session_state:
    st.session_state["transcripts"] = {}

uploaded = st.file_uploader("Upload audio/video/text", type=["mp4","mp3","wav","m4a","ogg","flac","txt"], accept_multiple_files=False)

if uploaded is not None:
    try:
        with st.spinner("Processing file..."):
            if uploaded.name.lower().endswith(".txt"):
                text = uploaded.read().decode("utf-8")
            else:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                result = model.transcribe(tmp_path)
                os.unlink(tmp_path)
                text = result["text"]
            st.session_state.transcripts[uploaded.name] = text
        st.success("File processed")
    except Exception:
        logger.exception("Failed to process uploaded file")
        st.error("Error processing file")

    with st.spinner("Processing file..."):
        if uploaded.name.lower().endswith(".txt"):
            text = uploaded.read().decode("utf-8")
        else:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            result = model.transcribe(tmp_path)
            os.unlink(tmp_path)
            text = result["text"]
        st.session_state.transcripts[uploaded.name] = text
    st.success("File processed")


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
        openai.api_base = centgpt_url
        response = openai.ChatCompletion.create(
            model="llama3-70b-instruct",
            messages=messages,
            stream=True,
        )
        placeholder = st.empty()
        collected = ""
        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                collected += delta["content"]
                placeholder.markdown(collected)
    except Exception:
        logger.exception("LLM request failed")
        st.error("Error contacting language model")

    openai.api_base = centgpt_url
    response = openai.ChatCompletion.create(
        model="llama3-70b-instruct",
        messages=messages,
        stream=True,
    )
    placeholder = st.empty()
    collected = ""
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {})
        if delta.get("content"):
            collected += delta["content"]
            placeholder.markdown(collected)


