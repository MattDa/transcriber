import io
import importlib
import sys
from pathlib import Path
import contextlib

import whisper
import torch


class DummyUploadedFile(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def test_no_reprocess_on_rerun(monkeypatch):
    class DummyModel:
        def transcribe(self, path):
            return {"text": "dummy"}

    monkeypatch.setattr(whisper, "load_model", lambda name: DummyModel())
    monkeypatch.setattr(torch, "compile", lambda model, mode=None: model)

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    app = importlib.import_module("app")

    app.st.session_state.clear()
    app.st.session_state["transcripts"] = {}

    calls = {"n": 0}

    def fake_process(file):
        calls["n"] += 1
        return "ok"

    monkeypatch.setattr(app, "process_uploaded_file", fake_process)
    monkeypatch.setattr(app.st, "spinner", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(app.st, "success", lambda msg: None)
    monkeypatch.setattr(app.st, "error", lambda msg: None)

    uploaded = DummyUploadedFile(b"data", "good.mp3")

    app.handle_new_upload(uploaded)
    app.handle_new_upload(uploaded)

    assert calls["n"] == 1
