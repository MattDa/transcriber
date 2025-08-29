import io
import importlib
import logging
import sys
from pathlib import Path

import whisper
import torch


class DummyUploadedFile(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def test_corrupted_upload_logs_error(monkeypatch, caplog):
    def fake_load_model(name):
        class DummyModel:
            def transcribe(self, path):
                raise AssertionError("transcribe should not be called")
        return DummyModel()

    monkeypatch.setattr(whisper, "load_model", fake_load_model)
    monkeypatch.setattr(torch, "compile", lambda model, mode=None: model)

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    app = importlib.import_module("app")

    error_called = {}
    monkeypatch.setattr(app.st, "error", lambda msg: error_called.setdefault("msg", msg))

    caplog.set_level(logging.ERROR)

    uploaded = DummyUploadedFile(b"not a real media file", "bad.mp3")
    text = app.process_uploaded_file(uploaded)

    assert text is None
    assert error_called.get("msg") == "Invalid or corrupted media file"
    assert any("ffprobe" in record.message for record in caplog.records)
