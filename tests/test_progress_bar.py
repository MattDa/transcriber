import io
import importlib
import sys
from pathlib import Path


class DummyUploadedFile(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def test_progress_bar_invoked(monkeypatch):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    app = importlib.import_module("app")

    class DummyModel:
        def transcribe(self, path, verbose=None):
            from tqdm import tqdm
            for _ in tqdm(range(2)):
                pass
            return {"text": "ok"}

    called = {"progress": 0}

    def fake_progress(val):
        called["progress"] += 1

        class PB:
            def progress(self, v):
                called["progress"] += 1

            def empty(self):
                pass

        return PB()

    monkeypatch.setattr(app, "model", DummyModel())
    monkeypatch.setattr(app.st, "progress", fake_progress)
    monkeypatch.setattr(app.st, "error", lambda msg: None)
    monkeypatch.setattr(app, "validate_media_file", lambda p: True)

    uploaded = DummyUploadedFile(b"data", "good.mp3")
    text = app.process_uploaded_file(uploaded)

    assert text == "ok"
    assert called["progress"] > 1

