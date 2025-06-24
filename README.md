# Transcriber

This project provides a Streamlit UI for uploading audio, video, or text files.

Files are transcribed using Whisper-large-v3 compiled with `torch.compile` and
sent to a Llama 3 endpoint for processing with either a Summarize or GEM
prompt. The container also includes the `faster-whisper` package for optional
high-performance inference.


## Building the container

```bash
docker build -t transcriber .
```

## Running

```bash
docker run -p 8501:8501 transcriber
```
