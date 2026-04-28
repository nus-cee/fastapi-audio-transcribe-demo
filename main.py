import os
import tempfile
import uuid
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aws_utils import upload_text_to_s3
from transcribe import _init_model, transcribe_single

app = FastAPI(title="Audio-to-Text API", version="1.0.0")

_model = None
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")


@app.on_event("startup")
def startup_event():
    global _model
    _model = _init_model(MODEL_SIZE)


class TranscribeRequest(BaseModel):
    presignedUrl: str = Field(..., description="Presigned URL of the audio file")
    language: str = Field(default="en", description="Language code (en, zh, ja, etc.)")
    model_size: str = Field(
        default="", description="Override model size (tiny/base/small/medium/large-v3)"
    )


@app.post("/api/transcribe")
def api_transcribe(req: TranscribeRequest):
    try:
        resp = requests.get(req.presignedUrl, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {exc}")

    content_type = resp.headers.get("Content-Type", "")
    ext = _guess_extension(content_type, req.presignedUrl)

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.flush()
        tmp.close()

        model = _model
        if req.model_size:
            model = _init_model(req.model_size)

        segments_data, full_text = transcribe_single(
            tmp.name, model, language=req.language
        )

        if not full_text:
            return JSONResponse(
                content={"segments": [], "full_text": "", "download_url": ""}
            )

        lines = []
        for seg in segments_data:
            start_mm, start_ss = divmod(seg["start"], 60)
            end_mm, end_ss = divmod(seg["end"], 60)
            lines.append(f"[{start_mm:06.3f} -> {end_mm:06.3f}] {seg['text']}")
        timestamped_text = "\n".join(lines)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        object_key = f"transcripts/{timestamp}_{short_id}.txt"
        download_url = upload_text_to_s3(timestamped_text, object_key)

        return JSONResponse(
            content={
                "segments": segments_data,
                "full_text": full_text,
                "download_url": download_url,
            }
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _guess_extension(content_type: str, url: str) -> str:
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/flac": ".flac",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "audio/aac": ".aac",
        "video/mp4": ".mp4",
    }
    if content_type:
        for key, ext in mapping.items():
            if key in content_type:
                return ext
    from urllib.parse import urlparse

    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    if ext and len(ext) <= 5:
        return ext
    return ".mp3"
