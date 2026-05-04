import os
from io import BytesIO

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from transcribe import _init_model, transcribe_single

app = FastAPI(title="Audio-to-Text API", version="1.0.0")

_model = None
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.on_event("startup")
def startup_event():
    global _model
    _model = _init_model(MODEL_SIZE)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(..., description="Audio file"),
    language: str = Form(default="en", description="Language code (en, zh, ja, etc.)"),
    model_size: str = Form(
        default="", description="Override model size (tiny/base/small/medium/large-v3)"
    ),
):
    content = await file.read()
    buf = BytesIO(content)
    buf.name = file.filename or "audio"

    model = _model
    if model_size:
        model = _init_model(model_size)

    segments_data, full_text = transcribe_single(buf, model, language=language)

    return JSONResponse(
        content={
            "segments": segments_data,
            "full_text": full_text,
        }
    )


class TranscribeUrlRequest(BaseModel):
    url: str = Field(..., description="URL of the audio file to download")
    language: str = Field(default="en", description="Language code (en, zh, ja, etc.)")
    model_size: str = Field(
        default="", description="Override model size (tiny/base/small/medium/large-v3)"
    )


@app.post("/api/transcribe-url")
async def api_transcribe_url(req: TranscribeUrlRequest):
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.get(req.url)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {exc}")

    buf = BytesIO(resp.content)
    buf.name = req.url

    model = _model
    if req.model_size:
        model = _init_model(req.model_size)

    segments_data, full_text = transcribe_single(
        buf, model, language=req.language
    )

    return JSONResponse(
        content={
            "segments": segments_data,
            "full_text": full_text,
        }
    )


def _guess_extension(content_type: str, filename: str) -> str:
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
    _, ext = os.path.splitext(filename)
    if ext and len(ext) <= 5:
        return ext
    return ".mp3"
