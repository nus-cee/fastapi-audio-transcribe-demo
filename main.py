import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

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
    ext = _guess_extension(file.content_type or "", file.filename or "")
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        model = _model
        if model_size:
            model = _init_model(model_size)

        segments_data, full_text = transcribe_single(
            tmp.name, model, language=language
        )

        return JSONResponse(
            content={
                "segments": segments_data,
                "full_text": full_text,
            }
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


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
