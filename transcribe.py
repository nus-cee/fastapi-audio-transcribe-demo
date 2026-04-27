"""
Local CPU speech-to-text script based on faster-whisper.

Usage:
    python transcribe.py <audio_file_path> [model_size] [language]
    python transcribe.py <audio_directory_path> [model_size] [language]

Examples:
    python transcribe.py recording.mp3
    python transcribe.py recording.mp3 small
    python transcribe.py recording.mp3 base en
    python transcribe.py ./audio_folder
"""

import os
import sys
from tqdm import tqdm
from faster_whisper import WhisperModel

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".ogg",
    ".wma",
    ".aac",
    ".opus",
    ".webm",
    ".mp4",
}


def _init_model(model_size: str = "base") -> WhisperModel:
    """
    Initialize the faster-whisper model.

    Args:
        model_size: Model size, one of tiny / base / small / medium / large-v2 etc.
                    Larger models are more accurate but slower and use more memory.

    Returns:
        WhisperModel instance

    Notes:
        device="cpu"
            Force CPU inference, no CUDA / GPU required.
            Suitable for machines without a dedicated GPU.

        compute_type="int8"
            Use INT8 quantization. Compared to default float32:
            - ~75% reduction in memory usage
            - ~2-4x faster CPU inference
            - Negligible accuracy loss
            This is the recommended quantization for CPU deployment.

        The model is automatically downloaded from Hugging Face Hub on first run
        and cached at ~/.cache/huggingface/hub/. Subsequent runs use the local cache.
    """
    print(f"[Init] Loading model: {model_size} (CPU / INT8) ...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("[Init] Model loaded.")
    return model


def transcribe_single(file_path: str, model: WhisperModel, language: str = "en") -> str:
    """
    Transcribe a single audio file using the given model.

    Args:
        file_path: Absolute or relative path to the audio file.
        model:     An initialized WhisperModel instance.
        language:  Language code to force, e.g. "en", "zh", "ja".
                   Set to None for auto-detection. Defaults to "en".

    Returns:
        Full transcript text with timestamps.

    Notes:
        beam_size=5 (default)
            Beam search width. Higher values improve accuracy slightly
            but slow down inference. 5 is a good balance.

        When language is explicitly set, the model skips language detection
        and uses the specified language directly, slightly improving speed
        and accuracy.

        segments is a lazy generator that yields one segment at a time,
        keeping memory usage low even for hours-long audio files.
    """
    segments_iter, info = model.transcribe(file_path, beam_size=5, language=language)

    print(f"\n{'=' * 60}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Language: {info.language} (probability: {info.language_probability:.2%})")
    print(f"Duration: {info.duration:.2f}s")
    print(f"{'=' * 60}")

    full_text = ""
    pbar = tqdm(total=info.duration, unit="s", desc="Transcribing", ncols=80)

    for segment in segments_iter:
        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
        print(line)
        if full_text:
            full_text += "\n"
        full_text += line

        pbar.update(segment.end - segment.start)

    pbar.close()

    base_name = os.path.splitext(file_path)[0]
    output_path = base_name + ".txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"[Done] Saved to: {output_path}\n")
    return full_text


def transcribe_audio(
    file_path: str, model_size: str = "base", language: str = "en"
) -> str:
    """
    Public API entry point: transcribe a single audio file.
    Handles model initialization, file validation, transcription and saving.

    Args:
        file_path:   Path to the audio file.
        model_size:  Model size, defaults to "base".
        language:    Language code to force, defaults to "en". Set to None for auto-detection.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model = _init_model(model_size)
    return transcribe_single(file_path, model, language=language)


def transcribe_batch(
    directory: str, model_size: str = "base", language: str = "en"
) -> list:
    """
    Batch transcribe: scan a directory for supported audio files and transcribe them.
    All files share a single model instance to avoid reloading.

    Args:
        directory:   Path to the directory containing audio files.
        model_size:  Model size, defaults to "base".
        language:    Language code to force, defaults to "en". Set to None for auto-detection.

    Returns:
        List of transcript texts, one per file.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Directory not found: {directory}")

    audio_files = []
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in AUDIO_EXTENSIONS:
            audio_files.append(os.path.join(directory, f))

    if not audio_files:
        print(f"[Warning] No supported audio files found in {directory}.")
        print(f"          Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        return []

    print(f"[Batch] Found {len(audio_files)} audio file(s).")

    model = _init_model(model_size)
    results = []
    for i, fp in enumerate(audio_files, 1):
        print(f"\n>>> [{i}/{len(audio_files)}]")
        try:
            text = transcribe_single(fp, model, language=language)
            results.append(text)
        except Exception as e:
            print(f"[Error] Failed to transcribe {fp}: {e}")
            results.append("")

    print(f"\n{'=' * 60}")
    print(f"Batch transcription complete! Processed {len(audio_files)} file(s).")
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python transcribe.py <audio_file_path> [model_size] [language]")
        print("  python transcribe.py <audio_dir_path> [model_size] [language]")
        print()
        print("Model sizes: tiny, base, small, medium, large-v2, large-v3")
        print("Language codes: en (default), zh, ja, ko, etc.")
        print("                Pass 'auto' for auto-detection")
        print("Example: python transcribe.py recording.mp3 small en")
        sys.exit(1)

    target = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    language = sys.argv[3] if len(sys.argv) > 3 else "en"
    if language == "auto":
        language = None

    try:
        if os.path.isdir(target):
            transcribe_batch(target, model_size, language=language)
        elif os.path.isfile(target):
            transcribe_audio(target, model_size, language=language)
        else:
            print(f"[Error] Path not found: {target}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted] Transcription cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)
