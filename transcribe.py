"""
基于 faster-whisper 的本地 CPU 语音转文字脚本。

用法:
    python transcribe.py <音频文件路径> [模型大小]
    python transcribe.py <音频目录路径> [模型大小]

示例:
    python transcribe.py recording.mp3
    python transcribe.py recording.mp3 small
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
    初始化 faster-whisper 模型。

    参数:
        model_size: 模型规格，可选 tiny / base / small / medium / large-v2 等。
                    模型越大准确率越高，但推理速度越慢、内存占用越大。

    返回:
        WhisperModel 实例

    说明:
        device="cpu"
            强制使用 CPU 推理，不依赖 CUDA / GPU。
            适用于没有独立显卡或仅需轻量推理的场景。

        compute_type="int8"
            使用 INT8 量化。相比默认的 float32：
            - 内存占用降低约 75%
            - CPU 推理速度提升约 2-4 倍
            - 转录精度损失极小，可忽略不计
            这是 CPU 部署场景下最推荐的量化方案。

        model_size 首次运行时会自动从 Hugging Face Hub 下载并缓存到:
            ~/.cache/huggingface/hub/
        后续运行直接读取本地缓存，无需重复下载。
    """
    print(f"[初始化] 正在加载模型: {model_size} (CPU / INT8) ...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("[初始化] 模型加载完成。")
    return model


def transcribe_single(file_path: str, model: WhisperModel, language: str = "en") -> str:
    """
    使用已初始化的模型转录单个音频文件。

    参数:
        file_path: 音频文件的绝对或相对路径
        model:     已初始化的 WhisperModel 实例
        language:  强制指定的语言代码，如 "en"(英语)、"zh"(中文)、"ja"(日语) 等。
                   设为 None 则自动检测语言。默认 "en"。

    返回:
        完整转录文本（带时间戳）

    说明:
        beam_size=5 (默认)
            Beam Search 的宽度。值越大搜索空间越广，准确率略高但速度变慢。
            5 是在速度和准确率之间的良好平衡点。

        language 强制指定后，模型跳过语言检测步骤，
            直接以指定语言的语音识别器进行转录，可略微提升速度和准确率。

        segments 是一个惰性生成器（generator），
            不会一次性加载全部结果到内存，而是逐段 yield，
            因此即使转录数小时的音频，内存占用也保持在较低水平。
    """
    segments_iter, info = model.transcribe(file_path, beam_size=5, language=language)

    print(f"\n{'=' * 60}")
    print(f"文件: {os.path.basename(file_path)}")
    print(f"检测到语言: {info.language} (概率: {info.language_probability:.2%})")
    print(f"音频时长: {info.duration:.2f}s")
    print(f"{'=' * 60}")

    full_text = ""
    pbar = tqdm(total=info.duration, unit="s", desc="转录进度", ncols=80)

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

    print(f"[完成] 已保存至: {output_path}\n")
    return full_text


def transcribe_audio(
    file_path: str, model_size: str = "base", language: str = "en"
) -> str:
    """
    公开 API 入口：转录单个音频文件。
    自动完成模型初始化、文件校验、转录和保存。

    参数:
        file_path:   音频文件路径
        model_size:  模型大小，默认 "base"
        language:    强制指定的语言代码，默认 "en"（英语）。设为 None 自动检测。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    model = _init_model(model_size)
    return transcribe_single(file_path, model, language=language)


def transcribe_batch(
    directory: str, model_size: str = "base", language: str = "en"
) -> list:
    """
    批量转录：扫描目录下所有支持的音频文件，依次转录。
    所有文件共享同一个模型实例，避免重复加载。

    参数:
        directory:   音频文件所在目录
        model_size:  模型大小，默认 "base"
        language:    强制指定的语言代码，默认 "en"（英语）。设为 None 自动检测。

    返回:
        各文件转录文本的列表
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"目录不存在: {directory}")

    audio_files = []
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in AUDIO_EXTENSIONS:
            audio_files.append(os.path.join(directory, f))

    if not audio_files:
        print(f"[警告] 目录 {directory} 中未找到支持的音频文件。")
        print(f"       支持的格式: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        return []

    print(f"[批量模式] 找到 {len(audio_files)} 个音频文件。")

    model = _init_model(model_size)
    results = []
    for i, fp in enumerate(audio_files, 1):
        print(f"\n>>> [{i}/{len(audio_files)}]")
        try:
            text = transcribe_single(fp, model, language=language)
            results.append(text)
        except Exception as e:
            print(f"[错误] 转录失败 {fp}: {e}")
            results.append("")

    print(f"\n{'=' * 60}")
    print(f"批量转录完成！共处理 {len(audio_files)} 个文件。")
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python transcribe.py <音频文件路径> [模型大小] [语言]")
        print("  python transcribe.py <音频目录路径> [模型大小] [语言]")
        print()
        print("可选模型大小: tiny, base, small, medium, large-v2, large-v3")
        print("可选语言代码: en(英语, 默认), zh(中文), ja(日语), ko(韩语) 等")
        print("              传 auto 表示自动检测语言")
        print("示例: python transcribe.py recording.mp3 small en")
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
            print(f"[错误] 路径不存在: {target}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消转录。")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] {e}")
        sys.exit(1)
