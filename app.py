import os
import sys
import uuid
import glob
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import List, Optional

import gradio as gr
import imageio_ffmpeg
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"
MODELS_DIR = APP_DIR / "models"
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MODEL_CACHE = {}

MODEL_CHOICES = {
    "Fast - small": "small",
    "Balanced - medium (recommended default)": "medium",
    "High Quality - large-v3 (slow / needs more VRAM)": "large-v3",
}

MODEL_REPOS = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

LANGUAGE_CHOICES = {
    "Arabic": "ar",
    "English": "en",
    "Auto detect": None,
}

DEVICE_CHOICES = [
    "Auto: try GPU then CPU",
    "GPU CUDA only",
    "CPU only",
]

COMPUTE_CHOICES = [
    "Auto",
    "float16 - GPU quality/speed",
    "int8_float16 - GPU lower memory",
    "int8 - safest / CPU friendly",
]


def sanitize_name(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").replace(":", "_")


def local_model_path(model_name: str) -> Path:
    return MODELS_DIR / sanitize_name(model_name)


def model_marker_path(model_name: str) -> Path:
    return MODELS_DIR / f".{sanitize_name(model_name)}.ready"


def is_model_downloaded(model_name: str) -> bool:
    path = local_model_path(model_name)
    if model_marker_path(model_name).exists() and path.exists():
        return True
    # Extra tolerant check in case the marker was deleted.
    if path.exists() and (path / "model.bin").exists() and (path / "config.json").exists():
        return True
    return False


def seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def seconds_to_readable(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def build_readable_transcript(text_segments: List[str]) -> str:
    paragraphs = []
    current_parts = []
    current_length = 0
    punctuation = tuple(".!?;:\u060c\u061b\u061f")
    min_paragraph_chars = 1500
    max_paragraph_chars = 2500

    for raw_text in text_segments:
        text = " ".join(str(raw_text).strip().split())
        if not text:
            continue

        current_parts.append(text)
        current_length += len(text) + (1 if current_length else 0)

        can_break = current_length >= min_paragraph_chars and text.endswith(punctuation)
        must_break = current_length >= max_paragraph_chars
        if can_break or must_break:
            paragraphs.append(" ".join(current_parts))
            current_parts = []
            current_length = 0

    if current_parts:
        paragraphs.append(" ".join(current_parts))

    return "\n\n".join(paragraphs)


def run_command(cmd: List[str], log_lines: List[str]) -> None:
    log_lines.append("Running: " + " ".join([str(x) for x in cmd]))
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout:
        log_lines.append(process.stdout[-6000:])
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}.")


def get_compute_value(compute_mode: str, device: str) -> str:
    if compute_mode.startswith("float16"):
        return "float16"
    if compute_mode.startswith("int8_float16"):
        return "int8_float16"
    if compute_mode.startswith("int8"):
        return "int8"

    # Auto: safer for limited-VRAM NVIDIA GPUs, and stable for CPU.
    if device == "cuda":
        return "int8_float16"
    return "int8"


def is_cuda_dependency_error(error: Exception) -> bool:
    msg = (str(error) + " " + repr(error)).lower()
    keywords = [
        "cublas", "cudnn", "cudart", "cuda", "dll", "out of memory",
        "cannot be loaded", "not found", "failed to load", "not enough memory"
    ]
    return any(k in msg for k in keywords)


def collect_segments(model, chunk_path: Path, language, vad_filter: bool):
    segments, info = model.transcribe(
        str(chunk_path),
        language=language,
        beam_size=5,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=700),
    )
    return list(segments), info


def load_model(model_size: str, device_mode: str, compute_mode: str, log_lines: List[str]):
    model_name = MODEL_CHOICES[model_size]

    if not is_model_downloaded(model_name):
        raise RuntimeError(
            f"Model '{model_name}' is not downloaded yet. Click 'Download selected model' first."
        )

    model_path = str(local_model_path(model_name))

    if device_mode == "CPU only":
        device_attempts = ["cpu"]
    elif device_mode == "GPU CUDA only":
        device_attempts = ["cuda"]
    else:
        device_attempts = ["cuda", "cpu"]

    last_error = None

    for device in device_attempts:
        compute = get_compute_value(compute_mode, device)
        cache_key = (model_name, device, compute)

        try:
            if cache_key not in MODEL_CACHE:
                log_lines.append(f"Loading model: {model_name} | device={device} | compute_type={compute}")
                MODEL_CACHE[cache_key] = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute,
                    local_files_only=True,
                )
            else:
                log_lines.append(f"Using cached model: {model_name} | device={device} | compute_type={compute}")

            return MODEL_CACHE[cache_key], model_name, device, compute

        except Exception as e:
            last_error = e
            log_lines.append(f"Failed to load on {device} with {compute}: {repr(e)}")
            if device_mode == "GPU CUDA only":
                break

    raise RuntimeError(
        "Could not load the model. Last error:\n"
        + repr(last_error)
        + "\n\nTry: CPU only + int8, or choose Balanced - medium."
    )


def make_chunks(audio_path: Path, chunks_dir: Path, chunk_minutes: int, log_lines: List[str]) -> List[Path]:
    chunks_dir.mkdir(exist_ok=True)
    for old in chunks_dir.glob("chunk_*.wav"):
        try:
            old.unlink()
        except Exception:
            pass

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    chunk_seconds = max(5, int(chunk_minutes) * 60)

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        str(chunks_dir / "chunk_%03d.wav")
    ]
    run_command(cmd, log_lines)

    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("No chunks were created. Please try a different video/audio file.")
    return chunks


def extract_audio(input_file: str, work_dir: Path, denoise: bool, log_lines: List[str]) -> Path:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    audio_path = work_dir / "audio_clean.wav"

    audio_filter = "loudnorm,afftdn=nf=-25" if denoise else "loudnorm"

    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_file,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", audio_filter,
        str(audio_path)
    ]
    run_command(cmd, log_lines)

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("Audio extraction failed. No audio file was created.")

    return audio_path


def check_model_status(model_size: str) -> str:
    model_name = MODEL_CHOICES[model_size]
    path = local_model_path(model_name)
    if is_model_downloaded(model_name):
        size_gb = folder_size_gb(path)
        return f"OK - Model is ready: {model_name}\nLocation: {path}\nApprox folder size: {size_gb:.2f} GB"
    return f"Not ready - Model is not downloaded yet: {model_name}\nClick 'Download selected model' first."


def folder_size_gb(path: Path) -> float:
    total = 0
    if path.exists():
        for file in path.rglob("*"):
            if file.is_file():
                try:
                    total += file.stat().st_size
                except Exception:
                    pass
    return total / (1024 ** 3)


def download_selected_model(model_size: str, progress=gr.Progress(track_tqdm=True)):
    model_name = MODEL_CHOICES[model_size]
    repo_id = MODEL_REPOS[model_name]
    target_dir = local_model_path(model_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    progress(0.05, desc=f"Preparing download: {model_name}")
    try:
        # Download only; do not start transcription here.
        # local_dir keeps the model in a simple readable folder instead of a hidden HF cache layout.
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if not (target_dir / "model.bin").exists():
            raise RuntimeError("Download finished but model.bin was not found.")
        marker = model_marker_path(model_name)
        marker.write_text(f"ready={model_name}\nrepo={repo_id}\npath={target_dir}\n", encoding="utf-8")
        progress(1.0, desc="Model downloaded")
        size_gb = folder_size_gb(target_dir)
        return (
            f"OK - Model downloaded and ready: {model_name}\n"
            f"Approx size: {size_gb:.2f} GB\n"
            f"You can now click Start Transcription."
        )
    except Exception as e:
        raise gr.Error(
            "Model download failed. Check internet connection and free disk space. "
            f"Error: {e}"
        )


def check_gpu_status() -> str:
    lines = []
    lines.append("GPU / CUDA check")
    lines.append("=" * 40)

    def run_quiet(cmd):
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", shell=True)
            return p.returncode, p.stdout.strip()
        except Exception as exc:
            return 1, repr(exc)

    code, out = run_quiet("nvidia-smi")
    if code == 0:
        lines.append("OK - NVIDIA driver is visible through nvidia-smi")
        lines.append(out[:1200])
    else:
        lines.append("Not found - nvidia-smi failed. Update/install NVIDIA driver first.")
        lines.append(out[:1000])

    checks = [
        "cublas64_12.dll",
        "cudnn_ops64_9.dll",
        "cudart64_12.dll",
    ]
    for dll in checks:
        code, out = run_quiet(f"where {dll}")
        if code == 0:
            lines.append(f"OK - {dll} found:")
            lines.append(out)
        else:
            lines.append(f"Warning - {dll} not found in system PATH. This may still be okay if the app added pip-installed NVIDIA DLLs at startup.")

    venv_paths = [
        APP_DIR / ".venv" / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        APP_DIR / ".venv" / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        APP_DIR / ".venv" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
    ]
    for p in venv_paths:
        lines.append(("OK -" if p.exists() else "Missing -") + f" venv DLL path: {p}")

    return "\n".join(lines)


def transcribe_file(
    video_file,
    language_choice: str,
    model_size: str,
    device_mode: str,
    compute_mode: str,
    chunk_minutes: int,
    denoise: bool,
    vad_filter: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if video_file is None:
        raise gr.Error("Please upload a meeting video or audio file first.")

    selected_model = MODEL_CHOICES[model_size]
    if not is_model_downloaded(selected_model):
        raise gr.Error(
            f"Model '{selected_model}' is not downloaded yet. Click 'Download selected model' first."
        )

    job_id = uuid.uuid4().hex[:8]
    work_dir = OUTPUTS_DIR / f"job_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = work_dir / "chunks"
    partials_dir = work_dir / "partials"
    partials_dir.mkdir(exist_ok=True)

    txt_path = work_dir / "transcript.txt"
    clean_txt_path = work_dir / "clean_transcript.txt"
    readable_txt_path = work_dir / "readable_transcript.txt"
    srt_path = work_dir / "subtitles.srt"
    log_path = work_dir / "process_log.txt"

    log_lines = []
    try:
        input_path = video_file if isinstance(video_file, str) else video_file.name
        log_lines.append(f"Input file: {input_path}")
        log_lines.append(f"Output folder: {work_dir}")

        progress(0.03, desc="Extracting and cleaning audio...")
        audio_path = extract_audio(input_path, work_dir, denoise, log_lines)

        progress(0.10, desc="Splitting audio into safe chunks...")
        chunks = make_chunks(audio_path, chunks_dir, chunk_minutes, log_lines)
        log_lines.append(f"Created {len(chunks)} chunk(s).")

        progress(0.15, desc="Loading speech-to-text model...")
        model, loaded_model, loaded_device, loaded_compute = load_model(
            model_size, device_mode, compute_mode, log_lines
        )

        language = LANGUAGE_CHOICES[language_choice]
        chunk_seconds = max(5, int(chunk_minutes) * 60)

        all_timestamped_lines = []
        all_clean_lines = []
        srt_blocks = []
        srt_index = 1

        log_lines.append(
            f"Starting transcription | model={loaded_model} | device={loaded_device} | "
            f"compute={loaded_compute} | language={language or 'auto'} | vad={vad_filter}"
        )

        for idx, chunk_path in enumerate(chunks):
            progress_value = 0.15 + (0.80 * idx / max(1, len(chunks)))
            progress(progress_value, desc=f"Transcribing chunk {idx + 1}/{len(chunks)}...")

            offset = idx * chunk_seconds
            partial_txt = partials_dir / f"chunk_{idx:03d}.txt"
            partial_srt = partials_dir / f"chunk_{idx:03d}.srt"

            try:
                chunk_segments, info = collect_segments(model, chunk_path, language, vad_filter)
            except Exception as e:
                if device_mode == "Auto: try GPU then CPU" and loaded_device == "cuda" and is_cuda_dependency_error(e):
                    log_lines.append(
                        "GPU failed during transcription. Falling back to CPU automatically. "
                        f"Original error: {repr(e)}"
                    )
                    model, loaded_model, loaded_device, loaded_compute = load_model(
                        model_size, "CPU only", "int8 - safest / CPU friendly", log_lines
                    )
                    chunk_segments, info = collect_segments(model, chunk_path, language, vad_filter)
                    log_lines.append("CPU fallback worked for this chunk.")
                else:
                    raise

            partial_lines = []
            partial_srt_blocks = []

            for seg in chunk_segments:
                text = seg.text.strip()
                if not text:
                    continue

                start_abs = seg.start + offset
                end_abs = seg.end + offset

                readable = f"[{seconds_to_readable(start_abs)} - {seconds_to_readable(end_abs)}] {text}"
                all_timestamped_lines.append(readable)
                all_clean_lines.append(text)
                partial_lines.append(readable)

                block = (
                    f"{srt_index}\n"
                    f"{seconds_to_srt_time(start_abs)} --> {seconds_to_srt_time(end_abs)}\n"
                    f"{text}\n"
                )
                srt_blocks.append(block)
                partial_srt_blocks.append(block)
                srt_index += 1

            partial_txt.write_text("\n".join(partial_lines), encoding="utf-8")
            partial_srt.write_text("\n\n".join(partial_srt_blocks), encoding="utf-8")
            log_lines.append(f"Done chunk {idx + 1}/{len(chunks)}: {chunk_path.name}")

            txt_path.write_text("\n".join(all_timestamped_lines), encoding="utf-8")
            readable_text = build_readable_transcript(all_clean_lines)
            clean_txt_path.write_text(readable_text, encoding="utf-8")
            readable_txt_path.write_text(readable_text, encoding="utf-8")
            srt_path.write_text("\n\n".join(srt_blocks), encoding="utf-8")
            log_path.write_text("\n".join(log_lines), encoding="utf-8")

        progress(0.98, desc="Saving files...")

        summary = (
            "Done\n\n"
            f"Model used: {loaded_model}\n"
            f"Device used: {loaded_device}\n"
            f"Compute type: {loaded_compute}\n"
            f"Chunks: {len(chunks)}\n"
            f"Output folder: {work_dir}\n\n"
            "Clean readable transcript saved as clean_transcript.txt and readable_transcript.txt.\n"
            "Download the files below."
        )

        readable_text = build_readable_transcript(all_clean_lines)
        clean_txt_path.write_text(readable_text, encoding="utf-8")
        readable_txt_path.write_text(readable_text, encoding="utf-8")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        progress(1.0, desc="Finished")

        return summary, str(txt_path), str(clean_txt_path), str(readable_txt_path), str(srt_path), str(log_path)

    except Exception as e:
        log_lines.append("\nERROR:\n" + traceback.format_exc())
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        friendly = (
            "Something failed. Try CPU only + int8, or Balanced - medium. "
            "If the error mentions cublas/cudnn/CUDA DLL, run the GPU setup script or use CPU. "
            f"Details saved in process_log.txt. Error: {e}"
        )
        raise gr.Error(friendly)


with gr.Blocks(title="Local Meeting Transcriber") as demo:
    gr.Markdown(
        """
# Local Meeting Transcriber

**Step 1:** choose a model, then click **Download selected model**. This is needed only once per model.  
**Step 2:** upload a meeting recording and click **Start Transcription**.

Works with recordings from Zoom, Google Meet, Teams, lectures, interviews, or recorded calls.

Recommended default settings:
- Start with **Balanced - medium**
- Device: **Auto: try GPU then CPU**
- Compute: **Auto**
- Keep **VAD filter** enabled for long meetings
"""
    )

    with gr.Row():
        language_choice = gr.Dropdown(
            list(LANGUAGE_CHOICES.keys()),
            value="Arabic",
            label="Language",
        )
        model_size = gr.Dropdown(
            list(MODEL_CHOICES.keys()),
            value="Balanced - medium (recommended default)",
            label="Quality / Model",
        )

    with gr.Row():
        check_model_btn = gr.Button("Check selected model")
        download_model_btn = gr.Button("Download selected model", variant="secondary")
        check_gpu_btn = gr.Button("Check GPU / CUDA")

    model_status = gr.Textbox(label="Model / GPU status", lines=10)

    gr.Markdown("---")

    with gr.Row():
        video_input = gr.File(
            label="Upload meeting video/audio",
            file_types=["video", "audio"],
        )

    with gr.Row():
        device_mode = gr.Dropdown(
            DEVICE_CHOICES,
            value="Auto: try GPU then CPU",
            label="Device",
        )
        compute_mode = gr.Dropdown(
            COMPUTE_CHOICES,
            value="Auto",
            label="Memory / Compute mode",
        )

    with gr.Row():
        chunk_minutes = gr.Slider(
            minimum=5,
            maximum=30,
            step=5,
            value=20,
            label="Chunk length in minutes",
        )
        denoise = gr.Checkbox(value=True, label="Clean / normalize audio")
        vad_filter = gr.Checkbox(value=True, label="VAD filter: skip silence")

    start_btn = gr.Button("Start Transcription", variant="primary")

    status = gr.Textbox(label="Status", lines=8)
    transcript_file = gr.File(label="Download transcript with timestamps")
    clean_transcript_file = gr.File(label="Download clean readable transcript")
    readable_transcript_file = gr.File(label="Download readable transcript")
    srt_file = gr.File(label="Download subtitles SRT")
    log_file = gr.File(label="Download process log")

    check_model_btn.click(
        fn=check_model_status,
        inputs=[model_size],
        outputs=[model_status],
    )

    download_model_btn.click(
        fn=download_selected_model,
        inputs=[model_size],
        outputs=[model_status],
    )

    check_gpu_btn.click(
        fn=check_gpu_status,
        inputs=[],
        outputs=[model_status],
    )

    start_btn.click(
        fn=transcribe_file,
        inputs=[
            video_input,
            language_choice,
            model_size,
            device_mode,
            compute_mode,
            chunk_minutes,
            denoise,
            vad_filter,
        ],
        outputs=[
            status,
            transcript_file,
            clean_transcript_file,
            readable_transcript_file,
            srt_file,
            log_file,
        ],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
