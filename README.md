# Local Meeting Transcriber

Local Gradio app for transcribing meeting recordings with faster-whisper.

## Overview

Local Meeting Transcriber is a Windows-friendly transcription tool for meeting recordings, lectures, interviews, and recorded calls. It runs on your own machine, uses faster-whisper locally, and exports transcript files from a simple Gradio interface.

It is designed for non-technical Windows users, so the project includes `.bat` files for installation and startup.

## Why I built it

Many transcription tools require uploading private recordings to cloud services. This project keeps the workflow local-first: choose a recording, download a Whisper model once, run transcription on your computer, and keep the outputs on your computer.

## Features

- Upload meeting recordings from Zoom, Google Meet, Teams, lectures, interviews, or recorded calls
- Download the selected Whisper model from the app interface
- Transcribe locally using faster-whisper
- GPU and CPU options
- Arabic, English, and auto language detection
- Audio cleanup and normalization option
- VAD filter option to skip silence
- Export `transcript.txt`
- Export `clean_transcript.txt`
- Export `readable_transcript.txt`
- Export `subtitles.srt`
- Export `process_log.txt`
- Simple Gradio interface
- Windows `.bat` scripts for non-technical users

## How it works

```text
Video/audio file
  -> audio extraction and cleanup
  -> faster-whisper transcription
  -> text transcript
  -> clean readable transcript
  -> SRT subtitles
  -> downloadable outputs
```

## Setup for non-technical Windows users

1. Download or unzip the project folder.
2. Open `00_INSTALL_APP_FIRST.bat`.
3. Wait for installation to finish.
4. Open `02_START_APP.bat`.
5. In the browser, choose a model such as `Balanced - medium`.
6. Click `Download selected model`.
7. Wait until the app says the model is ready.
8. Upload a meeting recording and click `Start Transcription`.

For safer CPU-only use, open `03_CPU_SAFE_START.bat`, then choose:

- Device: `CPU only`
- Compute: `int8 - safest / CPU friendly`

## Manual setup for developers

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Usage

1. Choose the language: Arabic, English, or Auto detect.
2. Choose the model size.
3. Click `Download selected model` if the model is not already downloaded.
4. Upload a video or audio recording.
5. Choose GPU/CPU and compute settings.
6. Keep audio cleanup and VAD enabled for most long recordings.
7. Click `Start Transcription`.
8. Download the generated files.

## Output files

Each run creates a local folder under `outputs/job_*`.

Common output files:

- `transcript.txt`: transcript with timestamps
- `clean_transcript.txt`: readable transcript text
- `readable_transcript.txt`: same readable transcript text for convenience
- `subtitles.srt`: subtitle file
- `process_log.txt`: technical log for troubleshooting

Temporary files such as cleaned audio, chunks, and partial transcripts may also be created inside the job folder.

## Project structure

```text
app.py                                  Main Gradio app
requirements.txt                        Python dependencies
00_INSTALL_APP_FIRST.bat                First-time Windows setup
01_INSTALL_CUDA_12_FOR_GPU.bat          Optional CUDA helper for GPU users
02_START_APP.bat                        Normal app startup
03_CPU_SAFE_START.bat                   CPU-safe startup helper
README_AR.txt                           Arabic instructions
QUICK_STEPS_FOR_NON_TECHNICAL_USER.txt  Short beginner instructions
CHANGELOG_V5.txt                        Version notes
outputs/                                Generated transcripts and logs, ignored by Git
models/                                 Downloaded Whisper models, ignored by Git
installers/                             Downloaded installers, ignored by Git
```

## GPU / CPU notes

- CPU mode is slower but usually safer and does not require CUDA.
- GPU mode can be faster, but CUDA/cuDNN compatibility can be fragile on Windows.
- The install script includes NVIDIA CUDA-related Python packages used by the current GPU workflow.
- If GPU mode fails with CUDA, cuBLAS, or cuDNN errors, use CPU mode or run `01_INSTALL_CUDA_12_FOR_GPU.bat`.
- Large models may fail on GPUs with limited VRAM. Use `Balanced - medium` if `large-v3` runs out of memory.

## Limitations

- Large recordings can take a long time.
- GPU setup may require compatible NVIDIA drivers, CUDA, and cuDNN.
- CPU mode is slower but safer.
- Transcription accuracy depends on audio quality, language, background noise, speaker clarity, and model size.
- This is a local prototype/tool, not a hosted transcription service.

## Privacy note

- Files are processed locally on your machine.
- No cloud transcription API is used.
- Whisper models are downloaded to the local `models/` folder when you request them.
- Only transcribe recordings you have permission to process.
- Meeting recordings may contain sensitive data, so handle generated transcripts and subtitles carefully.

## Tech stack

- Python
- Gradio
- faster-whisper
- Hugging Face Hub
- imageio-ffmpeg / FFmpeg
- Optional NVIDIA CUDA/cuDNN/cuBLAS packages for GPU support

## Future improvements

- Speaker diarization
- Better transcript formatting
- Search inside transcript
- Summary generation
- Translation
- Export DOCX/PDF
- Better progress reporting
- Packaged Windows installer

## Repository hygiene

Do not commit private media, generated transcripts, downloaded models, installer files, temporary audio chunks, or output folders. The included `.gitignore` is configured to keep those local-only files out of Git.
