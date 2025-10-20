# License Plate Detection Pipeline

This project provides a modularised version of the original `main.py` script that
runs license plate detection and OCR on videos or image folders. The new layout
splits responsibilities across multiple modules and adds optional HTTP Live
Streaming (HLS) output for headless environments.

## Features

- YOLO-based plate detection with EasyOCR or Tesseract recognition.
- CPU/GPU flag to control the inference device.
- Optional OpenCV preprocessing pipeline to improve detections.
- HLS streaming server for viewing results on remote machines.
- Background frame prefetching for improved throughput.

## Installation

### System packages

Install the required system dependencies:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg tesseract-ocr
```

### Python environment

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python-headless numpy easyocr pytesseract torch torchvision torchaudio
```

> **Tip:** When using GPU acceleration, install the CUDA-enabled PyTorch build
> from [pytorch.org](https://pytorch.org/) that matches your driver/toolkit.

## Running the application

Activate your virtual environment and execute the entry point:

```bash
source .venv/bin/activate
python main.py --device gpu --video-in ./data/input/traffic2.mp4
```

Key options:

- `--device {cpu,gpu}` toggles between CPU and GPU inference. If `gpu` is
  requested but unavailable, the script falls back to CPU automatically.
- `--ocr-engine {easyocr,tesseract}` selects the OCR backend.
- `--enable-hls` starts the HLS pipeline. Visit
  `http://<host>:<port>/stream.m3u8` from an HLS-capable player (e.g. VLC or a
  browser with hls.js) to view the live stream.
- `--input-mode {video,images}` switches between video files and image folders.

Run `python main.py --help` to inspect the full list of options.

## Project structure

```
config.py        # Argument parsing and configuration dataclass
capture.py       # Video capture helpers (prefetching wrapper)
preprocessing.py # OpenCV preprocessing steps
ocr.py           # OCR manager for EasyOCR/Tesseract
hud.py           # Heads-up display drawing utilities
streaming.py     # HLS streaming utilities
main.py          # Main loop orchestrating all modules
```

Input assets reside in `data/input`, while processed artefacts (videos and HLS
segments) are written to `data/output` and `data/hls` respectively.

## Testing

A quick sanity check that the Python files compile can be executed with:

```bash
python -m compileall .
```

Additional integration tests depend on the availability of the YOLO weights and
sample media files.
