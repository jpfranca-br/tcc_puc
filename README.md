# License Plate Detection Pipeline

This project provides a modularised version of the original `main.py` script that
runs license plate detection and OCR on videos or image folders. The new layout
splits responsibilities across multiple modules and adds optional HTTP Live
Streaming (HLS) output for headless environments.

## Features

- YOLO-based plate detection with EasyOCR, Tesseract, or ChatGPT Visio recognition.
- CPU/GPU flag to control the inference device.
- Asynchronous OCR processing via a dedicated HTTP endpoint with callback support.
- Real-time HUD showing the new plate lifecycle statuses (accumulating, waiting,
  error, ok, and no_match).
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
pip install ultralytics opencv-python-headless numpy easyocr pytesseract torch torchvision torchaudio openai \
            fastapi uvicorn requests
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
- `--ocr-engine {easyocr,tesseract,chatgpt_visio}` selects the OCR backend.
- `--enable-hls` starts the HLS pipeline. Visit
  `http://<host>:<port>/stream.m3u8` from an HLS-capable player (e.g. VLC or a
  browser with hls.js) to view the live stream.
- `--input-mode {video,images}` switches between video files and image folders.

Run `python main.py --help` to inspect the full list of options.

When selecting `chatgpt_visio`, the OpenAI Python client looks for an
`OPENAI_API_KEY` environment variable. Optionally override the model used for
vision OCR by defining `CHATGPT_VISION_MODEL` (defaults to `gpt-4o-mini`).

## Project structure

```
config.py        # Argument parsing and configuration dataclass
capture.py       # Video capture helpers (prefetching wrapper)
ocr.py           # OCR manager for EasyOCR/Tesseract/ChatGPT Visio
ocr_endpoint.py  # FastAPI service that accepts async OCR jobs and posts callbacks
callback_endpoint.py # FastAPI service receiving OCR callbacks and updating state
hud.py           # Heads-up display drawing utilities
streaming.py     # HLS streaming utilities
main.py          # Main loop orchestrating all modules
```

Input assets reside in `data/input`, while processed artefacts (videos and HLS
segments) are written to `data/output` and `data/hls` respectively.

## Asynchronous OCR workflow

OCR requests are no longer executed inline with the detection loop. Instead the
pipeline accumulates up to five of the largest plate crops for each track and
submits them to `ocr_endpoint.py`. The service evaluates the candidates in
descending size order and immediately responds through a callback to
`callback_endpoint.py`, allowing the main loop to continue tracking vehicles in
parallel.

Each plate moves through the following statuses which are also reflected in the
HUD and overlay colours:

| Status         | Colour | Description |
| -------------- | ------ | ----------- |
| `accumulating` | White  | Gathering crops before contacting the OCR service. |
| `ocr_waiting`  | Yellow | Awaiting a response from the OCR endpoint. |
| `ocr_error`    | Red    | Submission failed or the endpoint signalled an error. |
| `ocr_ok`       | Blue   | Validated plate (no further OCR calls required). |
| `ocr_no_match` | Orange | OCR returned but confidence was insufficient; a retry will occur. |

The callback server is started automatically by `main.py`. If you wish to run
the components independently, launch them in separate terminals:

```bash
python ocr_endpoint.py
python callback_endpoint.py
```

Use the `--ocr-endpoint-url`, `--callback-host`, `--callback-port`, and
`--callback-url` options to tailor the network topology when the services run on
different machines.

## Testing

A quick sanity check that the Python files compile can be executed with:

```bash
python -m compileall .
```

Additional integration tests depend on the availability of the YOLO weights and
sample media files.
