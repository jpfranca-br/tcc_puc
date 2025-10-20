"""Application configuration helpers.

This module centralizes default values and argument parsing so that the
application settings are no longer scattered throughout ``main.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path


@dataclass
class AppConfig:
    """Stores runtime options for the license plate pipeline."""

    # Input configuration
    input_mode: str = "video"
    video_in: str = "./data/input/traffic2.mp4"
    video_out: str | None = "./data/output/output.avi"
    images_dir: str = "./data/images_in"
    images_per_second: float = 30.0

    # Performance toggles
    fast_video: bool = True
    frame_queue_size: int = 512
    show_window: bool = True

    # Model toggles
    model_path: str = "./models/license_plate_detector.pt"
    opencv_pre_yolo: bool = False
    target_yolo_width: int = 640
    yolo_warmup: bool = True

    # OCR options
    ocr_engine: str = "easyocr"
    ocr_gpu: bool = True
    easyocr_multivariant: bool = False

    # Thresholds
    conf_threshold_low: float = 0.20
    char_count_low: int = 3
    conf_threshold_high: float = 0.80
    char_count_high: int = 6
    ocr_wait_multiplier: float = 3.0
    ocr_min_wait: float = 0.2

    # Display
    display_width: int = 1280
    display_height: int = 720

    # Runtime device control
    device_request: str = "gpu"

    # Streaming
    enable_hls: bool = False
    hls_directory: str = "./data/hls"
    hls_host: str = "0.0.0.0"
    hls_port: int = 8000

    def ensure_directories(self) -> None:
        """Create output directories that must exist at runtime."""

        if self.video_out:
            Path(self.video_out).parent.mkdir(parents=True, exist_ok=True)
        if self.enable_hls:
            Path(self.hls_directory).mkdir(parents=True, exist_ok=True)


def parse_config() -> AppConfig:
    """Parse command line arguments and build an :class:`AppConfig`."""

    parser = argparse.ArgumentParser(description="License plate detection pipeline")
    parser.add_argument("--input-mode", choices=["video", "images"], default="video")
    parser.add_argument("--video-in", default="./data/input/traffic.mp4")
    parser.add_argument("--video-out", default="./data/output/output.avi")
    parser.add_argument("--images-dir", default="./data/images_in")
    parser.add_argument("--images-per-second", type=float, default=5.0)
    parser.add_argument("--fast-video", action="store_true", default="False")
    parser.add_argument("--no-fast-video", action="store_false", dst="fast-video")
    parser.add_argument("--frame-queue-size", type=int, default=8)
    parser.add_argument("--show-window", action="store_true", default=True)
    parser.add_argument("--no-window", action="store_false", dest="show_window")
    parser.add_argument("--model-path", default="./models/license_plate_detector.pt")
    parser.add_argument("--opencv-pre-yolo", action="store_true", default=False)
    parser.add_argument("--target-yolo-width", type=int, default=640)
    parser.add_argument("--yolo-warmup", action="store_true", default=True)
    parser.add_argument("--no-yolo-warmup", action="store_false", dest="yolo_warmup")
    parser.add_argument(
        "--ocr-engine",
        choices=["easyocr", "tesseract", "chatgpt_visio"],
        default="easyocr",
    )
    parser.add_argument("--ocr-gpu", action="store_true", default=True)
    parser.add_argument("--ocr-cpu", action="store_false", dest="ocr_gpu")
    parser.add_argument("--easyocr-multivariant", action="store_true", default=False)
    parser.add_argument("--conf-threshold-low", type=float, default=0.50)
    parser.add_argument("--conf-threshold-high", type=float, default=0.80)
    parser.add_argument("--char-count-low", type=int, default=3)
    parser.add_argument("--char-count-high", type=int, default=6)
    parser.add_argument("--ocr-wait-multiplier", type=float, default=3.0)
    parser.add_argument("--ocr-min-wait", type=float, default=0.5)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--enable-hls", action="store_true", default=False)
    parser.add_argument("--hls-directory", default="./data/hls")
    parser.add_argument("--hls-host", default="0.0.0.0")
    parser.add_argument("--hls-port", type=int, default=8000)

    args = parser.parse_args()
    config = AppConfig(
        input_mode=args.input_mode,
        video_in=args.video_in,
        video_out=args.video_out if args.video_out.lower() != "none" else None,
        images_dir=args.images_dir,
        images_per_second=args.images_per_second,
        fast_video=args.fast_video,
        frame_queue_size=args.frame_queue_size,
        show_window=args.show_window,
        model_path=args.model_path,
        opencv_pre_yolo=args.opencv_pre_yolo,
        target_yolo_width=args.target_yolo_width,
        yolo_warmup=args.yolo_warmup,
        ocr_engine=args.ocr_engine,
        ocr_gpu=args.ocr_gpu,
        easyocr_multivariant=args.easyocr_multivariant,
        conf_threshold_low=args.conf_threshold_low,
        conf_threshold_high=args.conf_threshold_high,
        char_count_low=args.char_count_low,
        char_count_high=args.char_count_high,
        ocr_wait_multiplier=args.ocr_wait_multiplier,
        ocr_min_wait=args.ocr_min_wait,
        display_width=args.display_width,
        display_height=args.display_height,
        device_request=args.device,
        enable_hls=args.enable_hls,
        hls_directory=args.hls_directory,
        hls_host=args.hls_host,
        hls_port=args.hls_port,
    )
    config.ensure_directories()
    return config
