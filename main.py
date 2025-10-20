"""Entry point for the license plate detection pipeline."""
from __future__ import annotations

import glob
import os
import signal
import time
import atexit
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

from config import AppConfig, parse_config
from capture import PrefetchCapture
from hud import COLOR_GREEN, COLOR_ORANGE, COLOR_RED, draw_hud, natural_key
from ocr import OCRManager, OCRResult
from preprocessing import preprocess_for_yolo
from streaming import HLSStreamServer

SHOULD_STOP = False


def _handle_stop_signal(sig, frame):
    """Set a flag when receiving termination signals."""

    global SHOULD_STOP
    SHOULD_STOP = True


signal.signal(signal.SIGINT, _handle_stop_signal)
signal.signal(signal.SIGTERM, _handle_stop_signal)


def _initialise_detector(config: AppConfig) -> tuple[YOLO, dict]:
    """Load the YOLO model according to the configured device."""

    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    use_gpu = config.device_request == "gpu" and cuda_available
    if config.device_request == "gpu" and not cuda_available:
        print("GPU requested but not available. Falling back to CPU.")
    device_name = "cuda:0" if use_gpu else "cpu"

    detector = YOLO(config.model_path)
    detector.to(device_name)

    infer_kwargs: dict = {"device": device_name, "half": use_gpu}
    return detector, infer_kwargs


def _setup_input(config: AppConfig):
    """Prepare the frame source based on the chosen mode."""

    if config.input_mode == "video":
        if config.fast_video:
            cap = PrefetchCapture(config.video_in, queue_size=config.frame_queue_size)
            probe = cv2.VideoCapture(config.video_in)
            if not probe.isOpened():
                raise RuntimeError(f"Could not open video: {config.video_in}")
            frame_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
            probe.release()
        else:
            cap = cv2.VideoCapture(config.video_in)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {config.video_in}")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        video_writer = None
        if config.video_out is not None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_writer = cv2.VideoWriter(
                config.video_out, fourcc, fps, (frame_width, frame_height)
            )
        return cap, video_writer, frame_width, frame_height, float(fps), None

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(config.images_dir, pattern)))
    if not files:
        raise RuntimeError(f"No images found in: {config.images_dir}")
    files.sort(key=natural_key)

    first = cv2.imread(files[0])
    if first is None:
        raise RuntimeError(f"Failed to read first image: {files[0]}")
    frame_height, frame_width = first.shape[:2]
    fps = float(config.images_per_second) if config.images_per_second > 0 else 0.1
    return files, None, frame_width, frame_height, fps, 0


def main() -> None:
    """Execute the main processing loop."""

    global SHOULD_STOP
    config = parse_config()
    detector, infer_kwargs = _initialise_detector(config)

    ocr_gpu = config.ocr_gpu and infer_kwargs.get("device", "cpu").startswith("cuda")
    # The OCR manager hides the differences between EasyOCR and Tesseract
    # so the rest of the loop can treat them uniformly.
    ocr_manager = OCRManager(
        engine=config.ocr_engine,
        use_gpu=ocr_gpu,
        multivariant=config.easyocr_multivariant,
        conf_threshold_low=config.conf_threshold_low,
    )

    # Prepare input sources (video stream or image folder) and optional writers.
    source, video_writer, frame_width, frame_height, fps, image_index = _setup_input(config)

    hls_server = None
    if config.enable_hls:
        # The HLS server uses ffmpeg to transcode frames and a tiny HTTP server
        # to expose ``stream.m3u8`` for remote/headless environments.
        hls_server = HLSStreamServer(
            width=frame_width,
            height=frame_height,
            fps=fps or 25.0,
            directory=config.hls_directory,
            host=config.hls_host,
            port=config.hls_port,
        )
        try:
            hls_server.start()
            print(
                f"HLS server running at http://{config.hls_host}:{config.hls_port}/stream.m3u8"
            )
        except FileNotFoundError:
            print("ffmpeg not found. Disable --enable-hls or install ffmpeg.")
            hls_server = None

    if config.yolo_warmup:
        dummy = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        if config.input_mode == "images":
            detector.predict(dummy, verbose=False, **infer_kwargs)
        else:
            detector.track(dummy, persist=True, verbose=False, **infer_kwargs)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    # Track meta information for each object ID produced by the tracker.
    track_history: Dict[int, Dict[str, object]] = {}
    ocr_method_stats: Dict[int, Dict[str, Dict[str, object]]] = {}

    ocr_calls = 0
    ocr_success = 0
    processed_frames = 0
    processing_start_ts = None
    next_track_id = 0

    yolo_last_dt = 0.0
    ocr_last_dt = 0.0
    preproc_last_dt = 0.0

    loop_dt = 0.0
    loop_min = float("inf")
    loop_max = float("-inf")
    yolo_min = float("inf")
    yolo_max = float("-inf")
    ocr_min = float("inf")
    ocr_max = float("-inf")
    ocr_retry_min = float("inf")
    ocr_retry_max = float("-inf")
    preproc_min = float("inf")
    preproc_max = float("-inf")

    hud_lines: List[str] = []

    def print_run_summary():
        """Report aggregated metrics once the process exits."""

        try:
            print("\n==================== RUN SUMMARY ====================")
            print("\n".join(hud_lines))
            print("\nOCR method - bests per plate:")
            print("=====================================================\n")
            if not ocr_method_stats:
                print("  (no OCR stats recorded)")
            else:
                for tid, methods in ocr_method_stats.items():
                    print(f"Placa {tid}:")
                    for method, info in methods.items():
                        plate = info.get("plate", "")
                        conf = info.get("best_conf", 0.0) * 100
                        print(f"    plate='{plate}' ({conf:0.1f}%) - {method:<8}")
            print("=====================================================\n")
        except Exception as exc:
            print(f"[Summary error] {exc}")

    atexit.register(print_run_summary)

    processing_start_ts = time.perf_counter()

    try:
        # Main processing loop that stops on EOF or when a signal is received.
        while not SHOULD_STOP:
            loop_t0 = time.perf_counter()
            # ``source`` is a list in image mode; otherwise it behaves like a capture.
            if isinstance(source, list):
                if image_index >= len(source):
                    break
                frame = cv2.imread(source[image_index])
                if frame is None:
                    print(f"Skipping unreadable image: {source[image_index]}")
                    image_index += 1
                    continue
                ret = True
            else:
                ret, frame = source.read()
            if not ret:
                break

            processed_frames += 1
            if processing_start_ts is None:
                processing_start_ts = time.perf_counter()

            input_frame = frame
            scale = 1.0
            # Optionally resize frames so YOLO receives a consistent width.
            if config.target_yolo_width and frame.shape[1] != config.target_yolo_width:
                scale = config.target_yolo_width / frame.shape[1]
                resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            else:
                resized = frame

            if config.opencv_pre_yolo:
                pre_t0 = time.perf_counter()
                resized = preprocess_for_yolo(resized)
                preproc_last_dt = time.perf_counter() - pre_t0
                preproc_min = min(preproc_min, preproc_last_dt)
                preproc_max = max(preproc_max, preproc_last_dt)

            # Run the detector and keep timing statistics.
            yolo_t0 = time.perf_counter()
            if config.input_mode == "images":
                results = detector.predict(resized, verbose=False, **infer_kwargs)
            else:
                results = detector.track(resized, persist=True, verbose=False, **infer_kwargs)
            yolo_last_dt = time.perf_counter() - yolo_t0
            yolo_min = min(yolo_min, yolo_last_dt)
            yolo_max = max(yolo_max, yolo_last_dt)

            now = time.perf_counter()
            ocr_retry_wait = config.ocr_min_wait
            track_ids = np.empty(0, dtype=int)
            boxes = np.empty((0, 4), dtype=int)
            if results:
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    boxes = detections.xyxy.cpu().numpy().astype(int)
                    if detections.id is not None:
                        track_ids = detections.id.cpu().numpy().astype(int)
                    elif config.input_mode == "images":
                        track_ids = np.arange(next_track_id, next_track_id + len(boxes))
                        next_track_id += len(track_ids)
                    if scale != 1.0 and len(boxes) > 0:
                        boxes = (boxes / scale).astype(int)

            # Convert YOLO output back into the original frame space.
            if track_ids.size > 0:
                ocr_retry_wait = max(
                    config.ocr_min_wait,
                    config.ocr_wait_multiplier * loop_dt * track_ids.size if loop_dt else config.ocr_min_wait,
                )
                ocr_retry_max = max(ocr_retry_wait, ocr_retry_max)
                ocr_retry_min = min(ocr_retry_wait, ocr_retry_min)

                for box, track_id in zip(boxes, track_ids):
                    if track_id not in track_history:
                        track_history[track_id] = {
                            "name": f"Plate_{len(track_history) + 1}",
                            "confidence": "",
                            "ocr_done": False,
                            "ocr_certain": False,
                            "last_ocr_attempt": 0.0,
                        }

                    info = track_history[track_id]

                    if (not info["ocr_certain"]) and (now - info["last_ocr_attempt"] > ocr_retry_wait):
                        info["last_ocr_attempt"] = now
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_width, x2), min(frame_height, y2)
                        crop = input_frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        ocr_t0 = time.perf_counter()
                        try:
                            result: OCRResult = ocr_manager.run(crop)
                            if track_id not in ocr_method_stats:
                                ocr_method_stats[track_id] = {}
                            track_methods = ocr_method_stats[track_id]
                            current_best = track_methods.get(result.method, {"best_conf": 0.0, "plate": ""})
                            if result.confidence > current_best["best_conf"]:
                                track_methods[result.method] = {
                                    "best_conf": result.confidence,
                                    "plate": result.text,
                                }
                        except Exception as exc:
                            print(f"OCR error for track {track_id}: {exc}")
                            result = OCRResult(None, 0.0, "-")

                        ocr_last_dt = time.perf_counter() - ocr_t0
                        ocr_min = min(ocr_min, ocr_last_dt)
                        ocr_max = max(ocr_max, ocr_last_dt)
                        ocr_calls += 1

                        if result.text:
                            cleaned = result.text
                            info["method"] = result.method
                            if (
                                result.confidence >= config.conf_threshold_high
                                and len(cleaned) >= config.char_count_high
                            ):
                                info.update(
                                    {
                                        "name": cleaned,
                                        "confidence": f" ({result.confidence:.0%})",
                                        "ocr_done": True,
                                        "ocr_certain": True,
                                    }
                                )
                                ocr_success += 1
                            elif (
                                result.confidence >= config.conf_threshold_low
                                and len(cleaned) >= config.char_count_low
                            ):
                                info.update(
                                    {
                                        "name": cleaned,
                                        "confidence": f" ({result.confidence:.0%})",
                                        "ocr_done": True,
                                    }
                                )
                                ocr_success += 1

                for box, track_id in zip(boxes, track_ids):
                    info = track_history.get(track_id, {})
                    label = info.get("name", "") + info.get("confidence", "")
                    method_label = info.get("method", "")
                    if method_label:
                        label += f" [{method_label}]"
                    color = (
                        COLOR_GREEN
                        if info.get("ocr_certain")
                        else (COLOR_ORANGE if info.get("ocr_done") else COLOR_RED)
                    )
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.9,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            plates_total = len(track_history)
            plates_certain = sum(1 for v in track_history.values() if v.get("ocr_certain"))
            plates_done = sum(1 for v in track_history.values() if v.get("ocr_done") and not v.get("ocr_certain"))
            plates_pending = plates_total - (plates_done + plates_certain)
            num_tracked = int(track_ids.size)

            total_secs = max(1e-6, time.perf_counter() - (processing_start_ts or time.perf_counter()))
            overall_fps = processed_frames / total_secs

            input_desc = (
                f"video: {os.path.basename(config.video_in)}"
                if config.input_mode == "video"
                else f"images dir: {os.path.abspath(config.images_dir)}"
            )

            hud_lines = [
                f"Input source    | {input_desc}",
                f"Input source fps| {fps:.2f}" if fps else "Input source fps| N/A",
                f"OCR Engine      | {config.ocr_engine.upper()}  {ocr_manager.hud_status()}",
                f"GPU             | {infer_kwargs.get('device')}",
                f"Preprocessing   | {config.opencv_pre_yolo}",
                f"Multivariant    | {config.easyocr_multivariant}",
                f"Fast Video      | {config.fast_video} (queue={config.frame_queue_size})",
                f"Show Video      | {config.show_window}",
                f"Detectd         | {plates_total}",
                f"Tracked         | {num_tracked}",
                f"OCR calls       | {ocr_calls}",
                f"Plates          | bad={plates_pending} | average={plates_done} | ok={plates_certain}",
                f"Frames          | {processed_frames} processed in {total_secs:0.2f}s | fps={overall_fps:0.2f}",
                f"-----------------------------------------",
                f"Timings (ms)    |  min  |  cur  |  max  |",
                f"----------------------------------------",
                (
                    f"Pre Processing  | {preproc_min*1000:05.1f} | {preproc_last_dt*1000:05.1f} | {preproc_max*1000:05.1f}"
                    if config.opencv_pre_yolo
                    else "Pre Processing  |  N/A  |  N/A  |  N/A  |"
                ),
                f"YOLO Processing | {yolo_min*1000:05.1f} | {yolo_last_dt*1000:05.1f} | {yolo_max*1000:05.1f}",
                f"OCR Processing  | {ocr_min*1000:05.1f} | {ocr_last_dt*1000:05.1f} | {ocr_max*1000:05.1f}",
                f"OCR Wait (set)  | {ocr_retry_min*1000:05.1f} | {ocr_retry_wait*1000:05.1f} | {ocr_retry_max*1000:05.1f}",
                f"Total Loop      | {loop_min*1000:05.1f} | {loop_dt*1000:05.1f} | {loop_max*1000:05.1f}",
            ]

            draw_hud(frame, hud_lines, margin=5, padding=10, alpha=0.35)

            if config.input_mode == "video" and video_writer is not None:
                video_writer.write(frame)

            if hls_server:
                hls_server.push_frame(frame)

            if config.show_window:
                display_frame = cv2.resize(frame, (config.display_width, config.display_height))
                cv2.imshow("Annotated Output", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    while True:
                        key2 = cv2.waitKey(0) & 0xFF
                        if key2 == ord("p"):
                            break
                        if key2 == ord("q"):
                            SHOULD_STOP = True
                            break
                if key == ord("q"):
                    break

            if isinstance(source, list):
                image_index += 1
                if config.images_per_second > 0:
                    next_show_time = time.perf_counter() + (1.0 / float(max(config.images_per_second, 1e-6)))
                    while True:
                        if time.perf_counter() >= next_show_time:
                            break
                        if config.show_window:
                            if cv2.waitKey(10) & 0xFF == ord("q"):
                                SHOULD_STOP = True
                                break

            if SHOULD_STOP:
                break

            loop_dt = time.perf_counter() - loop_t0
            loop_min = min(loop_min, loop_dt)
            loop_max = max(loop_max, loop_dt)

    except KeyboardInterrupt:
        pass
    finally:
        print("Processing finished. Releasing resources.")
        # Release whichever capture implementation we used.
        if isinstance(source, PrefetchCapture):
            source.release()
        elif hasattr(source, "release"):
            source.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        if hls_server:
            hls_server.stop()


if __name__ == "__main__":
    main()
