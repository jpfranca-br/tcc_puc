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


def _initialise_detector(
    config: AppConfig,
    model_path: str,
    device_name: str | None = None,
    half_precision: bool | None = None,
) -> tuple[YOLO, dict, str, bool]:
    """Load a YOLO model according to the configured or provided device."""

    if device_name is None or half_precision is None:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        use_gpu = config.device_request == "gpu" and cuda_available
        if config.device_request == "gpu" and not cuda_available:
            print("GPU requested but not available. Falling back to CPU.")
        device_name = "cuda:0" if use_gpu else "cpu"
        half_precision = use_gpu

    detector = YOLO(model_path)
    detector.to(device_name)

    infer_kwargs: dict = {"device": device_name, "half": half_precision}
    return detector, infer_kwargs, device_name, half_precision


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
    detector, infer_kwargs, device_name, half_precision = _initialise_detector(
        config, config.model_path
    )

    model_names = getattr(detector, "names", {}) or {}
    if isinstance(model_names, dict):
        class_name_map = {int(idx): str(name).lower() for idx, name in model_names.items()}
    else:
        class_name_map = {int(idx): str(name).lower() for idx, name in enumerate(model_names)}

    plate_class_ids = {idx for idx, name in class_name_map.items() if "plate" in name}
    vehicle_keywords = (
        "vehicle",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "motorbike",
        "van",
        "automobile",
        "auto",
        "pickup",
        "suv",
        "carro",
        "veiculo",
        "caminhao",
        "caminhonete",
        "onibus",
        "moto",
    )
    vehicle_class_ids = {
        idx
        for idx, name in class_name_map.items()
        if any(keyword in name for keyword in vehicle_keywords)
    }
    primary_auto_vehicle_fallback = False
    if not plate_class_ids:
        plate_class_ids = {0}

    if not vehicle_class_ids:
        fallback_ids = {idx for idx in class_name_map.keys() if idx not in plate_class_ids}
        if fallback_ids:
            vehicle_class_ids = fallback_ids
            print(
                "No explicit vehicle classes advertised; treating remaining classes as vehicles."
            )
        else:
            primary_auto_vehicle_fallback = True

    use_secondary_vehicle_model = False
    secondary_vehicle_class_ids: set[int] = set()
    secondary_auto_vehicle_fallback = False
    vehicle_detector: YOLO | None = None
    vehicle_infer_kwargs: dict | None = None

    if primary_auto_vehicle_fallback and config.vehicle_model_path:
        try:
            vehicle_detector, vehicle_infer_kwargs, _, _ = _initialise_detector(
                config,
                config.vehicle_model_path,
                device_name=device_name,
                half_precision=half_precision,
            )
            vehicle_model_names = getattr(vehicle_detector, "names", {}) or {}
            if isinstance(vehicle_model_names, dict):
                vehicle_name_map = {
                    int(idx): str(name).lower() for idx, name in vehicle_model_names.items()
                }
            else:
                vehicle_name_map = {
                    int(idx): str(name).lower()
                    for idx, name in enumerate(vehicle_model_names)
                }

            secondary_vehicle_class_ids = {
                idx
                for idx, name in vehicle_name_map.items()
                if any(keyword in name for keyword in vehicle_keywords)
            }
            if not secondary_vehicle_class_ids:
                secondary_vehicle_class_ids = set(vehicle_name_map.keys())
                secondary_auto_vehicle_fallback = True
                print(
                    "Secondary vehicle model does not expose recognised vehicle class names;"
                    " treating all detections as vehicles."
                )
            else:
                use_secondary_vehicle_model = True
        except Exception as exc:
            print(
                "Failed to load secondary vehicle model "
                f"'{config.vehicle_model_path}': {exc}. Falling back to box association only."
            )
            vehicle_detector = None
            vehicle_infer_kwargs = None
            secondary_vehicle_class_ids = set()
            secondary_auto_vehicle_fallback = False
            primary_auto_vehicle_fallback = True

    if vehicle_detector is not None and not use_secondary_vehicle_model:
        use_secondary_vehicle_model = bool(secondary_vehicle_class_ids)

    if use_secondary_vehicle_model:
        print(f"Using secondary vehicle model '{config.vehicle_model_path}' for vehicle tracking.")

    if primary_auto_vehicle_fallback and not use_secondary_vehicle_model:
        print(
            "No vehicle detections available from the plate model; "
            "consider providing --vehicle-model-path for a dedicated vehicle detector."
        )

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

    # Track meta information for each detected plate and vehicle.
    plate_history: Dict[int, Dict[str, object]] = {}
    plate_id_map: Dict[int, int] = {}
    plate_to_vehicle: Dict[int, int] = {}
    vehicle_history: Dict[int, Dict[str, object]] = {}
    vehicle_id_map: Dict[int, int] = {}
    vehicle_to_plate: Dict[int, int] = {}
    ocr_method_stats: Dict[int, Dict[str, Dict[str, object]]] = {}
    ocr_accumulators: Dict[int, Dict[str, object]] = {}

    ocr_calls = 0
    ocr_success = 0
    processed_frames = 0
    processing_start_ts = None
    next_plate_track_id = 0
    next_vehicle_track_id = 0
    next_plate_global_id = 1
    next_vehicle_global_id = 1

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

    def _run_ocr_for_plate(plate_id: int, crop: np.ndarray, now_ts: float) -> OCRResult:
        """Execute OCR for a single track and update bookkeeping."""

        nonlocal ocr_calls, ocr_last_dt, ocr_min, ocr_max, ocr_success

        ocr_calls += 1
        ocr_t0 = time.perf_counter()
        try:
            result = ocr_manager.run(crop)
        except Exception as exc:
            print(f"OCR error for track {plate_id}: {exc}")
            result = OCRResult(None, 0.0, "-")
        ocr_last_dt = time.perf_counter() - ocr_t0
        ocr_min = min(ocr_min, ocr_last_dt)
        ocr_max = max(ocr_max, ocr_last_dt)

        info = plate_history.get(plate_id)
        if info is not None:
            info["last_ocr_attempt"] = now_ts

        if plate_id not in ocr_method_stats:
            ocr_method_stats[plate_id] = {}
        track_methods = ocr_method_stats[plate_id]
        current_best = track_methods.get(result.method, {"best_conf": 0.0, "plate": ""})
        if result.confidence > current_best["best_conf"]:
            track_methods[result.method] = {
                "best_conf": result.confidence,
                "plate": result.text,
            }

        if info is not None and result.text:
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

        return result

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
            classes = np.empty(0, dtype=int)
            if results:
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    boxes = detections.xyxy.cpu().numpy().astype(int)
                    if detections.id is not None:
                        track_ids = detections.id.cpu().numpy().astype(int)
                    else:
                        generated_ids = np.arange(
                            next_plate_track_id, next_plate_track_id + len(boxes)
                        )
                        track_ids = generated_ids.astype(int)
                        next_plate_track_id += len(generated_ids)
                    if detections.cls is not None:
                        classes = detections.cls.cpu().numpy().astype(int)
                    else:
                        classes = np.zeros(len(boxes), dtype=int)
                    if scale != 1.0 and len(boxes) > 0:
                        boxes = (boxes / scale).astype(int)

            detection_entries: List[Dict[str, object]] = []
            for idx in range(len(boxes)):
                entry = {
                    "box": boxes[idx],
                    "track_id": int(track_ids[idx]) if track_ids.size > idx else int(idx),
                    "class_id": int(classes[idx]) if classes.size > idx else 0,
                }
                detection_entries.append(entry)

            plate_detections: List[Dict[str, object]] = []
            vehicle_detections: List[Dict[str, object]] = []
            for det in detection_entries:
                class_id = det["class_id"]
                if class_id in plate_class_ids:
                    plate_detections.append(det)
                elif not use_secondary_vehicle_model:
                    if class_id in vehicle_class_ids or primary_auto_vehicle_fallback:
                        # When the model does not expose explicit vehicle classes we still
                        # want to keep the bounding boxes generated by the tracker so the
                        # vehicle state machine can operate. Treat any non-plate class as a
                        # vehicle in that scenario.
                        vehicle_detections.append(det)

            if use_secondary_vehicle_model and vehicle_detector is not None:
                secondary_results = None
                if config.input_mode == "images":
                    secondary_results = vehicle_detector.predict(
                        resized, verbose=False, **(vehicle_infer_kwargs or infer_kwargs)
                    )
                else:
                    secondary_results = vehicle_detector.track(
                        resized, persist=True, verbose=False, **(vehicle_infer_kwargs or infer_kwargs)
                    )
                if secondary_results:
                    sec_detections = secondary_results[0].boxes
                    if sec_detections is not None and len(sec_detections) > 0:
                        v_boxes = sec_detections.xyxy.cpu().numpy().astype(int)
                        if sec_detections.id is not None:
                            v_track_ids = sec_detections.id.cpu().numpy().astype(int)
                        else:
                            generated_ids = np.arange(
                                next_vehicle_track_id, next_vehicle_track_id + len(v_boxes)
                            )
                            v_track_ids = generated_ids.astype(int)
                            next_vehicle_track_id += len(generated_ids)
                        if sec_detections.cls is not None:
                            v_classes = sec_detections.cls.cpu().numpy().astype(int)
                        else:
                            v_classes = np.zeros(len(v_boxes), dtype=int)
                        if scale != 1.0 and len(v_boxes) > 0:
                            v_boxes = (v_boxes / scale).astype(int)

                        for idx in range(len(v_boxes)):
                            class_id = int(v_classes[idx]) if v_classes.size > idx else 0
                            if (
                                class_id in secondary_vehicle_class_ids
                                or secondary_auto_vehicle_fallback
                            ):
                                vehicle_detections.append(
                                    {
                                        "box": v_boxes[idx],
                                        "track_id": int(v_track_ids[idx])
                                        if v_track_ids.size > idx
                                        else int(idx),
                                        "class_id": class_id,
                                        "source": "secondary",
                                    }
                                )

            active_plate_ids: set[int] = set()
            active_vehicle_ids: set[int] = set()

            if vehicle_detections:
                current_vehicle_yolo_ids: set[int] = set()
                for det in vehicle_detections:
                    yolo_vehicle_id = int(det["track_id"])
                    current_vehicle_yolo_ids.add(yolo_vehicle_id)
                    vehicle_global_id = vehicle_id_map.get(yolo_vehicle_id)
                    if vehicle_global_id is None:
                        vehicle_global_id = next_vehicle_global_id
                        next_vehicle_global_id += 1
                        vehicle_id_map[yolo_vehicle_id] = vehicle_global_id
                        vehicle_history[vehicle_global_id] = {
                            "name": f"Vehicle_{vehicle_global_id}",
                            "last_seen": now,
                        }
                    vehicle_info = vehicle_history.setdefault(
                        vehicle_global_id, {"name": f"Vehicle_{vehicle_global_id}"}
                    )
                    vehicle_info["last_seen"] = now
                    det["global_id"] = vehicle_global_id
                    det["associated_plate"] = vehicle_to_plate.get(vehicle_global_id)
                    active_vehicle_ids.add(vehicle_global_id)

                for yolo_vehicle_id in list(vehicle_id_map.keys()):
                    if yolo_vehicle_id not in current_vehicle_yolo_ids:
                        vehicle_id_map.pop(yolo_vehicle_id, None)

            current_plate_yolo_ids: set[int] = set()
            for det in plate_detections:
                yolo_plate_id = int(det["track_id"])
                current_plate_yolo_ids.add(yolo_plate_id)

                vehicle_global_id = None
                if vehicle_detections:
                    px1, py1, px2, py2 = det["box"]
                    cx = (px1 + px2) / 2.0
                    cy = (py1 + py2) / 2.0
                    best_vehicle = None
                    best_area = -1
                    for veh in vehicle_detections:
                        vehicle_id = veh.get("global_id")
                        if vehicle_id is None:
                            continue
                        vx1, vy1, vx2, vy2 = veh["box"]
                        if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                            area = max(0, vx2 - vx1) * max(0, vy2 - vy1)
                            if area > best_area:
                                best_vehicle = vehicle_id
                                best_area = area
                    vehicle_global_id = best_vehicle

                plate_global_id = plate_id_map.get(yolo_plate_id)
                if plate_global_id is None and vehicle_global_id is not None:
                    plate_global_id = vehicle_to_plate.get(vehicle_global_id)
                if plate_global_id is None:
                    plate_global_id = next_plate_global_id
                    next_plate_global_id += 1

                plate_id_map[yolo_plate_id] = plate_global_id
                info = plate_history.setdefault(
                    plate_global_id,
                    {
                        "name": f"Plate_{plate_global_id}",
                        "confidence": "",
                        "ocr_done": False,
                        "ocr_certain": False,
                        "last_ocr_attempt": 0.0,
                    },
                )
                info.setdefault("confidence", "")
                info.setdefault("ocr_done", False)
                info.setdefault("ocr_certain", False)
                info.setdefault("last_ocr_attempt", 0.0)
                info["last_seen"] = now

                det["global_id"] = plate_global_id
                det["vehicle_global_id"] = vehicle_global_id
                active_plate_ids.add(plate_global_id)

                if vehicle_global_id is not None:
                    previous_vehicle = plate_to_vehicle.get(plate_global_id)
                    if previous_vehicle is not None and previous_vehicle != vehicle_global_id:
                        vehicle_to_plate.pop(previous_vehicle, None)
                    vehicle_to_plate[vehicle_global_id] = plate_global_id
                    plate_to_vehicle[plate_global_id] = vehicle_global_id

            for yolo_plate_id in list(plate_id_map.keys()):
                if yolo_plate_id not in current_plate_yolo_ids:
                    plate_id_map.pop(yolo_plate_id, None)

            for plate_id, vehicle_id in list(plate_to_vehicle.items()):
                if vehicle_id not in active_vehicle_ids:
                    plate_to_vehicle.pop(plate_id, None)
            for vehicle_id in list(vehicle_to_plate.keys()):
                if vehicle_id not in active_vehicle_ids:
                    vehicle_to_plate.pop(vehicle_id, None)

            num_plate_detections = len(plate_detections)
            if num_plate_detections > 0:
                if config.ocr_accumulate_best:
                    ocr_retry_wait = max(config.ocr_accumulation_seconds, 0.0)
                else:
                    ocr_retry_wait = max(
                        config.ocr_min_wait,
                        config.ocr_wait_multiplier * loop_dt * num_plate_detections if loop_dt else config.ocr_min_wait,
                    )
                ocr_retry_max = max(ocr_retry_wait, ocr_retry_max)
                ocr_retry_min = min(ocr_retry_wait, ocr_retry_min)

            for det in plate_detections:
                plate_id = det.get("global_id")
                if plate_id is None:
                    continue
                info = plate_history.get(plate_id, {})
                x1, y1, x2, y2 = det["box"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                crop = input_frame[y1:y2, x1:x2]

                if config.ocr_accumulate_best:
                    accumulator = ocr_accumulators.setdefault(
                        plate_id,
                        {
                            "start": now,
                            "best_area": 0,
                            "best_crop": None,
                            "last_seen": now,
                        },
                    )
                    if info.get("ocr_certain"):
                        ocr_accumulators.pop(plate_id, None)
                        continue

                    accumulator["last_seen"] = now
                    if crop.size > 0:
                        area = crop.shape[0] * crop.shape[1]
                        if area > accumulator.get("best_area", 0):
                            accumulator["best_area"] = area
                            accumulator["best_crop"] = crop.copy()
                    wait_elapsed = now - accumulator.get("start", now)
                    if (
                        not info.get("ocr_certain")
                        and wait_elapsed >= config.ocr_accumulation_seconds
                        and accumulator.get("best_crop") is not None
                    ):
                        _run_ocr_for_plate(plate_id, accumulator["best_crop"], now)
                        if info.get("ocr_certain"):
                            ocr_accumulators.pop(plate_id, None)
                        else:
                            accumulator["start"] = now
                            accumulator["best_area"] = 0
                            accumulator["best_crop"] = None
                    continue

                if crop.size == 0:
                    continue

                if (not info.get("ocr_certain")) and (
                    now - info.get("last_ocr_attempt", 0.0) > ocr_retry_wait
                ):
                    _run_ocr_for_plate(plate_id, crop, now)

            for det in plate_detections:
                plate_id = det.get("global_id")
                if plate_id is None:
                    continue
                info = plate_history.get(plate_id, {})
                label = info.get("name", "") + info.get("confidence", "")
                method_label = info.get("method", "")
                if method_label:
                    label += f" [{method_label}]"
                color = (
                    COLOR_GREEN
                    if info.get("ocr_certain")
                    else (COLOR_ORANGE if info.get("ocr_done") else COLOR_RED)
                )
                x1, y1, x2, y2 = det["box"]
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

            for det in vehicle_detections:
                vehicle_id = det.get("global_id")
                if vehicle_id is None:
                    continue
                x1, y1, x2, y2 = det["box"]
                associated_plate_id = vehicle_to_plate.get(vehicle_id)
                plate_info = plate_history.get(associated_plate_id) if associated_plate_id else None
                color = COLOR_RED
                if plate_info:
                    if plate_info.get("ocr_certain"):
                        color = COLOR_GREEN
                    elif plate_info.get("ocr_done"):
                        color = COLOR_ORANGE
                vehicle_label = vehicle_history.get(vehicle_id, {}).get(
                    "name", f"Vehicle_{vehicle_id}"
                )
                if plate_info:
                    vehicle_label += (
                        f" -> {plate_info.get('name', '')}{plate_info.get('confidence', '')}"
                    )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    vehicle_label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if config.ocr_accumulate_best:
                lost_plates = [pid for pid in list(ocr_accumulators.keys()) if pid not in active_plate_ids]
                for plate_id in lost_plates:
                    accumulator = ocr_accumulators.pop(plate_id)
                    info = plate_history.get(plate_id)
                    best_crop = accumulator.get("best_crop") if accumulator else None
                    if info and not info.get("ocr_certain") and best_crop is not None:
                        _run_ocr_for_plate(plate_id, best_crop, now)

            plates_total = len(plate_history)
            plates_certain = sum(1 for v in plate_history.values() if v.get("ocr_certain"))
            plates_done = sum(
                1 for v in plate_history.values() if v.get("ocr_done") and not v.get("ocr_certain")
            )
            plates_pending = plates_total - (plates_done + plates_certain)
            num_tracked = len(active_plate_ids)

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
                f"Tracked plates  | {num_tracked}",
                f"Vehicles active | {len(active_vehicle_ids)}",
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
