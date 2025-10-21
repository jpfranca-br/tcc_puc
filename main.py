"""Entry point for the license plate detection pipeline."""
from __future__ import annotations

import glob
import os
import signal
import time
import uuid
import atexit
from typing import Dict, List

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from config import AppConfig, parse_config
from capture import PrefetchCapture
from hud import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_hud,
    natural_key,
)
from callback_endpoint import CallbackServer, CallbackStore
from streaming import HLSStreamServer

SHOULD_STOP = False


def _handle_stop_signal(sig, frame):
    """Set a flag when receiving termination signals."""

    global SHOULD_STOP
    SHOULD_STOP = True


signal.signal(signal.SIGINT, _handle_stop_signal)
signal.signal(signal.SIGTERM, _handle_stop_signal)


PLATE_STATUS_ACCUMULATING = "accumulating"
PLATE_STATUS_WAITING = "ocr_waiting"
PLATE_STATUS_ERROR = "ocr_error"
PLATE_STATUS_OK = "ocr_ok"
PLATE_STATUS_NO_MATCH = "ocr_no_match"

STATUS_COLOR_MAP = {
    PLATE_STATUS_ACCUMULATING: COLOR_WHITE,
    PLATE_STATUS_WAITING: COLOR_YELLOW,
    PLATE_STATUS_ERROR: COLOR_RED,
    PLATE_STATUS_OK: COLOR_BLUE,
    PLATE_STATUS_NO_MATCH: COLOR_ORANGE,
}

MAX_OCR_IMAGES = 5


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


class AsyncOCRClient:
    """HTTP client that submits OCR jobs to the asynchronous endpoint."""

    def __init__(
        self,
        endpoint_url: str,
        default_engine: str,
        default_multivariant: bool,
        timeout: float = 10.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.default_engine = default_engine
        self.default_multivariant = default_multivariant
        self.timeout = timeout
        self._session = requests.Session()

    def submit(
        self,
        request_id: str,
        images: List[np.ndarray],
        callback_url: str,
        engine: str | None = None,
        use_multivariant: bool | None = None,
    ) -> bool:
        if not images:
            return False

        engine = (engine or self.default_engine).lower()
        multivariant = (
            self.default_multivariant if use_multivariant is None else bool(use_multivariant)
        )

        files = []
        for idx, image in enumerate(images):
            if image is None or image.size == 0:
                continue
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                continue
            files.append(
                (
                    "images",
                    (f"{request_id}_{idx}.png", encoded.tobytes(), "image/png"),
                )
            )

        if not files:
            return False

        data = {
            "request_id": request_id,
            "callback_url": callback_url,
            "ocr_engine": engine,
            "use_multivariant": "true" if multivariant else "false",
        }

        try:
            response = self._session.post(
                self.endpoint_url,
                data=data,
                files=files,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            print(f"Failed to submit OCR job {request_id}: {exc}")
            return False

    def close(self) -> None:
        self._session.close()


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
    if not plate_class_ids:
        plate_class_ids = {0}
    auto_vehicle_fallback = False
    if not vehicle_class_ids:
        fallback_ids = {idx for idx in class_name_map.keys() if idx not in plate_class_ids}
        if fallback_ids:
            vehicle_class_ids = fallback_ids
        else:
            auto_vehicle_fallback = True
            print(
                "No vehicle class names detected in the model metadata; "
                "treating non-plate detections as vehicles."
            )

    ocr_client = AsyncOCRClient(
        endpoint_url=config.ocr_endpoint_url,
        default_engine=config.ocr_engine,
        default_multivariant=config.easyocr_multivariant,
        timeout=config.ocr_submit_timeout,
    )

    callback_store = CallbackStore()
    callback_server: CallbackServer | None = None
    callback_url = config.callback_url or f"http://{config.callback_host}:{config.callback_port}/callback/ocr-result"
    try:
        callback_server = CallbackServer(
            callback_store,
            host=config.callback_host,
            port=config.callback_port,
            log_level=config.callback_log_level,
        )
        callback_server.start()
    except Exception as exc:
        print(f"Failed to start callback endpoint: {exc}")
        callback_server = None

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
    plate_accumulators: Dict[int, Dict[str, object]] = {}
    plate_jobs: Dict[str, int] = {}
    job_sent_ts: Dict[str, float] = {}

    ocr_jobs_submitted = 0
    ocr_jobs_completed = 0
    ocr_jobs_success = 0
    ocr_jobs_errors = 0
    processed_frames = 0
    processing_start_ts = None
    next_plate_track_id = 0
    next_vehicle_track_id = 0
    next_plate_global_id = 1
    next_vehicle_global_id = 1

    yolo_last_dt = 0.0

    loop_dt = 0.0
    loop_min = float("inf")
    loop_max = float("-inf")
    yolo_min = float("inf")
    yolo_max = float("-inf")
    ocr_response_last = 0.0
    ocr_response_min = float("inf")
    ocr_response_max = float("-inf")

    hud_lines: List[str] = []

    def print_run_summary() -> None:
        """Report aggregated metrics once the process exits."""

        try:
            print("\n==================== RUN SUMMARY ====================")
            print("\n".join(hud_lines))
            print("\nOCR statuses per plate:")
            print("=====================================================\n")
            if not plate_history:
                print("  (no plates detected)")
            else:
                for pid, info in plate_history.items():
                    status = info.get("status", PLATE_STATUS_ACCUMULATING)
                    name = info.get("name", f"Plate_{pid}")
                    confidence = info.get("confidence", "")
                    print(f"    plate_id={pid:<4} status={status:<14} name={name}{confidence}")
            print("=====================================================\n")
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"[Summary error] {exc}")

    atexit.register(print_run_summary)

    processing_start_ts = time.perf_counter()

    def _submit_plate_to_ocr(
        plate_id: int,
        accumulator: Dict[str, object],
        info: Dict[str, object],
        reason: str,
        now_ts: float,
    ) -> bool:
        """Send the accumulated crops to the asynchronous OCR endpoint."""

        nonlocal ocr_jobs_submitted, ocr_jobs_errors

        samples = accumulator.get("samples", []) if accumulator else []
        if not samples:
            return False

        samples = sorted(samples, key=lambda item: item["area"], reverse=True)
        top_samples = samples[:MAX_OCR_IMAGES]
        images = [sample["image"] for sample in top_samples if sample.get("image") is not None]
        if not images:
            return False

        request_id = f"{plate_id}-{uuid.uuid4().hex}"
        success = ocr_client.submit(
            request_id=request_id,
            images=images,
            callback_url=callback_url,
            engine=config.ocr_engine,
            use_multivariant=config.easyocr_multivariant,
        )

        if not success:
            info["status"] = PLATE_STATUS_ERROR
            info["job_id"] = None
            ocr_jobs_errors += 1
            return False

        accumulator["samples"] = []
        accumulator["start"] = now_ts
        accumulator["sent"] = True
        accumulator["job_id"] = request_id
        plate_jobs[request_id] = plate_id
        job_sent_ts[request_id] = time.perf_counter()

        info["status"] = PLATE_STATUS_WAITING
        info["job_id"] = request_id
        info["last_ocr_attempt"] = now_ts
        info["last_submission_reason"] = reason

        ocr_jobs_submitted += 1
        return True

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

            for callback in callback_store.consume():
                plate_id = plate_jobs.pop(callback.request_id, None)
                sent_ts = job_sent_ts.pop(callback.request_id, None)
                if plate_id is None:
                    continue
                info = plate_history.get(plate_id)
                if info is None:
                    continue

                ocr_jobs_completed += 1
                if sent_ts is not None:
                    elapsed = max(0.0, time.perf_counter() - sent_ts)
                    ocr_response_last = elapsed
                    ocr_response_min = min(ocr_response_min, elapsed)
                    ocr_response_max = max(ocr_response_max, elapsed)

                payload = callback.payload
                status = str(payload.get("status", PLATE_STATUS_ERROR))
                info["job_id"] = None
                info["attempts"] = payload.get("attempts", [])
                best = payload.get("best_result") or {}
                text = best.get("text") if best else None
                confidence = float(best.get("confidence", 0.0)) if best else 0.0
                method = best.get("method", "-")

                accumulator = plate_accumulators.get(plate_id)
                if accumulator is not None:
                    accumulator["sent"] = False
                    accumulator["job_id"] = None
                    accumulator["start"] = now

                if status == PLATE_STATUS_OK:
                    if text:
                        info["name"] = text
                        info["confidence"] = f" ({confidence:.0%})"
                    info["method"] = method
                    info["status"] = PLATE_STATUS_OK
                    info["ocr_done"] = True
                    info["ocr_certain"] = True
                    ocr_jobs_success += 1
                    plate_accumulators.pop(plate_id, None)
                elif status == PLATE_STATUS_NO_MATCH:
                    if text:
                        info["name"] = text
                        info["confidence"] = f" ({confidence:.0%})"
                    info["method"] = method
                    info["status"] = PLATE_STATUS_NO_MATCH
                    info["ocr_done"] = False
                    info["ocr_certain"] = False
                    info["pending_retry"] = True
                elif status == PLATE_STATUS_ERROR:
                    info["status"] = PLATE_STATUS_ERROR
                    info["ocr_done"] = False
                    info["ocr_certain"] = False
                    info["pending_retry"] = True
                    ocr_jobs_errors += 1
                else:
                    info["status"] = status
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
                        "status": PLATE_STATUS_ACCUMULATING,
                        "ocr_done": False,
                        "ocr_certain": False,
                        "job_id": None,
                        "attempts": [],
                        "pending_retry": False,
                        "last_ocr_attempt": 0.0,
                    },
                )
                info.setdefault("confidence", "")
                info.setdefault("status", PLATE_STATUS_ACCUMULATING)
                info.setdefault("job_id", None)
                info.setdefault("attempts", [])
                info.setdefault("pending_retry", False)
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

            for det in plate_detections:
                plate_id = det.get("global_id")
                if plate_id is None:
                    continue
                info = plate_history.get(plate_id)
                if info is None:
                    continue

                x1, y1, x2, y2 = det["box"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                crop = input_frame[y1:y2, x1:x2]

                accumulator = plate_accumulators.setdefault(
                    plate_id,
                    {
                        "samples": [],
                        "start": now,
                        "last_seen": now,
                        "sent": False,
                        "job_id": None,
                    },
                )
                accumulator["last_seen"] = now
                accumulator.setdefault("samples", [])

                if info.get("status") == PLATE_STATUS_OK:
                    plate_accumulators.pop(plate_id, None)
                    continue

                if info.pop("pending_retry", False):
                    accumulator["samples"] = []
                    accumulator["start"] = now
                    accumulator["sent"] = False
                    accumulator["job_id"] = None
                    info["job_id"] = None
                    info["status"] = PLATE_STATUS_ACCUMULATING

                if info.get("status") in (PLATE_STATUS_ERROR, PLATE_STATUS_NO_MATCH) and not accumulator.get("sent"):
                    info["status"] = PLATE_STATUS_ACCUMULATING

                if info.get("status") == PLATE_STATUS_WAITING and accumulator.get("sent"):
                    continue

                if crop.size == 0:
                    continue

                area = crop.shape[0] * crop.shape[1]
                accumulator["samples"].append(
                    {"image": crop.copy(), "area": area, "timestamp": now}
                )
                if len(accumulator["samples"]) > 20:
                    accumulator["samples"] = sorted(
                        accumulator["samples"], key=lambda item: item["area"], reverse=True
                    )[:20]

                if not accumulator.get("sent"):
                    elapsed = now - accumulator.get("start", now)
                    if elapsed >= config.ocr_accumulation_seconds:
                        _submit_plate_to_ocr(plate_id, accumulator, info, "accumulation", now)

                det["crop_area"] = area

            for det in plate_detections:
                plate_id = det.get("global_id")
                if plate_id is None:
                    continue
                info = plate_history.get(plate_id, {})
                name = info.get("name", "")
                confidence_text = info.get("confidence", "")
                status = info.get("status", PLATE_STATUS_ACCUMULATING)
                status_text = status.replace("_", " ")
                label = f"{name}{confidence_text}".strip()
                if label:
                    label += f" [{status_text}]"
                else:
                    label = f"[{status_text}]"
                method_label = info.get("method", "")
                if method_label:
                    label += f" ({method_label})"
                color = STATUS_COLOR_MAP.get(status, COLOR_WHITE)
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
                status_fragment = ""
                if plate_info:
                    status = plate_info.get("status", PLATE_STATUS_ACCUMULATING)
                    color = STATUS_COLOR_MAP.get(status, COLOR_RED)
                    status_fragment = f" [{status.replace('_', ' ')}]"
                vehicle_label = vehicle_history.get(vehicle_id, {}).get(
                    "name", f"Vehicle_{vehicle_id}"
                )
                if plate_info:
                    vehicle_label += (
                        f" -> {plate_info.get('name', '')}{plate_info.get('confidence', '')}{status_fragment}"
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

            for plate_id, accumulator in list(plate_accumulators.items()):
                info = plate_history.get(plate_id)
                if info is None:
                    plate_accumulators.pop(plate_id, None)
                    continue
                if info.get("status") == PLATE_STATUS_OK:
                    plate_accumulators.pop(plate_id, None)
                    continue
                if info.get("status") == PLATE_STATUS_WAITING and accumulator.get("sent"):
                    continue
                if plate_id not in active_plate_ids:
                    if accumulator.get("samples"):
                        submitted = _submit_plate_to_ocr(plate_id, accumulator, info, "lost", now)
                        if not submitted:
                            plate_accumulators.pop(plate_id, None)
                    else:
                        plate_accumulators.pop(plate_id, None)

            plates_total = len(plate_history)
            status_counts: Dict[str, int] = {}
            for info in plate_history.values():
                status = info.get("status", PLATE_STATUS_ACCUMULATING)
                status_counts[status] = status_counts.get(status, 0) + 1
            plates_ok = status_counts.get(PLATE_STATUS_OK, 0)
            plates_waiting = status_counts.get(PLATE_STATUS_WAITING, 0)
            plates_error = status_counts.get(PLATE_STATUS_ERROR, 0)
            plates_no_match = status_counts.get(PLATE_STATUS_NO_MATCH, 0)
            plates_accum = status_counts.get(PLATE_STATUS_ACCUMULATING, 0)
            num_tracked = len(active_plate_ids)

            total_secs = max(1e-6, time.perf_counter() - (processing_start_ts or time.perf_counter()))
            overall_fps = processed_frames / total_secs

            input_desc = (
                f"video: {os.path.basename(config.video_in)}"
                if config.input_mode == "video"
                else f"images dir: {os.path.abspath(config.images_dir)}"
            )

            response_min = ocr_response_min if ocr_response_min != float("inf") else 0.0
            response_max = ocr_response_max if ocr_response_max != float("-inf") else 0.0
            response_last = ocr_response_last

            hud_lines = [
                f"Input source    | {input_desc}",
                f"Input source fps| {fps:.2f}" if fps else "Input source fps| N/A",
                f"OCR Engine      | {config.ocr_engine.upper()}",
                f"GPU             | {infer_kwargs.get('device')}",
                f"Multivariant    | {config.easyocr_multivariant}",
                f"Fast Video      | {config.fast_video} (queue={config.frame_queue_size})",
                f"Show Video      | {config.show_window}",
                f"Detectd         | {plates_total}",
                f"Tracked plates  | {num_tracked}",
                f"Vehicles active | {len(active_vehicle_ids)}",
                (
                    f"OCR jobs       | submitted={ocr_jobs_submitted} done={ocr_jobs_completed} ok={ocr_jobs_success} err={ocr_jobs_errors}"
                ),
                (
                    f"OCR response ms| min={response_min*1000:05.1f} cur={response_last*1000:05.1f} max={response_max*1000:05.1f}"
                    if ocr_jobs_completed
                    else "OCR response ms| N/A"
                ),
                (
                    "Plate status   | "
                    f"ok={plates_ok} waiting={plates_waiting} accum={plates_accum} no_match={plates_no_match} err={plates_error}"
                ),
                f"Frames          | {processed_frames} processed in {total_secs:0.2f}s | fps={overall_fps:0.2f}",
                f"-----------------------------------------",
                f"Timings (ms)    |  min  |  cur  |  max  |",
                f"----------------------------------------",
                f"YOLO Processing | {yolo_min*1000:05.1f} | {yolo_last_dt*1000:05.1f} | {yolo_max*1000:05.1f}",
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
        ocr_client.close()
        if callback_server:
            callback_server.stop()
        cv2.destroyAllWindows()
        if hls_server:
            hls_server.stop()


if __name__ == "__main__":
    main()
