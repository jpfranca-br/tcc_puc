import os
import glob
import time
import cv2
import numpy as np
import threading, queue
import signal, atexit
from ultralytics import YOLO

# =========================
# I/O
# =========================
INPUT_MODE = "video"                   # "video" or "images"
VIDEO_IN = "./data/input/traffic2.mp4"  # input video
VIDEO_OUT = "./data/output/output.avi"  # only used in video mode
IMAGES_DIR = "./data/images_in"         # folder of images if INPUT_MODE == "images"
IMAGES_PER_SECOND = 30                  # can be fractional (e.g., 0.05 img/s)
# --- Speed / UI toggles
FAST_VIDEO = True                       # prefetch frames in a background thread for videos
FRAME_QUEUE_SIZE = 512                  # queue depth for prefetcher
SHOW_WINDOW = True                      # set False to disable user window (faster)

# =========================
# Models
# =========================
OPENCV_PRE_YOLO = False                 # OpenCV preprocessing before yolo
TARGET_YOLO_WIDTH = 640                 # resize for Yolo input. 960–1280 are good starters
YOLO_WARMUP = True                      # calls yolo once before starting the loop
license_plate_detector = YOLO("./models/license_plate_detector.pt")
license_plate_detector.to('cuda')       # if you have a CUDA GPU
TRACK_KW = dict(device=0, half=True)    # Ultralytics supports half on CUDA for most models

# =========================
# OCR Engine
# =========================
OCR_ENGINE = "easyocr"                  # "easyocr" or "tesseract"
GPU_PRESENT = True                      # only relevant for EasyOCR
EASYOCR_MULTIVARIANT = False            # try multiple preprocess variants with EasyOCR
# Detection/recognition thresholds
CONF_THRESHOLD_LOW = 0.20               # "bad" plate has a score below this
CHAR_COUNT_LOW = 3                      # "bad" plate has less than this number of chars
CONF_THRESHOLD_HIGH = 0.80              # "ok" plate has a score above this
CHAR_COUNT_HIGH = 6                     # "ok" plate has equal or more this number of chars
# OCR Wait parameters (for OCR throttling)
# wait period = multiplier * total loop time * number of objects being tracked.
# good idea to increase multiplier if Multivariant is enabled, since each plate will be sent to OCR multiple times
OCR_WAIT_MULTIPLIER = 3
OCR_MIN_WAIT = 0.2 #seconds

# Colors (BGR)
COLOR_RED    = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_GREEN  = (0, 255, 0)
COLOR_BLUE   = (255, 0, 0)
COLOR_WHITE  = (255, 255, 255)

DISPLAY_W, DISPLAY_H = 1280, 720

# =========================
# State
# =========================
track_history = {}
next_plate_id = 1
ocr_method_stats = {}
is_paused = False

frame_idx = 0
yolo_last_dt = 0.0
ocr_last_dt  = 0.0
ocr_calls = 0
ocr_success = 0
input_fps = None
preproc_last_dt = 0.0
preproc_min = float('inf')
preproc_max = float('-inf')
loop_dt = 0.0
loop_min = float('inf')
loop_max = float('-inf')
yolo_min = float('inf')
yolo_max = float('-inf')
ocr_min = float('inf')
ocr_max = float('-inf')
ocr_retry_min = float('inf')
ocr_retry_max = float('-inf')

# Throughput counters
processing_start_ts = None   # perf_counter at loop start
processed_frames = 0         # frames we actually processed (video/imgs)

# which variant won last (for HUD)
tess_last_mode = "-"
easy_last_mode = "-"

# ---- graceful-exit wiring (Ctrl+C etc.)
SHOULD_STOP = False
def _handle_stop_signal(sig, frame):
    global SHOULD_STOP
    SHOULD_STOP = True

signal.signal(signal.SIGINT, _handle_stop_signal)   # Ctrl+C
signal.signal(signal.SIGTERM, _handle_stop_signal)  # polite kill

########################
# --- OCR backends --- #
########################
reader = None

if OCR_ENGINE.lower() == "easyocr":
    import easyocr
    print("Loading EasyOCR Reader...")
    reader = easyocr.Reader(['en'], gpu=GPU_PRESENT)
    print("EasyOCR Reader loaded.")
elif OCR_ENGINE.lower() == "tesseract":
    import pytesseract
    # pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    print("Using Tesseract OCR backend.")
else:
    raise ValueError("OCR_ENGINE must be 'easyocr' or 'tesseract'")



def _put_text_multiline(img, text_lines, org=(10, 10), line_h=22, font_scale=0.55,
                        color=COLOR_WHITE, thickness=1):
    x, y = org
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (x, y + i * line_h),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)

def _draw_hud(frame, stats_lines, margin=5, padding=10, alpha=0.35):
    font_scale = 0.55
    thickness = 1
    line_h = 22
    box_w = 0
    for ln in stats_lines:
        (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        box_w = max(box_w, tw)
    box_h = padding * 2 + line_h * len(stats_lines)
    box_w = padding * 2 + box_w
    x1, y1 = margin, margin
    x2, y2 = x1 + box_w, y1 + box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_BLUE, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    _put_text_multiline(frame, stats_lines, org=(x1 + padding, y1 + padding),
                        line_h=line_h, font_scale=font_scale, color=COLOR_WHITE, thickness=1)

def _natural_key(path):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(path))]

# video prefetch
class PrefetchCapture:
    """Background-decoding wrapper around cv2.VideoCapture for file-based video."""
    def __init__(self, path, queue_size=8):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self._q = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stopped = False
        self._eof = False
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                if not self._eof:
                    self._eof = True
                    try:
                        self._q.put_nowait((False, None))
                    except queue.Full:
                        pass
                break
            try:
                self._q.put((True, frame), timeout=0.5)
            except queue.Full:
                continue

    def read(self):
        if self._eof and self._q.empty():
            return False, None
        ret, frame = self._q.get()
        return ret, frame

    def release(self):
        self._stopped = True
        try:
            self._q.put_nowait((False, None))
        except queue.Full:
            pass
        if self._t.is_alive():
            self._t.join(timeout=1.0)
        self._cap.release()

###########################################
### Preprocessing pipeline before yolo ###
###########################################

def _grayworld_white_balance(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    b = b * (mean_gray / (mean_b + 1e-6))
    g = g * (mean_gray / (mean_g + 1e-6))
    r = r * (mean_gray / (mean_r + 1e-6))
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def _clahe_on_l_channel(bgr, clip=2.0, tiles=(8,8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def _unsharp_mask(bgr, ksize=(0,0), sigma=1.0, amount=0.6):
    blurred = cv2.GaussianBlur(bgr, ksize, sigma)
    sharp = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)
    return sharp

def preprocess_for_yolo(bgr):
    x = bgr
    x = _grayworld_white_balance(x)
    x = _clahe_on_l_channel(x, clip=2.0, tiles=(8,8))
    x = _unsharp_mask(x, sigma=1.0, amount=0.6)
    x = cv2.bilateralFilter(x, d=5, sigmaColor=40, sigmaSpace=40)
    return x

####################################
### EasyOCR multi-variant helper ###
####################################

def easyocr_best(crop_bgr, reader):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 120:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray_dn = cv2.fastNlMeansDenoising(gray, None, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray_dn)

    def _score(img):
        try:
            res = reader.readtext(img)
        except Exception:
            res = []
        if not res:
            return ("", 0.0)
        best = max(res, key=lambda r: r[2])
        txt, conf = best[1], float(best[2])
        cleaned = "".join(filter(str.isalnum, txt)).upper()
        return (cleaned, conf)

    candidates = []
    for name, img in [("color", crop_bgr), ("gray", gray_dn), ("clahe", clahe)]:
        t, c = _score(img)
        candidates.append((name, t, c))

    best_name, best_txt, best_conf = max(candidates, key=lambda x: x[2])

    if best_conf < CONF_THRESHOLD_LOW:
        _, bw = cv2.threshold(gray_dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_adapt = cv2.adaptiveThreshold(gray_dn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 5)
        for name, img in [("otsu", bw), ("adapt", bw_adapt)]:
            t, c = _score(img)
            if c > best_conf:
                best_name, best_txt, best_conf = name, t, c

    return (best_txt if best_txt else None), float(best_conf), best_name

######################################
### Tesseract multi-variant helper ###
######################################

def tesseract_best(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 120:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=15)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 21, 5)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_inv = cv2.bitwise_not(bw)

    variants = {
        "color": crop_bgr,
        "gray": gray,
        "clahe": clahe,
        "adapt": adapt,
        "bw": bw,
        "bw_inv": bw_inv,
    }

    import pytesseract
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def _best_from_data(img):
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
        texts, confs = [], []
        for t, c in zip(data.get("text", []), data.get("conf", [])):
            try:
                c = float(c)
            except Exception:
                c = -1.0
            t_clean = "".join(filter(str.isalnum, t)).upper()
            if t_clean and c >= 0:
                texts.append(t_clean)
                confs.append(c / 100.0)
        if not texts:
            return None, 0.0
        return "".join(texts), max(confs) if confs else 0.0

    best_txt, best_conf, best_key = None, 0.0, None
    tie_pref = ["gray", "color", "clahe", "adapt", "bw", "bw_inv"]
    for k, img in variants.items():
        txt, conf = _best_from_data(img)
        if conf > best_conf or (abs(conf - best_conf) < 1e-6 and best_key and tie_pref.index(k) < tie_pref.index(best_key)):
            best_txt, best_conf, best_key = txt, conf, k

    return (best_txt if best_txt else None), float(best_conf), (best_key if best_key else "-")

##############################################################
### OCR caller (always returns (text, confidence, method)) ###
##############################################################

def run_ocr(crop_bgr):
    global tess_last_mode, easy_last_mode
    if OCR_ENGINE.lower() == "easyocr":
        if not EASYOCR_MULTIVARIANT:
            try:
                res = reader.readtext(crop_bgr)
            except Exception:
                res = []
            if not res:
                easy_last_mode = "-"
                return None, 0.0, "-"
            best = max(res, key=lambda r: r[2])
            txt, conf = best[1], float(best[2])
            cleaned = "".join(filter(str.isalnum, txt)).upper()
            easy_last_mode = "single"
            return (cleaned if cleaned else None), max(0.0, min(1.0, conf)), "single"
        else:
            txt, conf, mode = easyocr_best(crop_bgr, reader)
            mode = mode if mode else "-"
            easy_last_mode = mode
            return (txt if txt else None), (conf if conf else 0.0), mode
    elif OCR_ENGINE.lower() == "tesseract":
        txt, conf, mode = tesseract_best(crop_bgr)
        mode = mode if mode else "-"
        tess_last_mode = mode
        return (txt if txt else None), (conf if conf else 0.0), mode

#######################################################
### Input setup - Video / Fast Video / Image Folder ###
#######################################################

if INPUT_MODE == "video":
    if FAST_VIDEO:
        cap = PrefetchCapture(VIDEO_IN, queue_size=FRAME_QUEUE_SIZE)
        _probe = cv2.VideoCapture(VIDEO_IN)
        if not _probe.isOpened():
            raise RuntimeError(f"Could not open video: {VIDEO_IN}")
        frame_width  = int(_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _probe.get(cv2.CAP_PROP_FPS) or 25.0
        input_fps = float(fps)
        _probe.release()
    else:
        cap = cv2.VideoCapture(VIDEO_IN)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {VIDEO_IN}")
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        input_fps = float(fps)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    if VIDEO_OUT is not None:
        os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)
        out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (frame_width, frame_height))
    loop_avg = 1.0 / float(max(fps, 1e-6))
else:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(IMAGES_DIR, p)))
    if not files:
        raise RuntimeError(f"No images found in: {IMAGES_DIR}")
    files.sort(key=_natural_key)
    probe = cv2.imread(files[0])
    if probe is None:
        raise RuntimeError(f"Failed to read first image: {files[0]}")
    frame_height, frame_width = probe.shape[:2]
    fps = float(IMAGES_PER_SECOND) if IMAGES_PER_SECOND > 0 else 0.1
    input_fps = float(fps)
    out = None
    loop_avg = 1.0 / float(max(fps, 1e-6))

# pacing for image mode
if INPUT_MODE == "images":
    frame_interval = 1.0 / float(max(IMAGES_PER_SECOND, 1e-6))
    next_show_time = time.perf_counter()  # schedule

# -------------------------
# Summary printer (registered with atexit)
# -------------------------
def print_run_summary():
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
                    plate = info.get('plate', '')
                    conf = info.get('best_conf', 0.0)*100
                    print(f"    plate='{plate}' ({conf:0.1f}%) - {method:<8}")
        print("=====================================================\n")
    except Exception as e:
        print(f"[Summary error] {e}")

atexit.register(print_run_summary)

# Run Yolo Before the loop
if YOLO_WARMUP:
    dummy = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    _ = license_plate_detector.track(dummy, persist=True, verbose=False)
    # If using CUDA, make sure kernels finish before proceeding:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

###############################################
###                                         ###
### Main loop (wrapped for graceful Ctrl+C) ###
###                                         ###
###############################################

ret = True
image_index = 0
if processing_start_ts is None:
    processing_start_ts = time.perf_counter()

try:
    while ret:
        loop_t0 = time.perf_counter()
        frame_idx += 1

        if is_paused and SHOW_WINDOW:
            cv2.putText(
                probe if INPUT_MODE == "images" else np.zeros((100, 100, 3), dtype=np.uint8),
                "PAUSED", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3
            )
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('p'):
                    is_paused = False
                    break
                if key == ord('q'):
                    ret = False
                    break
            if not ret:
                break

        # ----- Fetch a frame -----
        if INPUT_MODE == "video":
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if image_index >= len(files):
                break
            frame = cv2.imread(files[image_index])
            if frame is None:
                image_index += 1
                continue
        processed_frames += 1
        now = time.perf_counter()

        # ----- OpenCV pre-processing (optional) BEFORE YOLO -----
        yolo_input = frame
        if OPENCV_PRE_YOLO:
            t0 = time.perf_counter()
            try:
                yolo_input = preprocess_for_yolo(frame)
            except Exception:
                yolo_input = frame  # fail-safe
            preproc_last_dt = time.perf_counter() - t0
            preproc_min = min(preproc_min,preproc_last_dt)
            preproc_max = max(preprox_max,preproc_last_dt)
        else:
            preproc_last_dt = 0.0

        # ----- YOLO track -----
        yolo_t0 = time.perf_counter()
        h, w = frame.shape[:2]
        scale = min(1.0, TARGET_YOLO_WIDTH / float(w))
        if scale < 1.0:
            yolo_input = cv2.resize(yolo_input, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
        results = license_plate_detector.track(yolo_input, persist=True, verbose=False, **TRACK_KW)
        yolo_last_dt = time.perf_counter() - yolo_t0
        yolo_min = min(yolo_min,yolo_last_dt)
        yolo_max = max(yolo_max,yolo_last_dt)

        # ----- OCR attempts -----
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            if scale != 1.0:
                boxes = boxes / scale  # map back to original frame coords
            boxes = boxes.astype(int)
            ocr_retry_wait = max(OCR_MIN_WAIT, OCR_WAIT_MULTIPLIER * loop_dt * len(results[0].boxes.id))
            ocr_retry_max = max(ocr_retry_wait,ocr_retry_max)
            ocr_retry_min = min(ocr_retry_wait,ocr_retry_min)

            for box, track_id in zip(boxes, track_ids):
                if track_id not in track_history:
                    track_history[track_id] = {
                        'name': f"Plate_{next_plate_id}",
                        'confidence': '',
                        'ocr_done': False,
                        'ocr_certain': False,
                        'last_ocr_attempt': 0.0
                    }
                    next_plate_id += 1

                info = track_history[track_id]

                if (not info['ocr_certain']) and (now - info['last_ocr_attempt'] > ocr_retry_wait):
                    info['last_ocr_attempt'] = now
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    ocr_t0 = time.perf_counter()

                    try:
                        txt, conf, method = run_ocr(crop)
                        print(f"OCR result for track {track_id}: {(txt, conf, method)}")
                        if track_id not in ocr_method_stats:
                            ocr_method_stats[track_id] = {}
                        track_methods = ocr_method_stats[track_id]
                        current_best = track_methods.get(method, {"best_conf": 0.0, "plate": ""})
                        if conf > current_best["best_conf"]:
                            track_methods[method] = {"best_conf": conf, "plate": txt}
                    except Exception as e:
                        print(f"OCR error for track {track_id}: {e}")
                        txt, conf, method = (None, 0.0, "-")

                    ocr_last_dt = time.perf_counter() - ocr_t0
                    ocr_min = min(ocr_min,ocr_last_dt)
                    ocr_max = max(ocr_max,ocr_last_dt)
                    ocr_calls += 1

                    if txt:
                        cleaned = txt  # already cleaned & uppercased inside run_ocr
                        info['method'] = method
                        if conf >= CONF_THRESHOLD_HIGH and len(cleaned) >= CHAR_COUNT_HIGH:
                            info.update({'name': cleaned, 'confidence': f" ({conf:.0%})",
                                         'ocr_done': True, 'ocr_certain': True})
                            ocr_success += 1
                        elif conf >= CONF_THRESHOLD_LOW and len(cleaned) >= CHAR_COUNT_LOW:
                            info.update({'name': cleaned, 'confidence': f" ({conf:.0%})",
                                         'ocr_done': True})
                            ocr_success += 1

            # draw boxes
            for box, track_id in zip(boxes, track_ids):
                info = track_history.get(track_id, {})
                label = info.get('name', '') + info.get('confidence', '')
                method_label = info.get('method', '')
                if method_label:
                    label += f" [{method_label}]"
                color = COLOR_GREEN if info.get('ocr_certain') else (COLOR_ORANGE if info.get('ocr_done') else COLOR_RED)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)

        # ----- HUD (always) -----
        plates_total   = len(track_history)
        plates_certain = sum(1 for v in track_history.values() if v['ocr_certain'])
        plates_done    = sum(1 for v in track_history.values() if v['ocr_done'] and not v['ocr_certain'])
        plates_pending = plates_total - (plates_done + plates_certain)
        if results and results[0].boxes.id is not None:
            num_tracked = len(results[0].boxes.id)
        else:
            num_tracked = 0
        if OCR_ENGINE.lower() == "tesseract":
            ocr_mode_str = f"Tess best={tess_last_mode}"
        else:
            if EASYOCR_MULTIVARIANT:
                ocr_mode_str = f"Easy best={easy_last_mode}"
            else:
                ocr_mode_str = "Easy single"
        try:
            total_secs = max(1e-6, time.perf_counter() - (processing_start_ts or time.perf_counter()))
            overall_fps = processed_frames / total_secs
        except Exception:
            total_secs, overall_fps = 0.0, 0.0

        if INPUT_MODE.lower() == "video":
            input_desc = f"video: {os.path.basename(VIDEO_IN)}"
        else:
            input_desc = f"images dir: {os.path.abspath(IMAGES_DIR)}"

        hud_lines = [
            f"Input source    | {input_desc}",
            f"Input source fps| {input_fps:.2f}" if input_fps is not None else "Input source fps| N/A",
            f"OCR Engine      | {OCR_ENGINE.upper()}  {ocr_mode_str}",
            f"GPU             | {GPU_PRESENT}",
            f"Preprocessing   | {OPENCV_PRE_YOLO}",
            f"Multivariant    | {EASYOCR_MULTIVARIANT}",
            f"Fast Video      | {FAST_VIDEO} (queue={FRAME_QUEUE_SIZE})",
            f"Show Video      | {SHOW_WINDOW}",
            f"Detectd         | {plates_total}",
            f"Tracked         | {num_tracked}",
            f"OCR calls       | {ocr_calls}",
            f"Plates          | bad={plates_pending} | average={plates_done} | ok={plates_certain}",
            f"Frames          | {processed_frames} processed in {total_secs:0.2f}s | fps={overall_fps:0.2f}",
            f"-----------------------------------------",
            f"Timings (ms)    |  min  |  cur  |  max  |",
            f"----------------------------------------",
            f"Pre Processing  | {preproc_min*1000:05.1f} | {preproc_last_dt*1000:05.1f} | {preproc_max*1000:05.1f}" if OPENCV_PRE_YOLO else "Pre Processing  |  N/A  |  N/A  |  N/A  |",
            f"YOLO Processing | {yolo_min*1000:05.1f} | {yolo_last_dt*1000:05.1f} | {yolo_max*1000:05.1f}",
            f"OCR Processing  | {ocr_min*1000:05.1f} | {ocr_last_dt*1000:05.1f} | {ocr_max*1000:05.1f}",
            f"OCR Wait (set)  | {ocr_retry_min*1000:05.1f} | {ocr_retry_wait*1000:05.1f} | {ocr_retry_max*1000:05.1f}",
            f"Total Loop      | {loop_min*1000:05.1f} | {loop_dt*1000:05.1f} | {loop_max*1000:05.1f}",
        ]
        _draw_hud(frame, hud_lines, margin=5, padding=10, alpha=0.35)

        # ----- Output / display -----
        if INPUT_MODE == "video" and VIDEO_OUT is not None:
            out.write(frame)

        if SHOW_WINDOW:
            display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            cv2.imshow("Annotated Output", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                is_paused = not is_paused
            if key == ord('q'):
                break
        # pacing for images mode
        if INPUT_MODE == "images":
            next_show_time = time.perf_counter() + (1.0 / float(max(IMAGES_PER_SECOND, 1e-6)))
            while True:
                now2 = time.perf_counter()
                if now2 >= next_show_time:
                    break
                if SHOW_WINDOW:
                    k2 = cv2.waitKey(10) & 0xFF
                    if k2 == ord('p'):
                        while True:
                            k3 = cv2.waitKey(0) & 0xFF
                            if k3 == ord('p'):
                                next_show_time = time.perf_counter() + (1.0 / float(max(IMAGES_PER_SECOND, 1e-6)))
                                break
                            if k3 == ord('q'):
                                ret = False
                                break
                        if not ret:
                            break
                    if k2 == ord('q'):
                        ret = False
                        break
            if not ret:
                break
            image_index += 1

        # Break cleanly if Ctrl+C was pressed
        if SHOULD_STOP:
            break

        # update loop time if you use it elsewhere
        loop_dt = time.perf_counter() - loop_t0
        loop_min = min(loop_min,loop_dt)
        loop_max = max(loop_max,loop_dt)

except KeyboardInterrupt:
    # Just in case; SHOULD_STOP + atexit will handle summary
    pass
finally:
    print("Processing finished. Releasing resources.")
    try:
        if INPUT_MODE == "video" and out is not None:
            out.release()
    except Exception:
        pass
    try:
        if INPUT_MODE == "video" and cap is not None:
            cap.release()  # works for both cv2.VideoCapture and PrefetchCapture
    except Exception:
        pass
    cv2.destroyAllWindows()
    # atexit will print the summary automatically
