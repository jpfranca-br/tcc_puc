"""OCR engine wrappers for license plate recognition."""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class OCRResult:
    """Returned by :meth:`OCRManager.run` with metadata."""

    text: Optional[str]
    confidence: float
    method: str


def _configure_logger() -> logging.Logger:
    """Create a dedicated logger for OCR debugging."""

    logger = logging.getLogger("ocr.chatgpt_visio")
    if logger.handlers:
        return logger

    log_path = Path(os.getenv("OCR_LOG_PATH", "logs/ocr_debug.log"))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    logger.debug("OCR logger initialised. Writing to %s", log_path)
    return logger


class OCRManager:
    """Centralises all OCR related logic and provides a simple interface."""

    def __init__(self, engine: str, use_gpu: bool, multivariant: bool,
                 conf_threshold_low: float, easyocr_confidence_cap: float = 1.0) -> None:
        self.engine = engine.lower()
        self.use_gpu = use_gpu
        self.multivariant = multivariant
        self.conf_threshold_low = conf_threshold_low
        self.easyocr_confidence_cap = easyocr_confidence_cap
        self.reader = None
        self.tess_last_mode = "-"
        self.easy_last_mode = "-"
        self.chatgpt_last_mode = "-"
        self._chatgpt_client = None
        self._logger: Optional[logging.Logger] = None
        self._chatgpt_model = os.getenv("CHATGPT_VISION_MODEL", "gpt-4o-mini")
        self._load_engine()

    def _load_engine(self) -> None:
        """Initialise the selected OCR backend."""

        if self.engine == "easyocr":
            import easyocr  # lazy import keeps dependency optional

            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        elif self.engine == "tesseract":
            import pytesseract  # noqa: F401  # imported for side effects only
        elif self.engine == "chatgpt_visio":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "ChatGPT Visio OCR requires the 'openai' package to be installed"
                ) from exc

            self._chatgpt_client = OpenAI()
            self._logger = _configure_logger()
        else:
            raise ValueError("OCR engine must be 'easyocr', 'tesseract' or 'chatgpt_visio'")

    # --- EasyOCR helpers -------------------------------------------------
    def _easyocr_best(self, crop_bgr: np.ndarray) -> Tuple[Optional[str], float, str]:
        """Try multiple preprocess variants to increase recognition accuracy."""

        assert self.reader is not None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if max(h, w) < 120:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray_dn = cv2.fastNlMeansDenoising(gray, None, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_dn)

        def _score(img: np.ndarray) -> Tuple[str, float]:
            try:
                res = self.reader.readtext(img)
            except Exception:
                res = []
            if not res:
                return "", 0.0
            best = max(res, key=lambda r: r[2])
            txt, conf = best[1], float(best[2])
            cleaned = "".join(filter(str.isalnum, txt)).upper()
            return cleaned, conf

        candidates = []
        for name, img in [("color", crop_bgr), ("gray", gray_dn), ("clahe", clahe)]:
            t, c = _score(img)
            candidates.append((name, t, c))

        best_name, best_txt, best_conf = max(candidates, key=lambda x: x[2])

        if best_conf < self.conf_threshold_low:
            _, bw = cv2.threshold(gray_dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw_adapt = cv2.adaptiveThreshold(
                gray_dn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
            )
            for name, img in [("otsu", bw), ("adapt", bw_adapt)]:
                t, c = _score(img)
                if c > best_conf:
                    best_name, best_txt, best_conf = name, t, c

        return (best_txt if best_txt else None), float(best_conf), best_name

    # --- Tesseract helpers -----------------------------------------------
    def _tesseract_best(self, crop_bgr: np.ndarray) -> Tuple[Optional[str], float, str]:
        """Run multiple thresholding approaches with pytesseract."""

        import pytesseract

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if max(h, w) < 120:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, None, h=15)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
        )
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

        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        def _best_from_data(img: np.ndarray) -> tuple[Optional[str], float]:
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
            if conf > best_conf or (
                abs(conf - best_conf) < 1e-6 and best_key and tie_pref.index(k) < tie_pref.index(best_key)
            ):
                best_txt, best_conf, best_key = txt, conf, k

        return (best_txt if best_txt else None), float(best_conf), best_key or "-"

    # --- ChatGPT Visio helpers -------------------------------------------
    def _chatgpt_visio_best(self, crop_bgr: np.ndarray) -> Tuple[Optional[str], float, str]:
        """Send the crop to the ChatGPT Visio API for transcription."""

        if self._chatgpt_client is None:
            return None, 0.0, "uninitialised"

        ok, encoded = cv2.imencode(".png", crop_bgr)
        if not ok:
            if self._logger is not None:
                self._logger.error("Failed to encode crop for ChatGPT Visio request")
            return None, 0.0, "encode_error"

        image_b64 = base64.b64encode(encoded).decode("utf-8")
        prompt = (
            "You are an OCR system specialised in reading vehicle license plates. "
            "Return only the plate characters you can confidently read. If nothing is readable, "
            "respond with UNKNOWN."
        )

        if self._logger is not None:
            self._logger.debug(
                "Dispatching ChatGPT Visio OCR request with model=%s", self._chatgpt_model
            )

        def _collect_fragments(value) -> list[str]:
            fragments: list[str] = []
            if value is None:
                return fragments
            if isinstance(value, str):
                if value:
                    fragments.append(value)
                return fragments
            if isinstance(value, dict):
                for key in ("text", "value"):
                    fragment = value.get(key)
                    if isinstance(fragment, str) and fragment:
                        fragments.append(fragment)
                for key in ("content", "contents"):
                    nested = value.get(key)
                    if isinstance(nested, (list, tuple, set)):
                        for item in nested:
                            fragments.extend(_collect_fragments(item))
                    elif isinstance(nested, str):
                        fragments.append(nested)
                return fragments
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    fragments.extend(_collect_fragments(item))
                return fragments

            for attr in ("text", "value"):
                fragment = getattr(value, attr, None)
                if isinstance(fragment, str) and fragment:
                    fragments.append(fragment)
            for attr in ("content", "contents"):
                nested = getattr(value, attr, None)
                if isinstance(nested, (list, tuple, set)):
                    for item in nested:
                        fragments.extend(_collect_fragments(item))
                elif isinstance(nested, str):
                    fragments.append(nested)
            return fragments

        try:
            if hasattr(self._chatgpt_client, "responses"):
                response = self._chatgpt_client.responses.create(
                    model=self._chatgpt_model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_base64": image_b64},
                            ],
                        }
                    ],
                )
            else:
                response = self._chatgpt_client.chat.completions.create(
                    model=self._chatgpt_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Read the license plate characters."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                                },
                            ],
                        },
                    ],
                )
        except Exception as exc:
            if self._logger is not None:
                self._logger.exception("ChatGPT Visio OCR request failed: %s", exc)
            return None, 0.0, "request_error"

        text_response = ""
        if hasattr(response, "output_text") and response.output_text:
            text_response = response.output_text
        else:
            output = getattr(response, "output", None) or getattr(response, "outputs", None)
            fragments = _collect_fragments(output)
            if fragments:
                text_response = " ".join(fragments)

        if not text_response:
            choices = getattr(response, "choices", None)
            fragments = _collect_fragments(choices)
            if fragments:
                text_response = " ".join(fragments)

        if self._logger is not None:
            self._logger.debug("ChatGPT Visio raw response text: %r", text_response)

        cleaned = "".join(filter(str.isalnum, (text_response or ""))).upper()

        if not cleaned or cleaned == "UNKNOWN":
            if self._logger is not None:
                self._logger.info(
                    "ChatGPT Visio OCR returned no usable text. raw=%r", text_response
                )
            return None, 0.0, "api"

        if self._logger is not None:
            self._logger.info("ChatGPT Visio OCR success: %s", cleaned)

        return cleaned, 1.0, "api"

    # ------------------------------------------------------------------
    def run(self, crop_bgr: np.ndarray) -> OCRResult:
        """Perform OCR on the provided crop."""

        if self.engine == "easyocr":
            if not self.multivariant:
                try:
                    res = self.reader.readtext(crop_bgr)
                except Exception:
                    res = []
                if not res:
                    self.easy_last_mode = "-"
                    return OCRResult(None, 0.0, "-")
                best = max(res, key=lambda r: r[2])
                txt, conf = best[1], float(best[2])
                cleaned = "".join(filter(str.isalnum, txt)).upper()
                self.easy_last_mode = "single"
                return OCRResult(cleaned or None, max(0.0, min(self.easyocr_confidence_cap, conf)), "single")
            txt, conf, mode = self._easyocr_best(crop_bgr)
            self.easy_last_mode = mode if mode else "-"
            return OCRResult(txt if txt else None, conf if conf else 0.0, self.easy_last_mode)

        if self.engine == "chatgpt_visio":
            txt, conf, mode = self._chatgpt_visio_best(crop_bgr)
            self.chatgpt_last_mode = mode if mode else "-"
            return OCRResult(txt if txt else None, conf if conf else 0.0, self.chatgpt_last_mode)

        txt, conf, mode = self._tesseract_best(crop_bgr)
        self.tess_last_mode = mode if mode else "-"
        return OCRResult(txt if txt else None, conf if conf else 0.0, self.tess_last_mode)

    def hud_status(self) -> str:
        """Return a short string describing the last OCR mode used."""

        if self.engine == "tesseract":
            return f"Tess best={self.tess_last_mode}"
        if self.engine == "chatgpt_visio":
            return f"GPT best={self.chatgpt_last_mode}"
        if self.multivariant:
            return f"Easy best={self.easy_last_mode}"
        return "Easy single"
