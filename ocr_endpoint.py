"""Asynchronous OCR submission endpoint."""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests

try:  # FastAPI is optional at import time for tooling.
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - module level guard
    FastAPI = None  # type: ignore
    UploadFile = None  # type: ignore
    File = None  # type: ignore
    Form = None  # type: ignore
    HTTPException = Exception  # type: ignore
    JSONResponse = dict  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from ocr import OCRBatchResult, OCRCriteria, OCRManager, process_image_batch

LOGGER = logging.getLogger("ocr.endpoint")


@dataclass
class OCRJob:
    """Represents a queued OCR request."""

    request_id: str
    images: List[np.ndarray]
    engine: str
    multivariant: bool
    callback_url: str


class OCRService:
    """Thin wrapper orchestrating :class:`OCRManager` executions."""

    def __init__(
        self,
        criteria: OCRCriteria,
        use_gpu: bool = False,
        easyocr_confidence_cap: float = 1.0,
        max_workers: Optional[int] = None,
        callback_timeout: float = 5.0,
        callback_retries: int = 3,
    ) -> None:
        self.criteria = criteria
        self.use_gpu = use_gpu
        self.easyocr_confidence_cap = easyocr_confidence_cap
        self.callback_timeout = callback_timeout
        self.callback_retries = max(1, callback_retries)
        self._executor = ThreadPoolExecutor(max_workers=max_workers or os.cpu_count() or 4)
        self._lock = threading.Lock()
        self._managers: Dict[tuple[str, bool], OCRManager] = {}

    def _get_manager(self, engine: str, multivariant: bool) -> OCRManager:
        key = (engine.lower(), bool(multivariant))
        with self._lock:
            manager = self._managers.get(key)
            if manager is None:
                manager = OCRManager(
                    engine=key[0],
                    use_gpu=self.use_gpu,
                    multivariant=key[1],
                    conf_threshold_low=self.criteria.confidence_low,
                    easyocr_confidence_cap=self.easyocr_confidence_cap,
                )
                self._managers[key] = manager
            else:
                manager.multivariant = key[1]
                manager.conf_threshold_low = self.criteria.confidence_low
        return manager

    def submit(self, job: OCRJob) -> None:
        LOGGER.debug("Queueing OCR job %s with %d images", job.request_id, len(job.images))
        self._executor.submit(self._process_job, job)

    def _dispatch_callback(self, url: str, payload: dict) -> None:
        for attempt in range(1, self.callback_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.callback_timeout)
                response.raise_for_status()
                LOGGER.debug("Callback to %s succeeded on attempt %d", url, attempt)
                return
            except Exception as exc:  # pragma: no cover - network errors are runtime concerns
                LOGGER.warning(
                    "Callback dispatch attempt %d/%d failed for %s: %s",
                    attempt,
                    self.callback_retries,
                    url,
                    exc,
                )
                time.sleep(min(2.0, 0.5 * attempt))
        LOGGER.error("Failed to dispatch callback to %s after %d attempts", url, self.callback_retries)

    def _process_job(self, job: OCRJob) -> None:
        payload: dict
        try:
            manager = self._get_manager(job.engine, job.multivariant)
            batch_result: OCRBatchResult = process_image_batch(manager, job.images, self.criteria)
            best = batch_result.best_result
            payload = {
                "request_id": job.request_id,
                "status": batch_result.status,
                "engine": job.engine,
                "multivariant": job.multivariant,
                "best_result": {
                    "text": best.text if best else None,
                    "confidence": best.confidence if best else 0.0,
                    "method": best.method if best else "-",
                },
                "attempts": [
                    {
                        "area": attempt.area,
                        "method": attempt.method,
                        "text": attempt.text,
                        "confidence": attempt.confidence,
                        "status": attempt.status,
                    }
                    for attempt in batch_result.attempts
                ],
            }
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.exception("OCR job %s failed: %s", job.request_id, exc)
            payload = {
                "request_id": job.request_id,
                "status": "ocr_error",
                "engine": job.engine,
                "multivariant": job.multivariant,
                "error": str(exc),
                "attempts": [],
            }

        self._dispatch_callback(job.callback_url, payload)


def _decode_upload(file: UploadFile) -> Optional[np.ndarray]:
    """Convert an uploaded image into a NumPy array."""

    try:
        data = file.file.read()
    except Exception:
        data = None
    finally:
        file.file.close()
    if not data:
        return None
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def create_app(service: OCRService | None = None) -> FastAPI:
    """Instantiate the FastAPI application."""

    if FastAPI is None:  # pragma: no cover - executed when dependency missing
        raise RuntimeError(
            "FastAPI is required to create the OCR endpoint application"
        ) from _IMPORT_ERROR

    if service is None:
        criteria = OCRCriteria(
            confidence_low=float(os.getenv("OCR_CONF_LOW", "0.20")),
            confidence_high=float(os.getenv("OCR_CONF_HIGH", "0.80")),
            chars_low=int(os.getenv("OCR_CHARS_LOW", "3")),
            chars_high=int(os.getenv("OCR_CHARS_HIGH", "6")),
        )
        use_gpu = os.getenv("OCR_USE_GPU", "false").lower() in {"1", "true", "yes"}
        workers = int(os.getenv("OCR_ENDPOINT_WORKERS", "0")) or None
        callback_timeout = float(os.getenv("OCR_CALLBACK_TIMEOUT", "5.0"))
        callback_retries = int(os.getenv("OCR_CALLBACK_RETRIES", "3"))
        service = OCRService(
            criteria=criteria,
            use_gpu=use_gpu,
            max_workers=workers,
            callback_timeout=callback_timeout,
            callback_retries=callback_retries,
        )

    app = FastAPI(title="Async OCR Endpoint", version="1.0.0")

    @app.post("/ocr/submit")
    async def submit_ocr(
        request_id: str = Form(...),
        callback_url: str = Form(...),
        ocr_engine: str = Form("easyocr"),
        use_multivariant: bool = Form(False),
        images: List[UploadFile] = File(...),
    ):
        if not images:
            raise HTTPException(status_code=400, detail="At least one image must be provided")

        decoded: List[np.ndarray] = []
        for upload in images:
            image = _decode_upload(upload)
            if image is not None:
                decoded.append(image)

        if not decoded:
            raise HTTPException(status_code=400, detail="No valid images provided")

        job = OCRJob(
            request_id=request_id,
            images=decoded,
            engine=ocr_engine,
            multivariant=use_multivariant,
            callback_url=callback_url,
        )
        service.submit(job)
        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "request_id": request_id,
                "images": len(decoded),
                "engine": ocr_engine,
                "multivariant": use_multivariant,
            },
        )

    return app


__all__ = ["create_app", "OCRService", "OCRJob"]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    if FastAPI is None:
        raise RuntimeError("FastAPI must be installed to run the OCR endpoint") from _IMPORT_ERROR
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("uvicorn must be installed to run the OCR endpoint") from exc

    port = int(os.getenv("OCR_ENDPOINT_PORT", "9101"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port)
