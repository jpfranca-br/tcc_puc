"""Callback endpoint used by the asynchronous OCR pipeline."""
from __future__ import annotations

import logging
import threading
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

try:  # FastAPI/uvicorn are optional at import time for tooling.
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    _FASTAPI_ERROR = exc
else:
    _FASTAPI_ERROR = None

try:
    import uvicorn
except ImportError as exc:  # pragma: no cover
    uvicorn = None  # type: ignore
    _UVICORN_ERROR = exc
else:
    _UVICORN_ERROR = None

LOGGER = logging.getLogger("ocr.callback")


@dataclass
class CallbackResult:
    """Stores the payload received from the OCR service."""

    request_id: str
    status: str
    payload: Dict[str, object] = field(default_factory=dict)


class CallbackStore:
    """Thread-safe container storing OCR callback results."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: Dict[str, CallbackResult] = {}
        self._queue: List[CallbackResult] = []

    def add(self, payload: Dict[str, object]) -> CallbackResult:
        request_id = str(payload.get("request_id", "")).strip()
        if not request_id:
            raise ValueError("Callback payload missing request_id")
        status = str(payload.get("status", "ocr_error"))
        result = CallbackResult(request_id=request_id, status=status, payload=payload)
        with self._lock:
            self._results[request_id] = result
            self._queue.append(result)
        LOGGER.debug("Callback stored for %s with status %s", request_id, status)
        return result

    def consume(self) -> List[CallbackResult]:
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items

    def get(self, request_id: str) -> Optional[CallbackResult]:
        with self._lock:
            return self._results.get(request_id)

    def values(self) -> Iterable[CallbackResult]:
        with self._lock:
            return list(self._results.values())


def create_app(store: CallbackStore | None = None) -> FastAPI:
    """Create the FastAPI application bound to ``store``."""

    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("FastAPI is required for the callback endpoint") from _FASTAPI_ERROR

    store = store or CallbackStore()
    app = FastAPI(title="OCR Callback Endpoint", version="1.0.0")

    @app.post("/callback/ocr-result")
    async def receive_callback(payload: Dict[str, object]):
        try:
            result = store.add(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "received", "request_id": result.request_id}

    @app.get("/callback/ocr-result/{request_id}")
    async def get_callback(request_id: str):
        result = store.get(request_id)
        if result is None:
            raise HTTPException(status_code=404, detail="request_id not found")
        return result.payload

    return app


class CallbackServer:
    """Utility to run the callback endpoint in a background thread."""

    def __init__(self, store: CallbackStore, host: str = "127.0.0.1", port: int = 9100, log_level: str = "warning") -> None:
        self.store = store
        self.host = host
        self.port = int(port)
        self.log_level = log_level
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[object] = None

    def start(self) -> None:
        if FastAPI is None:  # pragma: no cover
            raise RuntimeError("FastAPI must be installed to run the callback server") from _FASTAPI_ERROR
        if uvicorn is None:  # pragma: no cover
            raise RuntimeError("uvicorn must be installed to run the callback server") from _UVICORN_ERROR
        if self._thread and self._thread.is_alive():
            return

        app = create_app(self.store)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level=self.log_level)
        self._server = uvicorn.Server(config)

        def _run() -> None:
            assert self._server is not None
            LOGGER.info("Starting callback server at http://%s:%d", self.host, self.port)
            self._server.run()

        self._thread = threading.Thread(target=_run, name="OCRCallbackServer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is not None:
            setattr(server, "should_exit", True)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None


__all__ = ["CallbackStore", "CallbackServer", "create_app", "CallbackResult"]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    if FastAPI is None:
        raise RuntimeError("FastAPI must be installed to run the callback endpoint") from _FASTAPI_ERROR
    if uvicorn is None:
        raise RuntimeError("uvicorn must be installed to run the callback endpoint") from _UVICORN_ERROR

    host = os.getenv("CALLBACK_HOST", "0.0.0.0")
    port = int(os.getenv("CALLBACK_PORT", "9102"))
    uvicorn.run(create_app(CallbackStore()), host=host, port=port)
