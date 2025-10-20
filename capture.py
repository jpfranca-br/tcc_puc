"""Video capture helpers used by the main processing loop."""
from __future__ import annotations

import queue
import threading
import cv2


class PrefetchCapture:
    """Background-decoding wrapper around :class:`cv2.VideoCapture`.

    The class mirrors the behaviour of ``cv2.VideoCapture`` but fetches frames
    on a dedicated thread so the model can run without blocking on disk I/O.
    """

    def __init__(self, path: str, queue_size: int = 8) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self._q: queue.Queue[tuple[bool, cv2.typing.MatLike | None]] = queue.Queue(
            maxsize=max(1, int(queue_size))
        )
        self._stopped = False
        self._eof = False
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self) -> None:
        """Continuously pull frames from OpenCV and buffer them."""

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

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """Retrieve the next frame from the queue."""

        if self._eof and self._q.empty():
            return False, None
        ret, frame = self._q.get()
        return ret, frame

    def release(self) -> None:
        """Stop the reader thread and close the capture."""

        self._stopped = True
        try:
            self._q.put_nowait((False, None))
        except queue.Full:
            pass
        if self._t.is_alive():
            self._t.join(timeout=1.0)
        self._cap.release()
