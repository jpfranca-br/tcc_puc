"""Utilities for serving frames via HTTP Live Streaming (HLS)."""
from __future__ import annotations

import subprocess
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
from typing import Optional


class HLSStreamServer:
    """Spawn ``ffmpeg`` to produce HLS segments and serve them via HTTP."""

    def __init__(self, width: int, height: int, fps: float, directory: str,
                 host: str = "0.0.0.0", port: int = 8000) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.directory = Path(directory)
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch the ffmpeg pipeline and the HTTP server."""

        self.directory.mkdir(parents=True, exist_ok=True)
        playlist_path = self.directory / "stream.m3u8"
        if playlist_path.exists():
            # Remove stale segments from previous runs.
            for f in self.directory.glob("stream*.ts"):
                f.unlink(missing_ok=True)
            playlist_path.unlink(missing_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-f", "hls",
            "-hls_time", "1",
            "-hls_list_size", "3",
            "-hls_flags", "delete_segments",
            str(playlist_path),
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        handler = partial(SimpleHTTPRequestHandler, directory=str(self.directory))
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()

    def push_frame(self, frame) -> None:
        """Write a frame to ffmpeg via stdin."""

        if not self.process or not self.process.stdin:
            return
        try:
            self.process.stdin.write(frame.astype("uint8").tobytes())
        except BrokenPipeError:
            pass

    def stop(self) -> None:
        """Stop ffmpeg and the HTTP server."""

        if self.process and self.process.stdin:
            try:
                self.process.stdin.close()
            except Exception:
                pass
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.httpd:
            self.httpd.shutdown()
        if self.server_thread:
            self.server_thread.join(timeout=2)
