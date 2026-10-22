"""
FFmpeg-based video decoder.

Uses FFmpeg subprocess piping to perform hardware/software decoding.
Avoids OpenCV's VideoCapture software decoding which has high CPU usage.
Outputs raw BGR frames for OpenCV processing.

Supports three modes:
  - none   : Software decoding via FFmpeg (avcodec)
  - vaapi  : VA-API hardware decoding (Intel integrated GPU)
  - qsv    : Intel Quick Sync Video
"""

import subprocess
import sys
import time
import threading
from typing import Optional, Tuple
import numpy as np


class FFmpegDecoder:
    def __init__(
        self,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        hw_accel: str = "none",
        hw_device: str = "",
        pixel_format: str = "bgr24",
        ffmpeg_path: str = "ffmpeg",
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.hw_accel = hw_accel
        self.hw_device = hw_device
        self.pixel_format = pixel_format
        self.ffmpeg_path = ffmpeg_path
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._is_open = False
        self._frame_count = 0
        self._start_time = 0.0

    def open(self) -> bool:
        """Open the video source and start the FFmpeg decoder process."""
        if self._is_open:
            return True

        cmd = self._build_command()
        print(f"[Decoder] Starting FFmpeg: {' '.join(cmd[:8])}...")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._is_open = True
            self._start_time = time.time()
            return True
        except FileNotFoundError:
            print("[Decoder] ERROR: FFmpeg binary not found. Install ffmpeg first.")
            return False
        except Exception as e:
            print(f"[Decoder] ERROR: {e}")
            return False

    def _build_command(self) -> list:
        """Build the FFmpeg command line."""
        frame_size = self.width * self.height * 3

        cmd = [self.ffmpeg_path]

        # Hardware acceleration options
        if self.hw_accel == "vaapi" and self.hw_device:
            cmd.extend(["-hwaccel", "vaapi", "-hwaccel_device", self.hw_device,
                        "-hwaccel_output_format", "vaapi"])
        elif self.hw_accel == "qsv":
            cmd.extend(["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"])
        elif self.hw_accel == "cuda":
            cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])

        # Input source routing
        if self.source.isdigit() or self.source.startswith("/dev/video"):
            # USB camera (Video4Linux)
            cmd.extend([
                "-f", "v4l2",
                "-framerate", str(self.fps),
                "-video_size", f"{self.width}x{self.height}",
                "-i", self.source,
            ])
        elif self.source.startswith("rtsp://") or self.source.startswith("rtmp://"):
            # RTSP/RTMP network stream
            cmd.extend([
                "-rtsp_transport", "tcp",
                "-i", self.source,
            ])
        else:
            # Local video file
            cmd.extend(["-i", self.source])

        # Decode to raw BGR frames, output on stdout
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", self.pixel_format,
            "-vcodec", "rawvideo",
            "-an",
            "-sn",
            "-"
        ])

        return cmd

    def read(self) -> Optional[np.ndarray]:
        """Read one decoded frame as a numpy array (H, W, 3) in BGR format.

        Returns None on EOF or error.
        """
        if not self._is_open or self.process is None:
            return None

        with self._lock:
            try:
                frame_bytes = self.width * self.height * 3
                raw = self.process.stdout.read(frame_bytes)

                if len(raw) != frame_bytes:
                    return None

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                self._frame_count += 1
                return frame
            except Exception as e:
                print(f"[Decoder] Read error: {e}")
                return None

    @property
    def fps_actual(self) -> float:
        """Actual FPS based on frames read and elapsed time."""
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self):
        """Close the decoder and release resources."""
        if self.process:
            self.process.stdout.close()
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self._is_open = False
        print(f"[Decoder] Closed. Total frames: {self._frame_count}")


class OpenCVDecoder:
    """Fallback decoder using OpenCV VideoCapture for simple sources."""

    def __init__(self, source: str):
        self.source = source

    def open(self):
        import cv2
        if self.source.isdigit():
            src = int(self.source)
        else:
            src = self.source
        self.cap = cv2.VideoCapture(src)
        return self.cap.isOpened()

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        self.cap.release()

    @property
    def fps_actual(self):
        return self.cap.get(5) if hasattr(self, 'cap') else 0

    @property
    def width(self):
        return int(self.cap.get(3)) if hasattr(self, 'cap') else 0

    @property
    def height(self):
        return int(self.cap.get(4)) if hasattr(self, 'cap') else 0

    @property
    def frame_count(self):
        return int(self.cap.get(7)) if hasattr(self, 'cap') else 0
