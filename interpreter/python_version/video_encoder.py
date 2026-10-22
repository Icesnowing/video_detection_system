"""
FFmpeg-based video encoder.

Encodes processed BGR frames to H.264/H.265 video and optionally pushes
to an RTSP server via FFmpeg.

Encoding pipeline:
  BGR frame (numpy) -> FFmpeg stdin (rawvideo pipe) -> encoder -> file / RTSP
"""

import subprocess
import sys
import time
import threading
from typing import Optional
import numpy as np


class FFmpegEncoder:
    def __init__(
        self,
        output_path: str = "output.mp4",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        codec: str = "libx264",
        preset: str = "fast",
        crf: int = 23,
        rtsp_push_url: str = "",
        pixel_format: str = "bgr24",
        ffmpeg_path: str = "ffmpeg",
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.rtsp_push_url = rtsp_push_url
        self.pixel_format = pixel_format
        self.ffmpeg_path = ffmpeg_path

        self.process: Optional[subprocess.Popen] = None
        self._is_open = False
        self._frame_count = 0
        self._lock = threading.Lock()
        self._debug_printed = False

    def open(self) -> bool:
        """Start the FFmpeg encoder process."""
        cmd = self._build_command()
        print(f"[Encoder] Starting FFmpeg: {' '.join(cmd[:8])}...")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._is_open = True
            return True
        except FileNotFoundError:
            print("[Encoder] ERROR: FFmpeg binary not found.")
            return False
        except Exception as e:
            print(f"[Encoder] ERROR: {e}")
            return False

    def _build_command(self) -> list:
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", self.pixel_format,
            "-r", str(self.fps),
            "-i", "-",  # stdin
        ]

        # Video codec options
        cmd.extend(["-c:v", self.codec])

        if self.codec == "libx264":
            cmd.extend(["-preset", self.preset, "-crf", str(self.crf)])
            cmd.extend(["-pix_fmt", "yuv420p"])  # Compatible pixel format
        elif self.codec == "libx265":
            cmd.extend(["-preset", self.preset, "-crf", str(self.crf)])
            cmd.extend(["-pix_fmt", "yuv420p"])

        # If RTSP push is configured, use RTSP output format
        if self.rtsp_push_url:
            cmd.extend(["-f", "rtsp", self.rtsp_push_url])
        else:
            cmd.append(self.output_path)

        return cmd

    def write(self, frame: np.ndarray) -> bool:
        """Write a single BGR frame to the encoder."""
        if not self._is_open or self.process is None:
            return False

        with self._lock:
            try:
                if not self._debug_printed:
                    self._debug_printed = True
                    print(f"[Encoder] Frame shape={frame.shape} dtype={frame.dtype} "
                          f"contiguous={frame.flags['C_CONTIGUOUS']} "
                          f"bytes={frame.nbytes} expected={self.width*self.height*3}")

              #  if frame.ndim != 3 or frame.shape[2] != 3:
               #     if frame.ndim == 3 and frame.shape[2] == 4:
                #        frame = frame[:, :, :3]
                #    else:
                #        import cv2
               #         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
               # if frame.dtype != np.uint8:
                #    frame = frame.astype(np.uint8)
                data = np.ascontiguousarray(frame)
                self.process.stdin.write(data.tobytes())
                self._frame_count += 1
                return True
            except BrokenPipeError:
                print("[Encoder] Broken pipe - encoder process terminated.")
                self._is_open = False
                return False
            except Exception as e:
                print(f"[Encoder] Write error: {e}")
                self._is_open = False
                return False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self):
        """Flush and close the encoder."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
        self._is_open = False
        print(f"[Encoder] Closed. Total frames written: {self._frame_count}")
