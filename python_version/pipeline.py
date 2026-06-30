"""
Main detection pipeline orchestrator.

Wires together all modules:
  Decoder -> Preprocessor -> Inference -> PostProcessor -> Renderer -> Encoder

Supports three video source types:
  - USB camera (V4L2 device index or /dev/video*)
  - Local video file
  - RTSP network stream
"""

import time
import sys
import signal
import argparse
from typing import Optional

import cv2
import numpy as np

from video_decoder import FFmpegDecoder, OpenCVDecoder
from preprocessor import Preprocessor
from inference_onnx import InferenceEngine
from postprocessor import PostProcessor, Renderer
from video_encoder import FFmpegEncoder


class DetectionPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.running = False

        # --- Model config ---
        model_cfg = config.get("model", {})
        class_names_str = model_cfg.get("class_names", [])
        if isinstance(class_names_str, list):
            class_names_str = ";".join(class_names_str)
        self.class_names = class_names_str.split(";") if class_names_str else None

        # --- Decoder ---
        source_cfg = config.get("source", {})
        decoder_cfg = config.get("decoder", {})
        source_type = source_cfg.get("type", "video")
        source_path = source_cfg.get("path", "")

        self.use_ffmpeg_decoder = True
        if source_type == "usb_camera":
            src = source_path if source_path else "0"
        elif source_type == "rtsp":
            src = source_path
        else:
            src = source_path if source_path else "0"

        ffmpeg_bin = config.get("ffmpeg_path", "ffmpeg")
        self.use_ffmpeg_decoder = False

        # Prefer OpenCV decoder (reliable, works with all sources)
        # FFmpeg decoder reserved for hardware-accelerated scenarios
        self.decoder = OpenCVDecoder(src)

        # --- Preprocessor ---
        self.preprocessor = Preprocessor(
            input_width=model_cfg.get("input_width", 640),
            input_height=model_cfg.get("input_height", 640),
        )

        # --- Inference engine ---
        inf_cfg = config.get("inference", {})
        self.inference = InferenceEngine(
            model_path=model_cfg.get("onnx_path", "models/yolo11n.onnx"),
            provider=inf_cfg.get("provider", "CPUExecutionProvider"),
            intra_op_threads=int(inf_cfg.get("intra_op_threads", 0)),
            inter_op_threads=int(inf_cfg.get("inter_op_threads", 0)),
            graph_optimization_level=str(inf_cfg.get("graph_optimization_level", "99")),
        )

        # --- Postprocessor ---
        self.postprocessor = PostProcessor(
            confidence_threshold=float(model_cfg.get("confidence_threshold", 0.25)),
            nms_threshold=float(model_cfg.get("nms_threshold", 0.45)),
            max_detections=int(model_cfg.get("max_detections", 300)),
            class_names=self.class_names,
        )

        # --- Renderer ---
        display_cfg = config.get("display", {})
        self.renderer = Renderer(
            class_names=self.class_names,
            draw_labels=bool(display_cfg.get("draw_labels", True)),
            box_thickness=int(display_cfg.get("box_thickness", 2)),
            font_scale=float(display_cfg.get("font_scale", 0.5)),
        )

        # --- Encoder ---
        encoder_cfg = config.get("encoder", {})
        self.encoder_ffmpeg_path = ffmpeg_bin
        self.encoder_output_path = encoder_cfg.get("output_path", "video_output/output.mp4")
        self.encoder_fps = encoder_cfg.get("output_fps", 30)
        self.encoder_codec = encoder_cfg.get("codec", "libx264")
        self.encoder_preset = encoder_cfg.get("preset", "fast")
        self.encoder_crf = int(encoder_cfg.get("crf", 23))
        self.encoder_rtsp_url = encoder_cfg.get("rtsp_push_url", "")
        self.encoder_width = source_cfg.get("width", 1280)
        self.encoder_height = source_cfg.get("height", 720)
        self._encoder_initialized = False

        self.encoder = None

        # --- Stats ---
        self.frame_idx = 0
        self.total_inference_time = 0.0
        self.show_window = bool(display_cfg.get("show_window", True))
        self.window_name = display_cfg.get("window_name", "Video Detection")

    def setup(self) -> bool:
        """Initialize all components."""
        print("[Pipeline] Initializing...")

        if not self.decoder.open():
            print("[Pipeline] ERROR: Failed to open video source.")
            return False

        if not self.inference.load():
            print("[Pipeline] ERROR: Failed to load ONNX model.")
            self.decoder.close()
            return False

        self.inference.warmup(num_runs=5)

        self.running = True
        print("[Pipeline] Initialization complete. Starting loop...")
        return True

    def run(self):
        """Main processing loop."""
        if not self.setup():
            return

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        t_start = time.time()
        last_log = t_start

        while self.running:
            t_frame = time.perf_counter()

            # 1. Decode
            frame = self.decoder.read()
            if frame is None:
                print("[Pipeline] End of stream or read error.")
                break

            # Lazy-init encoder with auto-detected resolution
            if not self._encoder_initialized:
                self._encoder_initialized = True
                auto_w, auto_h = frame.shape[1], frame.shape[0]
                if self.encoder_width <= 0 or self.encoder_height <= 0:
                    self.encoder_width, self.encoder_height = auto_w, auto_h
                self.encoder = FFmpegEncoder(
                    output_path=self.encoder_output_path,
                    width=self.encoder_width,
                    height=self.encoder_height,
                    fps=self.encoder_fps,
                    codec=self.encoder_codec,
                    preset=self.encoder_preset,
                    crf=self.encoder_crf,
                    rtsp_push_url=self.encoder_rtsp_url,
                    ffmpeg_path=self.encoder_ffmpeg_path,
                )
                if self.encoder.open():
                    print(f"[Pipeline] Encoder opened: {self.encoder_width}x{self.encoder_height}")
                else:
                    print("[Pipeline] WARNING: Encoder failed, continuing without output video.")

            # 2. Preprocess
            tensor, ratio, pad = self.preprocessor(frame)

            # 3. Inference
            output, latency = self.inference.infer_timed(tensor)
            self.total_inference_time += latency

            # 4. Postprocess
            detections = self.postprocessor(
                output[self.inference.output_name],
                original_shape=frame.shape[:2],
                ratio=ratio,
                pad=pad,
            )

            # 5. Render
            rendered = self.renderer.draw(frame, detections)

            # 6. Encode
            self.encoder.write(rendered)

            # 7. Display
            if self.show_window:
                try:
                    cv2.imshow(self.window_name, rendered)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    self.show_window = False

            self.frame_idx += 1

            # --- Logging ---
            now = time.time()
            if now - last_log >= 2.0:
                elapsed = now - t_start
                fps = self.frame_idx / elapsed
                avg_lat = self.total_inference_time / max(self.frame_idx, 1)
                print(
                    f"[Pipeline] Frame {self.frame_idx:>6} | "
                    f"FPS: {fps:5.1f} | "
                    f"Inference: {avg_lat:6.1f}ms | "
                    f"Detections: {len(detections):>2}"
                )
                last_log = now

        self.shutdown()

    def _signal_handler(self, signum, frame):
        print(f"\n[Pipeline] Received signal {signum}, shutting down...")
        self.running = False

    def shutdown(self):
        """Clean up all resources."""
        self.running = False
        print("[Pipeline] Shutting down...")

        self.decoder.close()
        self.encoder.close()

        if self.show_window:
            cv2.destroyAllWindows()

        elapsed = time.time()
        print(f"[Pipeline] Shutdown complete.")
        print(f"  Total frames: {self.frame_idx}")
        if self.frame_idx > 0:
            print(f"  Avg inference: {self.total_inference_time / self.frame_idx:.2f} ms")
