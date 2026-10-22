#!/usr/bin/env python3
"""
Video Detection System - Python Version (CPU, ONNX Runtime)

Quick-start examples:

  # USB camera (device 0)
  python run.py --source usb_camera --path 0

  # Local video file
  python run.py --source video --path /path/to/video.mp4

  # RTSP network stream
  python run.py --source rtsp --path rtsp://admin:pass@192.168.1.100:554/stream

  # Benchmark mode (inference speed test only)
  python run.py --benchmark

  # Export ONNX then run detection
  python run.py --export-first
"""

import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python_version"))


def load_config(config_path: str) -> dict:
    """Load YAML config or return defaults."""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] Could not load config: {e}, using defaults.")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Video Detection System - Python (ONNX Runtime CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --source video --path test.mp4
  python run.py --source usb_camera --path 0
  python run.py --source rtsp --path rtsp://192.168.1.100:554/stream
  python run.py --benchmark
        """,
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--source", type=str, default="video",
                        choices=["usb_camera", "video", "rtsp"],
                        help="Video source type")
    parser.add_argument("--path", type=str, default="",
                        help="Video file path, camera index, or RTSP URL")
    parser.add_argument("--model", type=str, default="models/yolo11n.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--width", type=int, default=0,
                        help="Video capture/output width (0=auto-detect)")
    parser.add_argument("--height", type=int, default=0,
                        help="Video capture/output height (0=auto-detect)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable preview window")
    parser.add_argument("--output", type=str, default="video_output/output.mp4",
                        help="Output video path")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark only")
    parser.add_argument("--export-first", action="store_true",
                        help="Export ONNX model before running")
    parser.add_argument("--hw-accel", type=str, default="none",
                        choices=["none", "vaapi", "qsv", "cuda"],
                        help="Hardware acceleration for decoding")
    parser.add_argument("--ffmpeg", type=str, default="",
                        help="Path to ffmpeg binary (auto-detects bundled static build)")

    args = parser.parse_args()

    # Load and merge config
    config_path = os.path.join(PROJECT_ROOT, args.config)
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    cfg.setdefault("model", {})
    cfg.setdefault("source", {})
    cfg.setdefault("decoder", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("encoder", {})
    cfg.setdefault("display", {})

    # Auto-detect FFmpeg path
    ffmpeg_path = args.ffmpeg or "ffmpeg"
    if not ffmpeg_path or ffmpeg_path == "ffmpeg":
        # Check for bundled static FFmpeg
        bundled = os.path.join(PROJECT_ROOT, "third_party", "ffmpeg", "ffmpeg")
        if os.path.isfile(bundled):
            ffmpeg_path = bundled
    cfg["ffmpeg_path"] = ffmpeg_path

    # Auto-detect headless environment
    if not args.no_display:
        display = os.environ.get("DISPLAY", "")
        if not display:
            print("[WARN] No DISPLAY set. Auto-enabling --no-display mode.")
            args.no_display = True
        else:
            import subprocess
            try:
                subprocess.run(["xdpyinfo", "-display", display],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=3)
            except Exception:
                print(f"[WARN] DISPLAY={display} unreachable. Auto-enabling --no-display mode.")
                args.no_display = True

    # Override with CLI args
    cfg["source"]["type"] = args.source
    cfg["source"]["path"] = args.path or cfg["source"].get("path", "0")
    cfg["source"]["width"] = args.width
    cfg["source"]["height"] = args.height
    cfg["model"]["onnx_path"] = args.model
    cfg["model"]["confidence_threshold"] = args.confidence
    cfg["model"]["nms_threshold"] = args.nms
    cfg["display"]["show_window"] = not args.no_display
    cfg["encoder"]["output_path"] = args.output
    cfg["decoder"]["hw_accel"] = args.hw_accel

    # --- Export ONNX first if requested ---
    if args.export_first:
        from train_export import export_onnx, verify_onnx
        onnx_path = export_onnx("yolo11n.pt")
        verify_onnx(onnx_path)
        cfg["model"]["onnx_path"] = onnx_path

    # --- Benchmark mode ---
    if args.benchmark:
        print("\n=== Inference Benchmark ===\n")
        from inference_onnx import InferenceEngine

        engine = InferenceEngine(
            model_path=cfg["model"]["onnx_path"],
            provider=cfg.get("inference", {}).get("provider", "CPUExecutionProvider"),
        )
        if not engine.load():
            print("ERROR: Failed to load model.")
            sys.exit(1)

        engine.warmup(10)
        engine.benchmark(100)
        return

    # --- Normal detection pipeline ---
    from pipeline import DetectionPipeline

    pipeline = DetectionPipeline(cfg)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
