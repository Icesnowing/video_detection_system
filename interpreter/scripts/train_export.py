#!/usr/bin/env python3
"""
YOLOv11 small-sample fine-tuning + dynamic-shape ONNX export.

Usage:
  # Step 1: Download COCO-pretrained YOLOv11n weights (fast test without training)
  python scripts/train_export.py --download

  # Step 2: Fine-tune on custom dataset (COCO128 example or your own YAML)
  python scripts/train_export.py --train --data coco128.yaml --epochs 50

  # Step 3: Export trained model to dynamic-shape ONNX
  python scripts/train_export.py --export --weights runs/train/exp/weights/best.pt

  # Step 4: All-in-one: download -> train -> export
  python scripts/train_export.py --all --data coco128.yaml --epochs 30
"""

import argparse
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def download_pretrained():
    """Download YOLOv11n pretrained weights from Ultralytics."""
    print("[INFO] Downloading YOLOv11n pretrained weights...")
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    print(f"[INFO] Model loaded. Classes: {model.names}")
    dst = os.path.join(MODEL_DIR, "yolo11n.pt")
    if not os.path.exists(dst):
        import shutil
        src = os.path.join(os.getcwd(), "yolo11n.pt")
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"[INFO] Weights saved to {dst}")
    return dst


def train_model(data_yaml, epochs, imgsz, batch, device):
    """Fine-tune YOLOv11 on a custom dataset."""
    print(f"[INFO] Training YOLOv11n on {data_yaml} for {epochs} epochs...")
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        verbose=True,
    )
    best_pt = os.path.join(os.getcwd(), "runs", "detect", "train", "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"[INFO] Best weights: {best_pt}")
    else:
        best_pt = results.save_dir / "weights" / "best.pt"
        best_pt = str(best_pt)
        print(f"[INFO] Best weights: {best_pt}")
    return best_pt


def export_onnx(weights_path, imgsz=640, opset=12, simplify=True, dynamic=True):
    """
    Export YOLOv11 to dynamic-shape ONNX model.

    Key fixes applied:
      - `dynamic=True` exports with dynamic batch/height/width
      - `simplify=True` uses onnx-simplifier to prune redundant nodes
      - opset=12 for broad compatibility with ONNX Runtime
    """
    print(f"[INFO] Exporting {weights_path} to ONNX...")
    from ultralytics import YOLO
    model = YOLO(weights_path)

    # Export with dynamic axes: batch, channels, height, width
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=False,  # FP32 for CPU inference; set True if GPU available
    )
    print(f"[INFO] ONNX model exported to: {onnx_path}")

    # Copy to models/ directory
    dst = os.path.join(MODEL_DIR, os.path.basename(onnx_path))
    if os.path.abspath(onnx_path) != os.path.abspath(dst):
        import shutil
        shutil.copy(onnx_path, dst)
        print(f"[INFO] Copied to: {dst}")

    return dst


def verify_onnx(onnx_path):
    """Verify the exported ONNX model loads correctly with ONNX Runtime."""
    print(f"[INFO] Verifying ONNX model: {onnx_path}")
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    input_info = sess.get_inputs()[0]
    output_info = sess.get_outputs()[0]

    print(f"  Input  name: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    print(f"  Output name: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")

    # Test with a dummy frame (1, 3, 640, 640)
    dtype = np.float32
    test_shape = (1, 3, 640, 640)
    dummy = np.random.randn(*test_shape).astype(dtype)
    outputs = sess.run([output_info.name], {input_info.name: dummy})
    print(f"  Test inference output shape: {outputs[0].shape}")
    print(f"[INFO] ONNX model verified successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Training & ONNX Export")
    parser.add_argument("--download", action="store_true", help="Download YOLOv11n pretrained weights")
    parser.add_argument("--train", action="store_true", help="Fine-tune on custom dataset")
    parser.add_argument("--export", action="store_true", help="Export to ONNX")
    parser.add_argument("--verify", action="store_true", help="Verify exported ONNX model")
    parser.add_argument("--all", action="store_true", help="Run all steps: download -> train -> export -> verify")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")

    args = parser.parse_args()

    if args.all:
        args.download = True
        args.train = True
        args.export = True
        args.verify = True

    if args.download:
        download_pretrained()

    if args.train:
        best_pt = train_model(args.data, args.epochs, args.imgsz, args.batch, args.device)
        args.weights = best_pt

    if args.export:
        onnx_path = export_onnx(args.weights, args.imgsz, args.opset)
    else:
        onnx_path = os.path.join(MODEL_DIR, f"{os.path.splitext(os.path.basename(args.weights))[0]}.onnx")

    if args.verify and args.export:
        verify_onnx(onnx_path)
    elif args.verify:
        # Find latest onnx in models/
        onnx_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".onnx")])
        if onnx_files:
            verify_onnx(os.path.join(MODEL_DIR, onnx_files[-1]))
        else:
            print("[WARN] No ONNX model found in models/ directory.")


if __name__ == "__main__":
    main()
