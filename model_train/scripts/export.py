# -*- coding: utf-8 -*-
"""
模型导出脚本（对应方案第 7 节）

用法：
  python scripts/export.py --weights runs/detect/person_car/weights/best.pt
  python scripts/cli.py export --weights best.pt --classes person,car --int8 --format engine
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from coco_targets import DEFAULT_CLASSES, default_yaml_path, format_classes_help, parse_classes


def run_export(args: argparse.Namespace) -> None:
    if args.int8 and args.format not in ("engine", "openvino"):
        raise SystemExit(f"--int8 仅支持 engine / openvino 格式，当前为 {args.format}")

    model = YOLO(args.weights)
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        data=args.data if args.int8 else None,
        device=args.device if args.device else None,
    )
    print(f"导出完成: {out}")


def main():
    ap = argparse.ArgumentParser(description="导出 YOLO 模型")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="", help="INT8 校准时使用的 dataset.yaml")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help=format_classes_help())
    ap.add_argument("--format", default="onnx",
                    choices=["onnx", "engine", "torchscript", "openvino", "coreml"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 半精度")
    ap.add_argument("--int8", action="store_true", help="INT8 量化（校准集需覆盖全部目标类）")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    if not args.data:
        args.data = default_yaml_path(parse_classes(args.classes))
    run_export(args)


if __name__ == "__main__":
    main()
