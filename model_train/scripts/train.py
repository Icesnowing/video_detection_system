# -*- coding: utf-8 -*-
"""
YOLOv8 迁移学习训练脚本（对应方案第 3、4 节）

用法：
  python scripts/train.py --data data/person_car/dataset.yaml
  python scripts/cli.py train --classes person,car --epochs 100
  python scripts/train.py --model /model/yolo11n.pt --epochs 100 --patience 30 --imgsz 416 --batch 8 --lr0 1e-4 --resume /runs/detect/train-4/weights/last.pt --classes car --data data/car/dataset.yaml
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from coco_targets import DEFAULT_CLASSES, default_yaml_path, format_classes_help, parse_classes
from local_model import DEFAULT_PRETRAINED, resolve_pretrained_weight


def run_train(args: argparse.Namespace) -> None:
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    lr0 = args.lr0
    if lr0 is None and args.optimizer != "auto":
        lr0 = 0.001 if args.optimizer == "AdamW" else 0.01
        print(f"lr0 未指定，按 {args.optimizer} 自动取 {lr0}")

    weights = resolve_pretrained_weight(args.model)
    print(f"加载本地预训练权重: {weights}")
    model = YOLO(weights)
    
    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        cos_lr=True,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        lr0=lr0,
        lrf = args.lrf,
        freeze=args.freeze,
        mosaic=args.mosaic,
        fliplr=0.5,
        flipud=0.0,
        scale=args.scale,
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        degrees = args.degrees,
        translate=args.translate, shear=0.0, perspective=0.0,
        mixup=0.0, copy_paste=0.0,
        device=args.device if args.device else None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=42,
        verbose=True,
        amp=bool(getattr(args, "amp", True)),
        cache=bool(getattr(args, "cache", False)),
    )
    model.train(**train_args)


def main():
    ap = argparse.ArgumentParser(description="YOLO 目标检测训练")
    ap.add_argument("--data", default="", help="dataset.yaml；未指定时由 --classes 推导")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help=format_classes_help())
    ap.add_argument(
        "--model",
        default=DEFAULT_PRETRAINED,
        help=f"预训练权重文件名或路径，默认从 model/ 加载 {DEFAULT_PRETRAINED}",
    )
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW", "auto"])
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--lrf", type=float, default=None)
    ap.add_argument("--freeze", type=int, default=0, help="冻结前 n 层")
    ap.add_argument("--mosaic", type=float, default=1.0, help="数据增强：mosaic 等比缩放")
    ap.add_argument("--warmup_epochs", type=int, default=3, help="预热轮数")
    ap.add_argument("--warmup_bias_lr", type=float, default=0.01, help="预热轮数学习率")
    ap.add_argument("--degrees", type=float, default=5, help="数据增强：degrees 旋转角度")
    ap.add_argument("--translate", type=float, default=0.1, help="数据增强：translate 平移")
    ap.add_argument("--scale", type=float, default=0.5, help="数据增强：scale 缩放")
    ap.add_argument("--device", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="train")
    ap.add_argument("--resume", default="", help="断点续训权重路径（last.pt）")
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="CPU 快速训练：epochs≈32、imgsz=416、batch=4、关闭 mosaic/AMP",
    )
    args = ap.parse_args()
    if args.cpu:
        from cpu_profile import apply_cpu_train

        apply_cpu_train(args)
    if not args.data:
        args.data = default_yaml_path(parse_classes(args.classes))
    run_train(args)


if __name__ == "__main__":
    main()
