# -*- coding: utf-8 -*-
"""
YOLO 目标检测训练 CLI：通过命令行配置检测类别与训练流程。

默认检测目标：person、car。

用法：
  # CPU 约 5~6 小时：子采样 + 小图少 epoch
  python scripts/cli.py pipeline --coco-root D:/datasets/coco --cpu --overwrite

  # 从 COCO 构建 person+car 数据集并训练
  python scripts/cli.py pipeline --coco-root D:/datasets/coco

  # 仅构建数据集（可改类别）
  python scripts/cli.py prepare --coco-root D:/datasets/coco --classes person,car --out data/person_car

  # 仅训练（需已有 dataset.yaml）
  python scripts/cli.py train --data data/person_car/dataset.yaml --epochs 100

  # 评估 / 导出
  python scripts/cli.py evaluate --weights runs/detect/person_car/weights/best.pt --data data/person_car/dataset.yaml
  python scripts/cli.py export --weights runs/detect/person_car/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从 yoloDetectionModel 根目录或 scripts/ 下均可 import 同级模块
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from coco_targets import (  # noqa: E402
    DEFAULT_CLASSES,
    default_dataset_dir,
    default_yaml_path,
    format_classes_help,
    parse_classes,
    dataset_slug,
)
from local_model import DEFAULT_PRETRAINED  # noqa: E402
from prepare_dataset import build_dataset  # noqa: E402


def _add_classes_arg(p: argparse.ArgumentParser, *, required: bool = False) -> None:
    p.add_argument(
        "--classes",
        default=None if required else ",".join(DEFAULT_CLASSES),
        help=format_classes_help(),
    )


def _resolve_data_and_out(args, require_data: bool = False) -> tuple[list[str], str, str]:
    class_names = parse_classes(getattr(args, "classes", None))
    out = getattr(args, "out", None) or default_dataset_dir(class_names)
    data = getattr(args, "data", None) or default_yaml_path(class_names, out)
    if require_data and not Path(data).is_file():
        sys.exit(f"未找到数据集配置 {data}，请先运行 prepare 或指定 --data")
    return class_names, out, data


def cmd_prepare(args: argparse.Namespace) -> None:
    if getattr(args, "cpu", False):
        from cpu_profile import apply_cpu_prepare

        apply_cpu_prepare(args)
    class_names = parse_classes(args.classes)
    if not args.out:
        args.out = default_dataset_dir(class_names)
    build_dataset(args, class_names)


def cmd_train(args: argparse.Namespace) -> None:
    from train import run_train

    if getattr(args, "cpu", False):
        from cpu_profile import apply_cpu_train

        apply_cpu_train(args)
    _, _, data = _resolve_data_and_out(args)
    args.data = data
    if args.name == "train":
        args.name = dataset_slug(parse_classes(args.classes))
    run_train(args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate import run_evaluate

    _, _, data = _resolve_data_and_out(args)
    args.data = data
    run_evaluate(args)


def cmd_export(args: argparse.Namespace) -> None:
    from export import run_export

    _, _, data = _resolve_data_and_out(args)
    args.data = data
    run_export(args)


def cmd_pipeline(args: argparse.Namespace) -> None:
    from train import run_train
    from evaluate import run_evaluate

    if getattr(args, "cpu", False):
        from cpu_profile import apply_cpu_prepare

        apply_cpu_prepare(args)
    class_names = parse_classes(args.classes)
    args.out = args.out or default_dataset_dir(class_names)
    args.data = default_yaml_path(class_names, args.out)
    build_dataset(args, class_names)
    train_ns = argparse.Namespace(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=dataset_slug(class_names),
        resume="",
        mosaic=getattr(args, "mosaic", 1.0),
        amp=getattr(args, "amp", True),
        cache=getattr(args, "cache", False),
    )
    if getattr(args, "cpu", False):
        from cpu_profile import apply_cpu_train

        apply_cpu_train(train_ns)
    run_train(train_ns)
    if args.skip_eval:
        return
    eval_ns = argparse.Namespace(
        weights=str(Path(args.project) / dataset_slug(class_names) / "weights" / "best.pt"),
        data=args.data,
        split=args.eval_split,
        imgsz=args.imgsz,
        device=args.device,
    )
    if not Path(eval_ns.weights).is_file():
        print(f"[pipeline] 未找到 {eval_ns.weights}，跳过评估")
        return
    run_evaluate(eval_ns)


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO 目标检测：CLI 配置类别与训练流程")
    sub = ap.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="从 COCO 构建 YOLO 数据集")
    p_prep.add_argument("--coco-root", required=True)
    p_prep.add_argument("--out", default="", help="输出目录，默认 data/<类别名拼接>")
    _add_classes_arg(p_prep)
    p_prep.add_argument("--download", action="store_true")
    p_prep.add_argument("--demo", type=int, default=0, metavar="N")
    p_prep.add_argument("--neg-ratio", type=float, default=0.08)
    p_prep.add_argument("--max-per-class", type=int, default=0, metavar="N",
                        help="每 COCO 子集每类最多 N 张正样本（0=不限）")
    p_prep.add_argument("--cpu", action="store_true", help="CPU 快速构建预设")
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.add_argument("--overwrite", action="store_true")
    p_prep.set_defaults(func=cmd_prepare)

    p_train = sub.add_parser("train", help="迁移学习训练")
    p_train.add_argument("--data", default="", help="dataset.yaml，默认同 --classes 推导")
    _add_classes_arg(p_train)
    p_train.add_argument("--out", default="", help="与 prepare 一致时用于推导 --data")
    p_train.add_argument("--model", default=DEFAULT_PRETRAINED,
                         help=f"model/ 目录下的权重，默认 {DEFAULT_PRETRAINED}")
    p_train.add_argument("--epochs", type=int, default=150)
    p_train.add_argument("--patience", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW", "auto"])
    p_train.add_argument("--lr0", type=float, default=None)
    p_train.add_argument("--device", default="")
    p_train.add_argument("--workers", type=int, default=4)
    p_train.add_argument("--project", default="runs/detect")
    p_train.add_argument("--name", default="train")
    p_train.add_argument("--resume", default="")
    p_train.add_argument("--cpu", action="store_true", help="CPU 快速训练预设")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="分类别评估")
    p_eval.add_argument("--weights", required=True)
    p_eval.add_argument("--data", default="")
    _add_classes_arg(p_eval)
    p_eval.add_argument("--out", default="")
    p_eval.add_argument("--split", default="val", choices=["val", "test"])
    p_eval.add_argument("--imgsz", type=int, default=640)
    p_eval.add_argument("--device", default="")
    p_eval.set_defaults(func=cmd_evaluate)

    p_exp = sub.add_parser("export", help="导出 ONNX / TensorRT 等")
    p_exp.add_argument("--weights", required=True)
    p_exp.add_argument("--data", default="")
    _add_classes_arg(p_exp)
    p_exp.add_argument("--out", default="")
    p_exp.add_argument("--format", default="onnx",
                       choices=["onnx", "engine", "torchscript", "openvino", "coreml"])
    p_exp.add_argument("--imgsz", type=int, default=640)
    p_exp.add_argument("--half", action="store_true")
    p_exp.add_argument("--int8", action="store_true")
    p_exp.add_argument("--device", default="")
    p_exp.set_defaults(func=cmd_export)

    p_pipe = sub.add_parser("pipeline", help="prepare + train（可选评估）")
    p_pipe.add_argument("--coco-root", required=True)
    p_pipe.add_argument("--out", default="")
    _add_classes_arg(p_pipe)
    p_pipe.add_argument("--download", action="store_true")
    p_pipe.add_argument("--demo", type=int, default=0, metavar="N")
    p_pipe.add_argument("--neg-ratio", type=float, default=0.08)
    p_pipe.add_argument("--max-per-class", type=int, default=0, metavar="N")
    p_pipe.add_argument("--cpu", action="store_true", help="CPU：子采样数据集 + 快速训练参数")
    p_pipe.add_argument("--seed", type=int, default=42)
    p_pipe.add_argument("--overwrite", action="store_true")
    p_pipe.add_argument("--model", default=DEFAULT_PRETRAINED)
    p_pipe.add_argument("--epochs", type=int, default=150)
    p_pipe.add_argument("--patience", type=int, default=50)
    p_pipe.add_argument("--imgsz", type=int, default=640)
    p_pipe.add_argument("--batch", type=int, default=16)
    p_pipe.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW", "auto"])
    p_pipe.add_argument("--lr0", type=float, default=None)
    p_pipe.add_argument("--device", default="")
    p_pipe.add_argument("--workers", type=int, default=4)
    p_pipe.add_argument("--project", default="runs/detect")
    p_pipe.add_argument("--skip-eval", action="store_true")
    p_pipe.add_argument("--eval-split", default="val", choices=["val", "test"])
    p_pipe.set_defaults(func=cmd_pipeline)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
