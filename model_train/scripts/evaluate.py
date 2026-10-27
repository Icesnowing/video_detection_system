# -*- coding: utf-8 -*-
"""
分类别评估脚本（对应方案第 5 节）

用法：
  python scripts/evaluate.py --weights runs/detect/person_car/weights/best.pt --classes person,car
  python scripts/cli.py evaluate --weights best.pt --split val
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from coco_targets import DEFAULT_CLASSES, default_yaml_path, format_classes_help, parse_classes

ACCEPT_MAP50 = 0.85


def run_evaluate(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        plots=True,
    )

    map50 = metrics.box.map50
    map5095 = metrics.box.map
    names = metrics.names
    ap50, ap5095 = metrics.box.ap50, metrics.box.ap
    # 准确率
    precision = metrics.box.p
    recall = metrics.box.r
    print("\n===== 评估结果 =====")
    print(f"  mAP@0.5      : {map50:.4f}  (合格线 {ACCEPT_MAP50})")
    print(f"  mAP@0.5:0.95 : {map5095:.4f}")
    print("  分类别指标:")
    for i in sorted(names):
        if i < len(ap50):
            print(f"    {names[i]:10s}: AP@0.5 = {ap50[i]:.4f}   AP@0.5:0.95 = {ap5095[i]:.4f}   Precision = {precision[i]:.4f}   Recall = {recall[i]:.4f}")

    print("\n===== 验收结论 =====")
    if map50 >= ACCEPT_MAP50:
        print(f"  [合格] mAP@0.5 = {map50:.4f} >= {ACCEPT_MAP50}")
    else:
        print(f"  [未达标] mAP@0.5 = {map50:.4f} < {ACCEPT_MAP50}")
        print("  迭代路径：1) 看分类别 AP 定位短板  2) 看混淆矩阵区分漏检/误检/类别混淆")
        print("           3) 针对性处理：补数据 / 调增强 / 调损失权重")
    print(f"  混淆矩阵与 PR 曲线已保存至: {metrics.save_dir}")


def main():
    ap = argparse.ArgumentParser(description="YOLO 模型评估")
    ap.add_argument("--weights", required=True, help="模型权重路径")
    ap.add_argument("--data", default="", help="dataset.yaml")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help=format_classes_help())
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    if not args.data:
        args.data = default_yaml_path(parse_classes(args.classes))
    run_evaluate(args)


if __name__ == "__main__":
    main()
