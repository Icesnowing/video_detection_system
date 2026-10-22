# -*- coding: utf-8 -*-
"""
CPU 快速训练预设：缩小每类样本量 + 降低训练开销，目标约 5~6 小时内跑完（视 CPU 性能略有浮动）。
"""

from __future__ import annotations

import argparse

# prepare：每个 COCO 子集内，每类最多保留的正样本图片数（含该类目标的图，再取各类并集）
CPU_MAX_PER_CLASS = {
    "train2017": 150,
    "val2017": 40,
    "test2017": 40,
}

CPU_TRAIN = {
    "device": "cpu",
    "epochs": 32,
    "patience": 10,
    "imgsz": 416,
    "batch": 4,
    "workers": 0,
    "mosaic": 0.0,
    "amp": False,
    "cache": False,
}


def apply_cpu_prepare(args: argparse.Namespace) -> None:
    if getattr(args, "max_per_class", 0) <= 0:
        args.max_per_class = CPU_MAX_PER_CLASS["train2017"]
    if getattr(args, "max_per_class_by_split", None) is None:
        args.max_per_class_by_split = dict(CPU_MAX_PER_CLASS)
    if args.neg_ratio >= 0.08:
        args.neg_ratio = 0.05
    print(
        "[cpu] 数据集预设: "
        f"train 每类≤{args.max_per_class_by_split.get('train2017', args.max_per_class)} 张正样本, "
        f"val 每类≤{args.max_per_class_by_split.get('val2017', 40)}, "
        f"neg_ratio={args.neg_ratio}"
    )


def apply_cpu_train(args: argparse.Namespace) -> None:
    for k, v in CPU_TRAIN.items():
        setattr(args, k, v)
    print(
        "[cpu] 训练预设: "
        f"device=cpu, epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
        f"mosaic=0, amp=False, workers=0"
    )
