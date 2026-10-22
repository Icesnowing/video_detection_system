# -*- coding: utf-8 -*-
"""本地预训练权重目录（yoloDetectionModel/model），避免 Ultralytics 在线下载。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_PRETRAINED = "yolov8m.pt"


def resolve_pretrained_weight(spec: str | None) -> str:
    """
    将 --model 解析为本地 .pt 绝对路径。
    查找顺序：已是文件 -> model/<文件名> -> model/<spec>.pt
    """
    name = (spec or DEFAULT_PRETRAINED).strip()
    p = Path(name)
    if p.is_file():
        return str(p.resolve())

    for candidate in (MODEL_DIR / p.name, MODEL_DIR / name, MODEL_DIR / f"{name}.pt"):
        if candidate.is_file():
            return str(candidate.resolve())

    sys.exit(
        f"未找到预训练权重「{name}」。请将 .pt 放入 {MODEL_DIR}，或通过 --model 指定已有文件路径。"
    )
