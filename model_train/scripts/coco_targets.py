# -*- coding: utf-8 -*-
"""COCO 检测目标类别：CLI 解析与路径约定。"""

from __future__ import annotations

DEFAULT_CLASSES: tuple[str, ...] = ("person", "car")


def parse_classes(raw: str | None) -> list[str]:
    """解析 --classes：逗号分隔的 COCO 类别英文名；顺序即 YOLO 重映射 id 0,1,…"""
    if raw is None or not str(raw).strip():
        return list(DEFAULT_CLASSES)
    names: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    if not names:
        raise ValueError("至少指定一个检测类别，例如 --classes person,car")
    return names


def name2new_map(class_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(class_names)}


def dataset_slug(class_names: list[str]) -> str:
    return "_".join(class_names)


def default_dataset_dir(class_names: list[str]) -> str:
    return f"data/{dataset_slug(class_names)}"


def default_yaml_path(class_names: list[str], out_dir: str | None = None) -> str:
    base = out_dir or default_dataset_dir(class_names)
    return f"{base}/dataset.yaml"


def format_classes_help() -> str:
    return f"逗号分隔的 COCO 类别英文名，默认 {','.join(DEFAULT_CLASSES)}（如 person,car）"
