# -*- coding: utf-8 -*-
"""
COCO 2017 子集构建脚本（对应方案第 2 节）

功能：
1. 从 COCO 2017 各子集（train2017 / val2017 / test2017）中分别筛选含目标类别的图片作为正样本
2. 类别 id 按 --classes 重映射为 0~nc-1（按 categories 名称反查 COCO id）
3. 各 COCO 子集内保留一定比例无目标类别的图片作为背景负样本
4. 输出划分与 COCO 子集对齐：train2017->train、val2017->val、test2017->test
5. 输出 YOLO 格式数据集（images/ labels/ + dataset.yaml）

用法：
  python scripts/prepare_dataset.py --coco-root D:/datasets/coco --classes person,car
  python scripts/cli.py prepare --coco-root D:/datasets/coco --classes person,car
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from coco_targets import DEFAULT_CLASSES, format_classes_help, name2new_map, parse_classes

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
IMG_URL_TMPL = "http://images.cocodataset.org/{split}/{file_name}"

COMPLETE_SPLIT_MIN_IMGS = 4000

COCO_SPLITS = ("train2017", "val2017", "test2017")
COCO_TO_YOLO_SPLIT = {"train2017": "train", "val2017": "val", "test2017": "test"}


def download_file(url: str, dst: Path, retries: int = 3):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"  已存在，跳过: {dst}")
        return
    print(f"  下载 {url} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dst)
            return
        except Exception as e:
            print(f"  下载失败（第 {attempt}/{retries} 次）: {e}")
            if tmp.exists():
                tmp.unlink()
    sys.exit(f"下载失败，已达最大重试次数: {url}")


def split_dir_complete(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False
    if (split_dir / ".complete").exists():
        return True
    return sum(1 for _ in split_dir.glob("*.jpg")) >= COMPLETE_SPLIT_MIN_IMGS


def resolve_ann_dir(coco_root: Path) -> Path:
    for name in ("annotations", "annotations_trainval2017"):
        d = coco_root / name
        if (d / "instances_train2017.json").exists():
            return d
    return coco_root / "annotations"


def ensure_coco(coco_root: Path, download: bool, demo: bool):
    ann_dir = resolve_ann_dir(coco_root)
    train_ann = ann_dir / "instances_train2017.json"
    val_ann = ann_dir / "instances_val2017.json"
    if not (train_ann.exists() and val_ann.exists()):
        if not (download or demo):
            sys.exit(f"未找到标注文件 {ann_dir}，请准备 COCO 2017 标注或加 --download / --demo")
        zip_path = coco_root / "downloads" / "annotations_trainval2017.zip"
        download_file(COCO_URLS["annotations"], zip_path)
        print("  解压标注文件 ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(coco_root)
    if download and not demo:
        for split in ("train2017", "val2017"):
            if not split_dir_complete(coco_root / split):
                zip_path = coco_root / "downloads" / f"{split}.zip"
                download_file(COCO_URLS[split], zip_path)
                print(f"  解压 {split} ...")
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(coco_root)
                (coco_root / split / ".complete").touch()


def load_split(ann_dir: Path, split: str, class_names: list[str]):
    ann_path = ann_dir / f"instances_{split}.json"
    print(f"加载标注 {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    name2new = name2new_map(class_names)
    name2coco = {c["name"]: c["id"] for c in coco["categories"]}
    missing = [n for n in name2new if n not in name2coco]
    if missing:
        sys.exit(f"{ann_path} 的 categories 中未找到类别: {missing}")
    id2new = {name2coco[n]: new for n, new in name2new.items()}
    target_anns = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in id2new:
            target_anns[ann["image_id"]].append(ann)
    return images, target_anns, id2new


# COCO 框尺寸（像素面积）：小 < 32²，中 < 96²，大 ≥ 96²
_AREA_SMALL = 32 * 32
_AREA_LARGE = 96 * 96
_TIER_LABELS = ("小", "中", "大")


def _ann_area(ann) -> float:
    _x, _y, w, h = ann["bbox"]
    return max(0.0, float(w)) * max(0.0, float(h))


def _area_tier(area: float) -> int:
    if area < _AREA_SMALL:
        return 0
    if area < _AREA_LARGE:
        return 1
    return 2


def _class_size_tier(anns, coco_id: int) -> int | None:
    """一张图在某一类上的档位：取该类标注框面积的中位数。"""
    areas = [
        _ann_area(ann)
        for ann in anns
        if ann.get("category_id") == coco_id and _ann_area(ann) > 0
    ]
    if not areas:
        return None
    areas.sort()
    mid = len(areas) // 2
    median = areas[mid] if len(areas) % 2 else (areas[mid - 1] + areas[mid]) / 2
    return _area_tier(median)


def _tier_quotas(n: int) -> list[int]:
    """把 N 分成三档，余数分给靠前的档，使各档尽量相等。"""
    base, rem = divmod(max(n, 0), 3)
    return [base + (1 if i < rem else 0) for i in range(3)]


def _take_stratified(tiers: list[list[int]], n: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """每档按配额抽取；某档不足时从其他档补足，总数不超过 n。"""
    quotas = _tier_quotas(n)
    pools = [list(tier) for tier in tiers]
    for pool in pools:
        rng.shuffle(pool)
    chosen: list[int] = []
    taken = [0, 0, 0]
    for i, (pool, quota) in enumerate(zip(pools, quotas)):
        got = pool[:quota]
        taken[i] = len(got)
        chosen.extend(got)
        pools[i] = pool[len(got):]
    short = n - len(chosen)
    if short > 0:
        rest = [(i, idx) for i, pool in enumerate(pools) for idx in pool]
        rng.shuffle(rest)
        for i, idx in rest[:short]:
            chosen.append(idx)
            taken[i] += 1
    return chosen, taken


def subsample_positives_per_class(
    positives: list,
    class_names: list[str],
    id2new: dict[int, int],
    max_per_class: int,
    rng: random.Random,
) -> list:
    """每类最多保留 max_per_class 张图，按框面积小/中/大各约 1/3，再取各类并集。"""
    if max_per_class <= 0 or not positives:
        return positives
    new2coco = {new: coco for coco, new in id2new.items()}
    keep: set[int] = set()
    summaries: list[str] = []
    for cls_idx, name in enumerate(class_names):
        coco_id = new2coco.get(cls_idx)
        if coco_id is None:
            continue
        tiers: list[list[int]] = [[], [], []]
        for idx, (_split, _meta, anns) in enumerate(positives):
            tier = _class_size_tier(anns, coco_id)
            if tier is not None:
                tiers[tier].append(idx)
        if not any(tiers):
            continue
        picked, taken = _take_stratified(tiers, max_per_class, rng)
        keep.update(picked)
        pool = "/".join(str(len(t)) for t in tiers)
        got = "/".join(str(n) for n in taken)
        summaries.append(
            f"{name}: 池 {_TIER_LABELS[0]}/{_TIER_LABELS[1]}/{_TIER_LABELS[2]}={pool} "
            f"-> 抽到 {got}（共 {len(picked)}）"
        )
    out = [positives[i] for i in sorted(keep)]
    print(
        f"    [子采样] 每类≤{max_per_class}，按框面积小/中/大各约 1/3: "
        f"{len(positives)} -> {len(out)} 张正样本"
    )
    for line in summaries:
        print(f"      {line}")
    return out


def coco_ann_to_yolo(ann, img_w, img_h, id2new):
    x, y, w, h = ann["bbox"]
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    cx, cy = x1 + w / 2, y1 + h / 2
    return id2new[ann["category_id"]], cx / img_w, cy / img_h, w / img_w, h / img_h


def _max_per_class_for_split(args: argparse.Namespace, coco_split: str) -> int:
    by_split = getattr(args, "max_per_class_by_split", None) or {}
    if coco_split in by_split and by_split[coco_split] > 0:
        return int(by_split[coco_split])
    return int(getattr(args, "max_per_class", 0) or 0)


def build_dataset(args: argparse.Namespace, class_names: list[str] | None = None) -> None:
    class_names = class_names or parse_classes(getattr(args, "classes", None))
    print(f"检测类别 ({len(class_names)}): {', '.join(class_names)}")

    random.seed(args.seed)
    coco_root = Path(args.coco_root)
    out_dir = Path(args.out)
    demo = args.demo > 0

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    if any(p.is_dir() and any(p.iterdir()) for p in (out_images, out_labels)):
        if not args.overwrite:
            sys.exit(f"输出目录 {out_dir} 下已有数据。请加 --overwrite 清空重建，或更换 --out")
        for p in (out_images, out_labels):
            if p.is_dir():
                shutil.rmtree(p)

    ensure_coco(coco_root, args.download, demo)
    ann_dir = resolve_ann_dir(coco_root)

    per_coco = {}
    id2new_all = None
    for split in COCO_SPLITS:
        ann_path = ann_dir / f"instances_{split}.json"
        if not ann_path.exists():
            print(f"  跳过 {split}：无 {ann_path.name}"
                  f"（官方 test2017 通常仅 image_info，无公开检测标注）")
            continue
        images, target_anns, id2new = load_split(ann_dir, split, class_names)
        if id2new_all is None:
            id2new_all = id2new
        elif id2new_all != id2new:
            sys.exit(f"{split} 与先前子集的 COCO 类别 id 不一致")
        pos, neg_pool = [], []
        for img_id, meta in images.items():
            if img_id in target_anns:
                pos.append((split, meta, target_anns[img_id]))
            else:
                neg_pool.append((split, meta))
        limit = _max_per_class_for_split(args, split)
        if limit > 0:
            pos = subsample_positives_per_class(pos, class_names, id2new, limit, random)
        per_coco[split] = {"pos": pos, "neg_pool": neg_pool}
        print(f"  {split}: 含目标 {len(pos)} 张，候选负样本 {len(neg_pool)} 张")

    if not per_coco:
        sys.exit(f"未找到任何 instances_*.json，请检查 {ann_dir}")

    total_pos = sum(len(v["pos"]) for v in per_coco.values())
    print(f"含目标类别图片合计: {total_pos} 张")

    if demo:
        all_pos = [item for v in per_coco.values() for item in v["pos"]]
        random.shuffle(all_pos)
        all_pos = all_pos[: args.demo]
        per_coco = {s: {"pos": [], "neg_pool": per_coco[s]["neg_pool"]} for s in per_coco}
        for item in all_pos:
            per_coco[item[0]]["pos"].append(item)
        print(f"[demo] 抽样 {len(all_pos)} 张目标图片")

    splits = {name: [] for name in ("train", "val", "test")}
    total_neg = 0
    for coco_split, buckets in per_coco.items():
        positives = buckets["pos"]
        neg_pool = buckets["neg_pool"]
        n_neg = int(len(positives) * args.neg_ratio / (1 - args.neg_ratio)) if positives else 0
        random.shuffle(neg_pool)
        negatives = neg_pool[:n_neg]
        total_neg += len(negatives)
        yolo_split = COCO_TO_YOLO_SPLIT[coco_split]
        items = [(s, m, a) for s, m, a in positives] + [(s, m, []) for s, m in negatives]
        random.shuffle(items)
        splits[yolo_split] = items
        print(f"  -> YOLO {yolo_split}: {len(positives)} 正 + {len(negatives)} 负")

    print(f"背景负样本合计: {total_neg} 张（各子集内占比约 {args.neg_ratio:.0%}）")

    stats = defaultdict(lambda: defaultdict(int))
    written = defaultdict(int)
    for split_name, items in splits.items():
        img_out = out_images / split_name
        lab_out = out_labels / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        for src_split, meta, anns in items:
            fname = meta["file_name"]
            src_img = coco_root / src_split / fname
            dst_img = img_out / fname
            if not src_img.exists():
                if demo:
                    download_file(IMG_URL_TMPL.format(split=src_split, file_name=fname), src_img)
                else:
                    print(f"  [警告] 缺图跳过: {src_img}")
                    continue
            if not dst_img.exists():
                try:
                    dst_img.hardlink_to(src_img)
                except (OSError, AttributeError):
                    shutil.copy2(src_img, dst_img)
            lines = []
            for ann in anns:
                yolo = coco_ann_to_yolo(ann, meta["width"], meta["height"], id2new_all)
                if yolo is None:
                    continue
                cls, cx, cy, w, h = yolo
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats[split_name][class_names[cls]] += 1
            (lab_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            written[split_name] += 1

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(class_names)}\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(class_names)),
        encoding="utf-8",
    )

    print("\n===== 数据集构建完成 =====")
    for split_name, items in splits.items():
        boxes = ", ".join(f"{k}:{v}" for k, v in sorted(stats[split_name].items()))
        print(f"  {split_name:5s}: {written[split_name]:6d} 张图片（计划 {len(items)}）| 标注框 [{boxes}]")
    print(f"  配置文件: {yaml_path}")


def main():
    from coco_targets import default_dataset_dir

    ap = argparse.ArgumentParser(description="从 COCO 2017 构建 YOLO 检测数据集")
    ap.add_argument("--coco-root", required=True, help="COCO 根目录（含 annotations/ 与图片）")
    ap.add_argument("--out", default="", help="输出目录，默认 data/<类别名拼接>")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help=format_classes_help())
    ap.add_argument("--download", action="store_true", help="自动下载完整 COCO 2017（约19GB）")
    ap.add_argument("--demo", type=int, default=0, metavar="N",
                    help="演示模式：只抽取最多 N 张目标图片并单独下载")
    ap.add_argument("--neg-ratio", type=float, default=0.08, help="负样本占比（5%%~10%%）")
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        metavar="N",
        help="每个 COCO 子集内每类最多 N 张正样本图，按框面积小/中/大各约 1/3（0=不限制）",
    )
    ap.add_argument("--cpu", action="store_true", help="应用 CPU 快速构建预设（每类子采样 + 略减负样本）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true",
                    help="输出目录已有数据时先清空重建")
    args = ap.parse_args()
    if args.cpu:
        from cpu_profile import apply_cpu_prepare

        apply_cpu_prepare(args)
    class_names = parse_classes(args.classes)
    if not args.out:
        args.out = default_dataset_dir(class_names)
    build_dataset(args, class_names)


if __name__ == "__main__":
    main()
