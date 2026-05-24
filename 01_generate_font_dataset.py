"""
01_generate_font_dataset.py

作用：
    生成“标准字体 + 轻微扰动”的字符分类数据集，用于训练 LCD 算式识别的 CNN。

输出目录结构示例：
    dataset_symbols/
        0/000000.png
        1/000000.png
        ...
        plus/000000.png      -> '+'
        minus/000000.png     -> '-'
        mul/000000.png       -> '×'
        div/000000.png       -> '÷'
        eq/000000.png        -> '='
        labels.json          -> 文件夹名与真实字符的映射

核心思想：
    1. 用 PIL 按指定字体渲染黑白字符；
    2. 加入轻微旋转、平移、缩放、透视/剪切、模糊、噪声、膨胀/腐蚀等扰动；
    3. 统一输出为 28×28 的“黑底白字”灰度图，与后续二值化分割后的字符输入保持一致。

依赖：
    pip install pillow numpy opencv-python tqdm

示例：
    python 01_generate_font_dataset.py --out_dir dataset_symbols --samples_per_class 3000
    python 01_generate_font_dataset.py --out_dir dataset_symbols --fonts "C:/Windows/Fonts/times.ttf"
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm


# 文件夹名不能直接稳定使用特殊符号，因此用 folder_name -> char 的映射。
LABELS: Dict[str, str] = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "plus": "+",
    "minus": "-",
    "mul": "×",
    "div": "÷",
    "eq": "=",
}


def find_default_fonts() -> List[str]:
    """尽量自动寻找 Times New Roman 或相近的衬线字体。"""
    candidates = [
        # Windows: Times New Roman 常见路径
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/timesi.ttf",
        # Linux: 常见替代衬线字体
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        # macOS: 常见字体路径
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    ]
    fonts = [p for p in candidates if Path(p).exists()]
    if not fonts:
        # 兜底：PIL 默认字体也能运行，但风格不一定像 Times New Roman。
        print("[WARN] 未找到 Times New Roman 或常见衬线字体，将使用 PIL 默认字体。")
    return fonts


def load_font(font_paths: List[str], font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """随机加载一个字体；若 font_paths 为空，则使用 PIL 默认字体。"""
    if not font_paths:
        return ImageFont.load_default()
    return ImageFont.truetype(random.choice(font_paths), font_size)


def draw_centered_char(char: str, font_paths: List[str], canvas_size: int = 96) -> Image.Image:
    """在较大画布中心绘制白色字符，背景为黑色。"""
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)

    # 字号随机变化，模拟 LCD 屏幕拍摄时字符尺度变化。
    font_size = random.randint(58, 78)
    font = load_font(font_paths, font_size)

    # textbbox 比 textsize 更稳定，可获得真实字符边界。
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 让字符大致居中；减去 bbox 左上角偏移，避免某些字体基线导致偏心。
    x = (canvas_size - w) / 2 - bbox[0]
    y = (canvas_size - h) / 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    return img


def random_affine(img: Image.Image) -> Image.Image:
    """加入轻微仿射扰动：旋转、平移、缩放、剪切。"""
    arr = np.array(img)
    h, w = arr.shape
    center = (w / 2, h / 2)

    angle = random.uniform(-5.0, 5.0)       # 拍摄/ROI 透视矫正残差通常不会太大
    scale = random.uniform(0.88, 1.08)
    tx = random.uniform(-4.0, 4.0)
    ty = random.uniform(-4.0, 4.0)
    shear = random.uniform(-0.05, 0.05)

    mat = cv2.getRotationMatrix2D(center, angle, scale)
    mat[0, 1] += shear
    mat[0, 2] += tx
    mat[1, 2] += ty

    warped = cv2.warpAffine(
        arr,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return Image.fromarray(warped)


def random_degrade(img: Image.Image) -> Image.Image:
    """模拟拍摄退化：轻微模糊、噪声、笔画粗细变化。"""
    # 轻微高斯模糊，模拟相机虚焦/运动模糊后的边缘软化。
    if random.random() < 0.55:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    arr = np.array(img).astype(np.float32)

    # 加少量高斯噪声，模拟传感器噪声和压缩误差。
    if random.random() < 0.45:
        noise = np.random.normal(loc=0.0, scale=random.uniform(2.0, 10.0), size=arr.shape)
        arr += noise

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 随机膨胀/腐蚀，模拟阈值化后笔画变粗或变细。
    if random.random() < 0.35:
        k = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            arr = cv2.dilate(arr, k, iterations=1)
        else:
            arr = cv2.erode(arr, k, iterations=1)

    return Image.fromarray(arr)


def crop_and_resize(img: Image.Image, out_size: int = 28) -> Image.Image:
    """裁剪字符前景，等比例缩放到 out_size×out_size，并保留少量边距。"""
    arr = np.array(img)
    ys, xs = np.where(arr > 15)

    # 极端情况下没有前景，返回空图，避免程序崩溃。
    if len(xs) == 0 or len(ys) == 0:
        return Image.new("L", (out_size, out_size), color=0)

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    crop = arr[y1:y2, x1:x2]

    # 先放入正方形画布，避免横向符号 '-' '=' 被强行拉高。
    ch, cw = crop.shape
    side = max(ch, cw)
    pad = max(4, int(side * 0.18))
    square = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
    y0 = (square.shape[0] - ch) // 2
    x0 = (square.shape[1] - cw) // 2
    square[y0:y0 + ch, x0:x0 + cw] = crop

    resized = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)


def make_one_sample(char: str, font_paths: List[str], out_size: int) -> Image.Image:
    """生成单张训练样本。"""
    img = draw_centered_char(char, font_paths)
    img = random_affine(img)
    img = random_degrade(img)
    img = crop_and_resize(img, out_size=out_size)
    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="dataset_symbols", help="输出数据集目录")
    parser.add_argument("--samples_per_class", type=int, default=3000, help="每类生成样本数")
    parser.add_argument("--img_size", type=int, default=28, help="输出图像尺寸")
    parser.add_argument(
        "--fonts",
        type=str,
        nargs="*",
        default=None,
        help="字体文件路径，可传多个；例如 C:/Windows/Fonts/times.ttf",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，便于复现实验")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    font_paths = args.fonts if args.fonts else find_default_fonts()
    font_paths = [p for p in font_paths if Path(p).exists()]
    print(f"[INFO] 使用字体数量：{len(font_paths)}")
    for p in font_paths:
        print(f"       {p}")

    # 保存标签映射，训练和推理脚本都会读取它。
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)

    for folder_name, char in LABELS.items():
        class_dir = out_dir / folder_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(args.samples_per_class), desc=f"生成 {folder_name}({char})"):
            img = make_one_sample(char, font_paths, out_size=args.img_size)
            img.save(class_dir / f"{i:06d}.png")

    print(f"[DONE] 数据集已生成：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
