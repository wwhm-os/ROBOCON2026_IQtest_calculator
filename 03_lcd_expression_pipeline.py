"""
03_lcd_expression_pipeline.py

作用：
    对已经分割/透视矫正后的 LCD 算式图像做：
        1. 前处理：灰度化 + 二值化，得到黑底白字的前景图；
        2. DBSCAN：把前景像素聚成字符/笔画组件；
        3. 组件合并：将 '÷' 的“上点/横线/下点”、'=' 的两条横线合并成一个字符框；
        4. CNN：逐字符识别；
        5. 表达式求值：把 ×、÷ 转为 *、/ 后，用安全 AST 计算。

注意：
    不建议直接 eval(识别结果)，因为识别文本本质上是外部输入。
    这里用 ast 白名单方式只允许数字和 + - * / 运算，效果等价但更安全。

依赖：
    pip install opencv-python numpy scikit-learn torch torchvision pillow

示例：
    python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt --debug_dir debug
"""

from __future__ import annotations

import argparse
import ast
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


# -----------------------------
# 1. 与训练脚本一致的 CNN 结构
# -----------------------------


class SymbolCNN(nn.Module):
    """推理时必须与训练时的网络结构一致。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class Box:
    """字符或笔画组件的外接框。"""

    x1: int
    y1: int
    x2: int
    y2: int
    area: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def merge_boxes(boxes: Sequence[Box]) -> Box:
    """把多个 Box 合并为一个更大的 Box。"""
    return Box(
        x1=min(b.x1 for b in boxes),
        y1=min(b.y1 for b in boxes),
        x2=max(b.x2 for b in boxes),
        y2=max(b.y2 for b in boxes),
        area=sum(b.area for b in boxes),
    )


# -----------------------------
# 2. 图像前处理与 DBSCAN 分割
# -----------------------------


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """将 LCD 图像转为黑底白字二值图。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 轻微模糊可以抑制阈值化时的小噪点，但核不能太大，否则 ':'、'÷' 的小点会变弱。
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # LCD 是白底黑字，因此使用 THRESH_BINARY_INV：黑色字符 -> 白色前景。
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # 去掉非常细碎的噪声，不做复杂形态学，避免把相邻字符粘连。
    # kernel = np.ones((2, 2), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def dbscan_components(binary: np.ndarray, eps: float = 3.0, min_samples: int = 5) -> List[Box]:
    """对所有前景像素做 DBSCAN，得到基本笔画/字符组件。"""
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return []

    # DBSCAN 的输入是二维点坐标。这里用 (x, y)，方便理解空间距离。
    points = np.column_stack([xs, ys]).astype(np.float32)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    boxes: List[Box] = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        pts = points[labels == lab]
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int) + 1
        area = int(len(pts))
        boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, area=area))
    return boxes


def filter_noise_boxes(boxes: Sequence[Box], image_shape: Tuple[int, int]) -> List[Box]:
    """滤除明显噪声框以及屏幕边框/任务栏等超大非字符区域。"""
    h, w = image_shape
    min_area = max(8, int(h * w * 0.00002))
    min_side = max(2, int(min(h, w) * 0.003))

    kept = []
    for b in boxes:
        too_small = b.area < min_area or b.w < min_side or b.h < min_side

        # 对用户给出的 LCD 图，底部任务栏、屏幕黑边可能会被二值化成大块前景。
        # 这些区域通常远宽于单个字符，因此直接排除，避免进入 CNN。
        too_large = b.w > 0.55 * w or b.h > 0.35 * h

        if not too_small and not too_large:
            kept.append(b)
    return kept


def should_merge_vertical_stack(a: Box, b: Box, image_h: int) -> bool:
    """
    判断两个组件是否可能属于同一个字符。

    主要用于：
        - '÷'：上点、横线、下点本来是 3 个不连通组件；
        - '='：上下两条横线本来是 2 个不连通组件。

    条件强调“横向重叠/中心接近”，避免把左右相邻字符错误合并。
    """
    x_overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    min_w = max(1, min(a.w, b.w))
    overlap_ratio = x_overlap / min_w

    center_close = abs(a.cx - b.cx) < 0.65 * max(a.w, b.w)
    vertical_gap = max(0, max(a.y1, b.y1) - min(a.y2, b.y2))

    # 字符内部的上下笔画间隔不会特别大；这个阈值对图像高度做归一化。
    gap_ok = vertical_gap < max(6, int(image_h * 0.10))

    return (overlap_ratio > 0.25 or center_close) and gap_ok


def merge_division_and_equal(boxes: Sequence[Box], image_shape: Tuple[int, int]) -> List[Box]:
    """把 DBSCAN 分裂出来的 '÷' 和 '=' 子组件合并。"""
    if not boxes:
        return []

    image_h, _ = image_shape
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # 任意两个“上下堆叠且横向对齐”的组件都合并。
    for i in range(n):
        for j in range(i + 1, n):
            if should_merge_vertical_stack(boxes[i], boxes[j], image_h):
                union(i, j)

    groups: Dict[int, List[Box]] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)

    merged = [merge_boxes(g) for g in groups.values()]
    merged.sort(key=lambda b: b.x1)
    return merged


def keep_main_text_row(boxes: Sequence[Box], image_shape: Tuple[int, int]) -> List[Box]:
    """
    从候选框中保留主算式所在的水平行。

    原因：实际 LCD ROI 往往仍包含任务栏、边框、鼠标指针等黑色结构。
    它们也会被二值化成前景。主算式的字符通常位于同一水平行，且单字符面积较大，
    因此按 y 中心聚成若干“行”，选择总面积最大的那一行即可。
    """
    if len(boxes) <= 2:
        return list(boxes)

    h, _ = image_shape
    row_tol = max(12.0, 0.12 * h)
    rows: List[List[Box]] = []

    for b in sorted(boxes, key=lambda x: x.cy):
        placed = False
        for row in rows:
            row_cy = sum(x.cy for x in row) / len(row)
            if abs(b.cy - row_cy) < row_tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    # 优先选择字符面积总和最大的行。底部小图标即使数量多，单个面积通常明显小于大字号算式。
    best_row = max(rows, key=lambda row: sum(b.area for b in row))
    best_row.sort(key=lambda b: b.x1)
    return best_row


def segment_characters(binary: np.ndarray, eps: float, min_samples: int, keep_main_row: bool = True) -> List[Box]:
    """完整字符分割：DBSCAN -> 去噪 -> 合并特殊符号组件 -> 从左到右排序。"""
    boxes = dbscan_components(binary, eps=eps, min_samples=min_samples)
    boxes = filter_noise_boxes(boxes, binary.shape)
    boxes = merge_division_and_equal(boxes, binary.shape)
    boxes.sort(key=lambda b: b.x1)
    return boxes


# -----------------------------
# 3. 字符归一化与 CNN 推理
# -----------------------------


def crop_to_model_input(binary: np.ndarray, box: Box, img_size: int = 28) -> np.ndarray:
    """把一个字符框裁出并归一化到 CNN 输入尺寸。"""
    pad = max(2, int(0.12 * max(box.w, box.h)))
    h, w = binary.shape
    x1 = max(0, box.x1 - pad)
    y1 = max(0, box.y1 - pad)
    x2 = min(w, box.x2 + pad)
    y2 = min(h, box.y2 + pad)
    crop = binary[y1:y2, x1:x2]

    # 放入正方形画布，保持字符长宽比。
    ch, cw = crop.shape
    side = max(ch, cw)
    square = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - ch) // 2
    x0 = (side - cw) // 2
    square[y0:y0 + ch, x0:x0 + cw] = crop

    resized = cv2.resize(square, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized


def load_model(model_path: Path, device: torch.device) -> Tuple[SymbolCNN, Dict[int, str], int]:
    """加载训练好的 CNN 模型。"""
    ckpt = torch.load(model_path, map_location=device)

    # torch 保存 json-like dict 时，key 可能是 int，也可能被某些流程转成 str，这里统一转 int。
    idx_to_char = {int(k): v for k, v in ckpt["idx_to_char"].items()}
    img_size = int(ckpt.get("img_size", 28))

    model = SymbolCNN(num_classes=len(idx_to_char)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, idx_to_char, img_size


def predict_one(model: SymbolCNN, char_img: np.ndarray, idx_to_char: Dict[int, str], device: torch.device) -> Tuple[str, float]:
    """识别单个 28×28 字符图，返回字符和置信度。"""
    x = char_img.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5  # 与训练脚本 Normalize(mean=0.5, std=0.5) 一致
    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = F.softmax(logits, dim=1)[0]
        conf, idx = torch.max(prob, dim=0)

    return idx_to_char[int(idx.item())], float(conf.item())


def recognize_expression(binary: np.ndarray, boxes: Sequence[Box], model: SymbolCNN, idx_to_char: Dict[int, str], img_size: int, device: torch.device) -> Tuple[str, List[Tuple[Box, str, float]]]:
    """逐字符识别，并拼接成表达式字符串。"""
    results: List[Tuple[Box, str, float]] = []
    chars: List[str] = []

    for box in boxes:
        char_img = crop_to_model_input(binary, box, img_size=img_size)
        ch, conf = predict_one(model, char_img, idx_to_char, device)
        results.append((box, ch, conf))
        chars.append(ch)

    expr = "".join(chars)
    return expr, results


# -----------------------------
# 4. 安全表达式计算
# -----------------------------


ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def normalize_expression(expr: str) -> str:
    """把识别字符转为 Python 可解析的四则运算表达式。"""
    # '=' 只表示题目结束，不参与计算；例如 20÷20×1= -> 20/20*1。
    expr = expr.split("=")[0]
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = expr.replace(" ", "")
    return expr


def safe_eval_expr(expr: str) -> float:
    """仅允许数字和 + - * / 的安全求值。"""
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Python 3.8+ 数字常量是 ast.Constant。
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        # 兼容更早版本 Python 的 ast.Num。
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return float(node.n)       # type: ignore[attr-defined]

        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            return ALLOWED_BIN_OPS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPS:
            return ALLOWED_UNARY_OPS[type(node.op)](_eval(node.operand))

        raise ValueError(f"表达式包含不允许的语法：{ast.dump(node)}")

    return _eval(tree)


# -----------------------------
# 5. 调试可视化
# -----------------------------


def draw_debug(image_bgr: np.ndarray, results: Sequence[Tuple[Box, str, float]], out_path: Path) -> None:
    """保存检测框、识别字符和置信度，便于调参。"""
    vis = image_bgr.copy()
    for box, ch, conf in results:
        cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        text = f"{ch}:{conf:.2f}"
        cv2.putText(vis, text, (box.x1, max(18, box.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def save_char_debug(binary: np.ndarray, boxes: Sequence[Box], debug_dir: Path, img_size: int) -> None:
    """把每个归一化后的字符小图保存下来，检查 CNN 输入是否合理。"""
    char_dir = debug_dir / "chars"
    char_dir.mkdir(parents=True, exist_ok=True)
    for i, box in enumerate(boxes):
        char_img = crop_to_model_input(binary, box, img_size=img_size)
        cv2.imwrite(str(char_dir / f"{i:02d}.png"), char_img)


# -----------------------------
# 6. 主流程
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="已经分割/透视矫正后的 LCD 图像路径")
    parser.add_argument("--model", type=str, default="symbol_cnn.pt", help="训练好的 CNN 模型路径")
    parser.add_argument("--eps", type=float, default=3.0, help="DBSCAN 邻域半径；字符断裂可略增，相邻字符粘连则略减")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN 核心点最小邻居数")
    parser.add_argument("--debug-dir", type=str, default="debug", help="若非空，则保存二值图、字符小图、框选结果")
    parser.add_argument("--no-keep-main-row", action="store_true", help="关闭主算式行筛选；当输入已严格裁成单行算式时可关闭")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_char, img_size = load_model(model_path, device)

    binary = preprocess(image_bgr)
    boxes = segment_characters(binary, eps=args.eps, min_samples=args.min_samples, keep_main_row=not args.no_keep_main_row)
    raw_expr, results = recognize_expression(binary, boxes, model, idx_to_char, img_size, device)

    expr = normalize_expression(raw_expr)
    value = safe_eval_expr(expr)

    print(f"raw_expr = {raw_expr}")
    print(f"expr     = {expr}")
    print(f"value    = {value:g}")
    print("chars:")
    for i, (box, ch, conf) in enumerate(results):
        print(f"  {i:02d}: {ch!r}, conf={conf:.3f}, box=({box.x1},{box.y1},{box.x2},{box.y2})")

    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "binary.png"), binary)
        save_char_debug(binary, boxes, debug_dir, img_size)
        draw_debug(image_bgr, results, debug_dir / "recognized.png")
        print(f"[DEBUG] 调试结果已保存到：{debug_dir.resolve()}")


if __name__ == "__main__":
    main()
