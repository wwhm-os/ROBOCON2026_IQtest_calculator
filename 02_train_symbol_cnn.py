"""
02_train_symbol_cnn.py

作用：
    训练一个轻量 CNN，用于识别分割后的单个字符：0~9、+、-、×、÷、=。

输入：
    由 01_generate_font_dataset.py 生成的数据集目录。

输出：
    symbol_cnn.pt，包含：
        - model_state: CNN 权重
        - idx_to_char: 类别编号到真实字符的映射
        - img_size: 输入图像尺寸

依赖：
    pip install torch torchvision pillow tqdm

示例：
    python 02_train_symbol_cnn.py --data_dir dataset_symbols --out_path symbol_cnn.pt --epochs 12
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


class SymbolCNN(nn.Module):
    """一个足够小的 CNN：字符类别少、图像尺寸小，不需要很深的网络。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # 输入：1×28×28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32×14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64×7×7

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
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    """固定随机种子，便于复现实验结果。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(data_dir: Path, img_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """构造训练/验证 DataLoader，并生成 idx -> char 映射。"""
    label_file = data_dir / "labels.json"
    if not label_file.exists():
        raise FileNotFoundError(f"找不到 {label_file}，请先运行 01_generate_font_dataset.py")

    with open(label_file, "r", encoding="utf-8") as f:
        folder_to_char: Dict[str, str] = json.load(f)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # 自动归一化到 [0, 1]，白色前景数值更大
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 变为大致 [-1, 1]，训练更稳定
    ])

    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)

    # ImageFolder 会按文件夹名排序生成 class_to_idx；这里转成真实字符映射。
    idx_to_char = {
        idx: folder_to_char[folder]
        for folder, idx in dataset.class_to_idx.items()
        if folder in folder_to_char
    }

    if len(idx_to_char) != len(folder_to_char):
        raise RuntimeError("数据集文件夹与 labels.json 不一致，请检查目录。")

    val_len = max(1, int(len(dataset) * 0.15))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, idx_to_char


def run_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> Tuple[float, float]:
    """运行一个 epoch；optimizer 为 None 时表示验证模式。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_symbols", help="训练数据集目录")
    parser.add_argument("--out_path", type=str, default="symbol_cnn.pt", help="模型保存路径")
    parser.add_argument("--img_size", type=int, default=28, help="输入图像尺寸")
    parser.add_argument("--epochs", type=int, default=12, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    train_loader, val_loader, idx_to_char = build_dataloaders(
        data_dir=Path(args.data_dir),
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    model = SymbolCNN(num_classes=len(idx_to_char)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = run_one_epoch(model, val_loader, None, device)

        print(f"train loss={train_loss:.4f}, acc={train_acc:.4f} | val loss={val_loss:.4f}, acc={val_acc:.4f}")

        # 只保存验证集最好的模型，避免后期过拟合。
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "idx_to_char": idx_to_char,
                    "img_size": args.img_size,
                },
                out_path,
            )
            print(f"[SAVE] best_acc={best_acc:.4f} -> {out_path}")

    print(f"[DONE] 训练结束，最佳验证准确率：{best_acc:.4f}")


if __name__ == "__main__":
    main()
