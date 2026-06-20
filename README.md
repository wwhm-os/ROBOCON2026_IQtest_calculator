# LCD 算式识别

本文档汇总以下 3 个脚本的命令行使用方式和可选参数含义：

1. `01_generate_font_dataset.py`：生成标准字体字符数据集；
2. `02_train_symbol_cnn.py`：训练字符分类 CNN；
3. `03_lcd_expression_pipeline.py`：对 LCD 算式图像做二值化、DBSCAN 分割、CNN 识别和表达式计算。

整体运行顺序建议为：

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --samples-per-class 3000
python 02_train_symbol_cnn.py --data-dir dataset_symbols --out-path symbol_cnn.pt --epochs 12
python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt --debug-dir debug
```

---

## 0. 依赖安装

三个脚本合并所需依赖如下：

```bash
pip install pillow numpy opencv-python tqdm torch torchvision scikit-learn
```

如果你的环境中已经安装过其中一部分库，可以只补装缺失库。

---

## 1. `01_generate_font_dataset.py`：生成字体字符数据集

### 1.1 作用

该脚本用于生成 CNN 训练数据集。它会用指定字体渲染以下字符：

```text
0 1 2 3 4 5 6 7 8 9 + - × ÷ =
```

同时加入轻微旋转、平移、缩放、剪切、模糊、噪声、膨胀、腐蚀等扰动，使训练数据更接近相机拍摄后的 LCD 字符。

输出目录示例：

```text
dataset_symbols/
├── 0/
├── 1/
├── 2/
├── ...
├── plus/
├── minus/
├── mul/
├── div/
├── eq/
└── labels.json
```

其中 `labels.json` 用于记录文件夹名和真实字符的映射，例如 `mul -> ×`，`div -> ÷`。
**注：可在 [LABELS字典](01_generate_font_dataset.py#L47) 中添加更多字符类别，或修改现有类别的标签。**

### 1.2 基本命令

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --samples-per-class 3000
```

含义：在 `dataset_symbols` 目录下生成训练数据，每个类别生成 3000 张图像。

### 1.3 指定 Times New Roman 字体

Windows 下可以显式指定字体文件：

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --fonts "C:/Windows/Fonts/times.ttf"
```

也可以指定多个字体文件，用于增强泛化性：

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --fonts "C:/Windows/Fonts/times.ttf" "C:/Windows/Fonts/timesbd.ttf"
```

如果不指定 `--fonts`，脚本会尝试自动寻找 Times New Roman 或相近的衬线字体；若找不到，则退回 PIL 默认字体。

### 1.4 可选参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---:|---:|---|
| `--out-dir` | `str` | `dataset_symbols` | 输出数据集目录。后续训练脚本的 `--data-dir` 应指向这个目录。 |
| `--samples-per-class` | `int` | `2500` | 每个字符类别生成多少张样本。数值越大，训练更充分，但生成和训练时间更长。 |
| `--img-size` | `int` | `28` | 输出字符图像尺寸，默认生成 `28×28` 灰度图。需要和训练、推理脚本保持一致。 |
| `--fonts` | `str` 列表 | `None` | 字体文件路径，可以传入一个或多个字体。若不传，脚本自动搜索常见字体路径。 |
| `--seed` | `int` | `42` | 随机种子。固定后，数据生成过程相对可复现。 |

### 1.5 推荐用法

普通测试：

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --samples-per-class 1000
```

正式训练：

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --samples-per-class 3000 --img-size 28
```

Windows 指定 Times New Roman：

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --samples-per-class 3000 --fonts "C:/Windows/Fonts/times.ttf"
```

---

## 2. `02_train_symbol_cnn.py`：训练字符分类 CNN

### 2.1 作用

该脚本读取 `01_generate_font_dataset.py` 生成的数据集，训练一个轻量 CNN，用于识别单个字符。

输入：

```text
dataset_symbols/
```

输出：

```text
symbol_cnn.pt
```

模型文件中保存：

```text
model_state   CNN 权重
idx_to_char   类别编号到真实字符的映射
img_size      输入图像尺寸
```

### 2.2 基本命令

```bash
python 02_train_symbol_cnn.py --data-dir dataset_symbols --out-path symbol_cnn.pt --epochs 12
```

含义：读取 `dataset_symbols` 数据集，训练 12 轮，并将最佳验证集模型保存为 `symbol_cnn.pt`。

### 2.3 可选参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---:|---:|---|
| `--data-dir` | `str` | `dataset_symbols` | 训练数据集目录，应为 `01_generate_font_dataset.py` 的输出目录。 |
| `--out-path` | `str` | `symbol_cnn.pt` | 模型保存路径。后续 `03_lcd_expression_pipeline.py` 的 `--model` 应指向该文件。 |
| `--img-size` | `int` | `28` | CNN 输入图像尺寸。应与数据集生成脚本的 `--img-size` 保持一致。 |
| `--epochs` | `int` | `12` | 训练轮数。数据量较大时 8~15 轮通常够用。 |
| `--batch-size` | `int` | `128` | 批大小。显存不足时可减小，例如 64 或 32。 |
| `--lr` | `float` | `1e-3` | 学习率。训练不稳定时可尝试减小到 `5e-4` 或 `1e-4`。 |
| `--seed` | `int` | `42` | 随机种子，用于提高实验可复现性。 |

### 2.4 推荐用法

CPU 或普通训练：

```bash
python 02_train_symbol_cnn.py --data-dir dataset_symbols --out-path symbol_cnn.pt --epochs 12 --batch-size 128
```

显存不足时：

```bash
python 02_train_symbol_cnn.py --data-dir dataset_symbols --out-path symbol_cnn.pt --epochs 12 --batch-size 64
```

想训练更久：

```bash
python 02_train_symbol_cnn.py --data-dir dataset_symbols --out-path symbol_cnn.pt --epochs 20 --lr 5e-4
```

---

## 3. `03_lcd_expression_pipeline.py`：LCD 算式识别与计算

### 3.1 作用

该脚本用于对已经大致分割、透视矫正后的 LCD 算式图像进行完整识别。

流程为：

```text
输入 LCD 图像
  ↓
灰度化 + Otsu 二值化
  ↓
DBSCAN 聚类前景像素
  ↓
合并 ÷、= 等不连通符号组件
  ↓
裁剪单字符并归一化到 28×28
  ↓
CNN 逐字符识别
  ↓
拼接表达式
  ↓
把 ×、÷ 转为 *、/
  ↓
安全计算表达式结果
```

### 3.2 基本命令

```bash
python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt
```

含义：使用训练好的 `symbol_cnn.pt` 模型，对 `001460_screen_warped.png` 这张 LCD 算式图像进行识别和计算。

### 3.3 开启调试输出

建议调试阶段始终开启 `--debug-dir`：

```bash
python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt --debug-dir debug
```

输出目录示例：

```text
debug/
├── binary.png
├── recognized.png
└── chars/
    ├── 00.png
    ├── 01.png
    ├── 02.png
    └── ...
```

其中：

| 文件 | 用途 |
|---|---|
| `binary.png` | 查看二值化效果，确认字符是否完整、背景是否干净。 |
| `recognized.png` | 查看 DBSCAN 分割框、CNN 识别字符和置信度。 |
| `chars/*.png` | 查看送入 CNN 的每个单字符输入是否裁剪正确。 |

### 3.4 可选参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---:|---:|---|
| `--image` | `str` | 必填 | 输入 LCD 算式图像路径。建议输入已经完成 ROI 裁剪或透视矫正后的图像。 |
| `--model` | `str` | `symbol_cnn.pt` | 训练好的 CNN 模型路径。一般来自 `02_train_symbol_cnn.py` 的输出。 |
| `--eps` | `float` | `3.0` | DBSCAN 邻域半径。字符被拆碎时适当增大；相邻字符粘连时适当减小。 |
| `--min-samples` | `int` | `5` | DBSCAN 核心点最小邻居数。小符号点被丢失时可适当减小。 |
| `--debug-dir` | `str` | 空字符串 | 若非空，则保存二值图、字符小图和识别框可视化结果。 |
| `--no-keep-main-row` | flag | 默认不开启 | 关闭主算式行筛选。若输入图像已经严格裁成单行算式，可开启；若图像里有任务栏、边框等干扰，建议不要开启。 |

> 注意：当前版本脚本中已经定义了 `--no-keep-main-row` 参数，但需要确认 `segment_characters()` 内部实际调用了 `keep_main_text_row()`。如果尚未调用，该参数不会产生效果。

### 3.5 DBSCAN 参数调试建议

字符被分成多个小框时，例如 `8`、`×` 被拆开：

```bash
python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt --eps 4 --debug-dir debug
```

相邻字符被粘成一个框时，例如 `20` 被合成一个整体：

```bash
python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt --eps 2 --debug-dir debug
```

`÷` 的上下点丢失时：

```bash
python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt --min-samples 3 --debug-dir debug
```

同时调节：

```bash
python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt --eps 4 --min-samples 3 --debug-dir debug
```

---

## 4. 一套完整推荐流程

### 4.1 生成数据集

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --samples-per-class 3000 --img-size 28 --fonts "C:/Windows/Fonts/times.ttf"
```

如果不是 Windows，或者不确定字体路径，可以先不指定 `--fonts`：

```bash
python 01_generate_font_dataset.py --out-dir dataset_symbols --samples-per-class 3000 --img-size 28
```

### 4.2 训练 CNN

```bash
python 02_train_symbol_cnn.py --data-dir dataset_symbols --out-path symbol_cnn.pt --epochs 12 --batch-size 128 --lr 1e-3
```

### 4.3 识别 LCD 算式

```bash
python 03_lcd_expression_pipeline.py --image 001460_screen_warped.png --model symbol_cnn.pt --debug-dir debug
```

如果分割效果不好，再优先调整：

```bash
--eps
--min-samples
```

---

## 5. 输出结果说明

运行 `03_lcd_expression_pipeline.py` 后，终端通常会输出类似内容：

```text
raw_expr = 20÷20×1=
expr     = 20/20*1
value    = 1
```

含义：

| 字段 | 含义 |
|---|---|
| `raw_expr` | CNN 识别出的原始表达式，保留 `×`、`÷`、`=`。 |
| `expr` | 转换后的 Python 表达式，例如 `× -> *`，`÷ -> /`。 |
| `value` | 安全计算后的结果。 |


