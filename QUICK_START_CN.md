# Ultralytics YOLO 快速开始 - 运行自己的数据集

## 快速上手（5 分钟）

### 1. 安装

```bash
pip install ultralytics
```

### 2. 准备数据集

创建以下目录结构：

```
my_dataset/
├── images/
│   ├── train/       # 训练图像
│   └── val/         # 验证图像
└── labels/
    ├── train/       # 训练标注
    └── val/         # 验证标注
```

**标注格式**（每个图像一个 .txt 文件）：

```
0 0.5 0.5 0.3 0.4
```

格式：`类别ID 中心x 中心y 宽度 高度`（所有值归一化到 0-1）

### 3. 创建配置文件

创建 `my_dataset.yaml`：

```yaml
path: /path/to/my_dataset
train: images/train
val: images/val

names:
  0: 类别1
  1: 类别2
  2: 类别3
```

### 4. 开始训练

**Python 方式：**

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 训练
model.train(data="my_dataset.yaml", epochs=100, imgsz=640)
```

**命令行方式：**

```bash
yolo detect train data=my_dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```

### 5. 使用模型预测

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 预测
results = model("image.jpg")
results[0].show()
```

## 常用模型选择

| 模型 | 大小 | 速度 | 精度 | 用途 |
|------|------|------|------|------|
| yolo11n.pt | 最小 | 最快 | 一般 | 实时应用 |
| yolo11s.pt | 小 | 快 | 中等 | 边缘设备 |
| yolo11m.pt | 中 | 中等 | 好 | 通用场景 |
| yolo11l.pt | 大 | 慢 | 很好 | 高精度需求 |
| yolo11x.pt | 最大 | 最慢 | 最好 | 最高精度 |

## 重要参数

```python
model.train(
    data="my_dataset.yaml",  # 数据集配置
    epochs=100,              # 训练轮数
    imgsz=640,               # 图像尺寸
    batch=16,                # 批次大小（根据显存调整）
    device=0,                # GPU 编号（cpu 或 0,1,2...）
    patience=50,             # 早停等待轮数
    lr0=0.01,               # 初始学习率
)
```

## 不同任务类型

### 目标检测（默认）

```bash
yolo detect train data=my_dataset.yaml model=yolo11n.pt
```

### 实例分割

```bash
yolo segment train data=my_dataset.yaml model=yolo11n-seg.pt
```

### 图像分类

```bash
yolo classify train data=my_dataset.yaml model=yolo11n-cls.pt
```

### 姿态估计

```bash
yolo pose train data=my_dataset.yaml model=yolo11n-pose.pt
```

## 常见问题快速解决

**Q: 显存不足？**
```python
# 减小 batch 或 imgsz
model.train(data="my_dataset.yaml", batch=8, imgsz=416)
```

**Q: 训练太慢？**
```python
# 使用更小的模型或多GPU
model.train(data="my_dataset.yaml", model="yolo11n.pt", device=[0,1])
```

**Q: 精度不够？**
```python
# 使用更大的模型，增加训练轮数
model.train(data="my_dataset.yaml", model="yolo11l.pt", epochs=300)
```

**Q: 如何恢复训练？**
```python
model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True)
```

## 查看训练结果

训练完成后，结果保存在 `runs/detect/train/`：

- `weights/best.pt` - 最佳模型
- `weights/last.pt` - 最后一轮模型
- `results.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵

## 完整示例

```python
from ultralytics import YOLO

# 训练
model = YOLO("yolo11n.pt")
results = model.train(
    data="my_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="my_exp"
)

# 验证
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")

# 预测
results = model("test.jpg")
results[0].show()

# 导出
model.export(format="onnx")
```

## 获取帮助

- 📖 [完整文档](https://docs.ultralytics.com/)
- 💬 [Discord 社区](https://discord.com/invite/ultralytics)
- 🐛 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- 📺 [YouTube 教程](https://www.youtube.com/@Ultralytics)

---

**提示**: 查看[完整训练指南](./自定义数据集训练指南.md)了解更多详细信息。
