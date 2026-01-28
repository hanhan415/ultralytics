"""
创建一个示例 YOLO 数据集的脚本

此脚本演示如何创建一个简单的 YOLO 格式数据集，用于训练自己的模型。
您可以根据需要修改此脚本来适配您的数据。
"""

import os
import shutil
from pathlib import Path


def create_sample_dataset(dataset_path="my_dataset"):
    """
    创建一个示例 YOLO 数据集结构
    
    Args:
        dataset_path: 数据集保存路径
    """
    
    # 创建目录结构
    base_path = Path(dataset_path)
    
    # 创建目录
    dirs = [
        base_path / "images" / "train",
        base_path / "images" / "val",
        base_path / "labels" / "train",
        base_path / "labels" / "val",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 创建示例标注文件
    # 格式: class_id x_center y_center width height (归一化坐标，0-1)
    
    # 训练集示例标注
    train_annotations = [
        "0 0.5 0.5 0.3 0.4\n1 0.7 0.3 0.2 0.2\n",  # image1.txt - 2个对象
        "0 0.3 0.6 0.25 0.35\n",  # image2.txt - 1个对象
        "1 0.4 0.4 0.3 0.3\n0 0.8 0.8 0.15 0.15\n",  # image3.txt - 2个对象
    ]
    
    # 验证集示例标注
    val_annotations = [
        "0 0.6 0.5 0.3 0.4\n",  # image1.txt - 1个对象
        "1 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.15 0.15\n",  # image2.txt - 2个对象
    ]
    
    # 写入训练集标注
    for i, annotation in enumerate(train_annotations, 1):
        label_file = base_path / "labels" / "train" / f"image{i}.txt"
        with open(label_file, "w") as f:
            f.write(annotation)
        print(f"创建标注文件: {label_file}")
    
    # 写入验证集标注
    for i, annotation in enumerate(val_annotations, 1):
        label_file = base_path / "labels" / "val" / f"image{i}.txt"
        with open(label_file, "w") as f:
            f.write(annotation)
        print(f"创建标注文件: {label_file}")
    
    # 创建 YAML 配置文件
    yaml_content = f"""# 数据集配置文件
# 路径可以是绝对路径或相对路径

path: {dataset_path}  # 数据集根目录
train: images/train   # 训练图像目录 (相对于 path)
val: images/val       # 验证图像目录 (相对于 path)
test:                 # 测试图像目录 (可选)

# 类别定义
names:
  0: person  # 类别 0
  1: car     # 类别 1

# 类别数量 (可选，会自动计算)
nc: 2
"""
    
    yaml_file = base_path / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n创建配置文件: {yaml_file}")
    
    # 创建 README 文件
    readme_content = f"""# 示例数据集

此数据集是使用 `创建示例数据集.py` 脚本创建的示例数据集。

## 目录结构

```
{dataset_path}/
├── dataset.yaml          # 数据集配置文件
├── images/
│   ├── train/           # 训练图像 (需要您添加实际图像)
│   └── val/             # 验证图像 (需要您添加实际图像)
└── labels/
    ├── train/           # 训练标注文件 (已创建示例)
    └── val/             # 验证标注文件 (已创建示例)
```

## 下一步

1. **添加图像**: 将您的图像放入 `images/train/` 和 `images/val/` 目录
   - 确保图像文件名与标注文件名匹配（除了扩展名）
   - 例如: `image1.jpg` 对应 `image1.txt`

2. **修改标注**: 根据您的实际数据修改 `labels/` 目录下的标注文件
   - 格式: `class_id x_center y_center width height`
   - 所有坐标都应该归一化到 0-1 范围

3. **更新配置**: 编辑 `dataset.yaml` 文件
   - 修改类别名称以匹配您的数据
   - 确保路径正确

4. **开始训练**:
   ```python
   from ultralytics import YOLO
   
   model = YOLO("yolo11n.pt")
   results = model.train(data="{dataset_path}/dataset.yaml", epochs=100)
   ```

## 标注格式说明

每个 `.txt` 文件包含一个或多个对象的标注，每行一个对象：

```
class_id x_center y_center width height
```

- `class_id`: 类别索引 (从 0 开始)
- `x_center`: 边界框中心 x 坐标 (归一化，0-1)
- `y_center`: 边界框中心 y 坐标 (归一化，0-1)
- `width`: 边界框宽度 (归一化，0-1)
- `height`: 边界框高度 (归一化，0-1)

### 坐标归一化示例

如果图像尺寸为 1000x800 像素，对象边界框为:
- 左上角: (200, 100)
- 右下角: (500, 400)

则归一化坐标为:
- x_center = (200 + 500) / 2 / 1000 = 0.35
- y_center = (100 + 400) / 2 / 800 = 0.3125
- width = (500 - 200) / 1000 = 0.3
- height = (400 - 100) / 800 = 0.375

标注行: `0 0.35 0.3125 0.3 0.375`

## 工具推荐

标注工具:
- [LabelImg](https://github.com/tzutalin/labelImg) - 支持 YOLO 格式
- [Roboflow](https://roboflow.com/) - 在线标注和数据集管理
- [CVAT](https://cvat.org/) - 开源标注工具
- [Labelbox](https://labelbox.com/) - 企业级标注平台

## 参考

- [YOLO 数据集格式文档](https://docs.ultralytics.com/datasets/detect/)
- [训练自定义数据集教程](https://docs.ultralytics.com/modes/train/)
"""
    
    readme_file = base_path / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"创建说明文件: {readme_file}")
    
    print(f"\n✅ 数据集结构创建完成！")
    print(f"\n📋 下一步:")
    print(f"   1. 将图像添加到 {base_path}/images/train/ 和 {base_path}/images/val/")
    print(f"   2. 根据您的数据修改标注文件")
    print(f"   3. 更新 {yaml_file} 中的类别名称")
    print(f"   4. 开始训练您的模型！")
    print(f"\n📖 详细说明请查看: {readme_file}")


if __name__ == "__main__":
    import sys
    
    # 从命令行参数获取数据集路径，默认为 "my_dataset"
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "my_dataset"
    
    print("=" * 60)
    print("YOLO 数据集创建工具")
    print("=" * 60)
    print(f"\n正在创建数据集: {dataset_path}\n")
    
    create_sample_dataset(dataset_path)
    
    print("\n" + "=" * 60)
