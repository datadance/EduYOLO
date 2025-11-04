from datetime import datetime
import os
from pathlib import Path
import sys
sys.path.insert(0, r"")#文件根路径
import torch
import random
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")
# 0. 彻底禁用所有下载
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.environ["GITHUB_ASSETS"] = "none"


from ultralytics.utils import downloads

original_attempt_download = downloads.attempt_download_asset

def patched_attempt_download(*args, **kwargs):
    """完全禁用下载功能"""
    file = args[0] if args else kwargs.get("file")
    if str(file).startswith(("http://", "https://")):
        raise RuntimeError(f"自动下载已被禁用！阻止了URL: {file}")
    return original_attempt_download(*args, **kwargs)

downloads.attempt_download = patched_attempt_download

model = YOLO(r"")#模型配置文件


params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {params/1e6:.1f}M (11m规模应在20M左右，v8m的规模为25.9M)")

# 1. 设定模型结构路径
model_path = r""#模型结构路径
model_name = Path(model_path).stem  # 提取模型名，例如：yolo1v11m

# 1. 构造时间戳和日志目录名称
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
batch_size = 8
lr0 = 0.01

run_name = f"{model_name}_bs{batch_size}_lr{lr0}_{timestamp}"

log_dir = Path(f"/root/run/{run_name}")
log_dir.mkdir(parents=True, exist_ok=True)

# 2. 写入训练日志文件（手动记录信息）
log_path = log_dir / "train_info.txt"
with open(log_path, "w") as f:
    f.write(f"训练时间: {timestamp}\n")
    f.write(f"模型结构文件: \n")
    f.write(f"模型参数量: {params / 1e6:.2f} M\n")
    f.write(f"数据集配置: \n")
    f.write(f"训练轮数: 150\n")
    f.write(f"Batch size: 8\n")
    f.write(f"学习率 lr0: 0.01\n")
    f.write(f"使用设备: 0\n")
    f.write(f"优化器: AdamW\n")
    f.write(f"预训练权重: False\n")
    f.write(f"混合精度 AMP: False\n")
    f.write(f"保存周期: 每 {20} epoch\n")

# 3. 启动训练
model.train(
    data=r"",#数据集配置路径
    pretrained=False,
    resume=False,
    epochs=150,
    seed=42,
    batch=batch_size,
    lr0=lr0,
    imgsz=640,
    device="0",
    project=r"",#日志保存路径
    name=run_name,
    optimizer="AdamW",
    save_period=50,
    amp=False,
    workers=4,
)
