# YOLO 牛奶盒检测：训练 + C++ 实时推理部署

这是我拿来存 YOLO 摄像头实时检测训练数据的仓库。正好从家里带来了纯牛奶，就拿牛奶盒来训练——牛奶盒有两种（`RYmilk` 认养一头牛 / `MNmilk`蒙牛），目标是让模型在摄像头下把它们区分出来。

完整链路：**数据采集 → 标注 → 训练（ultralytics） → 导出 ONNX → C++ 实时推理部署**。

---

## 一、环境

| 组件 | 版本 / 位置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop（8GB，驱动 610.88 / CUDA 13.3） |
| Python | 3.11.17（miniforge conda 环境 `yolo`，位于 `C:\Users\28687\miniforge3\envs\yolo`） |
| PyTorch | 2.14.1+cu130 |
| ultralytics | 8.4.174 |
| C++ 编译器 | Visual Studio 2026（MSVC 14.51，装在 `D:\vs`） |
| OpenCV | 4.13.0（预编译 vc16，位于 `D:\opencv\opencv\build`） |
| ONNX Runtime | 1.30.0 |

## 二、用到的开源库（谁家的、什么协议）

| 环节 | 开源库 | 作者 / 来源 | 协议 |
|---|---|---|---|
| 数据标注 | **X-AnyLabeling** | CVHub520 | GPL-3.0 |
| 模型训练 | **ultralytics**（底层 **PyTorch**） | Ultralytics / Meta | AGPL-3.0 / BSD-3 |
| 预训练权重 | **yolov8n.pt**、**yolo26n.pt** | Ultralytics 官方发布 | AGPL-3.0 |
| 数据脚本模板 | `yolo_train` | Linmoqian（CSDN 教程配套仓库） | — |
| 推理引擎 | **ONNX Runtime** | Microsoft | MIT |
| 图像处理 / 摄像头 | **OpenCV** | OpenCV 团队 | Apache-2.0 |
| C++ 工程骨架 | **ultralytics/examples/cpp** | Ultralytics 官方示例 | AGPL-3.0 |

> 说明：整条链路没有自研算法，仅在官方示例的 `main.cpp` 上增加了摄像头/视频实时循环与 FPS 统计（约 20 行胶水代码）。

## 三、训练

### 1. 数据准备

1. 拍摄牛奶盒照片（本项目 30 张，尺寸 1280×1707，每图 1 个框）
2. X-AnyLabeling 标注，类别 `RYmilk` / `MNmilk`
3. 转换与划分：

```bash
python script/json2txt.py     # JSON 标注 → YOLO txt
python script/DataProess.py   # 划分 train/val
```

划分结果：**train 21 张（RYmilk 10 / MNmilk 11）、val 9 张（RYmilk 5 / MNmilk 4）**

### 2. 配置

`config/dataset.yaml`：`nc: 2`，`names: 0: RYmilk, 1: MNmilk`

`config/train.yaml`：`epochs: 200`、`imgsz: 640`、`batch: 16`、`name: milk_v1`

### 3. 训练

```bash
conda activate yolo
cd C:\Users\28687\Desktop\X\yolo_train
python main.py
```

### 4. 训练结果（milk_v1，imgsz 1280，yolov8n 底模）

| 指标 | 数值 |
|---|---|
| **mAP50** | **0.953** |
| mAP50-95 | 0.632 |
| Precision | 0.792 |
| Recall | 0.967 |
| RYmilk AP50 | 0.962 |
| MNmilk AP50 | 0.945 |

训练耗时 6.4 分钟，显存占用 2.42G。

### 5. 结果的可信度说明（重要）

上述指标**偏乐观**：验证集仅 9 张，且与训练图同批拍摄（相同角度/背景/光线）。
真实摄像头测试暴露出问题：**只有特定角度能稳定识别**，说明模型过拟合到拍摄时的角度分布。
改进方向见"六、待办"。

## 四、推理部署

### 方案 A：Python 快速版

```bash
python realitime.py    # 摄像头实时检测，按 q 退出
```

### 方案 B：C++ 部署（ONNX Runtime，性能版）

工程位置：`yolo_cpp/`

```bash
# 1. 导出 ONNX（与训练分辨率保持一致）
python -c "from ultralytics import YOLO; YOLO('runs/train/milk_v1/weights/best.pt').export(format='onnx', imgsz=640, dynamic=False)"

# 2. 拷进部署工程
cp runs/train/milk_v1/weights/best.onnx yolo_cpp/models/milk_640.onnx

# 3. 编译（普通终端里双击 build.bat，或手动）
cd yolo_cpp && build.bat

# 4. 运行
run_camera_milk_640.bat     # 摄像头实时，按 q 退出
```

**换模型不需要重新编译**，只改 `--model` 指向的 onnx 即可：

```bash
yolo_ort.exe --model models\milk_640.onnx --source 0 --show --conf 0.3
```

### 实测性能（CPU 推理，RTX 5060 笔记本）

| 模型 / 后端 | 输入尺寸 | 帧率 |
|---|---|---|
| 官方 yolo26n（ONNX Runtime） | 640 | 18.3 FPS |
| 牛奶盒模型（ONNX Runtime） | 640 | 6.3 FPS |
| 牛奶盒模型（ONNX Runtime） | 1280 | ~2 FPS |
| 官方 yolo26n（OpenCV DNN，已删除） | 640 | 8.4 FPS |

> ONNX Runtime 比 OpenCV DNN 快约 2.2 倍；未启用 CUDA（本机无 CUDA Toolkit）。

### 实测效果（C++ 部署 / 摄像头实时）

![摄像头实时识别 RYmilk](docs/demo_rymilk.jpg)

画面信息：左上角 HUD 显示 **FPS 6.4、objects 1**，蓝框为模型输出 —— **RYmilk 置信度 0.76**（640 版本模型）。
框与 HUD 均由 `yolo_cpp/main.cpp` 绘制，便于实时观察帧率与检测数量。

## 五、遇到的问题与解决方式

| # | 问题 | 原因 | 解决 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'ultralytics'` | 用系统 Python 3.12 运行（包只装在 conda `yolo` 环境） | 用 `miniforge3\envs\yolo\python.exe`，或 VS Code 选对解释器 |
| 2 | 找不到 `config/dataset.yaml` | 脚本用相对路径，工作目录不对 | `main.py` 改用 `Path(__file__)` 推导绝对路径 |
| 3 | `No images found in dataset/images/train` | 数据未转换划分 | 先跑 `json2txt.py` + `DataProess.py` |
| 4 | DataLoader 子进程递归启动训练并崩溃 | Windows 用 spawn，子进程会重跑顶层代码 | 训练逻辑包进 `main()`，加 `if __name__ == '__main__':` |
| 5 | RTX 50 系显卡 torch 报 `no kernel image` | Blackwell（sm_120）需 CUDA 12.8+ 的 torch | 装 `torch 2.14.1+cu130` |
| 6 | `cv2.imshow` 报 "not implemented" | 同时装了 `opencv-python` 与 headless 版，后者覆盖了 `cv2` | 卸载 headless 变体，重装完整版 |
| 7 | ONNX Runtime 下载包损坏（无中央目录） | GitHub 直连下载被截断 | 换镜像下载并校验 zip 完整性 |
| 8 | CMake 找不到 MSVC 编译器 | VS 装在 D 盘非默认位置，且环境探测被限制 | 直接调用 `cl.exe` 编译（并保留 `build.bat` 走标准 vcvars 流程） |
| 9 | 摄像头实时推理卡顿 | 1280 输入算力是 640 的 4 倍，而摄像头只有 640×480 | 导出 640 版本模型；后续计划换更快底模 |
| 10 | 训练指标高但真实场景只有特定角度能识别 | 数据量少且角度/背景/光照单一，模型过拟合 | 待办：数据增强（`degrees`/`scale`/`shear`/`perspective`）+ 补拍多角度多背景数据 |

## 六、待办

- [ ] 补数据：每类 40~60 张，覆盖多角度、多距离、多光照、**5 个以上不同背景**，并加入负样本
- [ ] 开启更强数据增强后重训
- [ ] 换 `yolo26n` 底模 @640 重训（预期推理速度提升约 3 倍）
- [ ] Web 端接入（FastAPI + WebSocket）

## 七、目录结构

```
yolo_train/
├── config/            # dataset.yaml（类别/路径）、train.yaml（超参）
├── script/            # json2txt.py、DataProess.py
├── dataset/           # 划分后的训练集/验证集（images + labels）
├── yolo_milkdata/     # 原始素材：30 张图 + 30 个 json 标注
├── logs/              # 工程日志
├── runs/train/milk_v1/  # 训练产物：权重、曲线、混淆矩阵
├── yolo_cpp/          # C++ 实时推理工程（ONNX Runtime）
│   ├── main.cpp / inference.cpp / common/   # 基于官方示例改造
│   ├── models/        # milk_640.onnx、milk_best.onnx、yolo26n.onnx
│   ├── yolo_ort.exe   # 编译产物
│   └── run_camera*.bat # 一键运行入口
├── weights/           # 预训练底模（yolo26n.pt、yolov8n.pt）
└── main.py            # 训练入口
```
