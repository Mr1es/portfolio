# YOLO C++ 实时推理（Windows）

当前保留 **ONNX Runtime** 一套后端，基于 Ultralytics 官方示例改造。
OpenCV DNN 后端曾作为对照实现并实测过（数据见下表），工程已删除。

## 来源与许可证

- 官方仓库：https://github.com/ultralytics/ultralytics（`examples/cpp/ONNXRuntime`、`examples/cpp/OpenCV-DNN`）
- ONNX Runtime：MIT；OpenCV：Apache-2.0；Ultralytics：AGPL-3.0（商用需注意）

**改造点只有两处**（其余 letterbox / 解码 / NMS / 画框全部复用官方实现）：

1. `main.cpp`：官方只支持单张图片，这里增加了摄像头 / 视频实时循环 + FPS 统计
2. `CMakeLists.txt`：修正 `common/` 头文件路径；`USE_CUDA` 默认改为 OFF（本机无 CUDA Toolkit）

## 本机实测性能（RTX 5060 笔记本 / CPU 推理 / yolo26n 640×640 / 摄像头实时）

| 后端 | 帧率 | 说明 |
|---|---|---|
| **ONNX Runtime 1.30.0（CPU）** | **18.3 FPS** | 推荐，比 OpenCV DNN 快约 2.2 倍 |
| OpenCV DNN 4.13（CPU） | 8.4 FPS | 对照实测（工程已删除），零额外依赖但慢 2.2 倍 |

> 注：未启用 CUDA（本机只有显卡驱动、无 CUDA Toolkit），若要 GPU 推理需装 CUDA + cuDNN 并换 `onnxruntime-win-x64-gpu-cuda12` 包。

## 依赖与本机位置

| 组件 | 位置 |
|---|---|
| Visual Studio 2026（MSVC 14.51） | `D:\vs` |
| OpenCV 4.13.0（预编译 vc16） | `D:\opencv\opencv\build` |
| ONNX Runtime 1.30.0（win-x64） | `D:\onnxruntime\onnxruntime-win-x64-1.30.0` |

## 构建

```bat
cd C:\Users\28687\Desktop\X\yolo_train\yolo_cpp
build.bat          :: 在普通终端运行（内部调用 vcvars64.bat 配好 MSVC 环境）
```

`build.bat` 内部等价于直接调用 `cl.exe`；用 CMake 也可以（普通终端里）：

```bat
cmake -B build -G "Visual Studio 18 2026" -A x64 ^
  -DOpenCV_DIR=D:\opencv\opencv\build ^
  -DONNXRUNTIME_ROOT=D:\onnxruntime\onnxruntime-win-x64-1.30.0 -DUSE_CUDA=OFF
cmake --build build --config Release
```

## 运行

```bat
run_camera.bat                                                     :: 摄像头实时（按 q 退出）
yolo_ort.exe --model models\yolo26n.onnx --source test.mp4 --show  :: 视频文件
yolo_ort.exe --model models\yolo26n.onnx --source bus.jpg --show   :: 单张图片
```

参数：`--conf 0.25` 置信度阈值、`--iou 0.45` NMS 阈值、`--cuda`（需 CUDA 版 ORT）。

## 换成自己训练的模型（如 milk_box）

```bash
C:/Users/28687/miniforge3/envs/yolo/python.exe -c "from ultralytics import YOLO; YOLO('runs/train/milk_v1/weights/best.pt').export(format='onnx', imgsz=640, dynamic=False)"
```

把导出的 onnx 放到 `models/`，运行时 `--model models\best.onnx` 即可。
