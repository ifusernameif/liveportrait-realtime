# macOS M4 实时性能说明

本项目的实时版本已经针对 Mac mini M4 做了可落地优化，但需要明确一个边界：当前实现仍然基于官方 LivePortrait PyTorch 网络，不是重写后的 Metal/CoreML 原生推理引擎。

## 已做的 macOS 优化

- 使用 PyTorch MPS 后端运行神经网络推理，也就是通过 Apple Metal 加速。
- 增加 `--macos-preset quality|m4-fast|m4-max`。
- 增加异步摄像头采集，只处理最新帧，避免推理慢时堆积旧帧。
- 增加 MPS 预热 `--mps-warmup`。
- 增加 `static` / `center` 裁剪模式，减少每帧 CPU landmark 跟踪开销。
- 增加 `--max-frames`，可自动跑固定帧数并输出平均 FPS。

## 为什么 30 FPS 不能保证

本地运行时可以看到 PyTorch 的提示：

```text
The operator 'aten::grid_sampler_3d' is not currently supported on the MPS backend and will fall back to run on the CPU.
```

LivePortrait 的 warping module 依赖 3D grid sampling。只要这个核心算子回退到 CPU，就会发生 GPU/CPU 同步和数据搬运，M4 的 Metal GPU 无法全程吃满。因此，完整官方 PyTorch 网络在 macOS MPS 上很难稳定达到 30 FPS。

## 当前建议命令

优先试 M4 快速模式：

```bash
python realtime_camera.py \
  --source personal.jpg \
  --camera 0 \
  --display generated \
  --target-fps 15 \
  --macos-preset m4-fast \
  --mirror-output
```

冲击 30 FPS：

```bash
python realtime_camera.py \
  --source personal.jpg \
  --camera 0 \
  --display generated \
  --target-fps 30 \
  --macos-preset m4-max \
  --mirror-output
```

自动跑 60 帧 benchmark：

```bash
python realtime_camera.py \
  --source personal.jpg \
  --camera 0 \
  --display generated \
  --target-fps 30 \
  --macos-preset m4-max \
  --no-window \
  --max-frames 60
```

如果 Terminal 没有 Continuity Camera 权限，使用 `.app` 启动器：

```bash
python tools/create_macos_camera_app.py \
  --source personal.jpg \
  --camera 0 \
  --fast \
  --macos-preset m4-max \
  --target-fps 30 \
  --mirror-output
open dist/LivePortraitCamera.app
```

## 真正稳定 30 FPS 的工程路线

要把完整 LivePortrait 做到 macOS 原生实时 30 FPS，后续需要做其中一种更深的推理引擎改造：

- 为 LivePortrait warping module 的 3D grid sampling 写自定义 Metal kernel。
- 把 motion extractor、warping module、SPADE generator 分段转换到 CoreML / MPSGraph，并验证动态 shape、grid sampling、精度和内存拷贝。
- 做模型蒸馏/裁剪/量化，降低 generator 和 warping module 的计算量。
- 牺牲画质，改用更轻的实时人脸驱动模型，而不是完整官方 LivePortrait 网络。

当前仓库的目标是：在不重写神经网络算子的前提下，提供 macOS M4 上尽可能低延迟、可用、可调参的实时 LivePortrait 方案。
