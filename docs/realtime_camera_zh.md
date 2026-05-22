# LivePortrait 实时摄像头驱动

这个仓库新增了 `realtime_camera.py`，用于把摄像头画面作为 driving 输入，实时驱动一张源人像的头部表情和姿态。

适用范围：

- 头部/脸部表情、眨眼、张嘴、轻微转头。
- 源角色建议使用你有权使用的 AI 生成女性角色图，或获得明确授权的人像照片。
- 只做头部，不捕捉手部、身体、商品动作。如果直播需要拿货、展示商品，需要额外的全身/手部方案。

## 环境

macOS Apple Silicon 按官方说明安装：

```bash
conda create -n LivePortrait python=3.10
conda activate LivePortrait
pip install -r requirements_macOS.txt
```

下载权重：

```bash
huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

如果 Hugging Face 访问不稳定：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

可选：如果要把生成结果作为虚拟摄像头输出给 OBS/直播软件：

```bash
pip install pyvirtualcam
```

## iPhone 作为 macOS 摄像头

先在 macOS 打开连续互通相机（Continuity Camera），让 iPhone 靠近 Mac 并保持同一 Apple ID、蓝牙和 Wi-Fi 开启。然后在项目目录列出 OpenCV 可读摄像头：

```bash
python realtime_camera.py --list-cameras
```

记下可读的索引，例如 `0` 或 `1`。

如果 macOS 没有弹出摄像头权限请求，主动打开一个摄像头预览：

```bash
python realtime_camera.py --test-camera --camera 0
```

如果仍然没有画面，到 `系统设置 -> 隐私与安全性 -> 相机`，允许你当前运行命令的 App（Terminal、iTerm 或 Codex）访问摄像头，然后重新运行上面的命令。

如果你使用的是 iPhone Continuity Camera，再看 AVFoundation 的底层设备列表：

```bash
python realtime_camera.py --list-avfoundation
```

如果输出里出现 `NSCameraUseContinuityCameraDeviceType`，说明 macOS 要求宿主 App 声明 Continuity Camera 权限。Terminal 直接启动的 Python/OpenCV 可能看不到 iPhone 相机。更稳定的方案：

- 使用普通 USB/UVC 摄像头。
- 使用 Camo、EpocCam、Iriun 等把 iPhone 暴露成普通虚拟摄像头。
- 使用 OBS 接入 iPhone/Continuity Camera，再开 OBS Virtual Camera，让本脚本读取 OBS 虚拟摄像头。
- 使用 iPhone RTSP/MJPEG 摄像头 App，把流地址传给 `--camera`，例如 `--camera http://手机IP:端口/video`。

也可以尝试创建一个带 Continuity Camera 声明的本地 `.app` 启动器。请在 `(LivePortrait)` 环境里运行：

```bash
python tools/create_macos_camera_app.py --camera 0
open dist/LivePortraitCamera.app
```

如果要直接启动完整实时驱动：

```bash
python tools/create_macos_camera_app.py \
  --source /path/to/female_character.jpg \
  --camera 0
open dist/LivePortraitCamera.app
```

运行日志在 `logs/LivePortraitCamera.log`。如果 macOS 阻止打开：

```bash
xattr -dr com.apple.quarantine dist/LivePortraitCamera.app
open dist/LivePortraitCamera.app
```

## 运行

基础运行：

```bash
python realtime_camera.py \
  --source /path/to/female_character.jpg \
  --camera 0
```

启动后保持第一帧为正脸、中性表情；这帧会作为动作基准。按键：

- `q` 退出。
- `r` 重置中性基准。嘴巴或头部已经动过时，先恢复中性正脸再按。

如果你想只看生成头像：

```bash
python realtime_camera.py \
  --source /path/to/female_character.jpg \
  --camera 0 \
  --display generated
```

如果要录制生成视频：

```bash
python realtime_camera.py \
  --source /path/to/female_character.jpg \
  --camera 0 \
  --output animations/realtime_test.mp4
```

如果要输出到虚拟摄像头：

```bash
python realtime_camera.py \
  --source /path/to/female_character.jpg \
  --camera 0 \
  --display generated \
  --virtual-camera
```

## 性能预期

官方 PyTorch 版在 macOS Apple Silicon 上可以跑人类模型，但不保证直播级 25/30 FPS。它更适合先验证效果、调源图和动作基准。

如果最终要稳定直播：

- Apple Silicon：先用当前脚本验证链路，降低摄像头分辨率、关闭 `--pasteback`，只输出 `--display generated`。
- NVIDIA GPU：更适合实时输出；后续可以把同样的摄像头裁剪和动作基准逻辑迁移到 TensorRT/FasterLivePortrait 一类实时实现。

低延迟优先可以先试：

```bash
python realtime_camera.py \
  --source personal.jpg \
  --camera 0 \
  --display generated \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --target-fps 15 \
  --driving-crop-mode static \
  --redetect-interval 240 \
  --motion-smoothing 0.15
```

如果通过 `.app` 启动器运行：

```bash
python tools/create_macos_camera_app.py \
  --source personal.jpg \
  --camera 0 \
  --fast \
  --target-fps 15
open dist/LivePortraitCamera.app
```

可以把 `--target-fps 15` 换成 `12`、`18`、`24` 等自己测试。实际 FPS 达不到目标时，说明当前机器/模型推理速度不够，需要继续降分辨率、用 `static/center` 裁剪，或换 TensorRT/FasterLivePortrait 方案。

`--driving-crop-mode static` 只用第一帧定位脸，后面复用裁剪框；启动后请保持正脸、中性表情，画面跑起来后按 `r` 可重新校准。若脸移出裁剪框，改回默认 `landmark`，或使用 `--driving-crop-mode center --center-crop-ratio 0.7` 并让脸保持在画面中心。

## 常用调参

- `--driving-scale 2.0` 到 `2.6`：调摄像头头部裁剪范围。
- `--source-scale 2.0` 到 `2.8`：调源图裁剪范围。
- `--crop-smoothing 0.65`：裁剪框平滑，值越大越稳但延迟越大。
- `--motion-smoothing 0.2`：动作关键点平滑，能减少抖动但会牺牲同步性。
- `--driving-multiplier 0.8` 到 `1.2`：降低或增强表情幅度。
- `--mirror-input`：镜像摄像头输入，会改变驱动方向。
- `--mirror-output`：镜像最终生成画面，只改变直播/预览方向。
- `--normalize-lip`：源图嘴巴微张时可尝试启用。

## 源照片建议

官方 LivePortrait 是单源图驱动，不会因为提供多张照片自动变得更准。优先选一张：

- 正脸或轻微侧脸。
- 清晰、高分辨率、无遮挡。
- 中性表情，嘴巴自然闭合。
- 头发不要遮挡眼睛和嘴巴。

多张同一个人的照片可以用于人工挑选最稳定的一张；如果要真正融合多图身份，需要另做身份建模或训练，不属于当前脚本范围。
