# LivePortrait 实时摄像头一键启动

本分支新增了 `realtime_camera.py`，可用摄像头实时驱动一张源人像照片。

## 一行命令

克隆仓库后，在项目目录运行：

```bash
bash scripts/run_realtime_macos.sh --source /path/to/portrait.jpg --camera 0 --target-fps 15 --mirror-output
```

脚本会自动：

- 创建或复用 `LivePortrait` conda 环境。
- 安装 macOS 依赖。
- 下载官方 LivePortrait 权重到 `pretrained_weights/`。
- 以低延迟预设启动实时驱动。

如果你使用 iPhone Continuity Camera，并且 Terminal 没有摄像头权限，可改用 `.app` 启动器：

```bash
bash scripts/run_realtime_macos.sh --source /path/to/portrait.jpg --camera 0 --target-fps 15 --mirror-output --app
```

## 常用参数

- `--camera 0`：主摄像头索引。桌上视角或虚拟摄像头可能是 `1`、`2`。
- `--target-fps 12|15|18|24`：目标处理帧率。机器算不过来时，实际 FPS 会低于目标。
- `--mirror-output`：镜像最终直播画面。
- `--mirror-input`：镜像摄像头输入，会影响动作方向。
- `--no-fast`：关闭低延迟预设，使用默认 landmark 跟踪。

## 摄像头检查

```bash
conda activate LivePortrait
python realtime_camera.py --list-avfoundation
python realtime_camera.py --test-camera --camera 0
```

启动后第一帧请保持正脸、中性表情。按 `r` 重新校准，按 `q` 退出。

## 不上传的内容

`personal.jpg`、`pretrained_weights/`、`dist/`、`logs/` 都被 `.gitignore` 排除。请不要把个人照片或模型权重提交到 GitHub。
