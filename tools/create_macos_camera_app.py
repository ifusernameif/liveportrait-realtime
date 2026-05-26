# coding: utf-8

"""
Create a small macOS .app launcher with Continuity Camera entitlements declared
in Info.plist. Run this from the Python environment that should execute
realtime_camera.py.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shlex
import stat
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a macOS launcher app for realtime_camera.py.")
    parser.add_argument("--name", default="LivePortraitCamera", help="app name without .app")
    parser.add_argument("--camera", default="0", help="camera index or stream URL passed to realtime_camera.py")
    parser.add_argument("--source", help="optional source portrait path; omit to create a camera-test launcher")
    parser.add_argument("--display", choices=["split", "generated"], default="generated")
    parser.add_argument("--mirror-input", action="store_true")
    parser.add_argument("--mirror-output", action="store_true")
    parser.add_argument("--fast", action="store_true", help="use a lower-latency realtime preset")
    parser.add_argument("--macos-preset", choices=["custom", "quality", "m4-fast", "m4-max"], help="macOS/MPS realtime preset")
    parser.add_argument("--target-fps", type=float, help="target realtime processing FPS passed to realtime_camera.py")
    parser.add_argument("--extra-arg", action="append", default=[], help="extra argument passed through to realtime_camera.py")
    parser.add_argument("--output-dir", default="dist", help="directory where the .app bundle is created")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    app_root = repo_root / args.output_dir / f"{args.name}.app"
    contents = app_root / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_exe),
        str(repo_root / "realtime_camera.py"),
        "--camera",
        args.camera,
    ]
    if args.source:
        command.extend(["--source", args.source, "--display", args.display])
    else:
        command.append("--test-camera")
    if args.mirror_input:
        command.append("--mirror-input")
    if args.mirror_output:
        command.append("--mirror-output")
    if args.fast:
        command.extend(
            [
                "--macos-preset",
                args.macos_preset or "m4-fast",
            ]
        )
    elif args.macos_preset:
        command.extend(["--macos-preset", args.macos_preset])
    if args.target_fps:
        command.extend(["--target-fps", str(args.target_fps), "--camera-fps", str(int(args.target_fps))])
    command.extend(args.extra_arg)

    log_path = repo_root / "logs" / f"{args.name}.log"
    executable = macos / args.name
    executable.write_text(
        "#!/bin/zsh\n"
        "set -e\n"
        f"cd {shlex.quote(str(repo_root))}\n"
        "mkdir -p logs\n"
        f"export PATH={shlex.quote(str(python_exe.parent))}:$PATH\n"
        "export PYTORCH_ENABLE_MPS_FALLBACK=1\n"
        f"exec {' '.join(shlex.quote(part) for part in command)} >> {shlex.quote(str(log_path))} 2>&1\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    plist = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": args.name,
        "CFBundleExecutable": args.name,
        "CFBundleIdentifier": f"local.liveportrait.{args.name.lower()}",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": args.name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "13.0",
        "NSCameraUsageDescription": "LivePortrait uses the camera to drive the generated portrait in real time.",
        "NSDesktopFolderUsageDescription": "LivePortrait may read project files from user folders.",
        "NSDocumentsFolderUsageDescription": "LivePortrait reads the local project, model weights, and source portrait from Documents.",
        "NSDownloadsFolderUsageDescription": "LivePortrait may read camera or portrait assets from user folders.",
        "NSMicrophoneUsageDescription": "Microphone access is not required by LivePortrait but may be requested by AVFoundation devices.",
        "NSCameraUseContinuityCameraDeviceType": True,
    }
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump(plist, f)

    codesign = subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(app_root)],
        text=True,
        capture_output=True,
        check=False,
    )
    if codesign.returncode != 0:
        print("Warning: ad-hoc codesign failed:")
        print((codesign.stderr or codesign.stdout).strip())

    print(f"Created: {app_root}")
    print(f"Log file: {log_path}")
    print("Open it with:")
    print(f"  open {shlex.quote(str(app_root))}")
    print("If macOS blocks it, run:")
    print(f"  xattr -dr com.apple.quarantine {shlex.quote(str(app_root))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
